#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "bathymetry/bezier_basis_2d_base.hpp"
#include "bathymetry/biharmonic_assembler.hpp"
#include "core/scoped_timer.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/refine_mask.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace drifter {

// =============================================================================
// Constructors
// =============================================================================

AdaptiveCGCubicBezierSmoother::AdaptiveCGCubicBezierSmoother(
    Real xmin, Real xmax, Real ymin, Real ymax, int nx, int ny,
    const AdaptiveCGCubicBezierConfig &config)
    : config_(config) {
    // Create owned octree with domain bounds
    // Use z = [-1, 0] as a dummy vertical extent (only bottom face matters)
    octree_owned_ = std::make_unique<OctreeAdapter>(xmin, xmax, ymin, ymax, -1.0, 0.0);
    octree_owned_->build_uniform(nx, ny, 1); // 1 layer in z
    octree_ = octree_owned_.get();

    // Create quadtree from bottom face
    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);

    // Initialize Gauss quadrature (base class method)
    init_gauss_quadrature(config_.ngauss_error);
}

AdaptiveCGCubicBezierSmoother::AdaptiveCGCubicBezierSmoother(
    OctreeAdapter &octree, const AdaptiveCGCubicBezierConfig &config)
    : config_(config) {
    octree_ = &octree;

    // Create quadtree from bottom face of provided octree
    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);

    // Initialize Gauss quadrature (base class method)
    init_gauss_quadrature(config_.ngauss_error);
}

// Data input methods (set_bathymetry_data, set_land_mask) inherited from base

// =============================================================================
// Internal: Smoother management
// =============================================================================

void AdaptiveCGCubicBezierSmoother::rebuild_smoother() {
    // Invalidate error cache since mesh is changing
    invalidate_error_cache();

    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->quadtree_build_ms : nullptr);
        // Sync quadtree with refined octree
        quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);
    }

    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->smoother_init_ms : nullptr);
        // Create new CG cubic smoother for updated mesh
        smoother_ =
            std::make_unique<CGCubicBezierBathymetrySmoother>(*quadtree_, config_.smoother_config);
    }

    // Pass profile pointer so assembly sub-timings are recorded
    smoother_->set_profile(current_profile_);

    // Apply bathymetry data (assembly timings go into profile via inner smoother)
    apply_bathymetry_to_smoother();
}

void AdaptiveCGCubicBezierSmoother::apply_bathymetry_to_smoother() {
    if (!smoother_) {
        throw std::runtime_error("AdaptiveCGCubicBezierSmoother: smoother not initialized");
    }
    if (!bathy_func_) {
        throw std::runtime_error("AdaptiveCGCubicBezierSmoother: bathymetry data not set");
    }
    smoother_->set_bathymetry_data(bathy_func_);
}

const BezierBasis2DBase &AdaptiveCGCubicBezierSmoother::get_basis_impl() const {
    return smoother_->get_basis();
}

// =============================================================================
// Error estimation
// =============================================================================

Real AdaptiveCGCubicBezierSmoother::compute_element_l2_error(Index elem) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGCubicBezierSmoother: must solve "
                                 "before computing errors");
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;

    Real error_sq = 0.0;
    int ngauss = config_.ngauss_error;

    // 2D Gauss-Legendre quadrature
    for (int j = 0; j < ngauss; ++j) {
        for (int i = 0; i < ngauss; ++i) {
            Real u = gauss_nodes_(i);
            Real v = gauss_nodes_(j);
            Real w = gauss_weights_(i) * gauss_weights_(j);

            // Map to physical coordinates
            Real x = bounds.xmin + u * dx;
            Real y = bounds.ymin + v * dy;

            // Raw bathymetry value
            Real z_data = bathy_func_(x, y);

            // CG cubic Bezier approximation value
            Real z_bezier = smoother_->evaluate(x, y);

            // Accumulate weighted squared difference
            Real diff = z_data - z_bezier;
            error_sq += w * diff * diff;
        }
    }

    // L2 error = sqrt(integral of diff^2 over element)
    // The quadrature weights sum to 1 on [0,1]^2, so multiply by area
    return std::sqrt(error_sq * dx * dy);
}

CGCubicElementErrorEstimate
AdaptiveCGCubicBezierSmoother::estimate_element_error(Index elem) const {
    CGCubicElementErrorEstimate result;
    result.element = elem;

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    Real area = dx * dy;

    result.l2_error = compute_element_l2_error(elem);
    // Normalize by sqrt(area) to get average error magnitude
    result.normalized_error = result.l2_error / std::sqrt(area);

    // Compute coarsening metrics (solution change from refinement)
    compute_coarsening_metrics(elem, result.mean_difference, result.volume_change);

    // Refinement decision based on selected metric
    result.should_refine = (error_metric(result) > config_.error_threshold);

    return result;
}

void AdaptiveCGCubicBezierSmoother::invalidate_error_cache() {
    errors_valid_ = false;
    cached_errors_.clear();
    cached_max_error_ = 0.0;
    cached_mean_error_ = 0.0;
}

void AdaptiveCGCubicBezierSmoother::ensure_errors_computed() const {
    if (errors_valid_) {
        return;
    }

    cached_errors_.clear();
    cached_errors_.reserve(quadtree_->num_elements());
    cached_max_error_ = 0.0;
    Real sum_err = 0.0;

    for (Index e = 0; e < quadtree_->num_elements(); ++e) {
        auto err = estimate_element_error(e);
        cached_max_error_ = std::max(cached_max_error_, err.normalized_error);
        sum_err += err.normalized_error;
        cached_errors_.push_back(std::move(err));
    }

    cached_mean_error_ = cached_errors_.empty()
                             ? 0.0
                             : sum_err / static_cast<Real>(cached_errors_.size());
    errors_valid_ = true;
}

std::vector<CGCubicElementErrorEstimate> AdaptiveCGCubicBezierSmoother::estimate_errors() const {
    ensure_errors_computed();
    return cached_errors_;
}

Real AdaptiveCGCubicBezierSmoother::max_error() const {
    ensure_errors_computed();
    return cached_max_error_;
}

Real AdaptiveCGCubicBezierSmoother::mean_error() const {
    ensure_errors_computed();
    return cached_mean_error_;
}

// =============================================================================
// Mesh refinement
// =============================================================================

void AdaptiveCGCubicBezierSmoother::refine_elements(const std::vector<Index> &elements_to_refine) {
    if (elements_to_refine.empty())
        return;

    // Create refinement masks (XY only for 2D bathymetry)
    std::vector<RefineMask> masks(elements_to_refine.size(), RefineMask::XY);

    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->refinement_ms : nullptr);
        // Refine octree (auto-balances to maintain 2:1 constraint)
        octree_->refine(elements_to_refine, masks);
    }

    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->rebuild_ms : nullptr);
        // Rebuild smoother for new mesh
        rebuild_smoother();
    }
}

// =============================================================================
// Adaptive solve
// =============================================================================

CGCubicAdaptationResult AdaptiveCGCubicBezierSmoother::adapt_once() {
    CGCubicAdaptationResult result;
    result.iteration = static_cast<int>(history_.size());

    // Create smoother if not exists
    if (!smoother_) {
        rebuild_smoother();
    }

    // Solve on current mesh
    smoother_->solve();
    invalidate_error_cache(); // Solution changed, errors need recomputation

    // Estimate errors
    auto errors = estimate_errors();

    // Compute statistics
    Real max_err = 0.0;
    Real sum_err = 0.0;
    for (const auto &err : errors) {
        max_err = std::max(max_err, err.normalized_error);
        sum_err += err.normalized_error;
    }

    result.num_elements = quadtree_->num_elements();
    result.max_error = max_err;
    result.mean_error = errors.empty() ? 0.0 : sum_err / static_cast<Real>(errors.size());

    // Check stopping criteria
    bool error_converged = (max_err <= config_.error_threshold);
    bool max_elements_reached = (static_cast<int>(result.num_elements) >= config_.max_elements);

    if (error_converged || max_elements_reached) {
        result.converged = true;
        result.elements_refined = 0;
        return result;
    }

    // Sort errors using configured error metric
    std::sort(errors.begin(), errors.end(),
              [this](const CGCubicElementErrorEstimate &a, const CGCubicElementErrorEstimate &b) {
                  return error_metric(a) > error_metric(b);
              });

    // Select top dorfler_theta fraction of elements for refinement
    Index num_to_refine =
        static_cast<Index>(std::ceil(config_.dorfler_theta * static_cast<Real>(errors.size())));
    num_to_refine = std::max(Index(1), num_to_refine); // At least 1 element

    // Collect elements to refine, filtering by refinement level limit and land mask
    std::vector<Index> valid_refine;
    for (Index i = 0; i < num_to_refine && i < static_cast<Index>(errors.size()); ++i) {
        Index elem = errors[i].element;
        QuadLevel level = quadtree_->element_level(elem);
        if (level.max_level() < config_.max_refinement_level &&
            !is_element_on_land(elem)) {
            valid_refine.push_back(elem);
        }
    }

    result.elements_refined = static_cast<Index>(valid_refine.size());

    if (valid_refine.empty()) {
        // Can't refine further due to level limit
        result.converged = true;
    } else {
        // Store current solution before refinement for coarsening error computation
        store_current_solution();
        refine_elements(valid_refine);
        result.converged = false;
    }

    return result;
}

CGCubicAdaptationResult AdaptiveCGCubicBezierSmoother::solve_adaptive() {
    if (!bathy_func_) {
        throw std::runtime_error("AdaptiveCGCubicBezierSmoother: bathymetry data not set");
    }

    // Clear previous profiling data
    iteration_profiles_.clear();
    preprocess_profile_ = CGCubicPreprocessProfile{};
    postprocess_ms_ = 0.0;

    CGCubicAdaptationResult result;

    // Preprocess: initial setup (GeoTIFF caching happens here)
    if (!smoother_) {
        if (config_.verbose) {
            CGCubicIterationProfile preprocess;
            current_profile_ = &preprocess;
            rebuild_smoother();
            preprocess_profile_.quadtree_build_ms = preprocess.quadtree_build_ms;
            preprocess_profile_.smoother_init_ms = preprocess.smoother_init_ms;
            preprocess_profile_.hessian_assembly_ms = preprocess.hessian_assembly_ms;
            preprocess_profile_.data_fitting_ms = preprocess.data_fitting_ms;
            current_profile_ = nullptr;
        } else {
            rebuild_smoother();
        }
    }

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        CGCubicIterationProfile profile;
        current_profile_ = config_.verbose ? &profile : nullptr;

        // Set up solve profiling
        CGCubicSolveProfile solve_profile;
        if (current_profile_) {
            smoother_->set_solve_profile(&solve_profile);
        }

        // Solve on current mesh
        {
            OptionalScopedTimer t(current_profile_ ? &profile.solve_ms : nullptr);
            smoother_->solve();
        }
        invalidate_error_cache(); // Solution changed, errors need recomputation

        // Copy solve breakdown to iteration profile
        if (current_profile_) {
            profile.matrix_build_ms = solve_profile.matrix_build_ms;
            profile.constraint_build_ms = solve_profile.constraint_build_ms;
            profile.kkt_assembly_ms = solve_profile.kkt_assembly_ms;
            profile.sparse_lu_compute_ms = solve_profile.sparse_lu_compute_ms;
            profile.sparse_lu_solve_ms = solve_profile.sparse_lu_solve_ms;
            profile.constraint_projection_ms = solve_profile.constraint_projection_ms;
            smoother_->set_solve_profile(nullptr);
        }

        // Estimate errors
        std::vector<CGCubicElementErrorEstimate> errors;
        {
            OptionalScopedTimer t(current_profile_ ? &profile.error_estimation_ms : nullptr);
            errors = estimate_errors();
        }

        // Compute statistics (marking phase)
        Real max_err = 0.0;
        Real sum_err = 0.0;
        std::vector<Index> valid_refine;
        {
            OptionalScopedTimer t(current_profile_ ? &profile.marking_ms : nullptr);

            for (const auto &err : errors) {
                Real metric = error_metric(err);
                max_err = std::max(max_err, metric);
                sum_err += metric;
            }

            // Sort errors using configured error metric
            std::sort(errors.begin(), errors.end(),
                      [this](const CGCubicElementErrorEstimate &a,
                             const CGCubicElementErrorEstimate &b) {
                          return error_metric(a) > error_metric(b);
                      });

            // Bootstrap: check if using coarsening metric with no previous solution
            bool is_coarsening_metric =
                (config_.error_metric_type == ErrorMetricType::MeanDifference ||
                 config_.error_metric_type == ErrorMetricType::VolumeChange);
            bool is_bootstrap = (is_coarsening_metric && prev_solutions_.empty());

            if (is_bootstrap) {
                // Bootstrap: refine ALL elements to establish a baseline for comparison
                for (Index e = 0; e < quadtree_->num_elements(); ++e) {
                    QuadLevel level = quadtree_->element_level(e);
                    if (level.max_level() < config_.max_refinement_level &&
                        !is_element_on_land(e)) {
                        valid_refine.push_back(e);
                    }
                }
            } else {
                // Normal Dorfler marking
                Index num_to_refine = static_cast<Index>(
                    std::ceil(config_.dorfler_theta * static_cast<Real>(errors.size())));
                num_to_refine = std::max(Index(1), num_to_refine);

                for (Index i = 0; i < num_to_refine && i < static_cast<Index>(errors.size());
                     ++i) {
                    Index elem = errors[i].element;
                    QuadLevel level = quadtree_->element_level(elem);
                    // Skip elements at max level or entirely on land
                    if (level.max_level() < config_.max_refinement_level &&
                        !is_element_on_land(elem)) {
                        valid_refine.push_back(elem);
                    }
                }
            }
        }

        // Write VTK if configured (after solve, before refinement)
        if (!config_.vtk_output_prefix.empty()) {
            std::string vtk_file =
                config_.vtk_output_prefix + "_iter_" + std::to_string(iter);

            // Collect per-element error indicators
            std::vector<Real> element_rms(quadtree_->num_elements(), 0.0);
            std::vector<Real> element_mean_diff(quadtree_->num_elements(), 0.0);
            std::vector<Real> element_volume_change(quadtree_->num_elements(), 0.0);
            std::vector<Real> refinement_levels(quadtree_->num_elements(), 0.0);

            for (const auto &e : errors) {
                element_rms[e.element] = e.normalized_error;
                element_mean_diff[e.element] = e.mean_difference;
                element_volume_change[e.element] = e.volume_change;
            }
            for (Index i = 0; i < quadtree_->num_elements(); ++i) {
                refinement_levels[i] = static_cast<Real>(quadtree_->element_level(i).max_level());
            }

            io::write_cg_bezier_surface_vtk(
                vtk_file, *quadtree_, [this](Real x, Real y) { return smoother_->evaluate(x, y); },
                10, "elevation",
                {{"rms_error", element_rms},
                 {"mean_difference", element_mean_diff},
                 {"volume_change", element_volume_change},
                 {"refinement_level", refinement_levels}});

            if (config_.verbose) {
                std::cout << "Wrote VTK: " << vtk_file << ".vtu\n";
            }
        }

        // Build result
        result.iteration = iter;
        result.num_elements = quadtree_->num_elements();
        result.max_error = max_err;
        result.mean_error = errors.empty() ? 0.0 : sum_err / static_cast<Real>(errors.size());

        // Check stopping criteria
        bool error_converged = (max_err <= config_.error_threshold);
        bool max_elements_reached = (static_cast<int>(result.num_elements) >= config_.max_elements);

        // Bootstrap: don't claim convergence on first iteration with coarsening
        // metrics (since all coarsening metrics = 0 when no previous solution exists)
        bool is_coarsening_metric =
            (config_.error_metric_type == ErrorMetricType::MeanDifference ||
             config_.error_metric_type == ErrorMetricType::VolumeChange);
        bool is_bootstrap = (is_coarsening_metric && prev_solutions_.empty());
        if (is_bootstrap) {
            error_converged = false;
        }

        if (error_converged) {
            result.converged = true;
            result.elements_refined = 0;
            result.convergence_reason = ConvergenceReason::ErrorThreshold;
        } else if (max_elements_reached) {
            result.converged = true;
            result.elements_refined = 0;
            result.convergence_reason = ConvergenceReason::MaxElements;
        } else if (valid_refine.empty()) {
            result.converged = true;
            result.elements_refined = 0;
            result.convergence_reason = ConvergenceReason::MaxRefinementLevel;
        } else {
            // Store current solution before refinement for coarsening error computation
            store_current_solution();
            // Refine elements (timing happens inside refine_elements)
            refine_elements(valid_refine);
            result.elements_refined = static_cast<Index>(valid_refine.size());
            result.converged = false;
        }

        // Record profile context
        if (current_profile_) {
            profile.num_elements = result.num_elements;
            profile.num_dofs = smoother_->num_global_dofs();
            profile.num_free_dofs = smoother_->num_free_dofs();
            profile.num_constraints = smoother_->num_constraints();
            iteration_profiles_.push_back(profile);
        }

        history_.push_back(result);

        if (config_.verbose) {
            std::cout << "Iteration " << iter << ": " << result.num_elements << " elements, "
                      << "max_error=" << result.max_error << " m, "
                      << "mean_error=" << result.mean_error << " m, "
                      << "refined=" << result.elements_refined;
            if (current_profile_) {
                std::cout << ", time=" << profile.total_ms() << " ms";
            }
            std::cout << "\n";
        }

        if (result.converged) {
            if (config_.verbose) {
                std::cout << "Converged after " << (iter + 1) << " iterations\n";
            }
            break;
        }
    }

    current_profile_ = nullptr;

    // Postprocess: ensure final smoother is solved (in case we stopped after refinement)
    if (smoother_ && !smoother_->is_solved()) {
        OptionalScopedTimer t(config_.verbose ? &postprocess_ms_ : nullptr);
        smoother_->solve();
    }

    // Print profiling report if verbose
    if (config_.verbose) {
        print_profile_report();
    }

    return result;
}

// =============================================================================
// Accessors
// =============================================================================

const CGCubicBezierBathymetrySmoother &AdaptiveCGCubicBezierSmoother::smoother() const {
    if (!smoother_) {
        throw std::runtime_error("AdaptiveCGCubicBezierSmoother: smoother not "
                                 "initialized (call solve first)");
    }
    return *smoother_;
}

// evaluate() is inherited from AdaptiveCGBezierSmootherBase

void AdaptiveCGCubicBezierSmoother::write_vtk(const std::string &filename, int resolution) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGCubicBezierSmoother: must solve before writing VTK");
    }
    smoother_->write_vtk(filename, resolution);
}

// =============================================================================
// Land mask checking
// =============================================================================

bool AdaptiveCGCubicBezierSmoother::is_element_on_land(Index elem) const {
    if (!land_mask_func_) {
        return false; // No land mask set, so element is not on land
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);

    // Sample at corners and center
    std::array<std::pair<Real, Real>, 5> sample_points = {{{bounds.xmin, bounds.ymin},
                                                           {bounds.xmax, bounds.ymin},
                                                           {bounds.xmin, bounds.ymax},
                                                           {bounds.xmax, bounds.ymax},
                                                           {(bounds.xmin + bounds.xmax) / 2, (bounds.ymin + bounds.ymax) / 2}}};

    for (const auto &[x, y] : sample_points) {
        if (!land_mask_func_(x, y)) {
            return false; // At least one point is not on land
        }
    }

    return true; // All sample points are on land
}

// =============================================================================
// Profiling report
// =============================================================================

void AdaptiveCGCubicBezierSmoother::print_profile_report() const {
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\nAdaptive CG Cubic Bezier Profile\n";
    std::cout << std::string(175, '=') << "\n";

    // Preprocess section
    std::cout << "Preprocess (initial setup):\n";
    std::cout << "  Qtree:   " << std::setw(10) << preprocess_profile_.quadtree_build_ms << " ms\n";
    std::cout << "  SmInit:  " << std::setw(10) << preprocess_profile_.smoother_init_ms << " ms\n";
    std::cout << "  Hess:    " << std::setw(10) << preprocess_profile_.hessian_assembly_ms << " ms\n";
    std::cout << "  DataFit: " << std::setw(10) << preprocess_profile_.data_fitting_ms
              << " ms (includes GeoTIFF caching)\n";
    std::cout << "  Total:   " << std::setw(10) << preprocess_profile_.total_ms() << " ms\n";
    std::cout << "\n";

    // Per-iteration table
    if (!iteration_profiles_.empty()) {
        // Accumulate cumulative totals
        CGCubicIterationProfile cumulative;
        for (const auto &p : iteration_profiles_) {
            cumulative.rebuild_ms += p.rebuild_ms;
            cumulative.solve_ms += p.solve_ms;
            cumulative.error_estimation_ms += p.error_estimation_ms;
            cumulative.marking_ms += p.marking_ms;
            cumulative.refinement_ms += p.refinement_ms;
            cumulative.quadtree_build_ms += p.quadtree_build_ms;
            cumulative.smoother_init_ms += p.smoother_init_ms;
            cumulative.hessian_assembly_ms += p.hessian_assembly_ms;
            cumulative.data_fitting_ms += p.data_fitting_ms;
            cumulative.matrix_build_ms += p.matrix_build_ms;
            cumulative.constraint_build_ms += p.constraint_build_ms;
            cumulative.kkt_assembly_ms += p.kkt_assembly_ms;
            cumulative.sparse_lu_compute_ms += p.sparse_lu_compute_ms;
            cumulative.sparse_lu_solve_ms += p.sparse_lu_solve_ms;
            cumulative.constraint_projection_ms += p.constraint_projection_ms;
        }

        double iterations_total = cumulative.total_ms();

        std::string sep(175, '-');
        std::cout << "Per-iteration:\n";
        std::cout << std::setw(4) << "Iter" << std::setw(7) << "Elems" << std::setw(7) << "DOFs"
                  << std::setw(6) << "Cstr" << std::setw(8) << "Qtree" << std::setw(8) << "SmInit"
                  << std::setw(8) << "Hess" << std::setw(9) << "DataFit" << std::setw(8) << "MatBld"
                  << std::setw(8) << "CstBld" << std::setw(10) << "KKT" << std::setw(9) << "LUComp"
                  << std::setw(8) << "LUSolv" << std::setw(8) << "CstPrj" << std::setw(8) << "Errors"
                  << std::setw(7) << "Mark" << std::setw(8) << "Refine" << std::setw(10) << "Total"
                  << "  [ms]\n";
        std::cout << sep << "\n";

        for (size_t i = 0; i < iteration_profiles_.size(); ++i) {
            const auto &p = iteration_profiles_[i];
            std::cout << std::setw(4) << i << std::setw(7) << p.num_elements << std::setw(7)
                      << p.num_dofs << std::setw(6) << p.num_constraints << std::setw(8)
                      << p.quadtree_build_ms << std::setw(8) << p.smoother_init_ms << std::setw(8)
                      << p.hessian_assembly_ms << std::setw(9) << p.data_fitting_ms << std::setw(8)
                      << p.matrix_build_ms << std::setw(8) << p.constraint_build_ms << std::setw(10)
                      << p.kkt_assembly_ms << std::setw(9) << p.sparse_lu_compute_ms << std::setw(8)
                      << p.sparse_lu_solve_ms << std::setw(8) << p.constraint_projection_ms
                      << std::setw(8) << p.error_estimation_ms << std::setw(7) << p.marking_ms
                      << std::setw(8) << p.refinement_ms << std::setw(10) << p.total_ms() << "\n";
        }

        std::cout << sep << "\n";
        std::cout << std::setw(24) << "Cumulative:" << std::setw(8) << cumulative.quadtree_build_ms
                  << std::setw(8) << cumulative.smoother_init_ms << std::setw(8)
                  << cumulative.hessian_assembly_ms << std::setw(9) << cumulative.data_fitting_ms
                  << std::setw(8) << cumulative.matrix_build_ms << std::setw(8)
                  << cumulative.constraint_build_ms << std::setw(10) << cumulative.kkt_assembly_ms
                  << std::setw(9) << cumulative.sparse_lu_compute_ms << std::setw(8)
                  << cumulative.sparse_lu_solve_ms << std::setw(8) << cumulative.constraint_projection_ms
                  << std::setw(8) << cumulative.error_estimation_ms << std::setw(7)
                  << cumulative.marking_ms << std::setw(8) << cumulative.refinement_ms << std::setw(10)
                  << iterations_total << "\n";
        std::cout << "\n";

        // Postprocess section
        std::cout << "Postprocess:\n";
        if (postprocess_ms_ > 0.0) {
            std::cout << "  Final solve: " << std::setw(10) << postprocess_ms_ << " ms\n";
        } else {
            std::cout << "  (none)\n";
        }
        std::cout << "\n";

        // Summary
        double grand_total = preprocess_profile_.total_ms() + iterations_total + postprocess_ms_;
        std::cout << std::string(175, '=') << "\n";
        std::cout << "Summary:\n";
        std::cout << "  Preprocess:  " << std::setw(10) << preprocess_profile_.total_ms() << " ms\n";
        std::cout << "  Iterations:  " << std::setw(10) << iterations_total << " ms\n";
        std::cout << "  Postprocess: " << std::setw(10) << postprocess_ms_ << " ms\n";
        std::cout << "  Grand Total: " << std::setw(10) << grand_total << " ms\n";
        std::cout << "\n";

        // Find bottleneck among top-level phases (including preprocess)
        auto pct = [&](double ms) -> double {
            return grand_total > 0 ? 100.0 * ms / grand_total : 0.0;
        };

        double rebuild_total = preprocess_profile_.total_ms() + cumulative.quadtree_build_ms +
                               cumulative.smoother_init_ms + cumulative.hessian_assembly_ms +
                               cumulative.data_fitting_ms;
        double solve_total = cumulative.matrix_build_ms + cumulative.constraint_build_ms +
                             cumulative.kkt_assembly_ms + cumulative.sparse_lu_compute_ms +
                             cumulative.sparse_lu_solve_ms + cumulative.constraint_projection_ms +
                             postprocess_ms_;
        struct Phase {
            const char* name;
            double ms;
        };
        Phase phases[] = {{"Rebuild (preprocess + Qtree+SmInit+Hess+DataFit)", rebuild_total},
                          {"Solve (MatBld+CstBld+KKT+LU+CstPrj + postprocess)", solve_total},
                          {"Error estimation", cumulative.error_estimation_ms},
                          {"Marking", cumulative.marking_ms},
                          {"Refinement", cumulative.refinement_ms}};
        const Phase* bottleneck = &phases[0];
        for (const auto &ph : phases) {
            if (ph.ms > bottleneck->ms)
                bottleneck = &ph;
        }
        std::cout << "Bottleneck: " << bottleneck->name << " (" << std::setprecision(1)
                  << pct(bottleneck->ms) << "% of total)\n";
    }

    std::cout << std::endl;
}

} // namespace drifter
