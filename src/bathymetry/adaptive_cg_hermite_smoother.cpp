#include "bathymetry/adaptive_cg_hermite_smoother.hpp"
#include "bathymetry/basis_2d_base.hpp"
#include "core/scoped_timer.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/refine_mask.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace drifter {

namespace {

const char *convergence_reason_string(ConvergenceReason reason) {
    switch (reason) {
    case ConvergenceReason::ErrorThreshold:
        return "error below threshold";
    case ConvergenceReason::MaxElements:
        return "maximum elements reached";
    case ConvergenceReason::MaxRefinementLevel:
        return "maximum refinement level reached";
    case ConvergenceReason::MaxIterations:
        return "maximum iterations reached";
    default:
        return "unknown";
    }
}

void write_error_csv(const std::string &dir, int iteration,
                     const std::vector<HermiteElementErrorEstimate> &errors,
                     const std::vector<Index> &marked_elements, const QuadtreeAdapter &mesh) {
    std::filesystem::create_directories(dir);

    const std::set<Index> marked_set(marked_elements.begin(), marked_elements.end());

    std::ostringstream fname;
    fname << dir << "/errors_iter_" << std::setw(3) << std::setfill('0') << iteration << ".csv";

    std::ofstream ofs(fname.str());
    if (!ofs) {
        std::cerr << "Warning: could not open " << fname.str() << " for writing\n";
        return;
    }

    ofs << "element_id,center_x,center_y,l2_error,normalized_error,"
           "mean_difference,volume_change,marked\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto &err : errors) {
        const QuadBounds &b = mesh.element_bounds(err.element);
        ofs << err.element << "," << 0.5 * (b.xmin + b.xmax) << "," << 0.5 * (b.ymin + b.ymax)
            << "," << err.l2_error << "," << err.normalized_error << "," << err.mean_difference
            << "," << err.volume_change << "," << (marked_set.count(err.element) ? 1 : 0) << "\n";
    }
}

} // namespace

// =============================================================================
// Construction
// =============================================================================

AdaptiveCGHermiteSmoother::AdaptiveCGHermiteSmoother(Real xmin, Real xmax, Real ymin, Real ymax,
                                                     int nx, int ny,
                                                     const AdaptiveCGHermiteConfig &config)
    : config_(config) {
    // z = [-1, 0] is a dummy vertical extent; only the bottom face matters
    octree_owned_ = std::make_unique<OctreeAdapter>(xmin, xmax, ymin, ymax, -1.0, 0.0);
    octree_owned_->build_uniform(nx, ny, 1);
    octree_ = octree_owned_.get();

    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);
    init_gauss_quadrature(config_.ngauss_error);
}

AdaptiveCGHermiteSmoother::AdaptiveCGHermiteSmoother(OctreeAdapter &octree,
                                                     const AdaptiveCGHermiteConfig &config)
    : config_(config) {
    octree_ = &octree;
    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);
    init_gauss_quadrature(config_.ngauss_error);
}

// =============================================================================
// Smoother management
// =============================================================================

void AdaptiveCGHermiteSmoother::rebuild_smoother() {
    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->quadtree_build_ms : nullptr);
        quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);
    }
    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->smoother_init_ms : nullptr);
        smoother_ = std::make_unique<CGHermiteBathymetrySmoother>(*quadtree_,
                                                                  config_.smoother_config);
    }

    // The element matrix cache exists for the Bezier multigrid preconditioner;
    // the Hermite path solves directly, so it is deliberately left unset.

    smoother_->set_profile(current_profile_);
    apply_bathymetry_to_smoother();
}

void AdaptiveCGHermiteSmoother::apply_bathymetry_to_smoother() {
    if (!smoother_) {
        throw std::runtime_error("AdaptiveCGHermiteSmoother: smoother not initialized");
    }
    if (!bathy_func_) {
        throw std::runtime_error("AdaptiveCGHermiteSmoother: bathymetry data not set");
    }
    smoother_->set_bathymetry_data(bathy_func_);
}

const Basis2DBase &AdaptiveCGHermiteSmoother::get_basis_impl() const {
    return smoother_->get_basis();
}

const CGHermiteBathymetrySmoother &AdaptiveCGHermiteSmoother::smoother() const {
    if (!smoother_) {
        throw std::runtime_error(
            "AdaptiveCGHermiteSmoother: smoother not initialized (call solve_adaptive first)");
    }
    return *smoother_;
}

// =============================================================================
// Error estimation
// =============================================================================

bool AdaptiveCGHermiteSmoother::is_element_on_land(Index elem) const {
    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    const Real cx = 0.5 * (bounds.xmin + bounds.xmax);
    const Real cy = 0.5 * (bounds.ymin + bounds.ymax);

    if (land_mask_func_) {
        const bool all_land = land_mask_func_(bounds.xmin, bounds.ymin) &&
                              land_mask_func_(bounds.xmax, bounds.ymin) &&
                              land_mask_func_(bounds.xmin, bounds.ymax) &&
                              land_mask_func_(bounds.xmax, bounds.ymax) && land_mask_func_(cx, cy);
        if (all_land) {
            return true;
        }
    }

    // Also treat an element as land when the bathymetry is identically zero over
    // it, which covers masked or outside-polygon areas that the corner test misses
    // on a non-convex mask.
    if (bathy_func_) {
        constexpr Real ZERO_DEPTH_THRESHOLD = 1e-6;
        const Real dx = bounds.xmax - bounds.xmin;
        const Real dy = bounds.ymax - bounds.ymin;
        for (int j = 0; j < config_.ngauss_error; ++j) {
            for (int i = 0; i < config_.ngauss_error; ++i) {
                const Real x = bounds.xmin + gauss_nodes_(i) * dx;
                const Real y = bounds.ymin + gauss_nodes_(j) * dy;
                if (std::abs(bathy_func_(x, y)) > ZERO_DEPTH_THRESHOLD) {
                    return false;
                }
            }
        }
        return true;
    }

    return false;
}

void AdaptiveCGHermiteSmoother::compute_element_error_statistics(Index elem,
                                                                 Real &l2_error) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGHermiteSmoother: must solve before computing errors");
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    const Real dx = bounds.xmax - bounds.xmin;
    const Real dy = bounds.ymax - bounds.ymin;

    Real sum_error_sq = 0.0;
    for (int j = 0; j < config_.ngauss_error; ++j) {
        for (int i = 0; i < config_.ngauss_error; ++i) {
            const Real x = bounds.xmin + gauss_nodes_(i) * dx;
            const Real y = bounds.ymin + gauss_nodes_(j) * dy;
            const Real w = gauss_weights_(i) * gauss_weights_(j);

            // Evaluate within this element, avoiding a point-location lookup that
            // could land on a neighbour at an element boundary
            const Real diff = bathy_func_(x, y) - smoother_->evaluate_in_element(elem, x, y);
            sum_error_sq += w * diff * diff;
        }
    }

    // Weights sum to 1 on [0,1]^2, so scale by the element area
    l2_error = std::sqrt(sum_error_sq * dx * dy);
}

HermiteElementErrorEstimate
AdaptiveCGHermiteSmoother::estimate_element_error(Index elem) const {
    HermiteElementErrorEstimate result;
    result.element = elem;

    if (is_element_on_land(elem)) {
        return result; // all zero, should_refine = false
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    const Real area = (bounds.xmax - bounds.xmin) * (bounds.ymax - bounds.ymin);

    compute_element_error_statistics(elem, result.l2_error);
    result.normalized_error = result.l2_error / std::sqrt(area);

    compute_coarsening_metrics(elem, result.mean_difference, result.volume_change);
    result.should_refine = (error_metric(result) > config_.error_threshold);

    return result;
}

std::vector<HermiteElementErrorEstimate> AdaptiveCGHermiteSmoother::estimate_errors() const {
    std::vector<HermiteElementErrorEstimate> errors;
    errors.reserve(static_cast<size_t>(quadtree_->num_elements()));
    for (Index e = 0; e < quadtree_->num_elements(); ++e) {
        errors.push_back(estimate_element_error(e));
    }
    return errors;
}

Real AdaptiveCGHermiteSmoother::max_error() const {
    Real max_err = 0.0;
    for (const auto &err : estimate_errors()) {
        max_err = std::max(max_err, error_metric(err));
    }
    return max_err;
}

Real AdaptiveCGHermiteSmoother::mean_error() const {
    const auto errors = estimate_errors();
    if (errors.empty()) {
        return 0.0;
    }
    Real sum = 0.0;
    for (const auto &err : errors) {
        sum += error_metric(err);
    }
    return sum / static_cast<Real>(errors.size());
}

// =============================================================================
// Marking and refinement
// =============================================================================

std::vector<Index> AdaptiveCGHermiteSmoother::select_elements_for_refinement(
    const std::vector<HermiteElementErrorEstimate> &errors) const {
    if (errors.empty()) {
        return {};
    }

    // Dorfler bulk marking, extended by symmetry: find the cutoff by greedy
    // accumulation of squared error, then include *every* element at or above it
    // so that symmetric configurations refine symmetrically.
    Real total_sq = 0.0;
    for (const auto &err : errors) {
        const Real m = error_metric(err);
        total_sq += m * m;
    }

    auto sorted = errors;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [this](const HermiteElementErrorEstimate &a,
                            const HermiteElementErrorEstimate &b) {
                         return error_metric(a) > error_metric(b);
                     });

    const Real target = config_.dorfler_theta * total_sq;
    Real accumulated = 0.0;
    Real cutoff_error = 0.0;
    for (const auto &err : sorted) {
        if (accumulated >= target) {
            break;
        }
        const Real m = error_metric(err);
        accumulated += m * m;
        cutoff_error = m;
    }

    std::vector<Index> selected;
    const Real threshold_val = cutoff_error * (1.0 - config_.symmetry_tolerance);
    for (const auto &err : errors) {
        if (error_metric(err) >= threshold_val) {
            selected.push_back(err.element);
        }
    }

    if (selected.empty()) {
        Real max_err = 0.0;
        Index max_elem = errors[0].element;
        for (const auto &err : errors) {
            const Real m = error_metric(err);
            if (m > max_err) {
                max_err = m;
                max_elem = err.element;
            }
        }
        selected.push_back(max_elem);
    }

    return selected;
}

void AdaptiveCGHermiteSmoother::refine_elements(const std::vector<Index> &elements_to_refine) {
    if (elements_to_refine.empty()) {
        return;
    }

    // XY only: bathymetry refinement is isotropic in the horizontal plane
    const std::vector<RefineMask> masks(elements_to_refine.size(), RefineMask::XY);

    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->refinement_ms : nullptr);
        octree_->refine(elements_to_refine, masks); // auto-balances to 2:1
    }
    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->rebuild_ms : nullptr);
        rebuild_smoother();
    }
}

// =============================================================================
// Adaptive solve
// =============================================================================

HermiteAdaptationResult AdaptiveCGHermiteSmoother::adapt_once() {
    HermiteIterationProfile profile;
    current_profile_ = config_.verbose ? &profile : nullptr;

    HermiteAdaptationResult result;
    result.iteration = static_cast<int>(history_.size());

    if (!smoother_) {
        {
            ScopedTimer t(profile.rebuild_ms);
            rebuild_smoother();
        }
        {
            ScopedTimer t(profile.solve_ms);
            smoother_->solve();
        }
    }

    profile.num_elements = quadtree_->num_elements();
    profile.num_dofs = smoother_->num_global_dofs();
    profile.num_free_dofs = smoother_->num_free_dofs();
    profile.num_constraints = smoother_->num_constraints();

    std::vector<HermiteElementErrorEstimate> errors;
    {
        ScopedTimer t(profile.error_estimation_ms);
        errors = estimate_errors();
    }

    Real max_err = 0.0;
    Real sum_err = 0.0;
    for (const auto &err : errors) {
        max_err = std::max(max_err, error_metric(err));
        sum_err += error_metric(err);
    }

    result.num_elements = quadtree_->num_elements();
    result.max_error = max_err;
    result.mean_error = errors.empty() ? 0.0 : sum_err / static_cast<Real>(errors.size());

    bool error_converged = (max_err <= config_.error_threshold);
    const bool max_elements_reached =
        (static_cast<int>(result.num_elements) >= config_.max_elements);

    // The coarsening metrics compare against the previous solution, so on the
    // first iteration they are all zero and would spuriously report convergence.
    const bool is_coarsening_metric =
        (config_.error_metric_type == ErrorMetricType::MeanDifference ||
         config_.error_metric_type == ErrorMetricType::VolumeChange);
    const bool is_bootstrap = (is_coarsening_metric && prev_solutions_.empty());
    if (is_bootstrap) {
        error_converged = false;
    }

    auto finish_converged = [&](ConvergenceReason reason) {
        result.converged = true;
        result.convergence_reason = reason;
        result.elements_refined = 0;
        profiles_.push_back(profile);
        current_profile_ = nullptr;
        return result;
    };

    if (error_converged) {
        return finish_converged(ConvergenceReason::ErrorThreshold);
    }
    if (max_elements_reached) {
        return finish_converged(ConvergenceReason::MaxElements);
    }

    std::vector<Index> selected;
    {
        ScopedTimer t(profile.marking_ms);
        if (is_bootstrap) {
            // Refine everything once to establish a baseline for comparison
            for (Index e = 0; e < quadtree_->num_elements(); ++e) {
                selected.push_back(e);
            }
        } else {
            selected = select_elements_for_refinement(errors);
        }
    }

    if (!config_.error_output_dir.empty()) {
        write_error_csv(config_.error_output_dir, result.iteration, errors, selected, *quadtree_);
    }

    if (!config_.vtk_output_prefix.empty()) {
        const std::string vtk_file =
            config_.vtk_output_prefix + "_iter_" + std::to_string(result.iteration);

        const size_t n = static_cast<size_t>(quadtree_->num_elements());
        std::vector<Real> element_rms(n, 0.0);
        std::vector<Real> element_mean_diff(n, 0.0);
        std::vector<Real> element_volume_change(n, 0.0);
        std::vector<Real> refinement_levels(n, 0.0);

        for (const auto &e : errors) {
            element_rms[static_cast<size_t>(e.element)] = e.normalized_error;
            element_mean_diff[static_cast<size_t>(e.element)] = e.mean_difference;
            element_volume_change[static_cast<size_t>(e.element)] = e.volume_change;
        }
        for (Index i = 0; i < quadtree_->num_elements(); ++i) {
            refinement_levels[static_cast<size_t>(i)] =
                static_cast<Real>(quadtree_->element_level(i).max_level());
        }

        io::write_cg_bezier_surface_vtk(
            vtk_file, *quadtree_, [this](Real x, Real y) { return smoother_->evaluate(x, y); }, 8,
            "elevation",
            {{"rms_error", element_rms},
             {"mean_difference", element_mean_diff},
             {"volume_change", element_volume_change},
             {"refinement_level", refinement_levels}});

        if (config_.verbose) {
            std::cout << "Wrote VTK: " << vtk_file << ".vtu\n";
        }
    }

    std::vector<Index> valid_refine;
    for (Index elem : selected) {
        if (quadtree_->element_level(elem).max_level() < config_.max_refinement_level) {
            valid_refine.push_back(elem);
        }
    }
    result.elements_refined = static_cast<Index>(valid_refine.size());

    if (valid_refine.empty()) {
        result.converged = true;
        result.convergence_reason = ConvergenceReason::MaxRefinementLevel;
    } else {
        store_current_solution();
        refine_elements(valid_refine);
        {
            ScopedTimer t(profile.solve_ms);
            smoother_->solve();
        }
        result.converged = false;
    }

    profiles_.push_back(profile);
    current_profile_ = nullptr;
    return result;
}

HermiteAdaptationResult AdaptiveCGHermiteSmoother::solve_adaptive() {
    if (!bathy_func_) {
        throw std::runtime_error("AdaptiveCGHermiteSmoother: bathymetry data not set");
    }

    profiles_.clear();
    HermiteAdaptationResult result;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        result = adapt_once();
        history_.push_back(result);

        if (config_.verbose) {
            std::cout << "Iteration " << iter << ": " << result.num_elements << " elements, "
                      << "max_error=" << result.max_error << " m, "
                      << "mean_error=" << result.mean_error << " m, "
                      << "refined=" << result.elements_refined << "\n";
        }

        if (result.converged) {
            if (config_.verbose) {
                std::cout << "Converged after " << (iter + 1) << " iterations ("
                          << convergence_reason_string(result.convergence_reason) << ")\n";
            }
            break;
        }
    }

    if (!result.converged) {
        result.convergence_reason = ConvergenceReason::MaxIterations;
        if (config_.verbose) {
            std::cout << "Stopped after " << config_.max_iterations << " iterations ("
                      << convergence_reason_string(result.convergence_reason) << ")\n";
        }
    }

    if (config_.verbose) {
        print_profile_report();
    }

    return result;
}

// =============================================================================
// Output
// =============================================================================

void AdaptiveCGHermiteSmoother::write_vtk(const std::string &filename, int resolution) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGHermiteSmoother: must solve before writing VTK");
    }

    io::write_cg_bezier_surface_vtk(
        filename, *quadtree_, [this](Real x, Real y) { return evaluate(x, y); },
        resolution > 0 ? resolution : 8, "elevation");
}

void AdaptiveCGHermiteSmoother::print_profile_report() const {
    if (profiles_.empty()) {
        return;
    }

    std::cout << "\n=== Adaptive CG Hermite profile ===\n"
              << std::left << std::setw(6) << "iter" << std::setw(10) << "elements"
              << std::setw(10) << "dofs" << std::setw(8) << "constr" << std::setw(10) << "solve_ms"
              << std::setw(10) << "assem_ms" << std::setw(10) << "error_ms" << std::setw(10)
              << "total_ms" << "\n";

    std::cout << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < profiles_.size(); ++i) {
        const auto &p = profiles_[i];
        std::cout << std::left << std::setw(6) << i << std::setw(10) << p.num_elements
                  << std::setw(10) << p.num_dofs << std::setw(8) << p.num_constraints
                  << std::setw(10) << p.solve_ms << std::setw(10)
                  << (p.hessian_assembly_ms + p.data_fitting_ms) << std::setw(10)
                  << p.error_estimation_ms << std::setw(10) << p.total_ms() << "\n";
    }
    std::cout << std::defaultfloat << std::endl;
}

} // namespace drifter
