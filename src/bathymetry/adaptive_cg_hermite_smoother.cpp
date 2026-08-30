#include "bathymetry/adaptive_cg_hermite_smoother.hpp"
#include "bathymetry/basis_2d_base.hpp"
#include "core/logger.hpp"
#include "core/scoped_timer.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/geotiff_reader.hpp"
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
    case ConvergenceReason::PixelResolution:
        return "data resolution limit reached";
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
    // Masks must be attached before the data, so the assembly (and, for Hermite,
    // the DOF manager's land Dirichlet pins) can see them.
    smoother_->set_data_masks(has_data_func_, is_land_func_);
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

Real AdaptiveCGHermiteSmoother::element_resolution(Index elem) const {
    if (resolution_func_) {
        const QuadBounds &bounds = quadtree_->element_bounds(elem);
        const Real res = resolution_func_(0.5 * (bounds.xmin + bounds.xmax),
                                          0.5 * (bounds.ymin + bounds.ymax));
        if (res > 0.0) {
            return res;
        }
    }
    if (bathy_data_) {
        return bathy_data_->min_element_size();
    }
    return 0.0;
}

bool AdaptiveCGHermiteSmoother::refinement_allowed(Index elem) const {
    // A non-water element is pinned to depth 0, or out of the system entirely.
    // There is no fit to improve there, so refining it only adds DOFs.
    if (smoother_) {
        if (const ElementDataMask *mask = smoother_->element_mask()) {
            if (mask->is_pinned(elem)) {
                return false;
            }
        }
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    const Real dx = bounds.xmax - bounds.xmin;
    const Real dy = bounds.ymax - bounds.ymin;

    // Refinement halves the element, so every test is on the would-be children
    const Real child_dx = 0.5 * dx;
    const Real child_dy = 0.5 * dy;

    const Real res = element_resolution(elem);

    if (config_.enforce_pixel_limit) {
        Real min_size = config_.min_element_size;
        if (min_size <= 0.0) {
            min_size = res;
        }
        if (min_size > 0.0 && std::min(child_dx, child_dy) < min_size) {
            return false;
        }
    }

    if (config_.min_data_points_per_element > 0 && res > 0.0) {
        const Real child_points = (child_dx / res) * (child_dy / res);
        if (child_points < static_cast<Real>(config_.min_data_points_per_element)) {
            return false;
        }
    }

    return true;
}

void AdaptiveCGHermiteSmoother::compute_element_error_statistics(Index elem, Real &l2_error,
                                                                 Real &valid_weight) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGHermiteSmoother: must solve before computing errors");
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    const Real dx = bounds.xmax - bounds.xmin;
    const Real dy = bounds.ymax - bounds.ymin;

    Real sum_error_sq = 0.0;
    valid_weight = 0.0;
    for (int j = 0; j < config_.ngauss_error; ++j) {
        for (int i = 0; i < config_.ngauss_error; ++i) {
            const Real x = bounds.xmin + gauss_nodes_(i) * dx;
            const Real y = bounds.ymin + gauss_nodes_(j) * dy;
            const Real w = gauss_weights_(i) * gauss_weights_(j);

            // The surface deliberately spans gaps and is pinned over land rather
            // than fitting the zeros there, so the difference against those zeros
            // is not a fitting error. Measuring it would drive refinement into
            // every hole and along every coastline.
            if (is_excluded_from_fit(x, y)) {
                continue;
            }
            valid_weight += w;

            // Evaluate within this element, avoiding a point-location lookup that
            // could land on a neighbour at an element boundary
            const Real diff = bathy_func_(x, y) - smoother_->evaluate_in_element(elem, x, y);
            sum_error_sq += w * diff * diff;
        }
    }

    // Weights sum to 1 on [0,1]^2, so scale by the element area. Renormalise by the
    // weight actually sampled, so a partly-pinned element is not credited with a
    // small error merely because most of it was skipped.
    if (valid_weight > 0.0) {
        sum_error_sq /= valid_weight;
    }
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

    Real valid_weight = 0.0;
    compute_element_error_statistics(elem, result.l2_error, valid_weight);
    if (valid_weight <= 0.0) {
        // Every quadrature point is a gap or land: no data to be wrong about.
        // Distinct from a genuinely zero error, so it must not be reported as a
        // perfect fit that a Dorfler pass could still mark.
        return result; // all zero, should_refine = false
    }
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

void AdaptiveCGHermiteSmoother::refine_octree_only(
    const std::vector<Index> &elements_to_refine) {
    if (elements_to_refine.empty()) {
        return;
    }

    const std::vector<RefineMask> masks(elements_to_refine.size(), RefineMask::XY);
    octree_->refine(elements_to_refine, masks); // auto-balances to 2:1
    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);
}

// =============================================================================
// Coastline pre-pass
// =============================================================================

void AdaptiveCGHermiteSmoother::set_coastline(std::shared_ptr<const CoastlineIndex> index,
                                              int max_level, Real min_curvature_radius) {
    coastline_index_ = std::move(index);
    coastline_max_level_ = max_level;
    coastline_min_curvature_radius_ = min_curvature_radius;
    coastline_refined_ = false;
}

int AdaptiveCGHermiteSmoother::refine_coastline() {
    coastline_refined_ = true;

    if (!coastline_index_ || coastline_index_->num_segments() == 0) {
        return 0;
    }

    int sweeps = 0;
    bool changed = true;

    while (changed) {
        changed = false;

        // Bounded here as well as in the error-driven loop: a min_curvature_radius
        // far below the mesh scale would otherwise refine the whole coast to the
        // pixel limit before the first solve.
        if (static_cast<int>(quadtree_->num_elements()) >= config_.max_elements) {
            LOG_INFO("Coastline refinement stopped at max_elements ("
                     << config_.max_elements << ")");
            break;
        }

        std::vector<Index> to_refine;
        for (Index elem = 0; elem < quadtree_->num_elements(); ++elem) {
            if (quadtree_->element_level(elem).max_level() >= coastline_max_level_) {
                continue;
            }
            // Shares the data-resolution limits with the error-driven loop. Its
            // pinned-element test is inert here because no smoother exists yet,
            // which is what we want: the coast is exactly where the pinned land
            // and beach elements are, and resolving them is the point.
            if (!refinement_allowed(elem)) {
                continue;
            }

            const QuadBounds &bounds = quadtree_->element_bounds(elem);

            // Refine while the tightest coastline feature inside the element is
            // smaller than the element itself. min_curvature_radius() clamps its
            // result from below, so this converges on elements of about
            // coastline_min_curvature_radius_ along the coast, and returns
            // infinity - refining nothing - where the element holds no coastline
            // vertex at all.
            const Real element_side =
                std::min(bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin);
            const Real min_curvature = coastline_index_->min_curvature_radius(
                bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax,
                coastline_min_curvature_radius_);

            if (min_curvature < element_side) {
                to_refine.push_back(elem);
            }
        }

        if (!to_refine.empty()) {
            // Refinement rebalances to 2:1 and invalidates every element index,
            // so the next sweep rescans from scratch.
            refine_octree_only(to_refine);
            changed = true;
            ++sweeps;
            LOG_INFO("Coastline sweep " << sweeps << ": refined " << to_refine.size()
                                        << " elements, " << quadtree_->num_elements()
                                        << " total");
        }
    }

    if (sweeps > 0) {
        // The mesh changed under any smoother built from an earlier mesh
        smoother_.reset();
        LOG_INFO("Coastline refinement: " << sweeps << " sweeps, "
                                          << quadtree_->num_elements() << " elements");
    }

    return sweeps;
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

        io::write_high_order_surface_vtk(
            vtk_file, *quadtree_,
            [this](Index elem, Real x, Real y) {
                return smoother_->evaluate_in_element(elem, x, y);
            },
            config_.vtk_order > 0 ? std::max(config_.vtk_order, smoother_->surface_degree())
                                  : smoother_->surface_degree(),
            "elevation", element_cell_data(errors));

        if (config_.verbose) {
            std::cout << "Wrote VTK: " << vtk_file << ".vtu\n";
        }
    }

    std::vector<Index> valid_refine;
    bool blocked_by_resolution = false;
    for (Index elem : selected) {
        if (quadtree_->element_level(elem).max_level() >= config_.max_refinement_level) {
            continue;
        }
        if (!refinement_allowed(elem)) {
            blocked_by_resolution = true;
            continue;
        }
        valid_refine.push_back(elem);
    }
    result.elements_refined = static_cast<Index>(valid_refine.size());

    if (valid_refine.empty()) {
        result.converged = true;
        result.convergence_reason = blocked_by_resolution
                                        ? ConvergenceReason::PixelResolution
                                        : ConvergenceReason::MaxRefinementLevel;
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

    // Resolve the coastline before the error-driven loop starts, so the first
    // solve already sees a mesh that follows the shoreline. A no-op when no
    // coastline was set.
    if (!coastline_refined_) {
        refine_coastline();
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

std::vector<std::pair<std::string, std::vector<Real>>>
AdaptiveCGHermiteSmoother::element_cell_data(
    const std::vector<HermiteElementErrorEstimate> &errors) const {
    const size_t n = static_cast<size_t>(quadtree_->num_elements());

    std::vector<Real> metric(n, 0.0);
    std::vector<Real> rms(n, 0.0);
    std::vector<Real> mean_diff(n, 0.0);
    std::vector<Real> volume_change(n, 0.0);
    std::vector<Real> levels(n, 0.0);

    for (const auto &e : errors) {
        const size_t i = static_cast<size_t>(e.element);
        metric[i] = error_metric(e);
        rms[i] = e.normalized_error;
        mean_diff[i] = e.mean_difference;
        volume_change[i] = e.volume_change;
    }
    for (Index i = 0; i < quadtree_->num_elements(); ++i) {
        levels[static_cast<size_t>(i)] =
            static_cast<Real>(quadtree_->element_level(i).max_level());
    }

    // "error_metric" is the quantity the refinement decisions were made on,
    // whichever ErrorMetricType is configured; the others are always written so
    // the alternatives can be compared in the same file.
    return {{"error_metric", std::move(metric)},
            {"rms_error", std::move(rms)},
            {"mean_difference", std::move(mean_diff)},
            {"volume_change", std::move(volume_change)},
            {"refinement_level", std::move(levels)}};
}

void AdaptiveCGHermiteSmoother::write_vtk(const std::string &filename, int order) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGHermiteSmoother: must solve before writing VTK");
    }

    const int requested = order > 0 ? order : config_.vtk_order;
    const int emit_order = requested > 0 ? std::max(requested, smoother_->surface_degree())
                                         : smoother_->surface_degree();

    // Inland elements carry no data and are not solved for, so they are left out of
    // the file entirely - a hole in the surface rather than a misleading flat patch.
    // Water and Beach are emitted, tagged so the rim is identifiable in ParaView.
    auto cell_data = element_cell_data(estimate_errors());
    std::function<bool(Index)> include_element;
    if (const ElementDataMask *mask = smoother_->element_mask()) {
        std::vector<Real> element_class(static_cast<size_t>(quadtree_->num_elements()));
        for (Index e = 0; e < quadtree_->num_elements(); ++e) {
            element_class[static_cast<size_t>(e)] =
                static_cast<Real>(static_cast<int>((*mask)[e]));
        }
        cell_data.emplace_back("element_class", std::move(element_class));
        include_element = [mask](Index elem) { return !mask->is_excluded(elem); };
    }

    io::write_high_order_surface_vtk(
        filename, *quadtree_,
        [this](Index elem, Real x, Real y) { return smoother_->evaluate_in_element(elem, x, y); },
        emit_order, "elevation", cell_data, include_element);
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

    // Restore the stream state afterwards: setprecision is sticky, and the
    // caller keeps printing physical quantities to the same stream.
    const std::streamsize saved_precision = std::cout.precision();
    const std::ios::fmtflags saved_flags = std::cout.flags();

    std::cout << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < profiles_.size(); ++i) {
        const auto &p = profiles_[i];
        std::cout << std::left << std::setw(6) << i << std::setw(10) << p.num_elements
                  << std::setw(10) << p.num_dofs << std::setw(8) << p.num_constraints
                  << std::setw(10) << p.solve_ms << std::setw(10)
                  << (p.hessian_assembly_ms + p.data_fitting_ms) << std::setw(10)
                  << p.error_estimation_ms << std::setw(10) << p.total_ms() << "\n";
    }
    std::cout.flags(saved_flags);
    std::cout.precision(saved_precision);
    std::cout << std::endl;
}

} // namespace drifter
