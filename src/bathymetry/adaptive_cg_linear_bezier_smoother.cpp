#include "bathymetry/adaptive_cg_linear_bezier_smoother.hpp"
#include "bathymetry/biharmonic_assembler.hpp"
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
#include <stdexcept>

namespace drifter {

namespace {
const char* convergence_reason_string(ConvergenceReason reason) {
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

/// Write per-element error statistics to CSV file
void write_error_csv(const std::string &dir, int iteration,
                     const std::vector<CGLinearElementErrorEstimate> &errors,
                     const std::vector<Index> &marked_elements, const QuadtreeAdapter &mesh) {
    namespace fs = std::filesystem;

    // Create directory if needed
    fs::create_directories(dir);

    // Build set of marked elements for O(1) lookup
    std::set<Index> marked_set(marked_elements.begin(), marked_elements.end());

    // Generate filename with zero-padded iteration number
    std::ostringstream fname;
    fname << dir << "/errors_iter_" << std::setw(3) << std::setfill('0') << iteration << ".csv";

    std::ofstream ofs(fname.str());
    if (!ofs) {
        std::cerr << "Warning: could not open " << fname.str() << " for writing\n";
        return;
    }

    // Write header
    ofs << "element_id,center_x,center_y,l2_error,normalized_error,"
           "mean_difference,volume_change,marked\n";

    // Write data
    ofs << std::scientific << std::setprecision(8);
    for (const auto &err : errors) {
        const QuadBounds &bounds = mesh.element_bounds(err.element);
        Real cx = 0.5 * (bounds.xmin + bounds.xmax);
        Real cy = 0.5 * (bounds.ymin + bounds.ymax);
        bool is_marked = marked_set.count(err.element) > 0;

        ofs << err.element << "," << cx << "," << cy << "," << err.l2_error << ","
            << err.normalized_error << "," << err.mean_difference << "," << err.volume_change << ","
            << (is_marked ? 1 : 0) << "\n";
    }

    ofs.close();
}
} // namespace

// =============================================================================
// Constructors
// =============================================================================

AdaptiveCGLinearBezierSmoother::AdaptiveCGLinearBezierSmoother(
    Real xmin, Real xmax, Real ymin, Real ymax, int nx, int ny,
    const AdaptiveCGLinearBezierConfig &config)
    : config_(config) {
    // Create owned octree with domain bounds
    // Use z = [-1, 0] as a dummy vertical extent (only bottom face matters)
    octree_owned_ = std::make_unique<OctreeAdapter>(xmin, xmax, ymin, ymax, -1.0, 0.0);
    octree_owned_->build_uniform(nx, ny, 1); // 1 layer in z
    octree_ = octree_owned_.get();

    // Create quadtree from bottom face
    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);

    // Initialize Gauss quadrature
    init_gauss_quadrature();
}

AdaptiveCGLinearBezierSmoother::AdaptiveCGLinearBezierSmoother(
    OctreeAdapter &octree, const AdaptiveCGLinearBezierConfig &config)
    : config_(config), octree_(&octree) {
    // Create quadtree from bottom face of provided octree
    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);

    // Initialize Gauss quadrature
    init_gauss_quadrature();
}

// =============================================================================
// Gauss quadrature initialization
// =============================================================================

void AdaptiveCGLinearBezierSmoother::init_gauss_quadrature() {
    int ngauss = config_.ngauss_error;
    gauss_nodes_.resize(ngauss);
    gauss_weights_.resize(ngauss);

    // Gauss-Legendre nodes and weights on [0, 1]
    // Precomputed values for common orders
    if (ngauss == 1) {
        gauss_nodes_ << 0.5;
        gauss_weights_ << 1.0;
    } else if (ngauss == 2) {
        Real a = 0.5 / std::sqrt(3.0);
        gauss_nodes_ << 0.5 - a, 0.5 + a;
        gauss_weights_ << 0.5, 0.5;
    } else if (ngauss == 3) {
        Real a = 0.5 * std::sqrt(0.6);
        gauss_nodes_ << 0.5 - a, 0.5, 0.5 + a;
        gauss_weights_ << 5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0;
    } else if (ngauss == 4) {
        // Nodes and weights for 4-point GL on [0,1]
        gauss_nodes_ << 0.0694318442029737, 0.3300094782075719, 0.6699905217924281,
            0.9305681557970262;
        gauss_weights_ << 0.1739274225687269, 0.3260725774312731, 0.3260725774312731,
            0.1739274225687269;
    } else if (ngauss == 5) {
        gauss_nodes_ << 0.0469100770306680, 0.2307653449471585, 0.5, 0.7692346550528415,
            0.9530899229693319;
        gauss_weights_ << 0.1184634425280945, 0.2393143352496832, 0.2844444444444444,
            0.2393143352496832, 0.1184634425280945;
    } else if (ngauss == 6) {
        gauss_nodes_ << 0.0337652428984240, 0.1693953067668677, 0.3806904069584015,
            0.6193095930415985, 0.8306046932331323, 0.9662347571015760;
        gauss_weights_ << 0.0856622461895852, 0.1803807865240693, 0.2339569672863455,
            0.2339569672863455, 0.1803807865240693, 0.0856622461895852;
    } else {
        throw std::invalid_argument("AdaptiveCGLinearBezierSmoother: "
                                    "ngauss_error must be between 1 and 6");
    }
}

// =============================================================================
// Data input
// =============================================================================

void AdaptiveCGLinearBezierSmoother::set_bathymetry_data(const BathymetrySource &source) {
    // Wrap BathymetrySource in a lambda that captures by reference
    // Note: The source must outlive this smoother
    bathy_func_ = [&source](Real x, Real y) -> Real { return source.evaluate(x, y); };
}

void AdaptiveCGLinearBezierSmoother::set_bathymetry_data(
    std::function<Real(Real, Real)> bathy_func) {
    bathy_func_ = std::move(bathy_func);
}

void AdaptiveCGLinearBezierSmoother::set_land_mask(std::function<bool(Real, Real)> is_land_func) {
    land_mask_func_ = std::move(is_land_func);
}

void AdaptiveCGLinearBezierSmoother::set_bathymetry_surface(const BathymetrySurface &surface) {
    // Set depth function
    bathy_func_ = [&surface](Real x, Real y) -> Real { return surface.depth(x, y); };

    // Set land mask function
    land_mask_func_ = [&surface](Real x, Real y) -> bool { return surface.is_land(x, y); };
}

bool AdaptiveCGLinearBezierSmoother::is_element_on_land(Index elem) const {
    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    Real cx = 0.5 * (bounds.xmin + bounds.xmax);
    Real cy = 0.5 * (bounds.ymin + bounds.ymax);

    // Method 1: Check land mask function at corners and center
    if (land_mask_func_) {
        // Check corners - if ALL corners AND center are on land, treat as land
        bool all_corners_land = land_mask_func_(bounds.xmin, bounds.ymin) &&
                                land_mask_func_(bounds.xmax, bounds.ymin) &&
                                land_mask_func_(bounds.xmin, bounds.ymax) &&
                                land_mask_func_(bounds.xmax, bounds.ymax) &&
                                land_mask_func_(cx, cy);
        if (all_corners_land)
            return true;
    }

    // Method 2: Check if bathymetry data is zero at ALL Gauss quadrature points
    // This handles cases where:
    // - depth_func returns 0 for masked/outside-polygon areas
    // - Land mask corners are inside polygon but interior is outside (non-convex)
    // Use same Gauss points as error calculation for consistency
    if (bathy_func_) {
        constexpr Real ZERO_DEPTH_THRESHOLD = 1e-6;
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        int ngauss = config_.ngauss_error;

        for (int j = 0; j < ngauss; ++j) {
            for (int i = 0; i < ngauss; ++i) {
                Real u = gauss_nodes_(i);
                Real v = gauss_nodes_(j);
                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;
                if (std::abs(bathy_func_(x, y)) > ZERO_DEPTH_THRESHOLD)
                    return false;
            }
        }

        return true; // ALL Gauss point bathymetry values are zero → land
    }

    return false;
}

// =============================================================================
// Internal: Smoother management
// =============================================================================

void AdaptiveCGLinearBezierSmoother::rebuild_smoother() {
    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->quadtree_build_ms : nullptr);
        // Sync quadtree with refined octree
        quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);
    }

    {
        OptionalScopedTimer t(current_profile_ ? &current_profile_->smoother_init_ms : nullptr);
        // Create new CG linear smoother for updated mesh
        smoother_ =
            std::make_unique<CGLinearBezierBathymetrySmoother>(*quadtree_, config_.smoother_config);
    }

    // Pass profile pointer so assembly sub-timings are recorded
    smoother_->set_profile(current_profile_);

    // Apply bathymetry data (assembly timings go into profile via inner
    // smoother)
    apply_bathymetry_to_smoother();
}

void AdaptiveCGLinearBezierSmoother::apply_bathymetry_to_smoother() {
    if (!smoother_) {
        throw std::runtime_error("AdaptiveCGLinearBezierSmoother: smoother not initialized");
    }
    if (!bathy_func_) {
        throw std::runtime_error("AdaptiveCGLinearBezierSmoother: bathymetry data not set");
    }
    smoother_->set_bathymetry_data(bathy_func_);
}

// =============================================================================
// Error estimation
// =============================================================================

void AdaptiveCGLinearBezierSmoother::compute_element_error_statistics(Index elem,
                                                                      Real &l2_error) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGLinearBezierSmoother: must solve "
                                 "before computing errors");
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    int ngauss = config_.ngauss_error;

    // Accumulate weighted sum of squared errors
    Real sum_error_sq = 0.0;

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

            // CG linear Bezier approximation value
            Real z_bezier = smoother_->evaluate(x, y);

            // Accumulate weighted squared error
            Real diff = z_data - z_bezier;
            sum_error_sq += w * diff * diff;
        }
    }

    // L2 error = sqrt(integral of diff^2 over element)
    // The quadrature weights sum to 1 on [0,1]^2, so multiply by area
    l2_error = std::sqrt(sum_error_sq * dx * dy);
}

CGLinearElementErrorEstimate
AdaptiveCGLinearBezierSmoother::estimate_element_error(Index elem) const {
    CGLinearElementErrorEstimate result;
    result.element = elem;

    // Skip refinement for land-only elements
    if (is_element_on_land(elem)) {
        result.l2_error = 0.0;
        result.normalized_error = 0.0;
        result.mean_difference = 0.0;
        result.volume_change = 0.0;
        result.should_refine = false;
        return result;
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    Real area = dx * dy;

    // Compute L2 error via Gauss quadrature
    compute_element_error_statistics(elem, result.l2_error);

    // Normalize by sqrt(area) to get average error magnitude (RMS)
    result.normalized_error = result.l2_error / std::sqrt(area);

    // Compute coarsening metrics (solution change from refinement)
    compute_coarsening_metrics(elem, result.mean_difference, result.volume_change);

    // Refinement decision based on selected metric
    result.should_refine = (error_metric(result) > config_.error_threshold);

    return result;
}

std::vector<CGLinearElementErrorEstimate> AdaptiveCGLinearBezierSmoother::estimate_errors() const {
    std::vector<CGLinearElementErrorEstimate> errors;
    errors.reserve(quadtree_->num_elements());

    for (Index e = 0; e < quadtree_->num_elements(); ++e) {
        errors.push_back(estimate_element_error(e));
    }

    return errors;
}

Real AdaptiveCGLinearBezierSmoother::max_error() const {
    auto errors = estimate_errors();
    Real max_err = 0.0;
    for (const auto &err : errors) {
        max_err = std::max(max_err, error_metric(err));
    }
    return max_err;
}

Real AdaptiveCGLinearBezierSmoother::mean_error() const {
    auto errors = estimate_errors();
    Real sum_err = 0.0;
    for (const auto &err : errors) {
        sum_err += error_metric(err);
    }
    return errors.empty() ? 0.0 : sum_err / static_cast<Real>(errors.size());
}

// =============================================================================
// Coarsening error computation
// =============================================================================

void AdaptiveCGLinearBezierSmoother::store_current_solution() {
    if (!smoother_ || !smoother_->is_solved()) {
        return;
    }

    // Store coefficients for each element, keyed by (Morton, level_x, level_y)
    // Morton alone doesn't uniquely identify an element - same Morton can exist
    // at different levels with different positions/sizes.
    //
    // Note: We don't clear prev_solutions_ because refined (parent) elements
    // no longer exist in the mesh, but their coefficients are needed for
    // children to walk up and find their parent's previous solution.
    for (Index e = 0; e < quadtree_->num_elements(); ++e) {
        const QuadtreeNode* node = quadtree_->elements()[e];
        auto key = std::make_tuple(node->morton, node->level.x, node->level.y);
        prev_solutions_[key] = smoother_->element_coefficients(e);
    }
}

Real AdaptiveCGLinearBezierSmoother::evaluate_prev_solution(Real x, Real y) const {
    // Find element containing point in current mesh
    Index elem = quadtree_->find_element(Vec2(x, y));
    if (elem < 0) {
        return 0.0; // Outside domain
    }

    QuadtreeNode* node = quadtree_->elements()[elem];
    uint64_t morton = node->morton;
    QuadLevel level = node->level;

    // Start from PARENT level - coarsening error compares to parent, not self.
    // This ensures we find the parent's coefficients, not the same element's
    // previous iteration coefficients which would give near-zero differences.
    if (level.x > 0 && level.y > 0) {
        morton = MortonUtil::parent_x(MortonUtil::parent_y(morton));
        level.x--;
        level.y--;
    } else if (level.x > 0) {
        morton = MortonUtil::parent_x(morton);
        level.x--;
    } else if (level.y > 0) {
        morton = MortonUtil::parent_y(morton);
        level.y--;
    } else {
        return 0.0; // At root level, no parent to compare to
    }

    // Walk up hierarchy looking for stored solution
    // Key is (morton, level_x, level_y) to uniquely identify elements
    while (true) {
        auto key = std::make_tuple(morton, level.x, level.y);
        auto it = prev_solutions_.find(key);
        if (it != prev_solutions_.end()) {
            // Found stored solution - evaluate at (x, y)
            const VecX &c = it->second;

            // Compute element bounds from Morton and level
            const QuadBounds &domain = quadtree_->domain_bounds();
            Real elem_dx = (domain.xmax - domain.xmin) / (1 << level.x);
            Real elem_dy = (domain.ymax - domain.ymin) / (1 << level.y);

            // Decode Morton to get grid indices
            uint32_t ix, iy, iz;
            Morton3D::decode(morton, ix, iy, iz);

            Real elem_xmin = domain.xmin + ix * elem_dx;
            Real elem_ymin = domain.ymin + iy * elem_dy;

            // Map to parameter space [0,1]²
            Real u = std::clamp((x - elem_xmin) / elem_dx, 0.0, 1.0);
            Real v = std::clamp((y - elem_ymin) / elem_dy, 0.0, 1.0);

            // Use basis to evaluate (avoids coefficient ordering bugs)
            return basis_.evaluate_scalar(c, u, v);
        }

        // Check if at root level
        if (level.x == 0 && level.y == 0) {
            break;
        }

        // Move to parent Morton and level
        if (level.x > 0 && level.y > 0) {
            morton = MortonUtil::parent_x(MortonUtil::parent_y(morton));
            level.x--;
            level.y--;
        } else if (level.x > 0) {
            morton = MortonUtil::parent_x(morton);
            level.x--;
        } else {
            morton = MortonUtil::parent_y(morton);
            level.y--;
        }
    }

    return 0.0; // No stored solution found
}

void AdaptiveCGLinearBezierSmoother::compute_coarsening_metrics(Index elem, Real &mean_difference,
                                                                Real &volume_change) const {
    // Default to zero
    mean_difference = 0.0;
    volume_change = 0.0;

    if (prev_solutions_.empty()) {
        return; // First iteration, no previous solution
    }

    if (!smoother_ || !smoother_->is_solved()) {
        return;
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    Real area = dx * dy;
    int ngauss = config_.ngauss_error;

    Real sum_abs_diff = 0.0; // For L1 norm

    for (int j = 0; j < ngauss; ++j) {
        for (int i = 0; i < ngauss; ++i) {
            Real u = gauss_nodes_(i);
            Real v = gauss_nodes_(j);
            Real w = gauss_weights_(i) * gauss_weights_(j);

            Real x = bounds.xmin + u * dx;
            Real y = bounds.ymin + v * dy;

            Real z_fine = smoother_->evaluate(x, y);
            Real z_coarse = evaluate_prev_solution(x, y);

            Real diff = z_fine - z_coarse;
            sum_abs_diff += w * std::abs(diff);
        }
    }

    // Mean difference: ∫∫|z_fine - z_coarse|dA / ∫∫dA [m]
    // Gauss weights sum to 1 on [0,1]², so sum_abs_diff is already the mean
    mean_difference = sum_abs_diff;

    // Volume change: ∫∫|z_fine - z_coarse|dA [m³]
    volume_change = sum_abs_diff * area;
}

// =============================================================================
// Mesh refinement
// =============================================================================

void AdaptiveCGLinearBezierSmoother::refine_elements(const std::vector<Index> &elements_to_refine) {
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
// Element selection strategies
// =============================================================================

std::vector<Index> AdaptiveCGLinearBezierSmoother::select_elements_for_refinement(
    const std::vector<CGLinearElementErrorEstimate> &errors) const {
    if (errors.empty())
        return {};

    std::vector<Index> selected;

    switch (config_.marking_strategy) {
    case MarkingStrategy::DorflerSymmetric: {
        // Dorfler bulk marking with symmetry extension:
        // 1. Find cutoff via greedy accumulation of squared errors
        // 2. Include ALL elements at or above the cutoff error

        Real total_sq = 0.0;
        for (const auto &err : errors) {
            Real metric = error_metric(err);
            total_sq += metric * metric;
        }

        // Sort copy by error descending (stable for deterministic ordering)
        auto sorted = errors;
        std::stable_sort(
            sorted.begin(), sorted.end(),
            [this](const CGLinearElementErrorEstimate &a, const CGLinearElementErrorEstimate &b) {
                return error_metric(a) > error_metric(b);
            });

        // Greedy selection to find cutoff error
        Real target = config_.dorfler_theta * total_sq;
        Real accumulated = 0.0;
        Real cutoff_error = 0.0;

        for (const auto &err : sorted) {
            if (accumulated >= target)
                break;
            Real metric = error_metric(err);
            accumulated += metric * metric;
            cutoff_error = metric;
        }

        // Symmetry extension: include ALL elements at or above cutoff
        Real threshold_val = cutoff_error * (1.0 - config_.symmetry_tolerance);
        for (const auto &err : errors) {
            if (error_metric(err) >= threshold_val) {
                selected.push_back(err.element);
            }
        }
        break;
    }

    case MarkingStrategy::Dorfler: {
        // Standard Dorfler without symmetry extension

        Real total_sq = 0.0;
        for (const auto &err : errors) {
            Real metric = error_metric(err);
            total_sq += metric * metric;
        }

        auto sorted = errors;
        std::stable_sort(
            sorted.begin(), sorted.end(),
            [this](const CGLinearElementErrorEstimate &a, const CGLinearElementErrorEstimate &b) {
                return error_metric(a) > error_metric(b);
            });

        Real target = config_.dorfler_theta * total_sq;
        Real accumulated = 0.0;

        for (const auto &err : sorted) {
            if (accumulated >= target)
                break;
            selected.push_back(err.element);
            accumulated += error_metric(err) * error_metric(err);
        }
        break;
    }

    case MarkingStrategy::RelativeThreshold: {
        // Refine all elements with error > alpha * max_error

        Real max_err = 0.0;
        for (const auto &err : errors) {
            max_err = std::max(max_err, error_metric(err));
        }
        Real threshold = config_.relative_alpha * max_err;

        for (const auto &err : errors) {
            if (error_metric(err) > threshold) {
                selected.push_back(err.element);
            }
        }
        break;
    }

    case MarkingStrategy::FixedFraction:
    default: {
        // Legacy: top refine_fraction elements

        auto sorted = errors;
        std::sort(
            sorted.begin(), sorted.end(),
            [this](const CGLinearElementErrorEstimate &a, const CGLinearElementErrorEstimate &b) {
                return error_metric(a) > error_metric(b);
            });
        Index num = static_cast<Index>(
            std::ceil(config_.refine_fraction * static_cast<Real>(sorted.size())));
        num = std::max(Index(1), num);
        for (Index i = 0; i < num && i < static_cast<Index>(sorted.size()); ++i) {
            selected.push_back(sorted[i].element);
        }
        break;
    }
    }

    // Ensure at least one element selected
    if (selected.empty()) {
        Real max_err = 0.0;
        Index max_elem = errors[0].element;
        for (const auto &err : errors) {
            Real metric = error_metric(err);
            if (metric > max_err) {
                max_err = metric;
                max_elem = err.element;
            }
        }
        selected.push_back(max_elem);
    }

    return selected;
}

// =============================================================================
// Adaptive solve
// =============================================================================

CGLinearAdaptationResult AdaptiveCGLinearBezierSmoother::adapt_once() {
    CGLinearIterationProfile profile;
    current_profile_ = config_.verbose ? &profile : nullptr;

    CGLinearAdaptationResult result;
    result.iteration = static_cast<int>(history_.size());

    // Create smoother if not exists
    if (!smoother_) {
        ScopedTimer t(profile.rebuild_ms);
        rebuild_smoother();
    }

    // Ensure profile pointer is set on smoother for solve sub-timings
    smoother_->set_profile(current_profile_);

    // Solve on current mesh
    {
        ScopedTimer t(profile.solve_ms);
        smoother_->solve();
    }

    smoother_->set_profile(nullptr);

    // Record context
    profile.num_elements = quadtree_->num_elements();
    profile.num_dofs = smoother_->num_global_dofs();
    profile.num_free_dofs = smoother_->num_free_dofs();
    profile.num_constraints = smoother_->num_constraints();

    // Estimate errors
    std::vector<CGLinearElementErrorEstimate> errors;
    {
        ScopedTimer t(profile.error_estimation_ms);
        errors = estimate_errors();
    }

    // Compute statistics using selected error metric
    Real max_err = 0.0;
    Real sum_err = 0.0;
    for (const auto &err : errors) {
        Real metric = error_metric(err);
        if (metric > max_err) {
            max_err = metric;
        }
        sum_err += metric;
    }

    result.num_elements = quadtree_->num_elements();
    result.max_error = max_err;
    result.mean_error = errors.empty() ? 0.0 : sum_err / static_cast<Real>(errors.size());

    // Check stopping criteria
    bool error_converged = (max_err <= config_.error_threshold);
    bool max_elements_reached = (static_cast<int>(result.num_elements) >= config_.max_elements);

    // Bootstrap: don't claim convergence on first iteration with coarsening
    // metrics (since all coarsening metrics = 0 when no previous solution exists)
    bool is_coarsening_metric = (config_.error_metric_type == ErrorMetricType::MeanDifference ||
                                 config_.error_metric_type == ErrorMetricType::VolumeChange);
    bool is_bootstrap = (is_coarsening_metric && prev_solutions_.empty());
    if (is_bootstrap) {
        error_converged = false;
    }

    if (error_converged) {
        result.converged = true;
        result.convergence_reason = ConvergenceReason::ErrorThreshold;
        result.elements_refined = 0;
        profiles_.push_back(profile);
        current_profile_ = nullptr;
        return result;
    }
    if (max_elements_reached) {
        result.converged = true;
        result.convergence_reason = ConvergenceReason::MaxElements;
        result.elements_refined = 0;
        profiles_.push_back(profile);
        current_profile_ = nullptr;
        return result;
    }

    // Select elements using configured marking strategy
    std::vector<Index> selected;
    {
        ScopedTimer t(profile.marking_ms);

        // Bootstrap: if using a coarsening metric and no previous solution exists,
        // refine ALL elements to establish a baseline for comparison
        if (is_bootstrap) {
            for (Index e = 0; e < quadtree_->num_elements(); ++e) {
                selected.push_back(e);
            }
        } else {
            selected = select_elements_for_refinement(errors);
        }
    }

    // Write error CSV if configured
    if (!config_.error_output_dir.empty()) {
        write_error_csv(config_.error_output_dir, result.iteration, errors, selected, *quadtree_);
    }

    // Write VTK if configured (after solve, before refinement)
    if (!config_.vtk_output_prefix.empty()) {
        std::string vtk_file =
            config_.vtk_output_prefix + "_iter_" + std::to_string(result.iteration);

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
            vtk_file, *quadtree_, [this](Real x, Real y) { return smoother_->evaluate(x, y); }, 6,
            "elevation",
            {{"rms_error", element_rms},
             {"mean_difference", element_mean_diff},
             {"volume_change", element_volume_change},
             {"refinement_level", refinement_levels}});

        if (config_.verbose) {
            std::cout << "Wrote VTK: " << vtk_file << ".vtu\n";
        }
    }

    // Filter by refinement level limit
    std::vector<Index> valid_refine;
    for (Index elem : selected) {
        QuadLevel level = quadtree_->element_level(elem);
        if (level.max_level() < config_.max_refinement_level) {
            valid_refine.push_back(elem);
        }
    }

    result.elements_refined = static_cast<Index>(valid_refine.size());

    if (valid_refine.empty()) {
        // Can't refine further due to level limit
        result.converged = true;
        result.convergence_reason = ConvergenceReason::MaxRefinementLevel;
    } else {
        // Store current solution before refinement for coarsening error computation
        store_current_solution();

        // refine_elements() internally times refinement_ms and rebuild_ms
        refine_elements(valid_refine);
        result.converged = false;
    }

    profiles_.push_back(profile);
    current_profile_ = nullptr;
    return result;
}

CGLinearAdaptationResult AdaptiveCGLinearBezierSmoother::solve_adaptive() {
    if (!bathy_func_) {
        throw std::runtime_error("AdaptiveCGLinearBezierSmoother: bathymetry data not set");
    }

    profiles_.clear();
    CGLinearAdaptationResult result;

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

    // Handle max iterations case
    if (!result.converged) {
        result.convergence_reason = ConvergenceReason::MaxIterations;
        if (config_.verbose) {
            std::cout << "Stopped after " << config_.max_iterations << " iterations ("
                      << convergence_reason_string(result.convergence_reason) << ")\n";
        }
    }

    // Ensure final smoother is solved (in case we stopped after refinement)
    if (smoother_ && !smoother_->is_solved()) {
        smoother_->solve();
    }

    if (config_.verbose) {
        print_profile_report();
    }

    return result;
}

// =============================================================================
// Accessors
// =============================================================================

const CGLinearBezierBathymetrySmoother &AdaptiveCGLinearBezierSmoother::smoother() const {
    if (!smoother_) {
        throw std::runtime_error("AdaptiveCGLinearBezierSmoother: smoother not "
                                 "initialized (call solve first)");
    }
    return *smoother_;
}

Real AdaptiveCGLinearBezierSmoother::evaluate(Real x, Real y) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGLinearBezierSmoother: must solve before evaluating");
    }
    return smoother_->evaluate(x, y);
}

void AdaptiveCGLinearBezierSmoother::write_vtk(const std::string &filename, int resolution) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGLinearBezierSmoother: must solve before writing VTK");
    }

    // Call VTK writer with empty cell data
    const auto &dof_mgr = smoother_->dof_manager();
    io::write_cg_bezier_surface_vtk(
        filename, *quadtree_, [this](Real x, Real y) { return evaluate(x, y); },
        dof_mgr.xmin_domain(), dof_mgr.ymin_domain(), dof_mgr.inv_quantization_tol(),
        resolution > 0 ? resolution : 6, "elevation", {});
}

// =============================================================================
// Profiling report
// =============================================================================

void AdaptiveCGLinearBezierSmoother::print_profile_report() const {
    if (profiles_.empty())
        return;

    // Accumulate cumulative totals
    CGLinearIterationProfile cumulative;
    for (const auto &p : profiles_) {
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
        cumulative.sparse_lu_compute_ms += p.sparse_lu_compute_ms;
        cumulative.sparse_lu_solve_ms += p.sparse_lu_solve_ms;
        cumulative.constraint_condense_ms += p.constraint_condense_ms;
    }

    double grand_total = cumulative.total_ms();
    auto pct = [&](double ms) -> double {
        return grand_total > 0 ? 100.0 * ms / grand_total : 0.0;
    };

    std::cout << std::fixed << std::setprecision(3);

    // Per-iteration table
    std::string sep(105, '-');
    std::cout << "\nAdaptive CG Linear Bezier Profile\n";
    std::cout << std::string(105, '=') << "\n";
    std::cout << std::setw(4) << "Iter" << std::setw(8) << "Elems" << std::setw(8) << "DOFs"
              << std::setw(7) << "Cstr" << std::setw(11) << "Rebuild" << std::setw(11) << "Solve"
              << std::setw(11) << "Errors" << std::setw(11) << "Mark" << std::setw(11) << "Refine"
              << std::setw(11) << "Total"
              << "  [ms]\n";
    std::cout << sep << "\n";

    for (size_t i = 0; i < profiles_.size(); ++i) {
        const auto &p = profiles_[i];
        std::cout << std::setw(4) << i << std::setw(8) << p.num_elements << std::setw(8)
                  << p.num_dofs << std::setw(7) << p.num_constraints << std::setw(11)
                  << p.rebuild_ms << std::setw(11) << p.solve_ms << std::setw(11)
                  << p.error_estimation_ms << std::setw(11) << p.marking_ms << std::setw(11)
                  << p.refinement_ms << std::setw(11) << p.total_ms() << "\n";
    }

    std::cout << sep << "\n";
    std::cout << std::setw(27) << "Cumulative:" << std::setw(11) << cumulative.rebuild_ms
              << std::setw(11) << cumulative.solve_ms << std::setw(11)
              << cumulative.error_estimation_ms << std::setw(11) << cumulative.marking_ms
              << std::setw(11) << cumulative.refinement_ms << std::setw(11) << grand_total << "\n";
    std::cout << std::string(105, '=') << "\n";

    // Find bottleneck among top-level phases
    struct Phase {
        const char* name;
        double ms;
    };
    Phase phases[] = {{"Rebuild", cumulative.rebuild_ms},
                      {"Solve", cumulative.solve_ms},
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

    std::cout << std::setprecision(3);

    // Rebuild breakdown
    double rebuild_total = cumulative.quadtree_build_ms + cumulative.smoother_init_ms +
                           cumulative.hessian_assembly_ms + cumulative.data_fitting_ms;
    auto rebuild_pct = [&](double ms) -> double {
        return rebuild_total > 0 ? 100.0 * ms / rebuild_total : 0.0;
    };

    std::cout << "\nRebuild breakdown (" << cumulative.rebuild_ms << " ms cumulative):\n";
    std::cout << "  Quadtree build:     " << std::setw(10) << cumulative.quadtree_build_ms
              << " ms (" << std::setprecision(1) << rebuild_pct(cumulative.quadtree_build_ms)
              << "%)\n";
    std::cout << std::setprecision(3);
    std::cout << "  Smoother init:      " << std::setw(10) << cumulative.smoother_init_ms << " ms ("
              << std::setprecision(1) << rebuild_pct(cumulative.smoother_init_ms) << "%)\n";
    std::cout << std::setprecision(3);
    std::cout << "  Hessian assembly:   " << std::setw(10) << cumulative.hessian_assembly_ms
              << " ms (" << std::setprecision(1) << rebuild_pct(cumulative.hessian_assembly_ms)
              << "%)\n";
    std::cout << std::setprecision(3);
    std::cout << "  Data fitting:       " << std::setw(10) << cumulative.data_fitting_ms << " ms ("
              << std::setprecision(1) << rebuild_pct(cumulative.data_fitting_ms) << "%)\n";
    std::cout << std::setprecision(3);

    // Solve breakdown
    double solve_total = cumulative.matrix_build_ms + cumulative.constraint_condense_ms +
                         cumulative.sparse_lu_compute_ms + cumulative.sparse_lu_solve_ms;
    auto solve_pct = [&](double ms) -> double {
        return solve_total > 0 ? 100.0 * ms / solve_total : 0.0;
    };

    std::cout << "\nSolve breakdown (" << cumulative.solve_ms << " ms cumulative):\n";
    std::cout << "  Matrix build:       " << std::setw(10) << cumulative.matrix_build_ms << " ms ("
              << std::setprecision(1) << solve_pct(cumulative.matrix_build_ms) << "%)\n";
    std::cout << std::setprecision(3);
    if (cumulative.constraint_condense_ms > 0) {
        std::cout << "  Constraint condense:" << std::setw(10) << cumulative.constraint_condense_ms
                  << " ms (" << std::setprecision(1) << solve_pct(cumulative.constraint_condense_ms)
                  << "%)\n";
        std::cout << std::setprecision(3);
    }
    std::cout << "  SparseLU compute:   " << std::setw(10) << cumulative.sparse_lu_compute_ms
              << " ms (" << std::setprecision(1) << solve_pct(cumulative.sparse_lu_compute_ms)
              << "%)\n";
    std::cout << std::setprecision(3);
    std::cout << "  SparseLU solve:     " << std::setw(10) << cumulative.sparse_lu_solve_ms
              << " ms (" << std::setprecision(1) << solve_pct(cumulative.sparse_lu_solve_ms)
              << "%)\n";
    std::cout << std::setprecision(3);

    std::cout << std::endl;
}

} // namespace drifter
