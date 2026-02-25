#include "bathymetry/adaptive_cg_bezier_smoother_base.hpp"
#include "bathymetry/bezier_basis_2d_base.hpp"
#include "bathymetry/biharmonic_assembler.hpp"
#include "mesh/morton.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace drifter {

// =============================================================================
// Data input
// =============================================================================

void AdaptiveCGBezierSmootherBase::set_bathymetry_data(const BathymetrySource &source) {
    // Wrap BathymetrySource in a lambda that captures by reference
    // Note: The source must outlive this smoother
    bathy_func_ = [&source](Real x, Real y) -> Real { return source.evaluate(x, y); };
}

void AdaptiveCGBezierSmootherBase::set_bathymetry_data(
    std::function<Real(Real, Real)> bathy_func) {
    bathy_func_ = std::move(bathy_func);
}

void AdaptiveCGBezierSmootherBase::set_land_mask(
    std::function<bool(Real, Real)> is_land_func) {
    land_mask_func_ = std::move(is_land_func);
}

// =============================================================================
// Evaluation
// =============================================================================

Real AdaptiveCGBezierSmootherBase::evaluate(Real x, Real y) const {
    if (!is_solved_impl()) {
        throw std::runtime_error(
            "AdaptiveCGBezierSmootherBase: must solve before evaluating");
    }
    return smoother_evaluate(x, y);
}

// =============================================================================
// Gauss quadrature initialization
// =============================================================================

void AdaptiveCGBezierSmootherBase::init_gauss_quadrature(int ngauss) {
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
        throw std::invalid_argument("AdaptiveCGBezierSmootherBase: "
                                    "ngauss must be between 1 and 6");
    }
}

// =============================================================================
// Mesh refinement
// =============================================================================

void AdaptiveCGBezierSmootherBase::refine_elements_impl(
    const std::vector<Index> &elements_to_refine) {
    if (elements_to_refine.empty())
        return;

    // Create refinement masks (XY only for 2D bathymetry)
    std::vector<RefineMask> masks(elements_to_refine.size(), RefineMask::XY);

    // Refine octree (auto-balances to maintain 2:1 constraint)
    octree_->refine(elements_to_refine, masks);

    // Rebuild smoother for new mesh
    rebuild_smoother();
}

// =============================================================================
// Coarsening error computation
// =============================================================================

void AdaptiveCGBezierSmootherBase::store_current_solution() {
    if (!is_solved_impl()) {
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
        const QuadtreeNode *node = quadtree_->elements()[e];
        auto key = std::make_tuple(node->morton, node->level.x, node->level.y);
        prev_solutions_[key] = get_element_coefficients_impl(e);
    }
}

Real AdaptiveCGBezierSmootherBase::evaluate_prev_solution(Real x, Real y) const {
    // Find element containing point in current mesh
    Index elem = quadtree_->find_element(Vec2(x, y));
    if (elem < 0) {
        return 0.0; // Outside domain
    }

    QuadtreeNode *node = quadtree_->elements()[elem];
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

            // Use polymorphic basis to evaluate
            return get_basis_impl().evaluate_scalar(c, u, v);
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

void AdaptiveCGBezierSmootherBase::compute_coarsening_metrics(Index elem, Real &mean_difference,
                                                              Real &volume_change) const {
    // Default to zero
    mean_difference = 0.0;
    volume_change = 0.0;

    if (prev_solutions_.empty()) {
        return; // First iteration, no previous solution
    }

    if (!is_solved_impl()) {
        return;
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    Real area = dx * dy;
    int ngauss = static_cast<int>(gauss_nodes_.size());

    Real sum_abs_diff = 0.0; // For L1 norm

    for (int j = 0; j < ngauss; ++j) {
        for (int i = 0; i < ngauss; ++i) {
            Real u = gauss_nodes_(i);
            Real v = gauss_nodes_(j);
            Real w = gauss_weights_(i) * gauss_weights_(j);

            Real x = bounds.xmin + u * dx;
            Real y = bounds.ymin + v * dy;

            Real z_fine = smoother_evaluate(x, y);
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

} // namespace drifter
