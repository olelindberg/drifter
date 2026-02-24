#include "bathymetry/adaptive_cg_bezier_smoother_base.hpp"
#include "bathymetry/biharmonic_assembler.hpp"
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

} // namespace drifter
