#include "bathymetry/cg_bezier_smoother_base.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "bathymetry/biharmonic_assembler.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace drifter {

// =============================================================================
// Data input
// =============================================================================

void CGBezierSmootherBase::set_bathymetry_data(const BathymetrySource &source) {
    set_bathymetry_data([&source](Real x, Real y) { return source.evaluate(x, y); });
}

void CGBezierSmootherBase::set_bathymetry_data(std::function<Real(Real, Real)> bathy_func) {
    set_bathymetry_data_impl(bathy_func);
    data_set_ = true;
}

void CGBezierSmootherBase::set_scattered_points(const std::vector<Vec3> &points) {
    std::vector<BathymetryPoint> bathy_points;
    bathy_points.reserve(points.size());
    for (const auto &p : points) {
        bathy_points.emplace_back(p(0), p(1), p(2), 1.0);
    }
    set_scattered_points(bathy_points);
}

void CGBezierSmootherBase::set_scattered_points(const std::vector<BathymetryPoint> &points) {
    std::vector<BathymetryPoint> pts = points;

    auto bathy_func = [pts](Real x, Real y) -> Real {
        Real min_dist = std::numeric_limits<Real>::max();
        Real value = 0.0;
        for (const auto &p : pts) {
            Real dx = x - p.x;
            Real dy = y - p.y;
            Real dist = dx * dx + dy * dy;
            if (dist < min_dist) {
                min_dist = dist;
                value = p.z;
            }
        }
        return value;
    };

    set_bathymetry_data(bathy_func);
}

// =============================================================================
// Element lookup
// =============================================================================

Index CGBezierSmootherBase::find_element(Real x, Real y) const {
    return quadtree_->find_element(Vec2(x, y));
}

Index CGBezierSmootherBase::find_element_with_fallback(Real x, Real y) const {
    Index elem = find_element(x, y);
    if (elem >= 0) {
        return elem;
    }

    // Point outside domain - find closest element by center distance
    Real min_dist = std::numeric_limits<Real>::max();
    Index closest = 0;
    for (Index e = 0; e < quadtree_->num_elements(); ++e) {
        const auto &b = quadtree_->element_bounds(e);
        Real cx = 0.5 * (b.xmin + b.xmax);
        Real cy = 0.5 * (b.ymin + b.ymax);
        Real dist = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        if (dist < min_dist) {
            min_dist = dist;
            closest = e;
        }
    }
    return closest;
}

// =============================================================================
// Evaluation
// =============================================================================

Real CGBezierSmootherBase::evaluate_in_element(Index elem, Real x, Real y) const {
    const auto &bounds = quadtree_->element_bounds(elem);

    Real u = (x - bounds.xmin) / (bounds.xmax - bounds.xmin);
    Real v = (y - bounds.ymin) / (bounds.ymax - bounds.ymin);

    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    VecX coeffs = element_coefficients(elem);
    return evaluate_scalar(coeffs, u, v);
}

Real CGBezierSmootherBase::evaluate(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("CGBezierSmootherBase: must call solve() before evaluate()");
    }

    Index elem = find_element_with_fallback(x, y);
    return evaluate_in_element(elem, x, y);
}

Vec2 CGBezierSmootherBase::evaluate_gradient_in_element(Index elem, Real x, Real y) const {
    const auto &bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;

    Real u = (x - bounds.xmin) / dx;
    Real v = (y - bounds.ymin) / dy;
    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    VecX coeffs = element_coefficients(elem);

    // Get gradient in parametric coordinates
    Vec2 grad_uv = evaluate_gradient_uv(coeffs, u, v);

    // Transform to physical coordinates
    return Vec2(grad_uv(0) / dx, grad_uv(1) / dy);
}

Vec2 CGBezierSmootherBase::evaluate_gradient(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGBezierSmootherBase: must call solve() before evaluate_gradient()");
    }

    Index elem = find_element_with_fallback(x, y);
    return evaluate_gradient_in_element(elem, x, y);
}

// =============================================================================
// Transfer
// =============================================================================

void CGBezierSmootherBase::transfer_to_seabed(SeabedSurface &seabed) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGBezierSmootherBase: must call solve() before transfer_to_seabed()");
    }

    for (Index elem = 0; elem < quadtree_->num_elements(); ++elem) {
        VecX coeffs = element_coefficients(elem);
        seabed.set_element_coefficients(elem, coeffs);
    }
}

// =============================================================================
// Diagnostics
// =============================================================================

Real CGBezierSmootherBase::data_residual() const {
    if (!solved_)
        return 0.0;
    return solution_.dot(BtWB_global_ * solution_) - 2.0 * solution_.dot(BtWd_global_) +
           dTWd_global_;
}

Real CGBezierSmootherBase::regularization_energy() const {
    if (!solved_)
        return 0.0;
    return solution_.dot(H_global_ * solution_);
}

} // namespace drifter
