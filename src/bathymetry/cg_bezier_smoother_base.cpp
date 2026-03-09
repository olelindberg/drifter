#include "bathymetry/cg_bezier_smoother_base.hpp"
#include "bathymetry/bezier_basis_2d_base.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "bathymetry/bezier_hessian_base.hpp"
#include "bathymetry/biharmonic_assembler.hpp"
#include <Eigen/SparseLU>
#ifdef DRIFTER_USE_METIS
#include <iostream>  // Required before Eigen/MetisSupport (Eigen bug)
#include <Eigen/MetisSupport>
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace drifter {

#ifdef DRIFTER_USE_METIS
using SparseSolver = Eigen::SparseLU<SpMat, Eigen::MetisOrdering<int>>;
#else
using SparseSolver = Eigen::SparseLU<SpMat>;
#endif

// =============================================================================
// Gauss-Legendre quadrature
// =============================================================================

void CGBezierSmootherBase::gauss_legendre_01(int n, std::vector<Real> &pts, std::vector<Real> &wts) {
    pts.resize(n);
    wts.resize(n);

    if (n == 1) {
        pts[0] = 0.5;
        wts[0] = 1.0;
    } else if (n == 2) {
        pts[0] = 0.5 - 0.5 / std::sqrt(3.0);
        pts[1] = 0.5 + 0.5 / std::sqrt(3.0);
        wts[0] = wts[1] = 0.5;
    } else if (n == 3) {
        pts[0] = 0.5 - 0.5 * std::sqrt(0.6);
        pts[1] = 0.5;
        pts[2] = 0.5 + 0.5 * std::sqrt(0.6);
        wts[0] = wts[2] = 5.0 / 18.0;
        wts[1] = 8.0 / 18.0;
    } else if (n >= 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        pts.resize(4);
        wts.resize(4);
        pts[0] = 0.5 * (1.0 - b);
        pts[1] = 0.5 * (1.0 - a);
        pts[2] = 0.5 * (1.0 + a);
        pts[3] = 0.5 * (1.0 + b);
        Real wa = (18.0 + std::sqrt(30.0)) / 72.0;
        Real wb = (18.0 - std::sqrt(30.0)) / 72.0;
        wts[0] = wts[3] = 0.5 * wb;
        wts[1] = wts[2] = 0.5 * wa;
    }
}

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
// Element coefficients and evaluation helpers
// =============================================================================

VecX CGBezierSmootherBase::element_coefficients(Index elem) const {
    const auto &global_dofs = element_global_dofs(elem);
    int ndof = basis().num_dofs();
    VecX coeffs(ndof);
    for (int i = 0; i < ndof; ++i) {
        coeffs(i) = solution_(global_dofs[i]);
    }
    return coeffs;
}

Real CGBezierSmootherBase::evaluate_scalar(const VecX &coeffs, Real u, Real v) const {
    return basis().evaluate_scalar(coeffs, u, v);
}

Vec2 CGBezierSmootherBase::evaluate_gradient_uv(const VecX &coeffs, Real u, Real v) const {
    VecX du = basis().evaluate_du(u, v);
    VecX dv = basis().evaluate_dv(u, v);

    Real dz_du = coeffs.dot(du);
    Real dz_dv = coeffs.dot(dv);

    return Vec2(dz_du, dz_dv);
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

Real CGBezierSmootherBase::objective_value() const {
    if (!solved_)
        return 0.0;
    return alpha_ * regularization_energy() + lambda() * data_residual();
}

// =============================================================================
// Hessian assembly
// =============================================================================

void CGBezierSmootherBase::assemble_hessian_global(const BezierHessianBase &hessian) {
    Index num_dofs = dof_manager_num_global_dofs();
    Index num_elements = quadtree_->num_elements();
    int ndof = hessian.num_dofs();

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_elements * ndof * ndof);

    // Initialize temporary storage for element matrices if caching is enabled
    if (element_matrix_cache_) {
        element_matrix_cache_temp_.resize(static_cast<size_t>(num_elements));
        for (auto &m : element_matrix_cache_temp_) {
            m = MatX::Zero(ndof, ndof);
        }
    }

    for (Index elem = 0; elem < num_elements; ++elem) {
        Vec2 size = quadtree_->element_size(elem);
        Real dx = size(0);
        Real dy = size(1);

        MatX H_local = hessian.scaled_hessian(dx, dy);
        const auto &global_dofs = element_global_dofs(elem);

        // Store hessian contribution in temporary cache
        if (element_matrix_cache_) {
            element_matrix_cache_temp_[static_cast<size_t>(elem)] = H_local;
        }

        for (int i = 0; i < ndof; ++i) {
            Index I = global_dofs[i];
            for (int j = 0; j < ndof; ++j) {
                Index J = global_dofs[j];
                if (std::abs(H_local(i, j)) > 1e-16) {
                    triplets.emplace_back(I, J, H_local(i, j));
                }
            }
        }
    }

    H_global_.resize(num_dofs, num_dofs);
    H_global_.setFromTriplets(triplets.begin(), triplets.end());
}

// =============================================================================
// Data fitting assembly
// =============================================================================

void CGBezierSmootherBase::assemble_data_fitting_global(
    std::function<Real(Real, Real)> bathy_func) {

    Index num_dofs = dof_manager_num_global_dofs();
    Index num_elements = quadtree_->num_elements();
    int ngauss = ngauss_data();
    int ndof = basis().num_dofs();

    std::vector<Real> gauss_pts, gauss_wts;
    gauss_legendre_01(ngauss, gauss_pts, gauss_wts);

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_elements * ngauss * ngauss * ndof * ndof);

    BtWd_global_.setZero(num_dofs);
    dTWd_global_ = 0.0;

    for (Index elem = 0; elem < num_elements; ++elem) {
        const auto &bounds = quadtree_->element_bounds(elem);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real jacobian = dx * dy;

        const auto &global_dofs = element_global_dofs(elem);

        // Accumulate local data fitting matrix for caching
        MatX B_local = MatX::Zero(ndof, ndof);

        for (int qi = 0; qi < static_cast<int>(gauss_pts.size()); ++qi) {
            Real u = gauss_pts[qi];
            for (int qj = 0; qj < static_cast<int>(gauss_pts.size()); ++qj) {
                Real v = gauss_pts[qj];
                Real weight = gauss_wts[qi] * gauss_wts[qj] * jacobian;

                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;
                Real d = bathy_func(x, y);

                dTWd_global_ += weight * d * d;

                VecX B = basis().evaluate(u, v);

                for (int i = 0; i < ndof; ++i) {
                    Index I = global_dofs[i];
                    for (int j = 0; j < ndof; ++j) {
                        Index J = global_dofs[j];
                        Real val = weight * B(i) * B(j);
                        triplets.emplace_back(I, J, val);
                        B_local(i, j) += val;
                    }
                    BtWd_global_(I) += weight * B(i) * d;
                }
            }
        }

        // Complete and cache element matrix: Q_elem = alpha*H + lambda*B + eps*I
        // Note: alpha is computed after this loop, so we cache without alpha scaling
        // The multigrid will use the raw element matrix directly
        if (element_matrix_cache_ && !element_matrix_cache_temp_.empty()) {
            // element_matrix_cache_temp_ contains H_local from assemble_hessian_global()
            // Add lambda * B_local + ridge_epsilon * I
            element_matrix_cache_temp_[static_cast<size_t>(elem)] += lambda() * B_local;
            element_matrix_cache_temp_[static_cast<size_t>(elem)] +=
                lambda() * ridge_epsilon() * MatX::Identity(ndof, ndof);
            cache_element_matrix(elem, element_matrix_cache_temp_[static_cast<size_t>(elem)]);
        }
    }

    BtWB_global_.resize(num_dofs, num_dofs);
    BtWB_global_.setFromTriplets(triplets.begin(), triplets.end());

    // Clear temporary storage
    element_matrix_cache_temp_.clear();

    // Compute scale normalization factor (used in solve and objective_value)
    Real norm_BtWB = BtWB_global_.norm();
    Real norm_H = H_global_.norm();
    alpha_ = (norm_H > 1e-14) ? norm_BtWB / norm_H : 0.0;
}

// =============================================================================
// KKT system assembly
// =============================================================================

SpMat CGBezierSmootherBase::assemble_Q() const {
    Index num_dofs = dof_manager_num_global_dofs();
    SpMat Q = alpha_ * H_global_ + lambda() * BtWB_global_;
    for (Index i = 0; i < num_dofs; ++i) {
        Q.coeffRef(i, i) += lambda() * ridge_epsilon();
    }
    return Q;
}

VecX CGBezierSmootherBase::assemble_b() const { return lambda() * BtWd_global_; }

// =============================================================================
// Solve unconstrained
// =============================================================================

void CGBezierSmootherBase::solve_unconstrained() {
    SpMat Q = assemble_Q();
    VecX b = assemble_b();

    SparseSolver solver;
    solver.compute(Q);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGBezierSmootherBase: SparseLU decomposition failed");
    }

    solution_ = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGBezierSmootherBase: SparseLU solve failed");
    }
}

// =============================================================================
// Element matrix caching
// =============================================================================

void CGBezierSmootherBase::cache_element_matrix(Index elem, const MatX &Q_local) {
    if (!element_matrix_cache_) {
        return;
    }

    const QuadtreeNode* node = quadtree_->elements()[static_cast<size_t>(elem)];
    auto key = std::make_tuple(node->morton, node->level.x, node->level.y);
    (*element_matrix_cache_)[key] = Q_local;
}

} // namespace drifter
