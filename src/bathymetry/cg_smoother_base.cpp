#include "bathymetry/cg_smoother_base.hpp"
#include "bathymetry/basis_2d_base.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "bathymetry/hessian_base.hpp"
#include "bathymetry/biharmonic_assembler.hpp"
#include <Eigen/SparseLU>
#ifdef DRIFTER_USE_METIS
#include <iostream>  // Required before Eigen/MetisSupport (Eigen bug)
#include <Eigen/MetisSupport>
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
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

void CGSmootherBase::gauss_legendre_01(int n, std::vector<Real> &pts, std::vector<Real> &wts) {
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

void CGSmootherBase::set_bathymetry_data(const BathymetrySource &source) {
    // Adopt the source's masks before assembling, so gaps and land are kept out of
    // the least-squares term instead of entering it as depth-0 observations.
    set_data_masks([&source](Real x, Real y) { return source.has_data(x, y); },
                   [&source](Real x, Real y) { return source.is_land_point(x, y); });
    set_bathymetry_data([&source](Real x, Real y) { return source.evaluate(x, y); });
}

void CGSmootherBase::set_bathymetry_data(std::function<Real(Real, Real)> bathy_func) {
    set_bathymetry_data_impl(bathy_func);
    data_set_ = true;
}

void CGSmootherBase::build_element_mask() {
    if (!quadtree_ || (!has_data_func_ && !is_land_func_)) {
        element_mask_.reset();
        return;
    }
    element_mask_ =
        std::make_shared<const ElementDataMask>(*quadtree_, has_data_func_, is_land_func_);
}

void CGSmootherBase::set_scattered_points(const std::vector<Vec3> &points) {
    std::vector<BathymetryPoint> bathy_points;
    bathy_points.reserve(points.size());
    for (const auto &p : points) {
        bathy_points.emplace_back(p(0), p(1), p(2), 1.0);
    }
    set_scattered_points(bathy_points);
}

void CGSmootherBase::set_scattered_points(const std::vector<BathymetryPoint> &points) {
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

Index CGSmootherBase::find_element(Real x, Real y) const {
    return quadtree_->find_element(Vec2(x, y));
}

Index CGSmootherBase::find_element_with_fallback(Real x, Real y) const {
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

VecX CGSmootherBase::element_dof_scaling(Real, Real) const {
    return VecX::Ones(basis().num_dofs());
}

VecX CGSmootherBase::ridge_diagonal() const {
    return VecX::Constant(dof_manager_num_global_dofs(), lambda() * ridge_epsilon());
}

VecX CGSmootherBase::element_coefficients(Index elem) const {
    const auto &global_dofs = element_global_dofs(elem);
    const auto &bounds = quadtree_->element_bounds(elem);
    int ndof = basis().num_dofs();
    VecX coeffs(ndof);
    for (int i = 0; i < ndof; ++i) {
        coeffs(i) = solution_(global_dofs[i]);
    }
    // Identity for Bernstein bases; the h_x^a h_y^b factors for Hermite
    return coeffs.cwiseProduct(
        element_dof_scaling(bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin));
}

VecX CGSmootherBase::element_bernstein_coefficients(Index elem) const {
    return element_coefficients(elem);
}

Real CGSmootherBase::evaluate_scalar(const VecX &coeffs, Real u, Real v) const {
    return basis().evaluate_scalar(coeffs, u, v);
}

Vec2 CGSmootherBase::evaluate_gradient_uv(const VecX &coeffs, Real u, Real v) const {
    VecX du = basis().evaluate_du(u, v);
    VecX dv = basis().evaluate_dv(u, v);

    Real dz_du = coeffs.dot(du);
    Real dz_dv = coeffs.dot(dv);

    return Vec2(dz_du, dz_dv);
}

// =============================================================================
// Element surface
// =============================================================================

ElementSurface::ElementSurface(const Basis2DBase &basis, const QuadBounds &bounds, VecX coeffs)
    : basis_(&basis), xmin_(bounds.xmin), ymin_(bounds.ymin), dx_(bounds.xmax - bounds.xmin),
      dy_(bounds.ymax - bounds.ymin), coeffs_(std::move(coeffs)) {}

Real ElementSurface::value(Real x, Real y) const {
    return value_uv(std::clamp((x - xmin_) / dx_, 0.0, 1.0),
                    std::clamp((y - ymin_) / dy_, 0.0, 1.0));
}

Real ElementSurface::value_uv(Real u, Real v) const {
    return basis_->evaluate_scalar(coeffs_, u, v);
}

Vec2 ElementSurface::gradient(Real x, Real y) const {
    const Real u = std::clamp((x - xmin_) / dx_, 0.0, 1.0);
    const Real v = std::clamp((y - ymin_) / dy_, 0.0, 1.0);
    return Vec2(coeffs_.dot(basis_->evaluate_du(u, v)) / dx_,
                coeffs_.dot(basis_->evaluate_dv(u, v)) / dy_);
}

MatX sample_basis(const Basis2DBase &basis, const std::vector<Vec2> &points) {
    MatX N(basis.num_dofs(), static_cast<Index>(points.size()));
    for (size_t p = 0; p < points.size(); ++p) {
        N.col(static_cast<Index>(p)) = basis.evaluate(points[p](0), points[p](1));
    }
    return N;
}

ElementSurface CGSmootherBase::element_surface(Index elem) const {
    return ElementSurface(basis(), quadtree_->element_bounds(elem), element_coefficients(elem));
}

std::function<Real(Index, Real, Real)> element_major_evaluator(const CGSmootherBase &smoother) {
    return [&smoother, cached = std::optional<ElementSurface>(),
            last = Index(-1)](Index elem, Real x, Real y) mutable -> Real {
        if (elem != last) {
            cached = smoother.element_surface(elem);
            last = elem;
        }
        return cached->value(x, y);
    };
}

// =============================================================================
// Evaluation
// =============================================================================

Real CGSmootherBase::evaluate_in_element(Index elem, Real x, Real y) const {
    return element_surface(elem).value(x, y);
}

Real CGSmootherBase::evaluate(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("CGSmootherBase: must call solve() before evaluate()");
    }

    Index elem = find_element_with_fallback(x, y);
    return evaluate_in_element(elem, x, y);
}

Vec2 CGSmootherBase::evaluate_gradient_in_element(Index elem, Real x, Real y) const {
    return element_surface(elem).gradient(x, y);
}

Vec2 CGSmootherBase::evaluate_gradient(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGSmootherBase: must call solve() before evaluate_gradient()");
    }

    Index elem = find_element_with_fallback(x, y);
    return evaluate_gradient_in_element(elem, x, y);
}

// =============================================================================
// Transfer
// =============================================================================

void CGSmootherBase::transfer_to_seabed(SeabedSurface &seabed) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGSmootherBase: must call solve() before transfer_to_seabed()");
    }

    for (Index elem = 0; elem < quadtree_->num_elements(); ++elem) {
        // SeabedSurface expects Bernstein control values
        seabed.set_element_coefficients(elem, element_bernstein_coefficients(elem));
    }
}

// =============================================================================
// Diagnostics
// =============================================================================

Real CGSmootherBase::data_residual() const {
    if (!solved_)
        return 0.0;
    return solution_.dot(BtWB_global_ * solution_) - 2.0 * solution_.dot(BtWd_global_) +
           dTWd_global_;
}

Real CGSmootherBase::regularization_energy() const {
    if (!solved_)
        return 0.0;
    return solution_.dot(H_global_ * solution_);
}

Real CGSmootherBase::objective_value() const {
    if (!solved_)
        return 0.0;
    return alpha_ * regularization_energy() + lambda() * data_residual();
}

// =============================================================================
// Hessian assembly
// =============================================================================

void CGSmootherBase::assemble_hessian_global(const HessianBase &hessian) {
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

    // A quadtree has one element size per refinement level, so the element
    // hessian takes only a handful of distinct values across the whole mesh
    // while scaled_hessian() is a Kronecker assembly of its own. Element sizes
    // are produced by the same halving arithmetic, so they compare bit-exactly
    // and a linear scan over the few entries beats hashing them.
    struct SizedHessian {
        Real dx;
        Real dy;
        MatX H;
    };
    std::vector<SizedHessian> hessian_by_size;
    auto element_hessian = [&](Real dx, Real dy) -> const MatX & {
        for (const auto &entry : hessian_by_size) {
            if (entry.dx == dx && entry.dy == dy) {
                return entry.H;
            }
        }
        hessian_by_size.push_back({dx, dy, hessian.scaled_hessian(dx, dy)});
        return hessian_by_size.back().H;
    };

    for (Index elem = 0; elem < num_elements; ++elem) {
        // A non-water element has every DOF pinned to 0, so contributing smoothness
        // energy it can never influence would only add rows condensation drops again.
        if (is_element_excluded(elem)) {
            continue;
        }

        Vec2 size = quadtree_->element_size(elem);
        Real dx = size(0);
        Real dy = size(1);

        const MatX &H_local = element_hessian(dx, dy);
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

void CGSmootherBase::assemble_data_fitting_global(
    std::function<Real(Real, Real)> bathy_func) {

    Index num_dofs = dof_manager_num_global_dofs();
    Index num_elements = quadtree_->num_elements();
    int ngauss = ngauss_data();
    int ndof = basis().num_dofs();

    std::vector<Real> gauss_pts, gauss_wts;
    gauss_legendre_01(ngauss, gauss_pts, gauss_wts);
    const int nq = static_cast<int>(gauss_pts.size()); // clamped to 4 above

    // The parametric basis at the quadrature points does not depend on the
    // element - only the physical scaling applied to it below does - so it is
    // evaluated once here instead of once per element per quadrature point.
    MatX Nhat(nq * nq, ndof);
    for (int qi = 0; qi < nq; ++qi) {
        for (int qj = 0; qj < nq; ++qj) {
            Nhat.row(qi * nq + qj) = basis().evaluate(gauss_pts[qi], gauss_pts[qj]).transpose();
        }
    }

    // One ndof x ndof block per element, not per quadrature point: the element
    // matrix is a sum over the quadrature points, so accumulating it densely
    // first and emitting it once cuts the triplet count - and with it the peak
    // memory and the sort in setFromTriplets - by a factor of ngauss^2.
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(static_cast<size_t>(num_elements) * static_cast<size_t>(ndof) *
                     static_cast<size_t>(ndof));

    BtWd_global_.setZero(num_dofs);
    dTWd_global_ = 0.0;
    num_excluded_quad_points_ = 0;

    for (Index elem = 0; elem < num_elements; ++elem) {
        // See assemble_hessian_global(): a non-water element is out of the system,
        // and skipping it here also skips its raster lookups, which is where the
        // per-element cost actually is.
        if (is_element_excluded(elem)) {
            continue;
        }

        const auto &bounds = quadtree_->element_bounds(elem);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real jacobian = dx * dy;

        const auto &global_dofs = element_global_dofs(elem);

        // Identity for Bernstein bases; the h_x^a h_y^b factors for Hermite.
        // Hoisted out of the quadrature loop - it depends only on element size.
        const VecX dof_scal = element_dof_scaling(dx, dy);

        // Local data fitting matrix and RHS, scattered to the global system once
        // the element's quadrature is complete
        MatX B_local = MatX::Zero(ndof, ndof);
        VecX d_local = VecX::Zero(ndof);
        VecX B(ndof);
        bool has_observation = false;

        for (int qi = 0; qi < nq; ++qi) {
            Real u = gauss_pts[qi];
            for (int qj = 0; qj < nq; ++qj) {
                Real v = gauss_pts[qj];

                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;

                // No observation here. A gap contributes nothing and the surface
                // is carried across it by the smoothness term; land is a known zero
                // imposed strongly as a Dirichlet condition instead. Either way,
                // feeding a depth of 0 into the least-squares term would make the
                // smoothness term fight it and overshoot around it.
                if (is_excluded_from_fit(x, y)) {
                    ++num_excluded_quad_points_;
                    continue;
                }

                // Apply boundary relaxation to reduce data fitting near boundaries
                Real relaxation = compute_relaxation_factor(x, y);
                Real weight = gauss_wts[qi] * gauss_wts[qj] * jacobian * relaxation;

                Real d = bathy_func(x, y);

                dTWd_global_ += weight * d * d;
                has_observation = true;

                B = Nhat.row(qi * nq + qj).transpose().cwiseProduct(dof_scal);

                // Grouped as (weight * B(i)) * B(j) and (weight * B(i)) * d, the
                // same association the per-point scatter used, so the accumulated
                // values are unchanged rather than merely equal to round-off.
                B_local.noalias() += weight * B * B.transpose();
                d_local.noalias() += (weight * B) * d;
            }
        }

        // An element whose every quadrature point was excluded contributed no
        // entries before this loop was hoisted, and must not start contributing
        // a block of explicit zeros now.
        if (has_observation) {
            for (int i = 0; i < ndof; ++i) {
                const Index I = global_dofs[i];
                for (int j = 0; j < ndof; ++j) {
                    triplets.emplace_back(I, global_dofs[j], B_local(i, j));
                }
                BtWd_global_(I) += d_local(i);
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

SpMat CGSmootherBase::assemble_Q() const {
    Index num_dofs = dof_manager_num_global_dofs();

    // The ridge is added as a sparse diagonal rather than through coeffRef(i, i).
    // A DOF that no assembled element touches - every pinned land DOF, and any
    // isolated one - has no diagonal entry to reach, so coeffRef would *insert*
    // it, and an insertion into a compressed sparse matrix shifts everything
    // after it. That is O(nnz) per pinned DOF, which dominated the whole solve.
    // Summing three matrices merges the patterns in one pass instead.
    const VecX ridge = ridge_diagonal();
    SpMat ridge_matrix(num_dofs, num_dofs);
    ridge_matrix.reserve(Eigen::VectorXi::Constant(num_dofs, 1));
    for (Index i = 0; i < num_dofs; ++i) {
        ridge_matrix.insert(i, i) = ridge(i);
    }
    ridge_matrix.makeCompressed();

    return alpha_ * H_global_ + lambda() * BtWB_global_ + ridge_matrix;
}

VecX CGSmootherBase::assemble_b() const { return lambda() * BtWd_global_; }

// =============================================================================
// Solve unconstrained
// =============================================================================

void CGSmootherBase::solve_unconstrained() {
    SpMat Q = assemble_Q();
    VecX b = assemble_b();

    SparseSolver solver;
    solver.compute(Q);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGSmootherBase: SparseLU decomposition failed");
    }

    solution_ = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGSmootherBase: SparseLU solve failed");
    }
}

// =============================================================================
// Element matrix caching
// =============================================================================

void CGSmootherBase::cache_element_matrix(Index elem, const MatX &Q_local) {
    if (!element_matrix_cache_) {
        return;
    }

    const QuadtreeNode* node = quadtree_->elements()[static_cast<size_t>(elem)];
    auto key = std::make_tuple(node->morton, node->level.x, node->level.y);
    (*element_matrix_cache_)[key] = Q_local;
}

// =============================================================================
// Boundary relaxation
// =============================================================================

Real CGSmootherBase::compute_relaxation_factor(Real x, Real y) const {
    if (!relaxation_config_.enabled || relaxation_config_.width <= 0.0) {
        return 1.0;
    }

    const auto &domain = quadtree_->domain_bounds();
    Real width = relaxation_config_.width;

    // Compute minimum distance to any enabled boundary
    Real dist = std::numeric_limits<Real>::max();
    const auto &edges = relaxation_config_.edge_enabled;
    if (edges[0]) dist = std::min(dist, x - domain.xmin);         // left
    if (edges[1]) dist = std::min(dist, domain.xmax - x);         // right
    if (edges[2]) dist = std::min(dist, y - domain.ymin);         // bottom
    if (edges[3]) dist = std::min(dist, domain.ymax - y);         // top

    // If outside relaxation zone, full data fitting
    if (dist >= width) {
        return 1.0;
    }

    // Smoothstep interpolation: f(t) = 3t^2 - 2t^3
    // Maps t in [0,1] to [0,1] with zero derivative at both ends (C1 continuity)
    Real t = std::clamp(dist / width, 0.0, 1.0);
    Real factor = t * t * (3.0 - 2.0 * t);

    // Scale to [min_factor, 1.0]
    return relaxation_config_.min_factor +
           (1.0 - relaxation_config_.min_factor) * factor;
}

} // namespace drifter
