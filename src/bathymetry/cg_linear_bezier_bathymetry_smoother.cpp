#include "bathymetry/cg_linear_bezier_bathymetry_smoother.hpp"
#include "bathymetry/adaptive_cg_linear_bezier_smoother.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "core/scoped_timer.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>

namespace drifter {

namespace {

void gauss_legendre_01_linear(int n, std::vector<Real> &pts, std::vector<Real> &wts) {
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

} // anonymous namespace

// =============================================================================
// Construction
// =============================================================================

CGLinearBezierBathymetrySmoother::CGLinearBezierBathymetrySmoother(
    const QuadtreeAdapter &mesh, const CGLinearBezierSmootherConfig &config)
    : config_(config) {
    quadtree_ = &mesh;
    init_components();
}

CGLinearBezierBathymetrySmoother::CGLinearBezierBathymetrySmoother(
    const OctreeAdapter &octree, const CGLinearBezierSmootherConfig &config)
    : config_(config) {
    quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
    quadtree_ = quadtree_owned_.get();
    init_components();
}

void CGLinearBezierBathymetrySmoother::init_components() {
    basis_ = std::make_unique<LinearBezierBasis2D>();
    dirichlet_hessian_ = std::make_unique<DirichletHessian>(config_.ngauss_energy);
    dof_manager_ = std::make_unique<CGLinearBezierDofManager>(*quadtree_);

    solution_.setZero(dof_manager_->num_global_dofs());
}

// =============================================================================
// Data input (virtual method implementation)
// =============================================================================

void CGLinearBezierBathymetrySmoother::set_bathymetry_data_impl(
    std::function<Real(Real, Real)> bathy_func) {
    {
        OptionalScopedTimer t(profile_ ? &profile_->hessian_assembly_ms : nullptr);
        assemble_dirichlet_hessian();
    }
    {
        OptionalScopedTimer t(profile_ ? &profile_->data_fitting_ms : nullptr);
        assemble_data_fitting(bathy_func);
    }
}

// =============================================================================
// Assembly
// =============================================================================

void CGLinearBezierBathymetrySmoother::assemble_dirichlet_hessian() {
    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_elements = quadtree_->num_elements();

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_elements * 4 * 4);

    for (Index elem = 0; elem < num_elements; ++elem) {
        Vec2 size = quadtree_->element_size(elem);
        Real dx = size(0);
        Real dy = size(1);

        MatX H_local = dirichlet_hessian_->scaled_hessian(dx, dy);
        const auto &global_dofs = dof_manager_->element_dofs(elem);

        for (int i = 0; i < LinearBezierBasis2D::NDOF; ++i) {
            Index I = global_dofs[i];
            for (int j = 0; j < LinearBezierBasis2D::NDOF; ++j) {
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

void CGLinearBezierBathymetrySmoother::assemble_data_fitting(
    std::function<Real(Real, Real)> bathy_func) {

    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_elements = quadtree_->num_elements();
    int ngauss = config_.ngauss_data;

    std::vector<Real> gauss_pts, gauss_wts;
    gauss_legendre_01_linear(ngauss, gauss_pts, gauss_wts);

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_elements * ngauss * ngauss * 4 * 4);

    BtWd_global_.setZero(num_dofs);
    dTWd_global_ = 0.0;

    for (Index elem = 0; elem < num_elements; ++elem) {
        const auto &bounds = quadtree_->element_bounds(elem);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real jacobian = dx * dy;

        const auto &global_dofs = dof_manager_->element_dofs(elem);

        for (int qi = 0; qi < static_cast<int>(gauss_pts.size()); ++qi) {
            Real u = gauss_pts[qi];
            for (int qj = 0; qj < static_cast<int>(gauss_pts.size()); ++qj) {
                Real v = gauss_pts[qj];
                Real weight = gauss_wts[qi] * gauss_wts[qj] * jacobian;

                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;
                Real d = bathy_func(x, y);

                dTWd_global_ += weight * d * d;

                VecX B = basis_->evaluate(u, v);

                for (int i = 0; i < LinearBezierBasis2D::NDOF; ++i) {
                    Index I = global_dofs[i];
                    for (int j = 0; j < LinearBezierBasis2D::NDOF; ++j) {
                        Index J = global_dofs[j];
                        triplets.emplace_back(I, J, weight * B(i) * B(j));
                    }
                    BtWd_global_(I) += weight * B(i) * d;
                }
            }
        }
    }

    BtWB_global_.resize(num_dofs, num_dofs);
    BtWB_global_.setFromTriplets(triplets.begin(), triplets.end());

    // Compute scale normalization factor once (used in solve and objective_value)
    Real norm_BtWB = BtWB_global_.norm();
    Real norm_H = H_global_.norm();
    alpha_ = (norm_H > 1e-14) ? norm_BtWB / norm_H : 0.0;
}

// =============================================================================
// Solve
// =============================================================================

void CGLinearBezierBathymetrySmoother::solve() {
    if (!data_set_) {
        throw std::runtime_error("CGLinearBezierBathymetrySmoother: bathymetry data not set");
    }

    Index num_constraints = dof_manager_->num_constraints();

    if (num_constraints == 0) {
        solve_unconstrained();
    } else {
        solve_with_constraints();
    }

    solved_ = true;
}

void CGLinearBezierBathymetrySmoother::solve_unconstrained() {
    Index num_dofs = dof_manager_->num_global_dofs();

    SpMat Q;
    {
        OptionalScopedTimer t(profile_ ? &profile_->matrix_build_ms : nullptr);
        // Q = alpha * H_dirichlet + lambda * BtWB (alpha computed during assembly)
        Q = alpha_ * H_global_ + config_.lambda * BtWB_global_;

        // Ridge regularization
        for (Index i = 0; i < num_dofs; ++i) {
            Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
        }
    }

    VecX c = -config_.lambda * BtWd_global_;

    Eigen::SparseLU<SpMat> solver;
    {
        OptionalScopedTimer t(profile_ ? &profile_->sparse_lu_compute_ms : nullptr);
        solver.compute(Q);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGLinearBezierBathymetrySmoother: SparseLU decomposition failed");
    }

    {
        OptionalScopedTimer t(profile_ ? &profile_->sparse_lu_solve_ms : nullptr);
        solution_ = solver.solve(-c);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGLinearBezierBathymetrySmoother: SparseLU solve failed");
    }
}

void CGLinearBezierBathymetrySmoother::solve_with_constraints() {
    // Use constraint elimination instead of KKT system.
    // For hanging node constraints x_slave = sum(w_i * x_master_i),
    // we eliminate slave DOFs and solve a smaller system on free DOFs only.

    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_free = dof_manager_->num_free_dofs();

    // Build lookup: slave_dof -> constraint index
    std::unordered_map<Index, size_t> slave_to_constraint;
    const auto &constraints = dof_manager_->constraints();
    for (size_t ci = 0; ci < constraints.size(); ++ci) {
        slave_to_constraint[constraints[ci].slave_dof] = ci;
    }

    SpMat Q;
    VecX c;
    {
        OptionalScopedTimer t(profile_ ? &profile_->matrix_build_ms : nullptr);
        // Build Q and c for all DOFs
        Q = alpha_ * H_global_ + config_.lambda * BtWB_global_;
        for (Index i = 0; i < num_dofs; ++i) {
            Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
        }
        c = -config_.lambda * BtWd_global_;
    }

    // Helper: expand a global DOF to (free_index, weight) pairs
    // Free DOF: returns {(free_idx, 1.0)}
    // Slave DOF: returns {(master_free_idx, weight), ...}
    auto expand_dof = [&](Index g) -> std::vector<std::pair<Index, Real>> {
        Index f = dof_manager_->global_to_free(g);
        if (f >= 0) {
            return {{f, 1.0}};
        }
        // Slave DOF - expand to masters
        auto it = slave_to_constraint.find(g);
        if (it == slave_to_constraint.end()) {
            return {};
        }
        const auto &hc = constraints[it->second];
        std::vector<std::pair<Index, Real>> result;
        for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
            Index mf = dof_manager_->global_to_free(hc.master_dofs[i]);
            if (mf >= 0) {
                result.emplace_back(mf, hc.weights[i]);
            }
        }
        return result;
    };

    SpMat Q_reduced;
    VecX c_reduced;
    {
        OptionalScopedTimer t(profile_ ? &profile_->constraint_condense_ms : nullptr);
        // Build reduced system by condensing slave DOFs
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(Q.nonZeros());
        c_reduced = VecX::Zero(num_free);

        // Condense Q matrix
        for (int k = 0; k < Q.outerSize(); ++k) {
            for (SpMat::InnerIterator it(Q, k); it; ++it) {
                Index I = it.row();
                Index J = it.col();
                Real val = it.value();

                auto I_expanded = expand_dof(I);
                auto J_expanded = expand_dof(J);

                for (const auto &[If, Iw] : I_expanded) {
                    for (const auto &[Jf, Jw] : J_expanded) {
                        triplets.emplace_back(If, Jf, val * Iw * Jw);
                    }
                }
            }
        }

        // Condense RHS vector
        for (Index g = 0; g < num_dofs; ++g) {
            auto g_expanded = expand_dof(g);
            for (const auto &[gf, gw] : g_expanded) {
                c_reduced(gf) += c(g) * gw;
            }
        }

        Q_reduced.resize(num_free, num_free);
        Q_reduced.setFromTriplets(triplets.begin(), triplets.end());
    }

    // Solve reduced system
    Eigen::SparseLU<SpMat> solver;
    {
        OptionalScopedTimer t(profile_ ? &profile_->sparse_lu_compute_ms : nullptr);
        solver.compute(Q_reduced);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGLinearBezierBathymetrySmoother: SparseLU decomposition failed");
    }

    VecX x_free;
    {
        OptionalScopedTimer t(profile_ ? &profile_->sparse_lu_solve_ms : nullptr);
        x_free = solver.solve(-c_reduced);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGLinearBezierBathymetrySmoother: SparseLU solve failed");
    }

    // Back-substitute: free DOFs directly, slave DOFs from masters
    solution_.setZero(num_dofs);
    for (Index f = 0; f < num_free; ++f) {
        solution_(dof_manager_->free_to_global(f)) = x_free(f);
    }
    for (const auto &hc : constraints) {
        Real val = 0.0;
        for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
            val += hc.weights[i] * solution_(hc.master_dofs[i]);
        }
        solution_(hc.slave_dof) = val;
    }
}

// =============================================================================
// Virtual method implementations for CGBezierSmootherBase
// =============================================================================

VecX CGLinearBezierBathymetrySmoother::element_coefficients(Index elem) const {
    const auto &global_dofs = dof_manager_->element_dofs(elem);
    VecX coeffs(LinearBezierBasis2D::NDOF);
    for (int i = 0; i < LinearBezierBasis2D::NDOF; ++i) {
        coeffs(i) = solution_(global_dofs[i]);
    }
    return coeffs;
}

Real CGLinearBezierBathymetrySmoother::evaluate_scalar(const VecX &coeffs, Real u, Real v) const {
    return basis_->evaluate_scalar(coeffs, u, v);
}

Vec2 CGLinearBezierBathymetrySmoother::evaluate_gradient_uv(const VecX &coeffs, Real u,
                                                            Real v) const {
    VecX du = basis_->evaluate_du(u, v);
    VecX dv = basis_->evaluate_dv(u, v);

    Real dz_du = coeffs.dot(du);
    Real dz_dv = coeffs.dot(dv);

    return Vec2(dz_du, dz_dv);
}

// =============================================================================
// Output
// =============================================================================

void CGLinearBezierBathymetrySmoother::write_vtk(const std::string &filename,
                                                 int resolution) const {
    if (!solved_) {
        throw std::runtime_error("CGLinearBezierBathymetrySmoother: must call solve() before "
                                 "write_vtk()");
    }

    // Use CG-aware VTK writer with DOF manager's quantization parameters
    // This ensures consistent vertex deduplication between DOF sharing and VTK
    io::write_cg_bezier_surface_vtk(
        filename, *quadtree_, [this](Real x, Real y) { return evaluate(x, y); },
        dof_manager_->xmin_domain(), dof_manager_->ymin_domain(),
        dof_manager_->inv_quantization_tol(), resolution > 0 ? resolution : 6, "elevation");
}

void CGLinearBezierBathymetrySmoother::write_control_points_vtk(const std::string &filename) const {
    if (!solved_) {
        throw std::runtime_error("CGLinearBezierBathymetrySmoother: must call solve() before "
                                 "write_control_points_vtk()");
    }

    io::write_bezier_control_points_vtk(
        filename, *quadtree_, [this](Index e) { return element_coefficients(e); },
        [](int dof) {
            // Linear: control points at corners
            int i = dof % 2;
            int j = dof / 2;
            return Vec2(static_cast<Real>(i), static_cast<Real>(j));
        },
        2); // 2x2 control points
}

// =============================================================================
// Diagnostics
// =============================================================================

Real CGLinearBezierBathymetrySmoother::objective_value() const {
    if (!solved_)
        return 0.0;
    return alpha_ * regularization_energy() + config_.lambda * data_residual();
}

Real CGLinearBezierBathymetrySmoother::constraint_violation() const {
    if (!solved_ || dof_manager_->num_constraints() == 0)
        return 0.0;

    // Compute violation directly from constraints without building sparse matrix
    Real sum_sq = 0.0;
    for (const auto &hc : dof_manager_->constraints()) {
        Real val = solution_(hc.slave_dof);
        for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
            val -= hc.weights[i] * solution_(hc.master_dofs[i]);
        }
        sum_sq += val * val;
    }
    return std::sqrt(sum_sq);
}

} // namespace drifter
