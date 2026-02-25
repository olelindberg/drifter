#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "bathymetry/constraint_condenser.hpp"
#include "core/scoped_timer.hpp"
#include "dg/basis_hexahedron.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/SparseLU>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace drifter {

// =============================================================================
// Construction
// =============================================================================

CGCubicBezierBathymetrySmoother::CGCubicBezierBathymetrySmoother(
    const QuadtreeAdapter &mesh, const CGCubicBezierSmootherConfig &config)
    : config_(config) {
    quadtree_ = &mesh;
    init_components();
}

CGCubicBezierBathymetrySmoother::CGCubicBezierBathymetrySmoother(
    const OctreeAdapter &octree, const CGCubicBezierSmootherConfig &config)
    : config_(config) {
    quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
    quadtree_ = quadtree_owned_.get();
    init_components();
}

void CGCubicBezierBathymetrySmoother::init_components() {
    basis_ = std::make_unique<CubicBezierBasis2D>();
    thin_plate_hessian_ =
        std::make_unique<CubicThinPlateHessian>(config_.ngauss_energy);
    dof_manager_ = std::make_unique<CGCubicBezierDofManager>(*quadtree_);
    dof_manager_->build_edge_derivative_constraints(config_.edge_ngauss);

    solution_.setZero(dof_manager_->num_global_dofs());
}

// =============================================================================
// CGBezierSmootherBase virtual method implementations
// =============================================================================

void CGCubicBezierBathymetrySmoother::set_bathymetry_data_impl(
    std::function<Real(Real, Real)> bathy_func) {
    {
        OptionalScopedTimer t(profile_ ? &profile_->hessian_assembly_ms : nullptr);
        assemble_hessian_global(*thin_plate_hessian_);
    }
    {
        OptionalScopedTimer t(profile_ ? &profile_->data_fitting_ms : nullptr);
        assemble_data_fitting_global(bathy_func);
    }
}

// =============================================================================
// Solve
// =============================================================================

void CGCubicBezierBathymetrySmoother::solve() {
    if (!data_set_) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: bathymetry data not set");
    }

    Index total_constraints =
        dof_manager_->num_constraints() + dof_manager_->num_edge_derivative_constraints();

    if (total_constraints == 0) {
        solve_unconstrained();
    } else if (config_.use_condensation) {
        solve_with_constraints();
    } else {
        solve_with_constraints_full_kkt();
    }

    solved_ = true;
}

void CGCubicBezierBathymetrySmoother::solve_with_constraints() {
    // Use constraint condensation for hanging nodes (like linear smoother),
    // then build smaller KKT system for C¹ edge constraints only.

    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_free = dof_manager_->num_free_dofs();
    Index num_hanging_constraints = dof_manager_->num_constraints();
    Index num_edge_constraints = dof_manager_->num_edge_derivative_constraints();

    // Build slave lookup for hanging nodes
    std::unordered_map<Index, size_t> slave_to_constraint;
    const auto &constraints = dof_manager_->constraints();
    for (size_t ci = 0; ci < constraints.size(); ++ci) {
        slave_to_constraint[constraints[ci].slave_dof] = ci;
    }

    SpMat Q;
    VecX c;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->matrix_build_ms : nullptr);

        Q = alpha_ * H_global_ + config_.lambda * BtWB_global_;
        for (Index i = 0; i < num_dofs; ++i) {
            Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
        }

        c = -config_.lambda * BtWd_global_;
    }

    // Define expand_dof to map global DOF to (free_index, weight) pairs
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

    // Condense hanging node constraints
    SpMat Q_reduced;
    VecX c_reduced;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->constraint_build_ms : nullptr);
        condense_matrix_and_rhs(Q, c, expand_dof, num_free, Q_reduced, c_reduced);
    }

    // Build edge constraint matrix on FREE DOFs
    SpMat A_edge;
    {
        std::vector<Eigen::Triplet<Real>> A_triplets;
        A_triplets.reserve(num_edge_constraints * 20);

        Index row = 0;
        for (const auto &ec : dof_manager_->edge_derivative_constraints()) {
            const auto &global_dofs1 = dof_manager_->element_dofs(ec.elem1);
            const auto &global_dofs2 = dof_manager_->element_dofs(ec.elem2);

            for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
                // Map global DOF to free index via expand_dof
                auto expanded1 = expand_dof(global_dofs1[k]);
                Real coeff1 = ec.coeffs1(k) / ec.scale1;
                if (std::abs(coeff1) > 1e-14) {
                    for (const auto &[free_idx, weight] : expanded1) {
                        A_triplets.emplace_back(row, free_idx, coeff1 * weight);
                    }
                }

                auto expanded2 = expand_dof(global_dofs2[k]);
                Real coeff2 = ec.coeffs2(k) / ec.scale2;
                if (std::abs(coeff2) > 1e-14) {
                    for (const auto &[free_idx, weight] : expanded2) {
                        A_triplets.emplace_back(row, free_idx, -coeff2 * weight);
                    }
                }
            }
            ++row;
        }

        A_edge.resize(num_edge_constraints, num_free);
        A_edge.setFromTriplets(A_triplets.begin(), A_triplets.end());
    }

    SpMat KKT;
    VecX rhs;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->kkt_assembly_ms : nullptr);

        // Build smaller KKT: (num_free + num_edge) instead of (num_dofs + num_hanging + num_edge)
        Index kkt_size = num_free + num_edge_constraints;
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(Q_reduced.nonZeros() + 2 * A_edge.nonZeros());

        // Q_reduced block
        for (int k = 0; k < Q_reduced.outerSize(); ++k) {
            for (SpMat::InnerIterator it(Q_reduced, k); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }

        // A_edge and A_edge^T blocks
        for (int k = 0; k < A_edge.outerSize(); ++k) {
            for (SpMat::InnerIterator it(A_edge, k); it; ++it) {
                triplets.emplace_back(num_free + it.row(), it.col(), it.value());
                triplets.emplace_back(it.col(), num_free + it.row(), it.value());
            }
        }

        KKT.resize(kkt_size, kkt_size);
        KKT.setFromTriplets(triplets.begin(), triplets.end());

        Real constraint_reg = 1e-10;
        for (Index i = num_free; i < kkt_size; ++i) {
            KKT.coeffRef(i, i) -= constraint_reg;
        }

        rhs.resize(kkt_size);
        rhs.head(num_free) = -c_reduced;
        rhs.tail(num_edge_constraints).setZero();
    }

    Eigen::SparseLU<SpMat> solver;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_compute_ms : nullptr);
        solver.compute(KKT);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: KKT "
                                 "SparseLU decomposition failed");
    }

    VecX sol;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_solve_ms : nullptr);
        sol = solver.solve(rhs);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: KKT SparseLU solve failed");
    }

    VecX x_free = sol.head(num_free);

    // Recover full solution: free DOFs directly, slave DOFs from masters
    solution_.setZero(num_dofs);
    for (Index f = 0; f < num_free; ++f) {
        solution_(dof_manager_->free_to_global(f)) = x_free(f);
    }
    back_substitute_slaves(solution_, constraints);

    // Project onto edge constraint manifold (for numerical accuracy)
    if (num_edge_constraints > 0) {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->constraint_projection_ms : nullptr);

        // Build full constraint matrix (on global DOFs) for projection
        std::vector<Eigen::Triplet<Real>> A_triplets;
        Index row = 0;

        for (const auto &ec : dof_manager_->edge_derivative_constraints()) {
            const auto &global_dofs1 = dof_manager_->element_dofs(ec.elem1);
            const auto &global_dofs2 = dof_manager_->element_dofs(ec.elem2);

            for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
                Real coeff1 = ec.coeffs1(k) / ec.scale1;
                if (std::abs(coeff1) > 1e-14) {
                    A_triplets.emplace_back(row, global_dofs1[k], coeff1);
                }
                Real coeff2 = ec.coeffs2(k) / ec.scale2;
                if (std::abs(coeff2) > 1e-14) {
                    A_triplets.emplace_back(row, global_dofs2[k], -coeff2);
                }
            }
            ++row;
        }

        SpMat A_full(num_edge_constraints, num_dofs);
        A_full.setFromTriplets(A_triplets.begin(), A_triplets.end());

        VecX Ax = A_full * solution_;
        SpMat AAt = A_full * A_full.transpose();

        for (Index i = 0; i < num_edge_constraints; ++i) {
            AAt.coeffRef(i, i) += 1e-14;
        }

        Eigen::SparseLU<SpMat> projector;
        projector.compute(AAt);

        if (projector.info() == Eigen::Success) {
            VecX lambda = projector.solve(Ax);
            if (projector.info() == Eigen::Success) {
                VecX correction = A_full.transpose() * lambda;
                solution_ -= correction;

                // Re-run back-substitution to restore hanging node constraints
                // (projection may have modified slave DOFs directly)
                back_substitute_slaves(solution_, constraints);
            }
        }
    }
}

void CGCubicBezierBathymetrySmoother::solve_with_constraints_full_kkt() {
    // Original implementation: full KKT system with all constraints
    // (hanging node + edge constraints) without condensation

    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_hanging_constraints = dof_manager_->num_constraints();
    Index num_edge_constraints = dof_manager_->num_edge_derivative_constraints();
    Index num_constraints = num_hanging_constraints + num_edge_constraints;

    SpMat Q;
    VecX c;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->matrix_build_ms : nullptr);

        Q = alpha_ * H_global_ + config_.lambda * BtWB_global_;
        for (Index i = 0; i < num_dofs; ++i) {
            Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
        }

        c = -config_.lambda * BtWd_global_;
    }

    SpMat A;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->constraint_build_ms : nullptr);

        std::vector<Eigen::Triplet<Real>> A_triplets;
        A_triplets.reserve(num_constraints * 20);

        Index row = 0;

        // Hanging node constraints
        for (const auto &hc : dof_manager_->constraints()) {
            A_triplets.emplace_back(row, hc.slave_dof, 1.0);
            for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
                A_triplets.emplace_back(row, hc.master_dofs[i], -hc.weights[i]);
            }
            ++row;
        }

        // C¹ edge derivative constraints
        for (const auto &ec : dof_manager_->edge_derivative_constraints()) {
            const auto &global_dofs1 = dof_manager_->element_dofs(ec.elem1);
            const auto &global_dofs2 = dof_manager_->element_dofs(ec.elem2);

            for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
                Real coeff = ec.coeffs1(k) / ec.scale1;
                if (std::abs(coeff) > 1e-14) {
                    A_triplets.emplace_back(row, global_dofs1[k], coeff);
                }
            }

            for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
                Real coeff = ec.coeffs2(k) / ec.scale2;
                if (std::abs(coeff) > 1e-14) {
                    A_triplets.emplace_back(row, global_dofs2[k], -coeff);
                }
            }

            ++row;
        }

        A.resize(num_constraints, num_dofs);
        A.setFromTriplets(A_triplets.begin(), A_triplets.end());
    }

    SpMat KKT;
    VecX rhs;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->kkt_assembly_ms : nullptr);

        // Build KKT system
        Index kkt_size = num_dofs + num_constraints;
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(Q.nonZeros() + 2 * A.nonZeros());

        // Copy Q block
        for (int k = 0; k < Q.outerSize(); ++k) {
            for (SpMat::InnerIterator it(Q, k); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }

        // Single pass over A: add both A (lower-left) and A^T (upper-right) blocks
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SpMat::InnerIterator it(A, k); it; ++it) {
                triplets.emplace_back(num_dofs + it.row(), it.col(), it.value());
                triplets.emplace_back(it.col(), num_dofs + it.row(), it.value());
            }
        }

        KKT.resize(kkt_size, kkt_size);
        KKT.setFromTriplets(triplets.begin(), triplets.end());

        Real constraint_reg = 1e-10;
        for (Index i = num_dofs; i < kkt_size; ++i) {
            KKT.coeffRef(i, i) -= constraint_reg;
        }

        rhs.resize(kkt_size);
        rhs.head(num_dofs) = -c;
        rhs.tail(num_constraints).setZero();
    }

    Eigen::SparseLU<SpMat> solver;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_compute_ms : nullptr);
        solver.compute(KKT);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: KKT "
                                 "SparseLU decomposition failed");
    }

    VecX sol;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_solve_ms : nullptr);
        sol = solver.solve(rhs);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: KKT SparseLU solve failed");
    }

    solution_ = sol.head(num_dofs);

    // Project onto constraint manifold (for numerical accuracy)
    if (num_constraints > 0) {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->constraint_projection_ms : nullptr);

        VecX Ax = A * solution_;
        SpMat AAt = A * A.transpose();

        for (Index i = 0; i < num_constraints; ++i) {
            AAt.coeffRef(i, i) += 1e-14;
        }

        Eigen::SparseLU<SpMat> projector;
        projector.compute(AAt);

        if (projector.info() == Eigen::Success) {
            VecX lambda = projector.solve(Ax);
            if (projector.info() == Eigen::Success) {
                VecX correction = A.transpose() * lambda;
                solution_ -= correction;
            }
        }
    }
}

// =============================================================================
// Output
// =============================================================================

void CGCubicBezierBathymetrySmoother::write_vtk(const std::string &filename, int resolution) const {
    if (!solved_) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: must call "
                                 "solve() before write_vtk()");
    }

    // Use CG-aware VTK writer that deduplicates shared vertices at element
    // boundaries, producing a properly connected mesh without visual gaps
    io::write_cg_bezier_surface_vtk(
        filename, *quadtree_, [this](Real x, Real y) { return evaluate(x, y); },
        resolution > 0 ? resolution : 9, "elevation");
}

void CGCubicBezierBathymetrySmoother::write_control_points_vtk(const std::string &filename) const {
    if (!solved_) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: must call "
                                 "solve() before write_control_points_vtk()");
    }

    io::write_bezier_control_points_vtk(
        filename, *quadtree_, [this](Index e) { return element_coefficients(e); },
        [](int dof) {
            int i = dof % 4;
            int j = dof / 4;
            return Vec2(static_cast<Real>(i) / 3, static_cast<Real>(j) / 3);
        },
        4);
}

// =============================================================================
// Diagnostics
// =============================================================================

Real CGCubicBezierBathymetrySmoother::constraint_violation() const {
    Index num_hanging = dof_manager_->num_constraints();
    Index num_edge = dof_manager_->num_edge_derivative_constraints();
    Index num_constraints = num_hanging + num_edge;

    if (!solved_ || num_constraints == 0)
        return 0.0;

    Index num_dofs = dof_manager_->num_global_dofs();
    std::vector<Eigen::Triplet<Real>> A_triplets;
    Index row = 0;

    for (const auto &hc : dof_manager_->constraints()) {
        A_triplets.emplace_back(row, hc.slave_dof, 1.0);
        for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
            A_triplets.emplace_back(row, hc.master_dofs[i], -hc.weights[i]);
        }
        ++row;
    }

    for (const auto &ec : dof_manager_->edge_derivative_constraints()) {
        const auto &global_dofs1 = dof_manager_->element_dofs(ec.elem1);
        const auto &global_dofs2 = dof_manager_->element_dofs(ec.elem2);

        for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
            Real coeff = ec.coeffs1(k) / ec.scale1;
            if (std::abs(coeff) > 1e-14) {
                A_triplets.emplace_back(row, global_dofs1[k], coeff);
            }
        }
        for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
            Real coeff = ec.coeffs2(k) / ec.scale2;
            if (std::abs(coeff) > 1e-14) {
                A_triplets.emplace_back(row, global_dofs2[k], -coeff);
            }
        }
        ++row;
    }

    SpMat A(num_constraints, num_dofs);
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());

    VecX violation = A * solution_;
    return violation.norm();
}

} // namespace drifter
