#include "bathymetry/cg_linear_bezier_bathymetry_smoother.hpp"
#include "bathymetry/adaptive_cg_linear_bezier_smoother.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "bathymetry/constraint_condenser.hpp"
#include "core/scoped_timer.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/SparseLU>
#ifdef DRIFTER_USE_METIS
#include <iostream>  // Required before Eigen/MetisSupport (Eigen bug)
#include <Eigen/MetisSupport>
#endif
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>

namespace drifter {

#ifdef DRIFTER_USE_METIS
using SparseSolver = Eigen::SparseLU<SpMat, Eigen::MetisOrdering<int>>;
#else
using SparseSolver = Eigen::SparseLU<SpMat>;
#endif

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
        assemble_hessian_global(*dirichlet_hessian_);
    }
    {
        OptionalScopedTimer t(profile_ ? &profile_->data_fitting_ms : nullptr);
        assemble_data_fitting_global(bathy_func);
    }
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
    VecX b;
    {
        OptionalScopedTimer t(profile_ ? &profile_->matrix_build_ms : nullptr);
        Q = assemble_Q();
        b = assemble_b();
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
    VecX b_reduced;
    {
        OptionalScopedTimer t(profile_ ? &profile_->constraint_condense_ms : nullptr);
        condense_matrix_and_rhs(Q, b, expand_dof, num_free, Q_reduced, b_reduced);
    }

    // Solve reduced system
    SparseSolver solver;
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
        x_free = solver.solve(b_reduced);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGLinearBezierBathymetrySmoother: SparseLU solve failed");
    }

    // Back-substitute: free DOFs directly, slave DOFs from masters
    solution_.setZero(num_dofs);
    for (Index f = 0; f < num_free; ++f) {
        solution_(dof_manager_->free_to_global(f)) = x_free(f);
    }
    back_substitute_slaves(solution_, constraints);
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
