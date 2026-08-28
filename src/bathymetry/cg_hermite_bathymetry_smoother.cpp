#include "bathymetry/cg_hermite_bathymetry_smoother.hpp"
#include "bathymetry/adaptive_cg_hermite_smoother.hpp" // HermiteIterationProfile
#include "bathymetry/constraint_condenser.hpp"
#include "core/logger.hpp"
#include "core/scoped_timer.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace drifter {

namespace {

/// Q_red is symmetric positive definite, so a Cholesky-type factorisation
/// replaces the SparseLU / Schur-complement machinery the Bezier path needs.
using HermiteSparseSolver = Eigen::SimplicialLDLT<SpMat>;

} // namespace

// =============================================================================
// Construction
// =============================================================================

CGHermiteBathymetrySmoother::CGHermiteBathymetrySmoother(const QuadtreeAdapter &mesh,
                                                         const CGHermiteSmootherConfig &config)
    : config_(config) {
    quadtree_ = &mesh;
    init_components();
}

CGHermiteBathymetrySmoother::CGHermiteBathymetrySmoother(const OctreeAdapter &octree,
                                                         const CGHermiteSmootherConfig &config)
    : config_(config) {
    quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
    quadtree_ = quadtree_owned_.get();
    init_components();
}

void CGHermiteBathymetrySmoother::init_components() {
    const int r = config_.continuity_order;
    if (r < 0 || r > 1) {
        throw std::invalid_argument(
            "CGHermiteBathymetrySmoother: continuity_order must be 0 (C0) or 1 (C1), got " +
            std::to_string(r));
    }

    basis_ = std::make_unique<HermiteBasis2D>(r);
    hessian_ = std::make_unique<HermiteHessian>(r);
    // Land only: a NoData gap must be interpolated across, never pinned to 0.
    dof_manager_ = std::make_unique<CGHermiteDofManager>(
        *quadtree_, r, config_.enable_zero_gradient_bc, is_land_func_);

    slave_to_constraint_.clear();
    const auto &constraints = dof_manager_->constraints();
    for (size_t ci = 0; ci < constraints.size(); ++ci) {
        slave_to_constraint_[constraints[ci].slave_dof] = ci;
    }

    solution_.setZero(dof_manager_->num_global_dofs());
}

// =============================================================================
// Basis-dependent hooks
// =============================================================================

VecX CGHermiteBathymetrySmoother::element_dof_scaling(Real dx, Real dy) const {
    return basis_->dof_scaling(dx, dy);
}

VecX CGHermiteBathymetrySmoother::element_bernstein_coefficients(Index elem) const {
    const auto &bounds = quadtree_->element_bounds(elem);
    const Real dx = bounds.xmax - bounds.xmin;
    const Real dy = bounds.ymax - bounds.ymin;

    // element_coefficients() returns Lambda_e * q_e; M_e expects the raw physical
    // derivative DOFs q_e, so undo the scaling before the change of basis.
    const VecX scaled = element_coefficients(elem);
    const VecX q = scaled.cwiseQuotient(basis_->dof_scaling(dx, dy));

    return basis_->bernstein_change_of_basis(dx, dy) * q;
}

VecX CGHermiteBathymetrySmoother::ridge_diagonal() const {
    // lambda*epsilon applied in the equilibrated space is lambda*epsilon*S^-2 in
    // original coordinates. A uniform ridge would penalise z and z_xy identically
    // despite their entries differing by orders of magnitude, distorting exactly
    // the null-space modes the ridge exists to control.
    const VecX S = dof_manager_->equilibration_scaling();
    return lambda() * ridge_epsilon() * S.cwiseProduct(S).cwiseInverse();
}

// =============================================================================
// Data input
// =============================================================================

void CGHermiteBathymetrySmoother::set_bathymetry_data_impl(
    std::function<Real(Real, Real)> bathy_func) {
    relaxation_config_ = config_.boundary_relaxation;

    // The land region is only known once the data source is attached, and it
    // changes the constraint set, so the DOF manager is rebuilt here rather than in
    // init_components(). Cheap next to the assembly that follows.
    if (is_land_func_) {
        init_components();
        LOG_DEBUG("Hermite land Dirichlet: " << dof_manager_->num_land_pinned_dofs()
                                             << " DOFs pinned to depth 0");
    }

    {
        OptionalScopedTimer t(profile_ ? &profile_->hessian_assembly_ms : nullptr);
        assemble_hessian_global(*hessian_);
    }
    {
        OptionalScopedTimer t(profile_ ? &profile_->data_fitting_ms : nullptr);
        assemble_data_fitting_global(bathy_func);
    }
}

// =============================================================================
// Constraint condensation
// =============================================================================

std::vector<std::pair<Index, Real>> CGHermiteBathymetrySmoother::expand_dof(Index g) const {
    const Index f = dof_manager_->global_to_free(g);
    if (f >= 0) {
        return {{f, 1.0}};
    }

    auto it = slave_to_constraint_.find(g);
    if (it == slave_to_constraint_.end()) {
        return {};
    }

    // Constraints are transitively closed, so every master here is free
    const auto &c = dof_manager_->constraints()[it->second];
    std::vector<std::pair<Index, Real>> result;
    result.reserve(c.master_dofs.size());
    for (size_t i = 0; i < c.master_dofs.size(); ++i) {
        const Index mf = dof_manager_->global_to_free(c.master_dofs[i]);
        if (mf >= 0) {
            result.emplace_back(mf, c.weights[i]);
        }
    }
    return result;
}

void CGHermiteBathymetrySmoother::build_condensed_system(SpMat &Q_reduced,
                                                         VecX &b_reduced) const {
    const SpMat Q = assemble_Q();
    const VecX b = assemble_b();
    condense_matrix_and_rhs(Q, b, [this](Index g) { return expand_dof(g); },
                            dof_manager_->num_free_dofs(), Q_reduced, b_reduced);
}

SpMat CGHermiteBathymetrySmoother::condensed_matrix() const {
    SpMat Q_reduced;
    VecX b_reduced;
    build_condensed_system(Q_reduced, b_reduced);
    return Q_reduced;
}

// =============================================================================
// Solve
// =============================================================================

void CGHermiteBathymetrySmoother::solve() {
    if (!data_set_) {
        throw std::runtime_error("CGHermiteBathymetrySmoother: bathymetry data not set");
    }

    const Index num_dofs = dof_manager_->num_global_dofs();
    const Index num_free = dof_manager_->num_free_dofs();

    SpMat Q_reduced;
    VecX b_reduced;
    {
        OptionalScopedTimer t(profile_ ? &profile_->matrix_build_ms : nullptr);
        OptionalScopedTimer t2(profile_ ? &profile_->constraint_condense_ms : nullptr);
        build_condensed_system(Q_reduced, b_reduced);
    }

    // Symmetric equilibration: solve (S Q S)(S^-1 x) = S b. A congruence, so
    // symmetry and positive definiteness are preserved.
    VecX S_free = VecX::Ones(num_free);
    if (config_.use_equilibration) {
        const VecX S = dof_manager_->equilibration_scaling();
        for (Index f = 0; f < num_free; ++f) {
            S_free(f) = S(dof_manager_->free_to_global(f));
        }
        Q_reduced = S_free.asDiagonal() * Q_reduced * S_free.asDiagonal();
        b_reduced = b_reduced.cwiseProduct(S_free);
    }

    HermiteSparseSolver solver;
    {
        OptionalScopedTimer t(profile_ ? &profile_->ldlt_compute_ms : nullptr);
        solver.compute(Q_reduced);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "CGHermiteBathymetrySmoother: SimplicialLDLT factorisation of the condensed system "
            "failed. The system should be symmetric positive definite; check that ridge_epsilon > 0"
            + std::string(config_.use_equilibration
                              ? "."
                              : ", or enable use_equilibration on a deeply refined mesh."));
    }

    VecX y;
    {
        OptionalScopedTimer t(profile_ ? &profile_->ldlt_solve_ms : nullptr);
        y = solver.solve(b_reduced);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGHermiteBathymetrySmoother: SimplicialLDLT solve failed");
    }

    // Undo the change of variables: x = S y
    const VecX x_free = config_.use_equilibration ? y.cwiseProduct(S_free) : y;

    solution_.setZero(num_dofs);
    for (Index f = 0; f < num_free; ++f) {
        solution_(dof_manager_->free_to_global(f)) = x_free(f);
    }
    back_substitute_slaves(solution_, dof_manager_->constraints());

    solved_ = true;

    if (config_.verbose) {
        std::cout << "CGHermiteBathymetrySmoother: r=" << config_.continuity_order
                  << " dofs=" << num_dofs << " free=" << num_free
                  << " constraints=" << dof_manager_->num_constraints()
                  << " data_residual=" << data_residual() << std::endl;
    }
}

// =============================================================================
// Output
// =============================================================================

void CGHermiteBathymetrySmoother::write_vtk(const std::string &filename, int order) const {
    if (!solved_) {
        throw std::runtime_error("CGHermiteBathymetrySmoother: must solve() before write_vtk()");
    }

    io::write_high_order_surface_vtk(
        filename, *quadtree_,
        [this](Index elem, Real x, Real y) { return evaluate_in_element(elem, x, y); },
        order > 0 ? std::max(order, surface_degree()) : surface_degree(), "elevation");
}

void CGHermiteBathymetrySmoother::write_control_points_vtk(const std::string &filename) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGHermiteBathymetrySmoother: must solve() before write_control_points_vtk()");
    }

    const int n1d = basis_->num_nodes_1d();
    io::write_bezier_control_points_vtk(
        filename, *quadtree_,
        [this](Index elem) { return element_bernstein_coefficients(elem); },
        [n1d](int dof) {
            return Vec2(static_cast<Real>(dof % n1d) / (n1d - 1),
                        static_cast<Real>(dof / n1d) / (n1d - 1));
        },
        n1d);
}

} // namespace drifter
