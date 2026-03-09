#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "bathymetry/constraint_condenser.hpp"
#include "core/scoped_timer.hpp"
#include "dg/basis_hexahedron.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#ifdef DRIFTER_USE_METIS
#include <Eigen/MetisSupport>
#include <iostream> // Required before Eigen/MetisSupport (Eigen bug)
#endif
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream> // Required before Eigen/MetisSupport (Eigen bug)
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace drifter {

#ifdef DRIFTER_USE_METIS
using SparseSolver = Eigen::SparseLU<SpMat, Eigen::MetisOrdering<int>>;
#else
using SparseSolver = Eigen::SparseLU<SpMat>;
#endif

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

  // Enable element matrix caching for multigrid CachedRediscretization
  if (config_.use_multigrid) {
    element_matrix_cache_ = &internal_element_cache_;
  }
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
    throw std::runtime_error(
        "CGCubicBezierBathymetrySmoother: bathymetry data not set");
  }

  Index total_constraints = dof_manager_->num_constraints() +
                            dof_manager_->num_edge_derivative_constraints();

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
  if (config_.use_iterative_solver) {
    solve_with_constraints_iterative();
  } else {
    solve_with_constraints_direct();
  }
}

// =============================================================================
// Shared helpers for constrained solve
// =============================================================================

CGCubicBezierBathymetrySmoother::CondensedSystem
CGCubicBezierBathymetrySmoother::build_condensed_system() {
  CondensedSystem sys;
  sys.num_dofs = dof_manager_->num_global_dofs();
  sys.num_free = dof_manager_->num_free_dofs();
  sys.num_edge = dof_manager_->num_edge_derivative_constraints();

  // Build slave lookup for hanging nodes
  std::unordered_map<Index, size_t> slave_to_constraint;
  const auto &constraints = dof_manager_->constraints();
  for (size_t ci = 0; ci < constraints.size(); ++ci) {
    slave_to_constraint[constraints[ci].slave_dof] = ci;
  }

  SpMat Q;
  VecX b;
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->matrix_build_ms
                                         : nullptr);
    Q = assemble_Q();
    b = assemble_b();
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
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->constraint_build_ms
                                         : nullptr);
    condense_matrix_and_rhs(Q, b, expand_dof, sys.num_free, sys.Q_reduced,
                            sys.b_reduced);
  }

  // Build edge constraint matrix on FREE DOFs
  {
    OptionalScopedTimer t(solve_profile_
                              ? &solve_profile_->edge_constraint_assembly_ms
                              : nullptr);
    sys.A_edge = assemble_A_edge_free(expand_dof);
  }

  return sys;
}

void CGCubicBezierBathymetrySmoother::recover_solution_from_free(
    const VecX &x_free, const CondensedSystem &sys) {
  const auto &constraints = dof_manager_->constraints();

  // Recover full solution: free DOFs directly, slave DOFs from masters
  solution_.setZero(sys.num_dofs);
  for (Index f = 0; f < sys.num_free; ++f) {
    solution_(dof_manager_->free_to_global(f)) = x_free(f);
  }
  back_substitute_slaves(solution_, constraints);
}

// =============================================================================
// Direct solver (SparseLU)
// =============================================================================

void CGCubicBezierBathymetrySmoother::solve_with_constraints_direct() {
  // Use constraint condensation for hanging nodes (like linear smoother),
  // then build smaller KKT system for C¹ edge constraints only.

  CondensedSystem sys = build_condensed_system();

  SpMat KKT;
  VecX rhs;
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->kkt_assembly_ms
                                         : nullptr);
    std::tie(KKT, rhs) = assemble_kkt(sys.Q_reduced, sys.A_edge, sys.b_reduced);
  }

  SparseSolver solver;
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_compute_ms
                                         : nullptr);
    solver.compute(KKT);
  }
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("CGCubicBezierBathymetrySmoother: KKT "
                             "SparseLU decomposition failed");
  }

  VecX sol;
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_solve_ms
                                         : nullptr);
    sol = solver.solve(rhs);
  }
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "CGCubicBezierBathymetrySmoother: KKT SparseLU solve failed");
  }

  VecX x_free = sol.head(sys.num_free);
  recover_solution_from_free(x_free, sys);
}

// =============================================================================
// Iterative solver (Augmented Lagrangian CG)
// =============================================================================

void CGCubicBezierBathymetrySmoother::solve_with_constraints_iterative() {
  // Schur complement CG with LU factorization for inner solves:
  //
  // KKT system:
  // [ Q   A^T ] [ x ]   [ b ]
  // [ A    0  ] [ λ ] = [ 0 ]
  //
  // Schur complement: S * λ = A * Q^{-1} * b  where S = A * Q^{-1} * A^T
  // Then: x = Q^{-1} * (b - A^T * λ)
  //
  // Use CG on S with Q^{-1} computed via LU factorization.

  CondensedSystem sys = build_condensed_system();

  VecX x_free;
  int iterations = 0;

  if (sys.num_edge == 0) {
    // No edge constraints - solve Q*x = b directly
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->outer_cg_total_ms
                                         : nullptr);

    if (config_.use_multigrid) {
      // Use PCG with multigrid preconditioner
      BezierMultigridPreconditioner mg_precond(config_.multigrid_config);
      {
        OptionalScopedTimer t_setup(
            solve_profile_ ? &solve_profile_->inner_cg_setup_ms : nullptr);
        // Wire MG profile if available
        if (solve_profile_ && solve_profile_->multigrid_profile) {
          mg_precond.set_profile(solve_profile_->multigrid_profile);
        }
        // Connect element matrix cache for CachedRediscretization strategy
        if (element_matrix_cache_) {
          mg_precond.set_element_matrix_cache(element_matrix_cache_);
        }
        mg_precond.setup(sys.Q_reduced, *quadtree_, *dof_manager_);
      }

      // Preconditioned CG: solve Q*x = b with M = MG preconditioner
      x_free = VecX::Zero(sys.num_free);
      VecX r = sys.b_reduced - sys.Q_reduced * x_free;

      VecX z;
      {
        OptionalScopedTimer t_qinv(
            solve_profile_ ? &solve_profile_->qinv_apply_total_ms : nullptr);
        if (solve_profile_)
          solve_profile_->qinv_apply_calls++;
        z = mg_precond.apply(r);
      }

      VecX p = z;
      Real rz_old = r.dot(z);

      Real b_norm = sys.b_reduced.norm();
      Real tol_sq = config_.tolerance * config_.tolerance * b_norm * b_norm;

      for (int iter = 0; iter < config_.max_iterations; ++iter) {
        VecX Qp = sys.Q_reduced * p;
        Real pQp = p.dot(Qp);

        if (std::abs(pQp) < 1e-30) {
          break; // Breakdown
        }

        Real alpha = rz_old / pQp;
        {
          OptionalScopedTimer t_vec(
              solve_profile_ ? &solve_profile_->cg_vector_ops_ms : nullptr);
          x_free += alpha * p;
          r -= alpha * Qp;
        }

        Real r_norm_sq = r.squaredNorm();
        iterations = iter + 1;

        if (r_norm_sq < tol_sq) {
          break; // Converged
        }

        {
          OptionalScopedTimer t_qinv(
              solve_profile_ ? &solve_profile_->qinv_apply_total_ms : nullptr);
          if (solve_profile_)
            solve_profile_->qinv_apply_calls++;
          z = mg_precond.apply(r);
        }

        {
          OptionalScopedTimer t_vec(
              solve_profile_ ? &solve_profile_->cg_vector_ops_ms : nullptr);
          Real rz_new = r.dot(z);
          p = z + (rz_new / rz_old) * p;
          rz_old = rz_new;
        }
      }
    } else {
      // Direct LU solve
      SparseSolver solver;
      solver.compute(sys.Q_reduced);
      if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "CGCubicBezierBathymetrySmoother: LU factorization failed");
      }
      x_free = solver.solve(sys.b_reduced);
      iterations = 1;
    }

    if (solve_profile_) {
      solve_profile_->outer_cg_iterations = iterations;
    }
  } else {

    std::cout << "Solving with edge constraints ..." << std::endl;
    // Setup solver for Q (either LU or multigrid)
    std::unique_ptr<SparseSolver> Q_solver;
    std::unique_ptr<BezierMultigridPreconditioner> mg_precond;

    // Lambda for applying Q^{-1} with timing
    std::function<VecX(const VecX &)> apply_Qinv;

    {
      OptionalScopedTimer t(solve_profile_ ? &solve_profile_->inner_cg_setup_ms
                                           : nullptr);

      if (config_.use_multigrid) {
        // Setup multigrid preconditioner
        mg_precond = std::make_unique<BezierMultigridPreconditioner>(
            config_.multigrid_config);
        // Wire MG profile if available
        if (solve_profile_ && solve_profile_->multigrid_profile) {
          mg_precond->set_profile(solve_profile_->multigrid_profile);
        }
        // Connect element matrix cache for CachedRediscretization strategy
        if (element_matrix_cache_) {
          mg_precond->set_element_matrix_cache(element_matrix_cache_);
        }
        mg_precond->setup(sys.Q_reduced, *quadtree_, *dof_manager_);

        // Use MG V-cycle as approximate Q^{-1}
        apply_Qinv = [this, &mg_precond](const VecX &v) -> VecX {
          OptionalScopedTimer t_qinv(
              solve_profile_ ? &solve_profile_->qinv_apply_total_ms : nullptr);
          if (solve_profile_)
            solve_profile_->qinv_apply_calls++;
          return mg_precond->apply(v);
        };
      } else {
        // Use LU factorization for exact Q^{-1}
        Q_solver = std::make_unique<SparseSolver>();
        Q_solver->compute(sys.Q_reduced);
        if (Q_solver->info() != Eigen::Success) {
          throw std::runtime_error(
              "CGCubicBezierBathymetrySmoother: Q LU factorization failed");
        }

        apply_Qinv = [this, &Q_solver](const VecX &v) -> VecX {
          OptionalScopedTimer t_qinv(
              solve_profile_ ? &solve_profile_->qinv_apply_total_ms : nullptr);
          if (solve_profile_)
            solve_profile_->qinv_apply_calls++;
          return Q_solver->solve(v);
        };
      }
    }

    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->outer_cg_total_ms
                                         : nullptr);

    Index m = sys.num_edge;

    // Schur complement matvec: S * v = A * Q^{-1} * A^T * v
    auto schur_matvec = [&](const VecX &v) -> VecX {
      OptionalScopedTimer t_schur(
          solve_profile_ ? &solve_profile_->schur_matvec_total_ms : nullptr);
      VecX Atv = sys.A_edge.transpose() * v;
      VecX Qinv_Atv = apply_Qinv(Atv);
      return sys.A_edge * Qinv_Atv;
    };

    // RHS: A * Q^{-1} * b
    VecX Qinv_b, rhs;
    {
      OptionalScopedTimer t_rhs(solve_profile_ ? &solve_profile_->schur_rhs_ms
                                               : nullptr);
      Qinv_b = apply_Qinv(sys.b_reduced);
      rhs = sys.A_edge * Qinv_b;
    }

    // CG on Schur complement S * λ = rhs
    VecX lambda = VecX::Zero(m);
    VecX r = rhs - schur_matvec(lambda);
    VecX p = r;
    Real rs_old = r.squaredNorm();
    Real rhs_norm = rhs.norm();

    Real tol_sq = config_.tolerance * config_.tolerance * rhs_norm * rhs_norm;

    if (rs_old > tol_sq) {
      for (int iter = 0; iter < config_.max_iterations; ++iter) {
        VecX Sp = schur_matvec(p);
        Real pSp = p.dot(Sp);

        if (std::abs(pSp) < 1e-30) {
          break; // Breakdown
        }

        Real alpha = rs_old / pSp;
        {
          OptionalScopedTimer t_vec(
              solve_profile_ ? &solve_profile_->cg_vector_ops_ms : nullptr);
          lambda += alpha * p;
          r -= alpha * Sp;
        }

        Real rs_new = r.squaredNorm();
        iterations = iter + 1;

        // Record per-iteration metrics
        if (solve_profile_) {
          CGIterationMetrics m;
          m.iteration = iter;
          m.schur_residual_norm = std::sqrt(rs_new);
          m.relative_residual = std::sqrt(rs_new) / rhs_norm;
          m.alpha = alpha;
          m.pSp = pSp;
          solve_profile_->iteration_history.push_back(m);
        }

        if (rs_new < tol_sq) {
          break; // Converged
        }

        {
          OptionalScopedTimer t_vec(
              solve_profile_ ? &solve_profile_->cg_vector_ops_ms : nullptr);
          p = r + (rs_new / rs_old) * p;
          rs_old = rs_new;
        }
      }
    }

    // Recover x: x = Q^{-1} * (b - A^T * λ)
    {
      OptionalScopedTimer t_recover(
          solve_profile_ ? &solve_profile_->solution_recovery_ms : nullptr);
      VecX rhs_x = sys.b_reduced - sys.A_edge.transpose() * lambda;
      x_free = apply_Qinv(rhs_x);
    }

    if (solve_profile_) {
      solve_profile_->outer_cg_iterations = iterations;
      solve_profile_->inner_cg_total_calls = solve_profile_->qinv_apply_calls;
    }
  }

  recover_solution_from_free(x_free, sys);
}

void CGCubicBezierBathymetrySmoother::solve_with_constraints_full_kkt() {
  // Original implementation: full KKT system with all constraints
  // (hanging node + edge constraints) without condensation

  Index num_dofs = dof_manager_->num_global_dofs();

  SpMat Q;
  VecX b;
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->matrix_build_ms
                                         : nullptr);
    Q = assemble_Q();
    b = assemble_b();
  }

  SpMat A;
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->constraint_build_ms
                                         : nullptr);
    A = assemble_A();
  }

  SpMat KKT;
  VecX rhs;
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->kkt_assembly_ms
                                         : nullptr);
    std::tie(KKT, rhs) = assemble_kkt(Q, A, b);
  }

  SparseSolver solver;
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_compute_ms
                                         : nullptr);
    solver.compute(KKT);
  }
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("CGCubicBezierBathymetrySmoother: KKT "
                             "SparseLU decomposition failed");
  }

  VecX sol;
  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_solve_ms
                                         : nullptr);
    sol = solver.solve(rhs);
  }
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "CGCubicBezierBathymetrySmoother: KKT SparseLU solve failed");
  }

  solution_ = sol.head(num_dofs);
}

// =============================================================================
// Output
// =============================================================================

void CGCubicBezierBathymetrySmoother::write_vtk(const std::string &filename,
                                                int resolution) const {
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

void CGCubicBezierBathymetrySmoother::write_control_points_vtk(
    const std::string &filename) const {
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
// Constraint matrix assembly
// =============================================================================

SpMat CGCubicBezierBathymetrySmoother::assemble_A_hanging() const {
  Index num_dofs = dof_manager_->num_global_dofs();
  Index num_hanging = dof_manager_->num_constraints();

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(num_hanging * 5);

  Index row = 0;
  for (const auto &hc : dof_manager_->constraints()) {
    triplets.emplace_back(row, hc.slave_dof, 1.0);
    for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
      triplets.emplace_back(row, hc.master_dofs[i], -hc.weights[i]);
    }
    ++row;
  }

  SpMat A(num_hanging, num_dofs);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

SpMat CGCubicBezierBathymetrySmoother::assemble_A_edge() const {
  Index num_dofs = dof_manager_->num_global_dofs();
  Index num_edge = dof_manager_->num_edge_derivative_constraints();

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(num_edge * 32);

  Index row = 0;
  for (const auto &ec : dof_manager_->edge_derivative_constraints()) {
    const auto &global_dofs1 = dof_manager_->element_dofs(ec.elem1);
    const auto &global_dofs2 = dof_manager_->element_dofs(ec.elem2);

    for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
      Real coeff = ec.coeffs1(k) / ec.scale1;
      if (std::abs(coeff) > 1e-14) {
        triplets.emplace_back(row, global_dofs1[k], coeff);
      }
    }
    for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
      Real coeff = ec.coeffs2(k) / ec.scale2;
      if (std::abs(coeff) > 1e-14) {
        triplets.emplace_back(row, global_dofs2[k], -coeff);
      }
    }
    ++row;
  }

  SpMat A(num_edge, num_dofs);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

SpMat CGCubicBezierBathymetrySmoother::assemble_A() const {
  Index num_dofs = dof_manager_->num_global_dofs();
  Index num_hanging = dof_manager_->num_constraints();
  Index num_edge = dof_manager_->num_edge_derivative_constraints();
  Index num_constraints = num_hanging + num_edge;

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(num_constraints * 20);

  Index row = 0;

  // Hanging node constraints
  for (const auto &hc : dof_manager_->constraints()) {
    triplets.emplace_back(row, hc.slave_dof, 1.0);
    for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
      triplets.emplace_back(row, hc.master_dofs[i], -hc.weights[i]);
    }
    ++row;
  }

  // Edge derivative constraints
  for (const auto &ec : dof_manager_->edge_derivative_constraints()) {
    const auto &global_dofs1 = dof_manager_->element_dofs(ec.elem1);
    const auto &global_dofs2 = dof_manager_->element_dofs(ec.elem2);

    for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
      Real coeff = ec.coeffs1(k) / ec.scale1;
      if (std::abs(coeff) > 1e-14) {
        triplets.emplace_back(row, global_dofs1[k], coeff);
      }
    }
    for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
      Real coeff = ec.coeffs2(k) / ec.scale2;
      if (std::abs(coeff) > 1e-14) {
        triplets.emplace_back(row, global_dofs2[k], -coeff);
      }
    }
    ++row;
  }

  SpMat A(num_constraints, num_dofs);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

SpMat CGCubicBezierBathymetrySmoother::assemble_A_edge_free(
    const std::function<std::vector<std::pair<Index, Real>>(Index)> &expand_dof)
    const {

  Index num_free = dof_manager_->num_free_dofs();
  Index num_edge = dof_manager_->num_edge_derivative_constraints();

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(num_edge * 20);

  Index row = 0;
  for (const auto &ec : dof_manager_->edge_derivative_constraints()) {
    const auto &global_dofs1 = dof_manager_->element_dofs(ec.elem1);
    const auto &global_dofs2 = dof_manager_->element_dofs(ec.elem2);

    for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
      auto expanded1 = expand_dof(global_dofs1[k]);
      Real coeff1 = ec.coeffs1(k) / ec.scale1;
      if (std::abs(coeff1) > 1e-14) {
        for (const auto &[free_idx, weight] : expanded1) {
          triplets.emplace_back(row, free_idx, coeff1 * weight);
        }
      }

      auto expanded2 = expand_dof(global_dofs2[k]);
      Real coeff2 = ec.coeffs2(k) / ec.scale2;
      if (std::abs(coeff2) > 1e-14) {
        for (const auto &[free_idx, weight] : expanded2) {
          triplets.emplace_back(row, free_idx, -coeff2 * weight);
        }
      }
    }
    ++row;
  }

  SpMat A(num_edge, num_free);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

// =============================================================================
// Diagnostics
// =============================================================================

Real CGCubicBezierBathymetrySmoother::constraint_violation() const {
  Index num_constraints = dof_manager_->num_constraints() +
                          dof_manager_->num_edge_derivative_constraints();

  if (!solved_ || num_constraints == 0)
    return 0.0;

  SpMat A = assemble_A();
  VecX violation = A * solution_;
  return violation.norm();
}

} // namespace drifter
