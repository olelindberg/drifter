#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "bathymetry/constraint_condenser.hpp"
#include "bathymetry/diagonal_approx_cg_schur_preconditioner.hpp"
#include "bathymetry/diagonal_schur_preconditioner.hpp"
#include "bathymetry/flexible_cg.hpp"
#include "bathymetry/gauss_seidel_schur_preconditioner.hpp"
#include "bathymetry/multigrid_schur_preconditioner.hpp"
#include "bathymetry/schwarz_colored_schur_preconditioner.hpp"
#include "bathymetry/physics_based_schur_preconditioner.hpp"
#include "bathymetry/schur_preconditioner.hpp"
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
  if (config_.enable_natural_bc) {
    dof_manager_->build_boundary_curvature_constraints(config_.edge_ngauss);
  }
  if (config_.enable_zero_gradient_bc) {
    dof_manager_->build_boundary_gradient_constraints(config_.edge_ngauss);
  }

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
  // Pass boundary relaxation config to base class
  relaxation_config_ = config_.boundary_relaxation;

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
                            dof_manager_->num_edge_derivative_constraints() +
                            dof_manager_->num_boundary_curvature_constraints() +
                            dof_manager_->num_boundary_gradient_constraints();

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
  sys.num_edge = dof_manager_->num_edge_derivative_constraints() +
                 dof_manager_->num_boundary_curvature_constraints() +
                 dof_manager_->num_boundary_gradient_constraints();

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

  // Build constraint matrices on FREE DOFs (edge + boundary curvature +
  // boundary gradient)
  {
    OptionalScopedTimer t(solve_profile_
                              ? &solve_profile_->edge_constraint_assembly_ms
                              : nullptr);
    SpMat A_edge = assemble_A_edge_free(expand_dof);
    SpMat A_boundary = assemble_A_boundary_free(expand_dof);
    SpMat A_gradient = assemble_A_gradient_free(expand_dof);

    // Stack edge, boundary curvature, and boundary gradient constraints
    // vertically
    Index num_edge = A_edge.rows();
    Index num_boundary = A_boundary.rows();
    Index num_gradient = A_gradient.rows();
    Index total_constraints = num_edge + num_boundary + num_gradient;

    if (total_constraints > 0) {
      std::vector<Eigen::Triplet<Real>> triplets;
      triplets.reserve(A_edge.nonZeros() + A_boundary.nonZeros() +
                       A_gradient.nonZeros());

      // Copy edge constraints
      for (int k = 0; k < A_edge.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A_edge, k); it; ++it) {
          triplets.emplace_back(it.row(), it.col(), it.value());
        }
      }

      // Copy boundary curvature constraints (offset by num_edge rows)
      for (int k = 0; k < A_boundary.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A_boundary, k); it; ++it) {
          triplets.emplace_back(num_edge + it.row(), it.col(), it.value());
        }
      }

      // Copy boundary gradient constraints (offset by num_edge + num_boundary
      // rows)
      for (int k = 0; k < A_gradient.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A_gradient, k); it; ++it) {
          triplets.emplace_back(num_edge + num_boundary + it.row(), it.col(),
                                it.value());
        }
      }

      sys.A_edge.resize(total_constraints, sys.num_free);
      sys.A_edge.setFromTriplets(triplets.begin(), triplets.end());
    } else {
      sys.A_edge.resize(0, sys.num_free);
    }
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
  // Preconditioned Schur complement CG:
  //
  // KKT system:
  // [ Q   A^T ] [ x ]   [ b ]
  // [ A    0  ] [ λ ] = [ 0 ]
  //
  // Schur complement: S * λ = A * Q^{-1} * b  where S = A * Q^{-1} * A^T
  // Then: x = Q^{-1} * (b - A^T * λ)
  //
  // Preconditioner options:
  // - None: Unpreconditioned CG (monitors ||r||_2)
  // - Diagonal: M_S = diag(S)
  // - PhysicsBased: M_S = C * K^{-1} * C^T where K = alpha * H (smoothness)
  // - MultigridVCycle: M_S^{-1} v = C * A_mg^{-1} * C^T * v (requires FCG)

  CondensedSystem sys = build_condensed_system();

  VecX x_free;
  int iterations = 0;

  // Setup solver for Q (either LU or multigrid)
  std::unique_ptr<SparseSolver> Q_solver;
  std::unique_ptr<BezierMultigridPreconditioner> mg_precond;

  // Lambda for applying Q^{-1} with timing
  std::function<VecX(const VecX &)> apply_Qinv;

  {
    OptionalScopedTimer t(solve_profile_ ? &solve_profile_->inner_cg_setup_ms
                                         : nullptr);

    // Determine if we need exact Q^{-1} for Schur matvec
    // When using MultigridVCycle preconditioner with use_exact_schur_matvec=true,
    // schur_matvec must use exact LU to ensure consistency with the preconditioner.
    // Otherwise, the CG iteration diverges because schur_matvec (1 V-cycle) and
    // preconditioner (5 V-cycles with iterative refinement) approximate different
    // Schur complements.
    bool need_exact_schur_matvec =
        config_.use_exact_schur_matvec &&
        config_.schur_preconditioner == SchurPreconditionerType::MultigridVCycle;

    // Setup MG preconditioner if requested (used for Schur preconditioner)
    if (config_.use_multigrid) {
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
    }

    // Setup LU factorization if needed (for schur_matvec or when MG not used)
    if (!config_.use_multigrid || need_exact_schur_matvec) {
      Q_solver = std::make_unique<SparseSolver>();
      Q_solver->compute(sys.Q_reduced);
      if (Q_solver->info() != Eigen::Success) {
        throw std::runtime_error(
            "CGCubicBezierBathymetrySmoother: Q LU factorization failed");
      }
    }

    // Choose apply_Qinv based on what's available and required
    if (Q_solver) {
      // Use exact LU for Q^{-1} (more accurate)
      apply_Qinv = [this, &Q_solver](const VecX &v) -> VecX {
        OptionalScopedTimer t_qinv(
            solve_profile_ ? &solve_profile_->qinv_apply_total_ms : nullptr);
        if (solve_profile_)
          solve_profile_->qinv_apply_calls++;
        return Q_solver->solve(v);
      };
    } else {
      // Use MG V-cycle as approximate Q^{-1}
      apply_Qinv = [this, &mg_precond](const VecX &v) -> VecX {
        OptionalScopedTimer t_qinv(
            solve_profile_ ? &solve_profile_->qinv_apply_total_ms : nullptr);
        if (solve_profile_)
          solve_profile_->qinv_apply_calls++;
        return mg_precond->apply(v);
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

  //-------------------------------------------------------------------------//
  // Setup Schur complement preconditioner
  //-------------------------------------------------------------------------//
  std::unique_ptr<ISchurPreconditioner> schur_precond;
  {
    OptionalScopedTimer t_precond(
        solve_profile_ ? &solve_profile_->schur_precond_setup_ms : nullptr);

    switch (config_.schur_preconditioner) {
    case SchurPreconditionerType::Diagonal:
      schur_precond =
          std::make_unique<DiagonalSchurPreconditioner>(schur_matvec, m);
      break;

    case SchurPreconditionerType::PhysicsBased:
      // Use sys.Q_reduced = alpha*H + lambda*BtWB (full system matrix)
      // This is already SPD and matches the actual Schur complement matrix
      schur_precond = std::make_unique<PhysicsBasedSchurPreconditioner>(
          sys.Q_reduced, sys.A_edge);
      break;

    case SchurPreconditionerType::MultigridVCycle:
      if (!mg_precond) {
        throw std::runtime_error(
            "CGCubicBezierBathymetrySmoother: MultigridVCycle Schur "
            "preconditioner requires use_multigrid=true");
      }
      // Pass Q matrix for proper iterative refinement with V-cycles
      schur_precond = std::make_unique<MultigridSchurPreconditioner>(
          *mg_precond, sys.Q_reduced, sys.A_edge);
      break;

    case SchurPreconditionerType::GaussSeidel:
      schur_precond =
          std::make_unique<GaussSeidelSchurPreconditioner>(schur_matvec, m);
      break;

    case SchurPreconditionerType::SchwarzColored:
      schur_precond = std::make_unique<SchwarzColoredSchurPreconditioner>(
          schur_matvec, m, dof_manager_->edge_derivative_constraints(),
          dof_manager_->boundary_curvature_constraints(),
          dof_manager_->boundary_gradient_constraints(),
          config_.schwarz_schur_iterations);
      break;

    case SchurPreconditionerType::DiagonalApproxCG:
      schur_precond = std::make_unique<DiagonalApproxCGSchurPreconditioner>(
          sys.Q_reduced, sys.A_edge, config_.inner_tolerance,
          config_.inner_max_iterations);
      break;

    case SchurPreconditionerType::None:
    default:
      // No preconditioner
      break;
    }
  }

  // Lambda for applying Schur preconditioner with timing
  auto apply_schur_precond = [&](const VecX &r) -> VecX {
    if (!schur_precond) {
      return r; // Identity (unpreconditioned)
    }
    OptionalScopedTimer t_precond(
        solve_profile_ ? &solve_profile_->schur_precond_apply_total_ms
                       : nullptr);
    if (solve_profile_) {
      solve_profile_->schur_precond_apply_calls++;
    }
    return schur_precond->apply(r);
  };

  //-------------------------------------------------------------------------//
  // Solve Schur complement system S * λ = rhs with PCG or FCG
  //-------------------------------------------------------------------------//
  VecX lambda = VecX::Zero(m);

  if (schur_precond && schur_precond->is_variable()) {
    //-----------------------------------------------------------------------//
    // Flexible CG for variable preconditioner (MG V-cycle)
    //-----------------------------------------------------------------------//
    FlexibleCG fcg(schur_matvec, *schur_precond, config_.tolerance,
                   config_.max_iterations);
    fcg.set_verbose(config_.verbose);
    FCGResult result = fcg.solve(lambda, rhs);
    iterations = result.iterations;

    // Record iteration history
    if (solve_profile_) {
      for (size_t i = 0; i < result.residual_history.size(); ++i) {
        CGIterationMetrics met;
        met.iteration = static_cast<int>(i);
        met.precond_residual_norm = result.residual_history[i];
        met.schur_residual_norm = 0.0; // Not tracked in FCG
        met.relative_residual =
            (i == 0) ? 1.0
                     : result.residual_history[i] / result.residual_history[0];
        met.alpha = 0.0;
        met.pSp = 0.0;
        solve_profile_->iteration_history.push_back(met);
      }
    }
  } else {
    //-----------------------------------------------------------------------//
    // Preconditioned CG (or unpreconditioned if no preconditioner)
    //-----------------------------------------------------------------------//
    VecX r = rhs - schur_matvec(lambda);
    VecX z = apply_schur_precond(r);
    VecX p = z;

    // Preconditioned residual: ||r||^2_{M^{-1}} = r^T z
    Real rz_old = r.dot(z);
    Real rz_init = rz_old;
    Real rhs_norm = rhs.norm();

    // Handle zero RHS
    if (std::abs(rz_init) < 1e-30) {
      iterations = 0;
    } else {
      Real tol_sq = config_.tolerance * config_.tolerance;

      for (int iter = 0; iter < config_.max_iterations; ++iter) {
        VecX Sp = schur_matvec(p);
        Real pSp = p.dot(Sp);

        if (std::abs(pSp) < 1e-30) {
          break; // Breakdown
        }

        Real alpha = rz_old / pSp;
        {
          OptionalScopedTimer t_vec(
              solve_profile_ ? &solve_profile_->cg_vector_ops_ms : nullptr);
          lambda += alpha * p;
          r -= alpha * Sp;
        }

        z = apply_schur_precond(r);
        Real rz_new = r.dot(z);
        iterations = iter + 1;

        // Preconditioned residual norm: ||r||_{M^{-1}} = sqrt(r^T z)
        Real precond_res = std::sqrt(std::abs(rz_new));
        Real unprecond_res = r.norm();

        // Record per-iteration metrics
        if (solve_profile_) {
          CGIterationMetrics met;
          met.iteration = iter;
          met.schur_residual_norm = unprecond_res;
          met.precond_residual_norm = precond_res;
          met.relative_residual = precond_res / std::sqrt(rz_init);
          met.alpha = alpha;
          met.pSp = pSp;
          solve_profile_->iteration_history.push_back(met);
        }

        if (config_.verbose) {
          std::cout << "[SchurPCG] iter=" << iterations
                    << " ||r||_2=" << unprecond_res
                    << " ||r||_{M^-1}=" << precond_res
                    << " relative=" << (precond_res / std::sqrt(rz_init))
                    << " alpha=" << alpha << "\n";
        }

        // Convergence: ||r||_{M^{-1}} / ||r_0||_{M^{-1}} < tol
        if (rz_new / rz_init < tol_sq) {
          break;
        }

        {
          OptionalScopedTimer t_vec(
              solve_profile_ ? &solve_profile_->cg_vector_ops_ms : nullptr);
          Real beta = rz_new / rz_old;
          p = z + beta * p;
          rz_old = rz_new;
        }
      }
    }
  }

  //-------------------------------------------------------------------------//
  // Recover x: x = Q^{-1} * (b - A^T * λ)
  //-------------------------------------------------------------------------//
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
  Index num_boundary = dof_manager_->num_boundary_curvature_constraints();
  Index num_gradient = dof_manager_->num_boundary_gradient_constraints();
  Index num_constraints = num_hanging + num_edge + num_boundary + num_gradient;

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

  // Boundary curvature constraints (natural BC)
  for (const auto &bc : dof_manager_->boundary_curvature_constraints()) {
    const auto &global_dofs = dof_manager_->element_dofs(bc.elem);

    for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
      Real coeff = bc.coeffs(k) * bc.scale;
      if (std::abs(coeff) > 1e-14) {
        triplets.emplace_back(row, global_dofs[k], coeff);
      }
    }
    ++row;
  }

  // Boundary gradient constraints (zero gradient BC)
  for (const auto &gc : dof_manager_->boundary_gradient_constraints()) {
    const auto &global_dofs = dof_manager_->element_dofs(gc.elem);

    for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
      Real coeff = gc.coeffs(k) * gc.scale;
      if (std::abs(coeff) > 1e-14) {
        triplets.emplace_back(row, global_dofs[k], coeff);
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

SpMat CGCubicBezierBathymetrySmoother::assemble_A_boundary_free(
    const std::function<std::vector<std::pair<Index, Real>>(Index)> &expand_dof)
    const {

  Index num_free = dof_manager_->num_free_dofs();
  Index num_bc = dof_manager_->num_boundary_curvature_constraints();

  if (num_bc == 0) {
    return SpMat(0, num_free);
  }

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(num_bc * CubicBezierBasis2D::NDOF);

  Index row = 0;
  for (const auto &bc : dof_manager_->boundary_curvature_constraints()) {
    const auto &global_dofs = dof_manager_->element_dofs(bc.elem);

    for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
      auto expanded = expand_dof(global_dofs[k]);
      Real coeff = bc.coeffs(k) * bc.scale; // scale = 1/dx² or 1/dy²
      if (std::abs(coeff) > 1e-14) {
        for (const auto &[free_idx, weight] : expanded) {
          triplets.emplace_back(row, free_idx, coeff * weight);
        }
      }
    }
    ++row;
  }

  SpMat A(num_bc, num_free);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

SpMat CGCubicBezierBathymetrySmoother::assemble_A_gradient_free(
    const std::function<std::vector<std::pair<Index, Real>>(Index)> &expand_dof)
    const {

  Index num_free = dof_manager_->num_free_dofs();
  Index num_gc = dof_manager_->num_boundary_gradient_constraints();

  if (num_gc == 0) {
    return SpMat(0, num_free);
  }

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(num_gc * CubicBezierBasis2D::NDOF);

  Index row = 0;
  for (const auto &gc : dof_manager_->boundary_gradient_constraints()) {
    const auto &global_dofs = dof_manager_->element_dofs(gc.elem);

    for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
      auto expanded = expand_dof(global_dofs[k]);
      Real coeff = gc.coeffs(k) * gc.scale; // scale = 1/dx or 1/dy
      if (std::abs(coeff) > 1e-14) {
        for (const auto &[free_idx, weight] : expanded) {
          triplets.emplace_back(row, free_idx, coeff * weight);
        }
      }
    }
    ++row;
  }

  SpMat A(num_gc, num_free);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

// =============================================================================
// Diagnostics
// =============================================================================

Real CGCubicBezierBathymetrySmoother::constraint_violation() const {
  Index num_constraints = dof_manager_->num_constraints() +
                          dof_manager_->num_edge_derivative_constraints() +
                          dof_manager_->num_boundary_curvature_constraints() +
                          dof_manager_->num_boundary_gradient_constraints();

  if (!solved_ || num_constraints == 0)
    return 0.0;

  SpMat A = assemble_A();
  VecX violation = A * solution_;
  return violation.norm();
}

} // namespace drifter
