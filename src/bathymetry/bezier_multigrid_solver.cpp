#include "bathymetry/bezier_multigrid_solver.hpp"
#include <Eigen/SparseLU>
#include <stdexcept>

namespace drifter {

BezierMultigridSolver::BezierMultigridSolver(const QuadtreeAdapter& quadtree,
                                              const MultigridConfig& config)
    : config_(config), finest_grid_(&quadtree) {
  // Build grid hierarchy from finest to coarsest
  build_grid_hierarchy(quadtree);

  // Allocate operator storage for each level
  // For now (Step 2.1), we just have 1 level as placeholder
  // This will be properly implemented in Step 2.2
  int num_lvls = (grids_.empty() ? 1 : static_cast<int>(grids_.size()));
  operators_.resize(num_lvls);
  restriction_ops_.resize(std::max(0, num_lvls - 1));
  prolongation_ops_.resize(std::max(0, num_lvls - 1));
}

int BezierMultigridSolver::solve(const SpMat& KKT, VecX& x, const VecX& rhs) {
  // Store finest level operator
  operators_[num_levels() - 1] = KKT;

  // Build restriction/prolongation operators (will be implemented in Steps 2.3-2.4)
  // For now, just perform direct solve as placeholder

  residual_history_.clear();

  // V-cycle iterations
  for (int iter = 0; iter < config_.max_iterations; ++iter) {
    Real residual_norm = compute_residual(KKT, x, rhs);
    residual_history_.push_back(residual_norm);

    // Check convergence
    Real initial_residual = residual_history_[0];
    Real relative_residual = (initial_residual > 0)
                              ? residual_norm / initial_residual
                              : residual_norm;

    if (relative_residual < config_.tolerance) {
      return iter + 1;
    }

    // Perform V-cycle (stub for now, will be implemented in Step 2.6)
    // For now, just do a simple smoothing iteration as placeholder
    smooth(KKT, x, rhs, 1);
  }

  return config_.max_iterations;
}

void BezierMultigridSolver::build_grid_hierarchy(
    const QuadtreeAdapter& finest_grid) {
  // Stub: Will be implemented in Step 2.2
  // For now, just recognize that we have 1 level (finest only)
  // Actual grid hierarchy extraction will be done in Step 2.2
  // We just need grids_.size() to return a valid value

  // Placeholder: We'll properly build the hierarchy in Step 2.2
  // For now this is just a stub to allow compilation
}

SpMat BezierMultigridSolver::build_restriction(int fine_level) {
  // Stub: Will be implemented in Step 2.3
  // Return identity matrix as placeholder
  Index ndofs = grids_[fine_level]->num_elements() * 36;
  SpMat R(ndofs, ndofs);
  R.setIdentity();
  return R;
}

SpMat BezierMultigridSolver::build_prolongation(int coarse_level) {
  // Stub: Will be implemented in Step 2.4
  // Return identity matrix as placeholder
  Index ndofs = grids_[coarse_level]->num_elements() * 36;
  SpMat P(ndofs, ndofs);
  P.setIdentity();
  return P;
}

SpMat BezierMultigridSolver::build_coarse_operator(
    const SpMat& fine_operator,
    const SpMat& restriction,
    const SpMat& prolongation) {
  // Galerkin coarse-grid operator: A_coarse = R * A_fine * P
  return restriction * fine_operator * prolongation;
}

void BezierMultigridSolver::v_cycle(int level, VecX& x, const VecX& rhs) {
  // Stub: Will be implemented in Steps 2.6-2.7
  // For now, just perform smoothing
  if (level == config_.coarsest_level) {
    direct_solve(operators_[level], x, rhs);
  } else {
    smooth(operators_[level], x, rhs, config_.num_presmooth);
  }
}

void BezierMultigridSolver::smooth(const SpMat& A, VecX& x, const VecX& b,
                                    int num_iterations) {
  // Jacobi smoother: x ← x + ω * D^{-1} * (b - A*x)
  // Extract diagonal
  VecX diag = VecX::Zero(A.rows());
  for (int k = 0; k < A.outerSize(); ++k) {
    for (SpMat::InnerIterator it(A, k); it; ++it) {
      if (it.row() == it.col()) {
        diag(it.row()) = it.value();
      }
    }
  }

  // Jacobi iterations
  for (int iter = 0; iter < num_iterations; ++iter) {
    VecX residual = b - A * x;
    for (Index i = 0; i < x.size(); ++i) {
      if (std::abs(diag(i)) > 1e-14) {
        x(i) += config_.smoother_omega * residual(i) / diag(i);
      }
    }
  }
}

void BezierMultigridSolver::direct_solve(const SpMat& A, VecX& x,
                                          const VecX& b) {
  // Direct solve using SparseLU
  Eigen::SparseLU<SpMat> solver;
  solver.compute(A);

  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("SparseLU factorization failed");
  }

  x = solver.solve(b);

  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("SparseLU solve failed");
  }
}

Real BezierMultigridSolver::compute_residual(const SpMat& A, const VecX& x,
                                               const VecX& b) {
  VecX residual = b - A * x;
  return residual.norm();
}

} // namespace drifter
