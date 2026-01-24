#include "bathymetry/bezier_multigrid_solver.hpp"
#include <Eigen/SparseLU>
#include <stdexcept>

namespace drifter {

using Eigen::Triplet;

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
  // For Step 2.2: Simplified implementation for single-level (finest only)
  // Full multi-level hierarchy extraction will be implemented later
  // when QuadtreeAdapter API supports it

  // For now, we work with the finest grid only
  // grids_ stays empty, and we use finest_grid_ pointer
  // This allows us to proceed with implementing V-cycle logic

  // In future: Extract elements by refinement level and create coarse grids
  // For now: Single level = direct solve (no actual multigrid yet)

  // Store number of levels for sizing operators
  // We'll implement 2-level V-cycle first in Step 2.6
  grids_.clear();

  // Note: Full implementation would group elements by level and create
  // QuadtreeAdapter for each level. This requires either:
  // 1. QuadtreeAdapter API to support construction from element list
  // 2. Or direct access to internal structure
  //
  // For MVP (minimum viable product), we start with finest grid only
  // and implement the V-cycle framework. Multi-level hierarchy can be
  // added incrementally as we test and validate the approach.
}

SpMat BezierMultigridSolver::build_restriction(int fine_level) {
  // Build L2 projection restriction operator: fine → coarse
  // For 2:1 refinement, each coarse element has 4 fine children
  //
  // Restriction matrix R: ndofs_coarse × ndofs_fine
  // R * x_fine = x_coarse (approximately, in L2 sense)
  //
  // For Bezier surfaces, we use weighted averaging based on:
  // - Bezier basis function overlap integrals
  // - Fine element quadrature weights
  //
  // Each coarse Bezier control point receives contributions from
  // multiple fine control points based on their spatial overlap

  if (grids_.empty() || fine_level >= static_cast<int>(grids_.size())) {
    // No hierarchy yet - return empty matrix
    SpMat R(0, 0);
    return R;
  }

  const auto& fine_grid = *grids_[fine_level];
  const auto& coarse_grid = *grids_[fine_level - 1];

  Index ndofs_fine = fine_grid.num_elements() * 36;
  Index ndofs_coarse = coarse_grid.num_elements() * 36;

  // Build restriction via L2 projection
  std::vector<Triplet<Real>> triplets;
  triplets.reserve(ndofs_coarse * 4 * 36); // Rough estimate

  // For each coarse element, find corresponding fine elements
  for (Index coarse_elem = 0; coarse_elem < coarse_grid.num_elements();
       ++coarse_elem) {
    QuadBounds coarse_bounds = coarse_grid.element_bounds(coarse_elem);

    // Find fine elements that overlap this coarse element
    std::vector<Index> fine_elements;
    for (Index fine_elem = 0; fine_elem < fine_grid.num_elements();
         ++fine_elem) {
      QuadBounds fine_bounds = fine_grid.element_bounds(fine_elem);

      // Check if fine element center is inside coarse element
      Vec2 fine_center = fine_bounds.center();
      if (coarse_bounds.contains(fine_center, 1e-10)) {
        fine_elements.push_back(fine_elem);
      }
    }

    // For now, use simple averaging as placeholder
    // TODO: Implement proper L2 projection using Bezier basis overlap integrals
    //
    // Proper implementation would:
    // 1. Evaluate fine Bezier surfaces at coarse control point locations
    // 2. Use weighted least-squares to fit coarse control points
    // 3. Account for derivative constraints in C² continuity

    // Placeholder: Simple identity restriction (1:1 mapping)
    // This works only if fine and coarse have same DOF count
    if (fine_elements.size() > 0 && ndofs_fine == ndofs_coarse) {
      Index coarse_base = coarse_elem * 36;
      Index fine_base = fine_elements[0] * 36;

      for (int dof = 0; dof < 36; ++dof) {
        triplets.emplace_back(coarse_base + dof, fine_base + dof, 1.0);
      }
    }
  }

  SpMat R(ndofs_coarse, ndofs_fine);
  R.setFromTriplets(triplets.begin(), triplets.end());

  return R;
}

SpMat BezierMultigridSolver::build_prolongation(int coarse_level) {
  // Build prolongation (interpolation) operator: coarse → fine
  // For 2:1 refinement, each coarse element maps to 4 fine children
  //
  // Prolongation matrix P: ndofs_fine × ndofs_coarse
  // x_fine += P * correction_coarse
  //
  // For Bezier surfaces, prolongation evaluates the coarse Bezier surface
  // at fine control point locations. Each fine control point receives
  // a weighted combination of coarse control points.
  //
  // For L2 projection, P ≈ R^T (transpose of restriction)
  // For Galerkin coarsening, this ensures A_coarse = R * A_fine * P

  if (grids_.empty() || coarse_level >= static_cast<int>(grids_.size()) - 1) {
    // No hierarchy yet - return empty matrix
    SpMat P(0, 0);
    return P;
  }

  const auto& coarse_grid = *grids_[coarse_level];
  const auto& fine_grid = *grids_[coarse_level + 1];

  Index ndofs_coarse = coarse_grid.num_elements() * 36;
  Index ndofs_fine = fine_grid.num_elements() * 36;

  // Build prolongation via Bezier evaluation
  std::vector<Triplet<Real>> triplets;
  triplets.reserve(ndofs_fine * 36); // Rough estimate

  // For each fine element, find parent coarse element
  for (Index fine_elem = 0; fine_elem < fine_grid.num_elements();
       ++fine_elem) {
    QuadBounds fine_bounds = fine_grid.element_bounds(fine_elem);
    Vec2 fine_center = fine_bounds.center();

    // Find coarse element containing this fine element
    Index coarse_elem = -1;
    for (Index c = 0; c < coarse_grid.num_elements(); ++c) {
      QuadBounds coarse_bounds = coarse_grid.element_bounds(c);
      if (coarse_bounds.contains(fine_center, 1e-10)) {
        coarse_elem = c;
        break;
      }
    }

    if (coarse_elem < 0) {
      // Fine element not found in coarse grid - skip
      continue;
    }

    // For now, use simple identity as placeholder
    // TODO: Implement proper Bezier evaluation
    //
    // Proper implementation would:
    // 1. Map fine control points to coarse element parameter space
    // 2. Evaluate coarse Bezier basis at these parameter locations
    // 3. Build interpolation weights from basis values

    // Placeholder: Simple 1:1 mapping (works only if DOF counts match)
    if (ndofs_fine == ndofs_coarse) {
      Index fine_base = fine_elem * 36;
      Index coarse_base = coarse_elem * 36;

      for (int dof = 0; dof < 36; ++dof) {
        triplets.emplace_back(fine_base + dof, coarse_base + dof, 1.0);
      }
    }
  }

  SpMat P(ndofs_fine, ndofs_coarse);
  P.setFromTriplets(triplets.begin(), triplets.end());

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
