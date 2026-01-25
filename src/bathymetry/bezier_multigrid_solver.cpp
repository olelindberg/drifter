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
  // Performance note: This multigrid solver achieves O(n) complexity for
  // properly configured hierarchies. For single-level grids, it falls back
  // to direct solve or simple smoothing.

  // Store finest level operator
  int finest_level = num_levels() - 1;
  if (finest_level >= 0 && finest_level < static_cast<int>(operators_.size())) {
    operators_[finest_level] = KKT;
  }

  // Early exit for single-level grids (no multigrid benefit)
  if (num_levels() <= 1) {
    if (config_.use_direct_coarse_solve) {
      direct_solve(KKT, x, rhs);
      residual_history_.push_back(0.0);
      return 1;
    }
    // Otherwise fall through to smoothing iterations
  }

  // Build grid hierarchy operators via Galerkin coarsening (Step 2.8)
  // A_coarse = R * A_fine * P
  // This preserves the constraint structure on coarse grids
  bool hierarchy_usable = false;
  if (num_levels() > 1 && !grids_.empty()) {
    for (int level = finest_level - 1; level >= 0; --level) {
      if (level >= 0 &&
          level < static_cast<int>(restriction_ops_.size()) &&
          level < static_cast<int>(prolongation_ops_.size()) &&
          level + 1 < static_cast<int>(operators_.size()) &&
          level < static_cast<int>(grids_.size()) &&
          level + 1 < static_cast<int>(grids_.size())) {

        // Build restriction and prolongation for this level
        restriction_ops_[level] = build_restriction(level + 1);
        prolongation_ops_[level] = build_prolongation(level);

        // Skip if operators are empty (no valid hierarchy)
        if (restriction_ops_[level].rows() == 0 ||
            prolongation_ops_[level].rows() == 0) {
          continue;
        }

        // Verify dimension compatibility before Galerkin coarsening
        const SpMat& A_fine = operators_[level + 1];
        const SpMat& R = restriction_ops_[level];
        const SpMat& P = prolongation_ops_[level];

        // Check: R.cols() == A_fine.rows() and A_fine.cols() == P.rows()
        if (R.cols() != A_fine.rows() || A_fine.cols() != P.rows()) {
          // Dimension mismatch - skip this level
          // This can happen with non-conforming meshes where hierarchy
          // extraction doesn't produce compatible grids
          continue;
        }

        // Build coarse operator via Galerkin projection
        operators_[level] = build_coarse_operator(A_fine, R, P);

        // Mark hierarchy as usable if we successfully built at least one coarse operator
        if (operators_[level].rows() > 0 && operators_[level].cols() > 0) {
          hierarchy_usable = true;
        }
      }
    }
  }

  // If hierarchy is not usable (dimension mismatches, non-conforming mesh, etc.),
  // fall back to direct solve
  if (!hierarchy_usable || num_levels() <= 1) {
    direct_solve(KKT, x, rhs);
    residual_history_.clear();
    residual_history_.push_back(compute_residual(KKT, x, rhs));
    return 1;
  }

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

    // Perform V-cycle on finest level
    v_cycle(finest_level, x, rhs);
  }

  return config_.max_iterations;
}

void BezierMultigridSolver::build_grid_hierarchy(
    const QuadtreeAdapter& finest_grid) {
  // Extract multi-level grid hierarchy from quadtree mesh
  // Groups elements by maximum refinement level and creates
  // separate QuadtreeAdapter for each level

  grids_.clear();

  // Find all unique refinement levels in the mesh
  std::map<int, std::vector<Index>> elements_by_level;
  int max_level = 0;

  for (Index e = 0; e < finest_grid.num_elements(); ++e) {
    QuadLevel level = finest_grid.element_level(e);
    int max_ref_level = level.max_level();
    elements_by_level[max_ref_level].push_back(e);
    max_level = std::max(max_level, max_ref_level);
  }

  // Determine which levels to include
  // Start from coarsest_level or 0, up to max_level
  int min_level = std::max(0, config_.coarsest_level);

  // Build grid for each level from coarsest to finest
  for (int target_level = min_level; target_level <= max_level; ++target_level) {
    auto grid = std::make_unique<QuadtreeAdapter>();

    // Check if this level has elements
    auto it = elements_by_level.find(target_level);
    if (it == elements_by_level.end() || it->second.empty()) {
      // No elements at this level - skip it
      // This can happen with AMR that has gaps in refinement levels
      continue;
    }

    // Add all elements at this level to the new grid
    const auto& element_indices = it->second;

    for (Index elem_idx : element_indices) {
      QuadBounds bounds = finest_grid.element_bounds(elem_idx);
      QuadLevel level = finest_grid.element_level(elem_idx);

      grid->add_element(bounds, level);
    }

    // Store this grid in hierarchy
    // grids_[0] = coarsest, grids_[n-1] = finest
    grids_.push_back(std::move(grid));
  }

  // If no multi-level hierarchy was built, store reference to finest grid
  // This happens for uniform meshes or when all elements are at same level
  if (grids_.empty()) {
    // Can't copy QuadtreeAdapter, so just leave grids_ empty
    // The solver will fall back to single-level operation
  }
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
  // Two-level V-cycle multigrid algorithm
  //
  // Algorithm:
  // 1. Pre-smooth: Reduce high-frequency error on fine grid
  // 2. Restrict: Project residual to coarse grid
  // 3. Coarse solve: Solve error equation on coarse grid
  // 4. Prolongate: Interpolate correction to fine grid
  // 5. Post-smooth: Clean up interpolation artifacts
  //
  // For recursive V-cycle (Step 2.7), replace coarse solve with
  // recursive v_cycle call

  if (level == config_.coarsest_level || grids_.empty()) {
    // Base case: Direct solve on coarsest grid
    if (!operators_.empty() && level < static_cast<int>(operators_.size())) {
      direct_solve(operators_[level], x, rhs);
    }
    return;
  }

  // Check that we have valid operators for this level
  if (level >= static_cast<int>(operators_.size())) {
    // No operator for this level - just smooth
    if (!operators_.empty()) {
      smooth(operators_.back(), x, rhs, config_.num_presmooth);
    }
    return;
  }

  const SpMat& A_fine = operators_[level];

  // Step 1: Pre-smoothing on fine grid
  smooth(A_fine, x, rhs, config_.num_presmooth);

  // Step 2: Compute residual on fine grid
  VecX residual_fine = rhs - A_fine * x;

  // Step 3: Restrict residual to coarse grid
  if (level - 1 >= 0 &&
      level - 1 < static_cast<int>(restriction_ops_.size()) &&
      restriction_ops_[level - 1].rows() > 0) {
    const SpMat& R = restriction_ops_[level - 1];
    VecX rhs_coarse = R * residual_fine;

    // Step 4: Solve coarse grid equation for error
    // A_coarse * e_coarse = rhs_coarse
    VecX e_coarse = VecX::Zero(rhs_coarse.size());

    if (level - 1 < static_cast<int>(operators_.size())) {
      const SpMat& A_coarse = operators_[level - 1];

      if (config_.use_direct_coarse_solve) {
        // Direct solve on coarse grid
        direct_solve(A_coarse, e_coarse, rhs_coarse);
      } else {
        // Recursive V-cycle (Step 2.7)
        v_cycle(level - 1, e_coarse, rhs_coarse);
      }

      // Step 5: Prolongate correction to fine grid
      if (level - 1 < static_cast<int>(prolongation_ops_.size()) &&
          prolongation_ops_[level - 1].rows() > 0) {
        const SpMat& P = prolongation_ops_[level - 1];
        VecX correction_fine = P * e_coarse;

        // Apply correction
        x += correction_fine;
      }
    }
  }

  // Step 6: Post-smoothing on fine grid
  smooth(A_fine, x, rhs, config_.num_postsmooth);
}

void BezierMultigridSolver::smooth(const SpMat& A, VecX& x, const VecX& b,
                                    int num_iterations) {
  // Jacobi smoother for KKT systems
  //
  // KKT saddle-point structure:
  //   [Q   A^T] [x]   [f]
  //   [A    0 ] [λ] = [g]
  //
  // Standard Jacobi: x_new = x + ω·D^{-1}·(b - A·x)
  // where D = diag(A)
  //
  // For KKT, diagonal has zeros in Lagrange multiplier block.
  // We skip updates for zero diagonal entries.
  //
  // Better approach (future): Uzawa iteration
  //   x ← x + ω·D_Q^{-1}·(f - Q·x - A^T·λ)
  //   λ ← λ + ω·S^{-1}·(g - A·x)  where S = A·D_Q^{-1}·A^T

  // Extract diagonal
  VecX diag = VecX::Zero(A.rows());
  for (int k = 0; k < A.outerSize(); ++k) {
    for (SpMat::InnerIterator it(A, k); it; ++it) {
      if (it.row() == it.col()) {
        diag(it.row()) = it.value();
      }
    }
  }

  // Damped Jacobi iterations
  const Real omega = config_.smoother_omega;
  const Real tol = 1e-14; // Threshold for treating diagonal as zero

  for (int iter = 0; iter < num_iterations; ++iter) {
    VecX residual = b - A * x;

    // Update primal variables (nonzero diagonal)
    for (Index i = 0; i < x.size(); ++i) {
      if (std::abs(diag(i)) > tol) {
        x(i) += omega * residual(i) / diag(i);
      }
      // Skip Lagrange multipliers (zero diagonal) for now
      // TODO: Implement proper Uzawa update for multipliers
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
