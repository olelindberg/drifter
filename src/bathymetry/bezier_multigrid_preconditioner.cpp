#include "bathymetry/bezier_multigrid_preconditioner.hpp"
#include "bathymetry/iterative_method_factory.hpp"
#include "bathymetry/jacobi_method.hpp"
#include "core/scoped_timer.hpp"
#include "mesh/morton.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace drifter {

namespace {

/// @brief Compute 2D parent Morton code for quadtree
/// @details Since quadtree uses Morton3D::encode(x, y, 0), parent is
/// encode(x/2, y/2, 0)
uint64_t parent_2d(uint64_t code) {
  uint32_t x, y, z;
  Morton3D::decode(code, x, y, z);
  return Morton3D::encode(x / 2, y / 2, 0);
}

/// @brief Get child index within parent (0-3) for 2D quadtree
/// @return Child index: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
int child_index_2d(uint64_t code) {
  uint32_t x, y, z;
  Morton3D::decode(code, x, y, z);
  return (x & 1) + 2 * (y & 1);
}

} // namespace

BezierMultigridPreconditioner::BezierMultigridPreconditioner(
    const MultigridConfig &config)
    : config_(config) {
  // Precompute Bezier subdivision matrices
  CubicBezierBasis2D basis;
  S_left_ = basis.compute_1d_extraction_matrix(0.0, 0.5);
  S_right_ = basis.compute_1d_extraction_matrix(0.5, 1.0);

  // Compute Bernstein mass matrices for L2 projection
  M_1D_ = build_bernstein_mass_1d();
  M_2D_ = kronecker_product(M_1D_, M_1D_);
  M_2D_inv_ = M_2D_.inverse(); // 16x16 dense inverse is cheap

  // Compute local L2 operators (restriction and prolongation)
  build_l2_operators_local();
}

MatX BezierMultigridPreconditioner::kronecker_product(const MatX &A,
                                                      const MatX &B) {
  const int m = static_cast<int>(A.rows());
  const int n = static_cast<int>(A.cols());
  const int p = static_cast<int>(B.rows());
  const int q = static_cast<int>(B.cols());

  MatX result(m * p, n * q);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      result.block(i * p, j * q, p, q) = A(i, j) * B;
    }
  }
  return result;
}

MatX BezierMultigridPreconditioner::build_bernstein_mass_1d() {
  // M[i,j] = C(n,i) · C(n,j) / ((2n+1) · C(2n, i+j))
  // For n=3 (cubic): 4x4 matrix
  constexpr int n = 3;
  MatX M(4, 4);

  // Binomial coefficients for n=3: C(3,0)=1, C(3,1)=3, C(3,2)=3, C(3,3)=1
  std::array<Real, 4> C3 = {1, 3, 3, 1};
  // Binomial coefficients for 2n=6
  std::array<Real, 7> C6 = {1, 6, 15, 20, 15, 6, 1};

  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      M(i, j) = C3[i] * C3[j] / (7.0 * C6[i + j]);
    }
  }
  return M;
}

void BezierMultigridPreconditioner::build_l2_operators_local() {
  // R_local = M_c^{-1} * P_bezier^T * M_f
  // P_local = R_local^T
  //
  // P_bezier is 64x16 (4 children × 16 DOFs each, Bezier subdivision)
  // M_f is 64x64 block diagonal (4 blocks of 16x16, each scaled by 1/4)

  // Build P_bezier for all 4 children (de Casteljau subdivision)
  MatX P_bezier(64, 16);
  std::array<MatX, 4> P_child;
  P_child[0] = kronecker_product(S_left_, S_left_);   // Child 0 (SW)
  P_child[1] = kronecker_product(S_right_, S_left_);  // Child 1 (SE)
  P_child[2] = kronecker_product(S_left_, S_right_);  // Child 2 (NW)
  P_child[3] = kronecker_product(S_right_, S_right_); // Child 3 (NE)

  for (int c = 0; c < 4; ++c) {
    P_bezier.block(c * 16, 0, 16, 16) = P_child[c];
  }

  // Build M_f (block diagonal, each child has area 1/4)
  MatX M_f = MatX::Zero(64, 64);
  for (int c = 0; c < 4; ++c) {
    M_f.block(c * 16, c * 16, 16, 16) = 0.25 * M_2D_;
  }

  // R_local = M_c^{-1} * P_bezier^T * M_f (16x64)
  R_L2_local_ = M_2D_inv_ * P_bezier.transpose() * M_f;

  // P_local = R_local^T (64x16) - for symmetric multigrid
  P_L2_local_ = R_L2_local_.transpose();
}

void BezierMultigridPreconditioner::setup(
    const SpMat &Q_fine, const QuadtreeAdapter &mesh,
    const CGCubicBezierDofManager &dof_manager) {
  OptionalScopedTimer t_total(profile_ ? &profile_->setup_total_ms : nullptr);

  levels_.clear();

  // Level numbering:
  // - MG level 0 = coarsest
  // - MG level num_levels-1 = finest = ALL leaves (at any tree depth)
  //
  // For adaptive meshes, we use composite grids where:
  // - Finest level contains all leaves regardless of tree depth
  // - Coarser levels are built by coarsening complete sibling groups
  // - Incomplete sibling groups pass through unchanged (identity prolongation)

  int max_depth = mesh.max_depth();
  int min_depth = compute_min_leaf_depth(mesh);
  // With tree-based coarsening (recursive prolongation), we can coarsen all the
  // way to min_tree_level regardless of min_depth, since we traverse the tree
  // structure
  int available_levels = max_depth - config_.min_tree_level + 1;
  int num_levels = std::min(config_.num_levels, std::max(1, available_levels));

  if (config_.verbose) {
    std::cerr << "[Multigrid] Tree max_depth: " << max_depth
              << ", min_leaf_depth: " << min_depth
              << ", min_tree_level: " << config_.min_tree_level << "\n";
    std::cerr << "[Multigrid] Building " << num_levels
              << " levels (composite grid)\n";
  }

  // Resize levels vector (level 0 = coarsest, level num_levels-1 = finest)
  levels_.resize(num_levels);

  // Build finest level (level num_levels-1 = ALL leaves)
  int finest_level = num_levels - 1;
  {
    OptionalScopedTimer t(profile_ ? &profile_->setup_finest_init_ms : nullptr);
    levels_[finest_level].Q = Q_fine;
    levels_[finest_level].num_dofs = Q_fine.rows();
  }

  if (config_.verbose) {
    std::cerr << "[Multigrid] Level " << finest_level
              << " (finest): " << levels_[finest_level].num_dofs << " DOFs\n";
  }

  // Build composite grid hierarchy (initializes all composite_grid structures)
  build_composite_hierarchy(mesh, dof_manager);

  // Build coarser levels using composite grid structure
  // Each MG level L has operators: R maps L+1 -> L (restriction), P maps L ->
  // L+1 (prolongation)
  for (int mg_level = finest_level - 1; mg_level >= 0; --mg_level) {
    OptionalScopedTimer t(profile_ ? &profile_->setup_prolongation_ms
                                   : nullptr);

    Index coarse_num_dofs = levels_[mg_level].composite_grid.num_dofs;

    // Check if coarsening produced a usable level
    if (coarse_num_dofs == 0) {
      // Can't use this level - truncate hierarchy (discard this level)
      int levels_to_keep = finest_level - mg_level;
      std::vector<MultigridLevel> new_levels(levels_to_keep);
      for (int i = 0; i < levels_to_keep; ++i) {
        new_levels[i] = std::move(levels_[mg_level + 1 + i]);
      }
      levels_ = std::move(new_levels);

      if (config_.verbose) {
        std::cerr << "[Multigrid] Truncated to " << levels_.size() << " levels"
                  << " (no coarse DOFs)\n";
      }
      break;
    }

    // Level is valid - process it
    levels_[mg_level].num_dofs = coarse_num_dofs;

    // Build transfer operators based on strategy
    SpMat P, R;
    if (config_.transfer_strategy ==
        TransferOperatorStrategy::BezierSubdivision) {
      // Pure Bezier subdivision: P from de Casteljau, R = P^T normalized
      if (config_.verbose) {
        std::cerr << "[Multigrid] Using BezierSubdivision transfer for level "
                  << mg_level << "\n";
      }
      P = build_prolongation_composite(mg_level);
      R = P.transpose();

      // Normalize R rows to sum to 1 (partition of unity)
      Eigen::SparseMatrix<Real, Eigen::RowMajor> R_row(R);
      VecX row_sums = R_row * VecX::Ones(R.cols());
      for (Index i = 0; i < R_row.rows(); ++i) {
        Real scale = row_sums(i);
        if (std::abs(scale) > 1e-14 && std::abs(scale - 1.0) > 1e-10) {
          Real inv_scale = 1.0 / scale;
          for (Eigen::SparseMatrix<Real, Eigen::RowMajor>::InnerIterator it(
                   R_row, i);
               it; ++it) {
            it.valueRef() *= inv_scale;
          }
        }
      }
      R = SpMat(R_row);
    } else {
      // L2 projection: R from integral matching, P = R^T
      R = build_restriction_l2(mg_level);
      P = R.transpose();
    }
    levels_[mg_level + 1].R =
        R;                   // R maps mg_level+1 -> mg_level (fine -> coarse)
    levels_[mg_level].P = P; // P maps mg_level -> mg_level+1 (coarse -> fine)

    // Build coarse level system matrix
    {
      OptionalScopedTimer t2(profile_ ? &profile_->setup_galerkin_ms : nullptr);

      if (config_.coarse_grid_strategy ==
          CoarseGridStrategy::CachedRediscretization) {
        if (element_matrix_cache_ == nullptr) {
          throw std::runtime_error(
              "CachedRediscretization requires element_matrix_cache to be set");
        }
        // Assemble from cached element matrices (exact, includes data fitting)
        if (config_.verbose) {
          std::cerr << "[Multigrid] Using CachedRediscretization for level "
                    << mg_level << "\n";
        }
        levels_[mg_level].Q = assemble_from_cached_matrices(mg_level);
      } else {
        if (config_.verbose) {
          std::cerr << "[Multigrid] Using Galerkin for level " << mg_level
                    << "\n";
        }
        // Galerkin projection: Q_coarse = P^T * Q_fine * P
        // Use exact transpose P^T, not normalized R, to preserve variational
        // properties
        SpMat Pt = P.transpose();
        SpMat temp = levels_[mg_level + 1].Q * P;
        levels_[mg_level].Q = Pt * temp;
        // Symmetrize
        levels_[mg_level].Q = 0.5 * (levels_[mg_level].Q +
                                     SpMat(levels_[mg_level].Q.transpose()));
      }
    }

    if (config_.verbose) {
      std::cerr << "[Multigrid] Level " << mg_level << ": "
                << levels_[mg_level].num_dofs << " DOFs\n";
    }

    // Check if we've reached the minimum tree level - if so, keep this level
    // as coarsest and stop building more levels.
    // Use max_depth (deepest node) to allow intermediate mixed-depth levels.
    // Only stop when ALL nodes are at or below min_tree_level.
    int max_node_level = levels_[mg_level].composite_grid.max_depth;

    if (config_.verbose) {
      std::cerr << "[Multigrid] Level " << mg_level
                << " max_node_level=" << max_node_level << "\n";
    }

    if (max_node_level <= config_.min_tree_level) {
      // This is the coarsest level we want - truncate any remaining levels
      // below
      if (mg_level > 0) {
        int levels_to_keep = finest_level - mg_level + 1;
        std::vector<MultigridLevel> new_levels(levels_to_keep);
        for (int i = 0; i < levels_to_keep; ++i) {
          new_levels[i] = std::move(levels_[mg_level + i]);
        }
        levels_ = std::move(new_levels);

        if (config_.verbose) {
          std::cerr << "[Multigrid] Reached min_tree_level="
                    << config_.min_tree_level << ", keeping " << levels_.size()
                    << " levels\n";
        }
      }
      break;
    }
  }

  // Setup direct solver for coarsest level (level 0)
  {
    OptionalScopedTimer t(profile_ ? &profile_->setup_coarse_lu_ms : nullptr);

    levels_[0].solver = std::make_unique<Eigen::SparseLU<SpMat>>();
    levels_[0].solver->compute(levels_[0].Q);

    if (levels_[0].solver->info() != Eigen::Success) {
      throw std::runtime_error("BezierMultigridPreconditioner: Coarsest level "
                               "LU factorization failed");
    }
  }

  // Build element blocks for Schwarz smoother on all levels (if configured)
  finest_level = static_cast<int>(levels_.size()) - 1;
  if (config_.smoother_type != SmootherType::Jacobi) {
    OptionalScopedTimer t(profile_ ? &profile_->setup_element_blocks_ms
                                   : nullptr);
    for (int level = 1; level <= finest_level; ++level) {
      build_element_blocks(level);

      if (config_.smoother_type == SmootherType::ColoredMultiplicativeSchwarz) {
        build_element_coloring(level);
      }

      if (config_.verbose) {
        std::cerr << "[Multigrid] Level " << level << ": built "
                  << levels_[level].element_free_dofs.size()
                  << " element blocks\n";
      }
    }
  }

  // Create smoothers for all levels (except level 0 which uses direct solve)
  for (int level = 1; level <= finest_level; ++level) {
    if (config_.smoother_type != SmootherType::Jacobi) {
      // Verify element blocks were built
      if (levels_[level].element_free_dofs.empty()) {
        throw std::runtime_error(
            "BezierMultigridPreconditioner: Schwarz smoother requires element "
            "blocks at level " +
            std::to_string(level));
      }
      levels_[level].smoother = IterativeMethodFactory::create(
          config_.smoother_type, levels_[level].Q,
          levels_[level].element_free_dofs, levels_[level].element_block_lu,
          levels_[level].elements_by_color, config_.jacobi_omega);
    } else {
      levels_[level].smoother =
          std::make_unique<JacobiMethod>(levels_[level].Q, config_.jacobi_omega);
    }
  }
}

SpMat BezierMultigridPreconditioner::assemble_from_cached_matrices(
    int mg_level) {
  const auto &grid = levels_[mg_level].composite_grid;
  Index num_dofs = grid.num_dofs;

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(grid.nodes.size() * 16 * 16);

  for (const auto &node : grid.nodes) {
    // Build key from tree node
    const QuadtreeNode *tn = node.tree_node;
    auto key = std::make_tuple(tn->morton, tn->level.x, tn->level.y);

    auto it = element_matrix_cache_->find(key);
    MatX Q_local;
    if (it == element_matrix_cache_->end()) {
      // Subdivision node - compute from children via local Galerkin projection
      if (!tn->children.empty()) {
        Q_local = compute_subdivision_element_matrix(tn);
      } else {
        // Actual bug - leaf without cached matrix
        throw std::runtime_error(
            "BezierMultigridPreconditioner::assemble_from_cached_matrices: "
            "Missing cached element matrix for leaf node at level " +
            std::to_string(tn->level.max_level()) +
            " (morton=" + std::to_string(tn->morton) + ").");
      }
    } else {
      Q_local = it->second;
    }

    for (int i = 0; i < 16; ++i) {
      Index I = node.dof_indices[i];
      if (I < 0)
        continue;

      for (int j = 0; j < 16; ++j) {
        Index J = node.dof_indices[j];
        if (J < 0)
          continue;

        if (std::abs(Q_local(i, j)) > 1e-16) {
          triplets.emplace_back(I, J, Q_local(i, j));
        }
      }
    }
  }

  SpMat Q(num_dofs, num_dofs);
  Q.setFromTriplets(triplets.begin(), triplets.end());
  return Q;
}

MatX BezierMultigridPreconditioner::get_element_matrix_recursive(
    const QuadtreeNode *node) const {

  auto key = std::make_tuple(node->morton, node->level.x, node->level.y);
  auto it = element_matrix_cache_->find(key);

  if (it != element_matrix_cache_->end()) {
    return it->second; // Leaf node with cached matrix
  }

  // Subdivision node - compute recursively via Galerkin
  return compute_subdivision_element_matrix(node);
}

MatX BezierMultigridPreconditioner::compute_subdivision_element_matrix(
    const QuadtreeNode *node) const {

  // Build local Bezier subdivision matrices for each child quadrant
  std::array<MatX, 4> P_local;
  P_local[0] = kronecker_product(S_left_, S_left_);   // (0,0)
  P_local[1] = kronecker_product(S_right_, S_left_);  // (1,0)
  P_local[2] = kronecker_product(S_left_, S_right_);  // (0,1)
  P_local[3] = kronecker_product(S_right_, S_right_); // (1,1)

  MatX Q_coarse = MatX::Zero(16, 16);

  for (const auto &child : node->children) {
    int child_idx = get_child_quadrant(child.get(), node);
    MatX Q_child = get_element_matrix_recursive(child.get());
    Q_coarse += P_local[child_idx].transpose() * Q_child * P_local[child_idx];
  }

  return Q_coarse;
}

SpMat BezierMultigridPreconditioner::build_prolongation_for_level(
    int tree_level, const QuadtreeAdapter &mesh,
    const CGCubicBezierDofManager &leaf_dof_manager, Index &coarse_num_dofs,
    Index expected_fine_dofs) {

  // Build prolongation from tree_level to tree_level+1
  // tree_level 0 = root (coarsest), max_depth = leaves (finest)
  //
  // P maps coarse DOFs (at tree_level) -> fine DOFs (at tree_level+1)
  // Dimension: (fine_dofs x coarse_dofs)

  static constexpr Real POSITION_SCALE = 1e8;
  auto quantize = [](Real x, Real y) -> std::pair<int64_t, int64_t> {
    return {static_cast<int64_t>(std::round(x * POSITION_SCALE)),
            static_cast<int64_t>(std::round(y * POSITION_SCALE))};
  };

  // Bezier subdivision matrices for each child quadrant
  std::array<MatX, 4> P_local;
  P_local[0] = kronecker_product(S_left_, S_left_);   // (0,0)
  P_local[1] = kronecker_product(S_right_, S_left_);  // (1,0)
  P_local[2] = kronecker_product(S_left_, S_right_);  // (0,1)
  P_local[3] = kronecker_product(S_right_, S_right_); // (1,1)

  // Get nodes at coarse level and fine level
  auto fine_nodes = mesh.nodes_at_level(tree_level + 1);

  int max_depth = mesh.max_depth();
  bool fine_is_leaf_level = (tree_level + 1 == max_depth);

  // Collect coarse nodes that actually have children at the fine level
  // (For adaptive meshes, some coarse nodes might be leaves themselves)
  std::set<const QuadtreeNode *> active_coarse_nodes;
  for (const QuadtreeNode *fine_node : fine_nodes) {
    const QuadtreeNode *parent = fine_node->parent;
    if (parent && parent->level.max_level() == tree_level) {
      active_coarse_nodes.insert(parent);
    }
  }

  // If no coarse nodes have children at the fine level, coarsening is not
  // possible
  if (active_coarse_nodes.empty()) {
    coarse_num_dofs = 0;
    return SpMat(0, 0);
  }

  // Build coarse DOF numbering (position-based deduplication)
  // Only include active coarse nodes (those with children at fine level)
  std::map<const QuadtreeNode *, std::vector<Index>> coarse_node_to_dofs;
  std::map<std::pair<int64_t, int64_t>, Index> coarse_position_to_dof;
  coarse_num_dofs = 0;

  for (const QuadtreeNode *node : active_coarse_nodes) {
    const auto &bounds = node->bounds;
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;

    std::vector<Index> dofs(16);
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        int local_dof = i + 4 * j;
        Real u = i / 3.0;
        Real v = j / 3.0;
        Real x = bounds.xmin + u * dx;
        Real y = bounds.ymin + v * dy;

        auto key = quantize(x, y);
        auto it = coarse_position_to_dof.find(key);
        if (it != coarse_position_to_dof.end()) {
          dofs[local_dof] = it->second;
        } else {
          coarse_position_to_dof[key] = coarse_num_dofs;
          dofs[local_dof] = coarse_num_dofs;
          coarse_num_dofs++;
        }
      }
    }
    coarse_node_to_dofs[node] = std::move(dofs);
  }

  // Build fine DOF numbering
  std::map<const QuadtreeNode *, std::vector<Index>> fine_node_to_dofs;
  std::map<std::pair<int64_t, int64_t>, Index> fine_position_to_dof;
  Index fine_num_dofs = 0;

  if (fine_is_leaf_level) {
    // Fine level is leaf level - use the actual DOF manager
    fine_num_dofs = leaf_dof_manager.num_free_dofs();
  } else {
    // Fine level is intermediate - build position-based DOF numbering
    for (const QuadtreeNode *node : fine_nodes) {
      const auto &bounds = node->bounds;
      Real dx = bounds.xmax - bounds.xmin;
      Real dy = bounds.ymax - bounds.ymin;

      std::vector<Index> dofs(16);
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          int local_dof = i + 4 * j;
          Real u = i / 3.0;
          Real v = j / 3.0;
          Real x = bounds.xmin + u * dx;
          Real y = bounds.ymin + v * dy;

          auto key = quantize(x, y);
          auto it = fine_position_to_dof.find(key);
          if (it != fine_position_to_dof.end()) {
            dofs[local_dof] = it->second;
          } else {
            fine_position_to_dof[key] = fine_num_dofs;
            dofs[local_dof] = fine_num_dofs;
            fine_num_dofs++;
          }
        }
      }
      fine_node_to_dofs[node] = std::move(dofs);
    }

    // For intermediate levels, check DOF consistency with previous iteration
    // If there's a mismatch, the tree has adaptive leaves at different depths
    // and we can't reliably coarsen further
    if (expected_fine_dofs > 0 && fine_num_dofs != expected_fine_dofs) {
      coarse_num_dofs = 0;
      return SpMat(0, 0); // Signal to truncate hierarchy
    }
  }

  // Helper to determine child index from position within parent
  auto get_child_index = [](const QuadtreeNode *child,
                            const QuadtreeNode *parent) -> int {
    Real parent_cx = 0.5 * (parent->bounds.xmin + parent->bounds.xmax);
    Real parent_cy = 0.5 * (parent->bounds.ymin + parent->bounds.ymax);
    Real child_cx = 0.5 * (child->bounds.xmin + child->bounds.xmax);
    Real child_cy = 0.5 * (child->bounds.ymin + child->bounds.ymax);

    int cx = (child_cx > parent_cx) ? 1 : 0;
    int cy = (child_cy > parent_cy) ? 1 : 0;
    return cx + 2 * cy;
  };

  // Build prolongation matrix P: (fine_dofs x coarse_dofs)
  // P(fine, coarse) = weight for mapping coarse DOF -> fine DOF
  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(fine_nodes.size() * 16 * 16);

  for (const QuadtreeNode *fine_node : fine_nodes) {
    const QuadtreeNode *coarse_node = fine_node->parent;
    if (!coarse_node)
      continue;

    // Skip if parent is at a different level (adaptive mesh)
    if (coarse_node_to_dofs.find(coarse_node) == coarse_node_to_dofs.end()) {
      continue;
    }

    int child_idx = get_child_index(fine_node, coarse_node);
    const MatX &P_elem = P_local[child_idx];
    const auto &coarse_dofs = coarse_node_to_dofs[coarse_node];

    // Get fine DOFs
    std::vector<Index> fine_dofs(16);
    if (fine_is_leaf_level) {
      // Use leaf DOF manager
      Index leaf_idx = fine_node->leaf_index;
      if (leaf_idx < 0)
        continue;
      const auto &global_dofs = leaf_dof_manager.element_dofs(leaf_idx);
      for (int k = 0; k < 16; ++k) {
        fine_dofs[k] = leaf_dof_manager.global_to_free(global_dofs[k]);
      }
    } else {
      fine_dofs = fine_node_to_dofs[fine_node];
    }

    // P_elem(fine_local, coarse_local) = subdivision weight
    for (int fine_local = 0; fine_local < 16; ++fine_local) {
      Index fine_dof = fine_dofs[fine_local];
      if (fine_dof < 0)
        continue;

      for (int coarse_local = 0; coarse_local < 16; ++coarse_local) {
        Real weight = P_elem(fine_local, coarse_local);
        if (std::abs(weight) < 1e-14)
          continue;

        Index coarse_dof = coarse_dofs[coarse_local];
        triplets.emplace_back(fine_dof, coarse_dof, weight);
      }
    }
  }

  SpMat P(fine_num_dofs, coarse_num_dofs);
  P.setFromTriplets(triplets.begin(), triplets.end());

  return P;
}

void BezierMultigridPreconditioner::build_element_blocks(int level) {
  auto &L = levels_[level];
  const auto &composite_grid = L.composite_grid;
  size_t num_elements = composite_grid.nodes.size();

  L.element_free_dofs.resize(num_elements);
  L.element_block_lu.resize(num_elements);

  for (size_t e = 0; e < num_elements; ++e) {
    const auto &node = composite_grid.nodes[e];

    // Collect valid (non-negative) DOF indices from the composite grid node
    std::vector<Index> free_dofs;
    free_dofs.reserve(16);
    for (int local = 0; local < 16; ++local) {
      Index dof = node.dof_indices[local];
      if (dof >= 0) {
        free_dofs.push_back(dof);
      }
    }

    L.element_free_dofs[e] = free_dofs;

    // Extract element block from Q
    int block_size = static_cast<int>(free_dofs.size());
    if (block_size == 0) {
      // All DOFs constrained, store empty factorization
      L.element_block_lu[e] = Eigen::PartialPivLU<MatX>();
      continue;
    }

    MatX Q_block(block_size, block_size);
    for (int i = 0; i < block_size; ++i) {
      for (int j = 0; j < block_size; ++j) {
        Q_block(i, j) = L.Q.coeff(free_dofs[i], free_dofs[j]);
      }
    }

    // LU factorize for local solves
    L.element_block_lu[e] = Q_block.partialPivLu();
  }
}

VecX BezierMultigridPreconditioner::apply(const VecX &r) const {
  if (levels_.empty()) {
    throw std::runtime_error(
        "BezierMultigridPreconditioner::apply: setup() not called");
  }

  OptionalScopedTimer t(profile_ ? &profile_->apply_total_ms : nullptr);
  if (profile_) {
    profile_->apply_calls++;
  }

  // Start V-cycle at finest level (highest index)
  int finest_level = static_cast<int>(levels_.size()) - 1;
  VecX x = VecX::Zero(r.size());
  v_cycle(finest_level, x, r);
  return x;
}

void BezierMultigridPreconditioner::v_cycle(int level, VecX &x,
                                            const VecX &b) const {
  // Level numbering: 0 = coarsest (root), num_levels-1 = finest (leaves)
  if (profile_) {
    profile_->vcycle_calls++;
  }

  const auto &L = levels_[level];

  // Coarsest level (level 0): direct solve
  if (level == 0) {
    OptionalScopedTimer t(profile_ ? &profile_->vcycle_coarse_solve_ms
                                   : nullptr);
    if (profile_) {
      profile_->coarse_solves++;
    }
    x = L.solver->solve(b);
    return;
  }

  // Pre-smoothing
  {
    OptionalScopedTimer t(profile_ ? &profile_->vcycle_pre_smooth_ms : nullptr);
    if (profile_) {
      profile_->smoothing_iterations += config_.pre_smoothing;
    }
    L.smoother->apply(x, b, config_.pre_smoothing);
  }

  // Compute residual: r = b - Q*x
  VecX r;
  {
    OptionalScopedTimer t(profile_ ? &profile_->vcycle_residual_ms : nullptr);
    if (profile_) {
      profile_->matvec_products++;
    }
    r = b - L.Q * x;
  }

  // Restrict to coarser level (level - 1): r_c = R * r
  // R is stored in current level (maps level -> level-1)
  const auto &Lc = levels_[level - 1];
  VecX r_coarse;
  {
    OptionalScopedTimer t(profile_ ? &profile_->vcycle_restrict_ms : nullptr);
    r_coarse = L.R * r;
  }

  // Solve on coarse level (recursively)
  VecX e_coarse = VecX::Zero(Lc.num_dofs);
  v_cycle(level - 1, e_coarse, r_coarse);

  // Prolongate correction: e = P * e_c
  // P is stored in coarse level (maps level-1 -> level)
  VecX e;
  {
    OptionalScopedTimer t(profile_ ? &profile_->vcycle_prolong_ms : nullptr);
    e = Lc.P * e_coarse;
  }

  // Update solution: x = x + e
  x += e;

  // Post-smoothing
  {
    OptionalScopedTimer t(profile_ ? &profile_->vcycle_post_smooth_ms
                                   : nullptr);
    if (profile_) {
      profile_->smoothing_iterations += config_.post_smoothing;
    }
    L.smoother->apply(x, b, config_.post_smoothing);
  }
}

void BezierMultigridPreconditioner::build_element_coloring(int level) {
  auto &L = levels_[level];

  // Clear existing coloring
  L.elements_by_color.clear();
  L.num_colors = 0;

  size_t num_elements = L.element_free_dofs.size();
  if (num_elements == 0)
    return;

  // Build DOF -> elements adjacency map
  std::map<Index, std::vector<Index>> dof_to_elements;
  for (size_t e = 0; e < num_elements; ++e) {
    for (Index dof : L.element_free_dofs[e]) {
      dof_to_elements[dof].push_back(static_cast<Index>(e));
    }
  }

  // Greedy graph coloring based on DOF adjacency
  std::vector<int> element_color(num_elements, -1);
  int max_color = -1;

  for (size_t e = 0; e < num_elements; ++e) {
    // Find colors used by adjacent elements (elements sharing DOFs)
    std::set<int> neighbor_colors;
    for (Index dof : L.element_free_dofs[e]) {
      for (Index neighbor : dof_to_elements[dof]) {
        if (neighbor != static_cast<Index>(e) && element_color[neighbor] >= 0) {
          neighbor_colors.insert(element_color[neighbor]);
        }
      }
    }

    // Find lowest available color
    int color = 0;
    while (neighbor_colors.count(color)) {
      color++;
    }

    element_color[e] = color;
    max_color = std::max(max_color, color);
  }

  // Build color groups
  L.num_colors = max_color + 1;
  L.elements_by_color.resize(L.num_colors);

  for (size_t e = 0; e < num_elements; ++e) {
    L.elements_by_color[element_color[e]].push_back(static_cast<Index>(e));
  }

  if (config_.verbose) {
    std::cerr << "[Multigrid] Graph coloring: " << L.num_colors
              << " colors for " << num_elements << " elements (";
    for (int c = 0; c < L.num_colors; ++c) {
      std::cerr << L.elements_by_color[c].size();
      if (c < L.num_colors - 1)
        std::cerr << "/";
    }
    std::cerr << ")\n";
  }
}

// =============================================================================
// Composite grid methods for adaptive meshes
// =============================================================================

int BezierMultigridPreconditioner::compute_min_leaf_depth(
    const QuadtreeAdapter &mesh) {
  int min_depth = std::numeric_limits<int>::max();
  for (const QuadtreeNode *leaf : mesh.elements()) {
    min_depth = std::min(min_depth, leaf->level.max_level());
  }
  return min_depth;
}

int BezierMultigridPreconditioner::get_child_quadrant(
    const QuadtreeNode *child, const QuadtreeNode *parent) {
  Real parent_cx = 0.5 * (parent->bounds.xmin + parent->bounds.xmax);
  Real parent_cy = 0.5 * (parent->bounds.ymin + parent->bounds.ymax);
  Real child_cx = 0.5 * (child->bounds.xmin + child->bounds.xmax);
  Real child_cy = 0.5 * (child->bounds.ymin + child->bounds.ymax);

  int cx = (child_cx > parent_cx) ? 1 : 0;
  int cy = (child_cy > parent_cy) ? 1 : 0;
  return cx + 2 * cy;
}

void BezierMultigridPreconditioner::assign_dofs_from_bounds(
    CompositeGridNode &node, const QuadBounds &bounds,
    std::map<std::pair<int64_t, int64_t>, Index> &position_to_dof,
    Index &num_dofs) {

  static constexpr Real POSITION_SCALE = 1e8;
  auto quantize = [](Real x, Real y) -> std::pair<int64_t, int64_t> {
    return {static_cast<int64_t>(std::round(x * POSITION_SCALE)),
            static_cast<int64_t>(std::round(y * POSITION_SCALE))};
  };

  Real dx = bounds.xmax - bounds.xmin;
  Real dy = bounds.ymax - bounds.ymin;

  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      int local_dof = i + 4 * j;
      Real u = i / 3.0;
      Real v = j / 3.0;
      Real x = bounds.xmin + u * dx;
      Real y = bounds.ymin + v * dy;

      auto key = quantize(x, y);
      auto it = position_to_dof.find(key);
      if (it != position_to_dof.end()) {
        node.dof_indices[local_dof] = it->second;
      } else {
        position_to_dof[key] = num_dofs;
        node.dof_indices[local_dof] = num_dofs;
        num_dofs++;
      }
    }
  }
}

void BezierMultigridPreconditioner::build_composite_hierarchy(
    const QuadtreeAdapter &mesh, const CGCubicBezierDofManager &dof_manager) {

  // Store references for use by other methods
  mesh_ = &mesh;
  dof_manager_ = &dof_manager;

  int finest_level = static_cast<int>(levels_.size()) - 1;

  // Initialize finest level with ALL leaves (regardless of tree depth)
  auto &finest_grid = levels_[finest_level].composite_grid;
  finest_grid.nodes.clear();
  finest_grid.node_index_map.clear();
  finest_grid.num_dofs = dof_manager.num_free_dofs();
  finest_grid.max_depth = 0;

  for (const QuadtreeNode *leaf : mesh.elements()) {
    CompositeGridNode node;
    node.tree_node = leaf;
    node.is_subdivided = false; // Leaves are never subdivided at finest level

    // Copy DOF indices from the actual DOF manager
    const auto &global_dofs = dof_manager.element_dofs(leaf->leaf_index);
    for (int k = 0; k < 16; ++k) {
      node.dof_indices[k] = dof_manager.global_to_free(global_dofs[k]);
    }

    finest_grid.max_depth =
        std::max(finest_grid.max_depth, leaf->level.max_level());
    finest_grid.node_index_map[leaf] =
        static_cast<Index>(finest_grid.nodes.size());
    finest_grid.nodes.push_back(std::move(node));
  }

  // Finest level has no subdivision/passthrough - these only describe
  // how a coarse level maps to a finer level above it
  levels_[finest_level].subdivision_node_indices.clear();
  levels_[finest_level].passthrough_node_indices.clear();

  if (config_.verbose) {
    std::cerr << "[Multigrid] Composite level " << finest_level << ": "
              << finest_grid.nodes.size()
              << " nodes (0 subdivision, 0 pass-through), "
              << finest_grid.num_dofs
              << " DOFs, max_depth=" << finest_grid.max_depth << "\n";
  }

  // Build coarser levels iteratively
  for (int mg_level = finest_level - 1; mg_level >= 0; --mg_level) {
    build_composite_level_from_finer(mg_level);
  }

  // Clear cached references
  mesh_ = nullptr;
  dof_manager_ = nullptr;
}

namespace {
/// Recursively collect all nodes at a specific tree depth
void collect_nodes_at_depth(const QuadtreeNode *node, int target_depth,
                            std::vector<const QuadtreeNode *> &result) {
  if (node->level.max_level() == target_depth) {
    result.push_back(node);
    return;
  }
  if (node->level.max_level() > target_depth) {
    return; // Too deep, shouldn't happen if called correctly
  }
  // Recurse into children
  for (const auto &child : node->children) {
    collect_nodes_at_depth(child.get(), target_depth, result);
  }
}
} // namespace

void BezierMultigridPreconditioner::build_composite_level_from_finer(
    int mg_level) {
  const auto &fine_grid = levels_[mg_level + 1].composite_grid;
  auto &coarse_grid = levels_[mg_level].composite_grid;

  coarse_grid.nodes.clear();
  coarse_grid.node_index_map.clear();
  coarse_grid.num_dofs = 0;
  coarse_grid.max_depth = 0;

  levels_[mg_level].subdivision_node_indices.clear();
  levels_[mg_level].passthrough_node_indices.clear();

  // Position-based DOF deduplication
  std::map<std::pair<int64_t, int64_t>, Index> position_to_dof;

  // Determine target tree depth for this MG level
  // Each coarser MG level is one tree level coarser than the fine grid above it
  int target_depth = fine_grid.max_depth - 1;

  // Collect ALL nodes at target depth from the quadtree
  std::vector<const QuadtreeNode *> nodes_at_depth;
  collect_nodes_at_depth(mesh_->root(), target_depth, nodes_at_depth);

  // Build set of fine-grid tree nodes for quick lookup
  std::set<const QuadtreeNode *> fine_grid_nodes;
  for (const auto &node : fine_grid.nodes) {
    fine_grid_nodes.insert(node.tree_node);
  }

  // Classify and add each node at target depth
  for (const QuadtreeNode *tree_node : nodes_at_depth) {
    CompositeGridNode node;
    node.tree_node = tree_node;

    if (fine_grid_nodes.count(tree_node) > 0) {
      // This node IS in the fine grid - pass-through (identity prolongation)
      node.is_subdivided = false;

      // For pass-through nodes, copy DOF structure from fine level
      auto fine_it = fine_grid.node_index_map.find(tree_node);
      const auto &fine_node = fine_grid.nodes[fine_it->second];

      const auto &bounds = tree_node->bounds;
      Real dx = bounds.xmax - bounds.xmin;
      Real dy = bounds.ymax - bounds.ymin;

      static constexpr Real POSITION_SCALE = 1e8;
      auto quantize = [](Real x, Real y) -> std::pair<int64_t, int64_t> {
        return {static_cast<int64_t>(std::round(x * POSITION_SCALE)),
                static_cast<int64_t>(std::round(y * POSITION_SCALE))};
      };

      for (int j = 0; j < 4; ++j) {
        for (int ii = 0; ii < 4; ++ii) {
          int local_dof = ii + 4 * j;
          Index fine_dof = fine_node.dof_indices[local_dof];

          if (fine_dof >= 0) {
            Real u = ii / 3.0;
            Real v = j / 3.0;
            Real x = bounds.xmin + u * dx;
            Real y = bounds.ymin + v * dy;

            auto key = quantize(x, y);
            auto it = position_to_dof.find(key);
            if (it != position_to_dof.end()) {
              node.dof_indices[local_dof] = it->second;
            } else {
              position_to_dof[key] = coarse_grid.num_dofs;
              node.dof_indices[local_dof] = coarse_grid.num_dofs;
              coarse_grid.num_dofs++;
            }
          } else {
            node.dof_indices[local_dof] = -1;
          }
        }
      }

      coarse_grid.max_depth =
          std::max(coarse_grid.max_depth, tree_node->level.max_level());
      coarse_grid.node_index_map[tree_node] =
          static_cast<Index>(coarse_grid.nodes.size());
      levels_[mg_level].passthrough_node_indices.push_back(
          static_cast<Index>(coarse_grid.nodes.size()));
      coarse_grid.nodes.push_back(std::move(node));
    } else {
      // This node has descendants in fine grid - subdivision
      node.is_subdivided = true;

      assign_dofs_from_bounds(node, tree_node->bounds, position_to_dof,
                              coarse_grid.num_dofs);

      coarse_grid.max_depth =
          std::max(coarse_grid.max_depth, tree_node->level.max_level());
      coarse_grid.node_index_map[tree_node] =
          static_cast<Index>(coarse_grid.nodes.size());
      levels_[mg_level].subdivision_node_indices.push_back(
          static_cast<Index>(coarse_grid.nodes.size()));
      coarse_grid.nodes.push_back(std::move(node));
    }
  }

  if (config_.verbose) {
    std::cerr << "[Multigrid] Composite level " << mg_level << ": "
              << coarse_grid.nodes.size() << " nodes ("
              << levels_[mg_level].subdivision_node_indices.size()
              << " subdivision, "
              << levels_[mg_level].passthrough_node_indices.size()
              << " pass-through), " << coarse_grid.num_dofs
              << " DOFs, max_depth=" << coarse_grid.max_depth << "\n";
  }
}

void BezierMultigridPreconditioner::add_recursive_prolongation(
    const QuadtreeNode *tree_node, const CompositeGridNode &coarse_node,
    const CompositeGridLevel &fine_grid, const MatX &P_accumulated,
    const std::array<MatX, 4> &P_local,
    std::vector<Eigen::Triplet<Real>> &triplets,
    const QuadtreeNode *parent) const {

  // Check if this tree node is in the fine grid (i.e., it's a leaf at fine
  // level)
  auto it = fine_grid.node_index_map.find(tree_node);
  if (it != fine_grid.node_index_map.end()) {
    // Found in fine grid - add prolongation entries
    // Note: We allow duplicate triplets for the same (fine_dof, coarse_dof)
    // pair. setFromTriplets() will sum them, and we normalize P rows
    // afterwards.
    const auto &fine_node = fine_grid.nodes[it->second];

    for (int fine_local = 0; fine_local < 16; ++fine_local) {
      Index fine_dof = fine_node.dof_indices[fine_local];
      if (fine_dof < 0)
        continue;

      for (int coarse_local = 0; coarse_local < 16; ++coarse_local) {
        Real weight = P_accumulated(fine_local, coarse_local);
        if (std::abs(weight) < 1e-14)
          continue;

        Index coarse_dof = coarse_node.dof_indices[coarse_local];
        if (coarse_dof < 0)
          continue;

        triplets.emplace_back(fine_dof, coarse_dof, weight);
      }
    }
    return;
  }

  // Not in fine grid - recurse into children
  if (tree_node->children.empty()) {
    // Leaf but not in fine grid - shouldn't happen if mesh is consistent
    return;
  }

  for (const auto &child : tree_node->children) {
    int child_quadrant = get_child_quadrant(child.get(), tree_node);
    MatX P_new = P_local[child_quadrant] * P_accumulated;
    add_recursive_prolongation(child.get(), coarse_node, fine_grid, P_new,
                               P_local, triplets, tree_node);
  }
}

SpMat BezierMultigridPreconditioner::build_prolongation_composite(
    int mg_level) {
  const auto &coarse_grid = levels_[mg_level].composite_grid;
  const auto &fine_grid = levels_[mg_level + 1].composite_grid;

  Index fine_dofs = fine_grid.num_dofs;
  Index coarse_dofs = coarse_grid.num_dofs;

  // Bezier subdivision matrices for each child quadrant
  std::array<MatX, 4> P_local;
  P_local[0] = kronecker_product(S_left_, S_left_);   // (0,0)
  P_local[1] = kronecker_product(S_right_, S_left_);  // (1,0)
  P_local[2] = kronecker_product(S_left_, S_right_);  // (0,1)
  P_local[3] = kronecker_product(S_right_, S_right_); // (1,1)

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(coarse_grid.nodes.size() * 16 * 16);

  // Part 1: Subdivision contributions
  // For each coarse subdivision node, map to all descendant leaves in fine grid
  for (Index coarse_idx : levels_[mg_level].subdivision_node_indices) {
    const auto &coarse_node = coarse_grid.nodes[coarse_idx];
    const QuadtreeNode *parent = coarse_node.tree_node;

    // Recursively process all 4 children (whether they're leaves or refined)
    for (const auto &tree_child : parent->children) {
      int child_quadrant = get_child_quadrant(tree_child.get(), parent);
      MatX P_child = P_local[child_quadrant]; // Start with single subdivision
      add_recursive_prolongation(tree_child.get(), coarse_node, fine_grid,
                                 P_child, P_local, triplets, parent);
    }
  }

  // Part 2: Pass-through (identity) contributions
  // Nodes that appear at both levels map identically
  for (Index coarse_idx : levels_[mg_level].passthrough_node_indices) {
    const auto &coarse_node = coarse_grid.nodes[coarse_idx];

    // Find corresponding node in fine grid
    auto it = fine_grid.node_index_map.find(coarse_node.tree_node);
    if (it == fine_grid.node_index_map.end())
      continue;

    Index fine_idx = it->second;
    const auto &fine_node = fine_grid.nodes[fine_idx];

    // Identity mapping: P(fine_dof[k], coarse_dof[k]) = 1
    // Note: setFromTriplets sums duplicate entries, so interface DOFs that
    // receive contributions from both subdivision and pass-through will
    // accumulate and be normalized below.
    for (int local = 0; local < 16; ++local) {
      Index fine_dof = fine_node.dof_indices[local];
      Index coarse_dof = coarse_node.dof_indices[local];
      if (fine_dof >= 0 && coarse_dof >= 0) {
        triplets.emplace_back(fine_dof, coarse_dof, 1.0);
      }
    }
  }

  SpMat P(fine_dofs, coarse_dofs);
  P.setFromTriplets(triplets.begin(), triplets.end());

  // Normalize P rows to sum to 1 (partition of unity for interpolation)
  // This handles interface DOFs that received contributions from multiple
  // sources
  Eigen::SparseMatrix<Real, Eigen::RowMajor> P_row(P);
  VecX row_sums = P_row * VecX::Ones(P.cols());
  for (Index i = 0; i < P_row.rows(); ++i) {
    Real scale = row_sums(i);
    if (std::abs(scale) > 1e-14 && std::abs(scale - 1.0) > 1e-10) {
      Real inv_scale = 1.0 / scale;
      for (Eigen::SparseMatrix<Real, Eigen::RowMajor>::InnerIterator it(P_row,
                                                                        i);
           it; ++it) {
        it.valueRef() *= inv_scale;
      }
    }
  }
  P = SpMat(P_row);

  if (config_.verbose) {
    // Count rows with exactly one non-zero (identity) vs multiple (subdivision)
    // SpMat is column-major, so count nnz per row manually
    std::vector<Index> row_nnz(P.rows(), 0);
    for (int k = 0; k < P.outerSize(); ++k) {
      for (SpMat::InnerIterator it(P, k); it; ++it) {
        row_nnz[it.row()]++;
      }
    }
    Index identity_rows = 0;
    Index subdivision_rows = 0;
    for (Index row = 0; row < P.rows(); ++row) {
      if (row_nnz[row] == 1) {
        identity_rows++;
      } else if (row_nnz[row] > 1) {
        subdivision_rows++;
      }
    }
    std::cerr << "[Multigrid] Prolongation " << mg_level << "->"
              << (mg_level + 1) << ": " << P.rows() << "x" << P.cols() << ", "
              << identity_rows << " identity rows, " << subdivision_rows
              << " subdivision rows\n";
  }

  return P;
}

SpMat BezierMultigridPreconditioner::build_restriction_l2(int mg_level) {
  const auto &coarse_grid = levels_[mg_level].composite_grid;
  const auto &fine_grid = levels_[mg_level + 1].composite_grid;

  Index fine_dofs = fine_grid.num_dofs;
  Index coarse_dofs = coarse_grid.num_dofs;

  // Bezier subdivision matrices for each child quadrant (for recursive
  // traversal)
  std::array<MatX, 4> P_local;
  P_local[0] = kronecker_product(S_left_, S_left_);   // (0,0)
  P_local[1] = kronecker_product(S_right_, S_left_);  // (1,0)
  P_local[2] = kronecker_product(S_left_, S_right_);  // (0,1)
  P_local[3] = kronecker_product(S_right_, S_right_); // (1,1)

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(coarse_grid.nodes.size() * 16 * 16);

  // Track which (coarse_dof, fine_dof) pairs have been processed to avoid
  // duplicates
  std::set<std::pair<Index, Index>> processed_pairs;

  // Identity matrix for starting recursive traversal
  MatX I = MatX::Identity(16, 16);

  // Part 1: Subdivision contributions using L2 projection
  for (Index coarse_idx : levels_[mg_level].subdivision_node_indices) {
    const auto &coarse_node = coarse_grid.nodes[coarse_idx];
    const QuadtreeNode *parent = coarse_node.tree_node;

    // Process all 4 children recursively
    for (const auto &tree_child : parent->children) {
      int child_quadrant = get_child_quadrant(tree_child.get(), parent);
      MatX P_accumulated =
          P_local[child_quadrant]; // Start with single subdivision
      int depth = 1;               // Depth from coarse node

      // Lambda to recursively process descendants
      std::function<void(const QuadtreeNode *, const MatX &, int)>
          process_descendants = [&](const QuadtreeNode *node, const MatX &P_acc,
                                    int d) {
            // Check if this node is in the fine grid
            auto it = fine_grid.node_index_map.find(node);
            if (it != fine_grid.node_index_map.end()) {
              // Found in fine grid - compute L2 restriction weights
              const auto &fine_node = fine_grid.nodes[it->second];

              // Area scale = (1/4)^d where d is depth from coarse
              Real area_scale = std::pow(0.25, d);

              // L2 restriction: R_eff = M_c^{-1} * P_acc^T * (area_scale * M_f)
              //                       = area_scale * M_2D_inv * P_acc^T * M_2D
              MatX R_eff = area_scale * M_2D_inv_ * P_acc.transpose() * M_2D_;

              // Add triplets for R(coarse_dof, fine_dof)
              for (int coarse_local = 0; coarse_local < 16; ++coarse_local) {
                Index coarse_dof = coarse_node.dof_indices[coarse_local];
                if (coarse_dof < 0)
                  continue;

                for (int fine_local = 0; fine_local < 16; ++fine_local) {
                  Index fine_dof = fine_node.dof_indices[fine_local];
                  if (fine_dof < 0)
                    continue;

                  Real weight = R_eff(coarse_local, fine_local);
                  if (std::abs(weight) < 1e-14)
                    continue;

                  // Skip if already processed
                  auto pair = std::make_pair(coarse_dof, fine_dof);
                  if (processed_pairs.count(pair) > 0)
                    continue;
                  processed_pairs.insert(pair);

                  triplets.emplace_back(coarse_dof, fine_dof, weight);
                }
              }
              return;
            }

            // Not in fine grid - recurse into children
            if (node->children.empty())
              return;

            for (const auto &child : node->children) {
              int child_quad = get_child_quadrant(child.get(), node);
              MatX P_new = P_local[child_quad] * P_acc;
              process_descendants(child.get(), P_new, d + 1);
            }
          };

      process_descendants(tree_child.get(), P_accumulated, depth);
    }
  }

  // Part 2: Pass-through (identity) contributions
  for (Index coarse_idx : levels_[mg_level].passthrough_node_indices) {
    const auto &coarse_node = coarse_grid.nodes[coarse_idx];

    // Find corresponding node in fine grid
    auto it = fine_grid.node_index_map.find(coarse_node.tree_node);
    if (it == fine_grid.node_index_map.end())
      continue;

    Index fine_idx = it->second;
    const auto &fine_node = fine_grid.nodes[fine_idx];

    // Identity mapping: R(coarse_dof[k], fine_dof[k]) = 1
    for (int local = 0; local < 16; ++local) {
      Index fine_dof = fine_node.dof_indices[local];
      Index coarse_dof = coarse_node.dof_indices[local];
      if (fine_dof >= 0 && coarse_dof >= 0) {
        auto pair = std::make_pair(coarse_dof, fine_dof);
        if (processed_pairs.count(pair) > 0)
          continue;
        processed_pairs.insert(pair);
        triplets.emplace_back(coarse_dof, fine_dof, 1.0);
      }
    }
  }

  SpMat R(coarse_dofs, fine_dofs);
  R.setFromTriplets(triplets.begin(), triplets.end());

  // Normalize rows to sum to 1 (partition of unity)
  // This ensures constant fields are preserved during restriction.
  // For adaptive meshes with partial children, the raw L2 projection
  // doesn't automatically satisfy this property.
  {
    // Convert to RowMajor for efficient row access
    Eigen::SparseMatrix<Real, Eigen::RowMajor> R_row(R);
    VecX row_sums = R_row * VecX::Ones(fine_dofs);

    // Debug: report row sum statistics before normalization
    if (config_.verbose) {
      Real min_sum = row_sums.minCoeff();
      Real max_sum = row_sums.maxCoeff();
      Real mean_sum = row_sums.mean();
      Index num_needing_norm = 0;
      for (Index i = 0; i < row_sums.size(); ++i) {
        if (std::abs(row_sums(i) - 1.0) > 1e-10)
          ++num_needing_norm;
      }
      std::cerr << "[Multigrid] R row sums before normalization: min="
                << min_sum << ", max=" << max_sum << ", mean=" << mean_sum
                << ", rows needing normalization=" << num_needing_norm << "/"
                << row_sums.size() << "\n";
    }

    for (Index i = 0; i < R_row.rows(); ++i) {
      Real scale = row_sums(i);
      if (std::abs(scale) > 1e-14 && std::abs(scale - 1.0) > 1e-10) {
        Real inv_scale = 1.0 / scale;
        for (Eigen::SparseMatrix<Real, Eigen::RowMajor>::InnerIterator it(R_row,
                                                                          i);
             it; ++it) {
          it.valueRef() *= inv_scale;
        }
      }
    }

    R = SpMat(R_row); // Convert back to ColMajor
  }

  if (config_.verbose) {
    std::cerr << "[Multigrid] L2 Restriction " << (mg_level + 1) << "->"
              << mg_level << ": " << R.rows() << "x" << R.cols()
              << ", nnz=" << R.nonZeros() << "\n";
  }

  return R;
}

} // namespace drifter
