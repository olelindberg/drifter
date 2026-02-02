#pragma once

/// @file cg_linear_bezier_dof_manager.hpp
/// @brief CG DOF management for linear Bezier basis bathymetry smoothing
///
/// Manages Continuous Galerkin (CG) degrees of freedom for linear Bezier
/// elements (degree 1, 2x2 = 4 DOFs per element). CG shares DOFs at interfaces:
///   - Corner DOFs: shared by all elements meeting at the vertex
///   - No edge interior DOFs (all DOFs are corners for linear elements)
///
/// For C0 continuity only (no derivative constraints needed).

#include "bathymetry/linear_bezier_basis_2d.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include <map>
#include <set>
#include <vector>

namespace drifter {

/// @brief Constraint for a hanging node at non-conforming interface (linear)
struct LinearHangingNodeConstraint {
  Index slave_dof;
  std::vector<Index> master_dofs;
  std::vector<Real> weights;
};

/// @brief CG DOF manager for linear Bezier elements
///
/// Manages global DOF numbering with sharing at element interfaces.
/// For linear elements, all DOFs are corners, so only vertex sharing is needed.
class CGLinearBezierDofManager {
public:
  /// @brief Construct DOF manager
  /// @param mesh 2D quadtree mesh
  explicit CGLinearBezierDofManager(const QuadtreeAdapter &mesh);

  // =========================================================================
  // DOF queries
  // =========================================================================

  Index num_global_dofs() const { return num_global_dofs_; }
  Index num_free_dofs() const { return num_free_dofs_; }
  int num_element_dofs() const { return LinearBezierBasis2D::NDOF; }

  Index global_dof(Index elem, int local_dof) const;
  const std::vector<Index> &element_dofs(Index elem) const;
  const std::vector<std::vector<Index>> &all_element_dofs() const {
    return elem_to_global_;
  }

  bool is_boundary_dof(Index dof) const;
  const std::vector<Index> &boundary_dofs() const { return boundary_dofs_; }

  // =========================================================================
  // Constraint handling
  // =========================================================================

  const std::vector<LinearHangingNodeConstraint> &constraints() const {
    return constraints_;
  }
  Index num_constraints() const {
    return static_cast<Index>(constraints_.size());
  }

  bool is_constrained(Index dof) const;
  SpMat build_constraint_matrix() const;

  Index global_to_free(Index global_dof) const;
  Index free_to_global(Index free_dof) const;

  // =========================================================================
  // Mesh and basis access
  // =========================================================================

  const QuadtreeAdapter &mesh() const { return mesh_; }
  const LinearBezierBasis2D &basis() const { return basis_; }

private:
  const QuadtreeAdapter &mesh_;
  LinearBezierBasis2D basis_;

  Index num_global_dofs_ = 0;
  Index num_free_dofs_ = 0;

  std::vector<std::vector<Index>> elem_to_global_;
  std::vector<Index> boundary_dofs_;
  std::set<Index> boundary_dof_set_;
  std::vector<LinearHangingNodeConstraint> constraints_;
  std::set<Index> constrained_dofs_;
  std::vector<Index> global_to_free_;
  std::vector<Index> free_to_global_;

  // DOF assignment (single pass - corners only)
  void assign_vertex_dofs();
  void identify_boundary_dofs();

  // Constraint handling
  void build_hanging_node_constraints();
  void build_dof_mappings();

  // Helpers
  Vec2 get_dof_position(Index elem, int local_dof) const;
  std::map<std::pair<int64_t, int64_t>, Index> position_to_dof_;
  std::pair<int64_t, int64_t> quantize_position(const Vec2 &pos) const;
  Index find_dof_at_position(const Vec2 &pos) const;
};

} // namespace drifter
