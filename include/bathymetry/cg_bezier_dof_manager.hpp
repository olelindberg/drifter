#pragma once

/// @file cg_bezier_dof_manager.hpp
/// @brief CG DOF management for Bezier basis bathymetry smoothing
///
/// Manages Continuous Galerkin (CG) degrees of freedom for quintic Bezier
/// elements. Unlike DG where each element has independent DOFs, CG shares
/// DOFs at element interfaces:
///   - Corner DOFs: shared by all elements meeting at the vertex
///   - Edge DOFs: shared by the 2 elements on either side
///   - Interior DOFs: unique per element
///
/// For non-conforming meshes (hanging nodes), this class provides constraints
/// that relate hanging DOFs to their coarse neighbors via Bezier interpolation.

#include "bathymetry/bezier_basis_2d.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include <map>
#include <set>
#include <vector>

namespace drifter {

/// @brief Constraint for a hanging node at non-conforming interface
///
/// Hanging DOFs on fine elements are constrained to interpolate from
/// the coarse element's Bezier curve along the shared edge.
struct HangingNodeConstraint {
    /// Index of the constrained (slave) DOF
    Index slave_dof;

    /// Indices of the master DOFs (on coarse edge)
    std::vector<Index> master_dofs;

    /// Interpolation weights for each master DOF
    std::vector<Real> weights;
};

/// @brief Constraint for derivative continuity along shared edge at Gauss
/// points
///
/// Enforces C¹/C² continuity along shared edges at Gauss quadrature points.
/// This complements vertex constraints by ensuring smoothness along the entire
/// edge, not just at corners. The normal derivative across the edge should
/// match.
struct EdgeDerivativeConstraint {
    /// Adjacent elements sharing the edge
    Index elem1, elem2;

    /// Edge indices for each element (0=left, 1=right, 2=bottom, 3=top)
    int edge1, edge2;

    /// Parameter position along the edge (0 to 1)
    Real t;

    /// Derivative order in normal direction (1 = first derivative, 2 = second)
    int deriv_order;

    /// Basis derivative coefficients for each element (36 values each)
    /// These are the normal derivatives at the Gauss point
    VecX coeffs1, coeffs2;

    /// Element size scale factors (dx or dy depending on edge orientation)
    Real scale1, scale2;
};

/// @brief CG DOF manager for quintic Bezier elements
///
/// Manages global DOF numbering with sharing at element interfaces.
/// The numbering follows a 3-pass algorithm:
///   1. Vertex DOFs (corners) - shared by multiple elements
///   2. Edge DOFs - shared by 2 adjacent elements
///   3. Interior DOFs - unique per element
class CGBezierDofManager {
public:
    /// @brief Construct DOF manager
    /// @param mesh 2D quadtree mesh
    CGBezierDofManager(const QuadtreeAdapter &mesh);

    // =========================================================================
    // DOF queries
    // =========================================================================

    /// Total number of global DOFs
    Index num_global_dofs() const { return num_global_dofs_; }

    /// Number of free DOFs (after constraint elimination)
    Index num_free_dofs() const { return num_free_dofs_; }

    /// Number of DOFs per element (always 36 for quintic Bezier)
    int num_element_dofs() const { return BezierBasis2D::NDOF; }

    /// Get global DOF index for a local DOF in an element
    /// @param elem Element index
    /// @param local_dof Local DOF index (0-35)
    Index global_dof(Index elem, int local_dof) const;

    /// Get all global DOF indices for an element
    const std::vector<Index> &element_dofs(Index elem) const;

    /// Get local-to-global DOF mapping for all elements
    const std::vector<std::vector<Index>> &all_element_dofs() const { return elem_to_global_; }

    /// Check if a DOF is on the domain boundary
    bool is_boundary_dof(Index dof) const;

    /// Get all DOFs on domain boundary
    const std::vector<Index> &boundary_dofs() const { return boundary_dofs_; }

    // =========================================================================
    // Constraint handling for non-conforming meshes
    // =========================================================================

    /// Get all hanging node constraints
    const std::vector<HangingNodeConstraint> &constraints() const { return constraints_; }

    /// Number of hanging node constraints
    Index num_constraints() const { return static_cast<Index>(constraints_.size()); }

    /// Get edge derivative constraints for C² continuity along edges
    const std::vector<EdgeDerivativeConstraint> &edge_derivative_constraints() const {
        return edge_derivative_constraints_;
    }

    /// Number of edge derivative constraints
    Index num_edge_derivative_constraints() const {
        return static_cast<Index>(edge_derivative_constraints_.size());
    }

    /// Build edge derivative constraints for C² continuity along shared edges
    /// Enforces matching of normal derivatives at Gauss points along conforming
    /// edges.
    /// @param ngauss Number of Gauss points per edge (default: 4)
    void build_edge_derivative_constraints(int ngauss = 4);

    /// Check if a DOF is constrained (hanging node)
    bool is_constrained(Index dof) const;

    /// Build constraint matrix A such that A*x = 0 for valid solutions
    /// @return Sparse matrix (num_constraints x num_global_dofs)
    SpMat build_constraint_matrix() const;

    /// @brief Map from global DOF to free DOF index
    /// @return -1 if DOF is constrained, otherwise the free DOF index
    Index global_to_free(Index global_dof) const;

    /// @brief Map from free DOF to global DOF index
    Index free_to_global(Index free_dof) const;

    // =========================================================================
    // Mesh and basis access
    // =========================================================================

    /// Get the mesh
    const QuadtreeAdapter &mesh() const { return mesh_; }

    /// Get the Bezier basis
    const BezierBasis2D &basis() const { return basis_; }

private:
    const QuadtreeAdapter &mesh_;
    BezierBasis2D basis_;

    /// Number of DOFs
    Index num_global_dofs_ = 0;
    Index num_free_dofs_ = 0;

    /// Local to global DOF mapping for each element
    /// elem_to_global_[elem][local_dof] = global_dof
    std::vector<std::vector<Index>> elem_to_global_;

    /// Boundary DOFs
    std::vector<Index> boundary_dofs_;
    std::set<Index> boundary_dof_set_;

    /// Hanging node constraints
    std::vector<HangingNodeConstraint> constraints_;

    /// Edge derivative constraints for C² continuity along edges
    std::vector<EdgeDerivativeConstraint> edge_derivative_constraints_;

    /// Constrained DOF set (for fast lookup)
    std::set<Index> constrained_dofs_;

    /// Global to free DOF mapping (-1 if constrained)
    std::vector<Index> global_to_free_;

    /// Free to global DOF mapping
    std::vector<Index> free_to_global_;

    // =========================================================================
    // DOF assignment methods (3-pass algorithm)
    // =========================================================================

    /// Assign DOFs at element vertices (corners)
    void assign_vertex_dofs();

    /// Assign DOFs along element edges
    void assign_edge_dofs();

    /// Assign DOFs at non-conforming edges using index-based sharing
    /// For 2:1 refinement, every 2nd fine DOF shares with a coarse DOF
    void assign_edge_dofs_nonconforming();

    /// Assign interior DOFs (unique per element)
    void assign_interior_dofs();

    /// Build boundary DOF list
    void identify_boundary_dofs();

    // =========================================================================
    // Constraint handling methods
    // =========================================================================

    /// Detect and build constraints for hanging nodes
    void build_hanging_node_constraints();

    /// Build the global-to-free and free-to-global mappings
    void build_dof_mappings();

    // =========================================================================
    // Helper methods
    // =========================================================================

    /// Get physical position for a local DOF in an element
    Vec2 get_dof_position(Index elem, int local_dof) const;

    /// Position lookup for DOF sharing (quantized position -> DOF)
    std::map<std::pair<int64_t, int64_t>, Index> position_to_dof_;

    /// Convert position to quantized key for map lookup
    std::pair<int64_t, int64_t> quantize_position(const Vec2 &pos) const;

    /// Find existing DOF at a position (or return -1)
    Index find_dof_at_position(const Vec2 &pos) const;

    /// Check if a local DOF is on a corner
    bool is_corner_dof(int local_dof) const;

    /// Check if a local DOF is on an edge (excluding corners)
    bool is_edge_dof(int local_dof) const;

    /// Check if a local DOF is interior
    bool is_interior_dof(int local_dof) const;

    /// Get the edge ID (0-3) for an edge DOF, or -1 if not on edge
    int get_edge_for_dof(int local_dof) const;
};

} // namespace drifter
