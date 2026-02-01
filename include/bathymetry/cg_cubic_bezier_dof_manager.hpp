#pragma once

/// @file cg_cubic_bezier_dof_manager.hpp
/// @brief CG DOF management for cubic Bezier basis bathymetry smoothing
///
/// Manages Continuous Galerkin (CG) degrees of freedom for cubic Bezier
/// elements (degree 3, 4×4 = 16 DOFs per element). CG shares DOFs at interfaces:
///   - Corner DOFs: shared by all elements meeting at the vertex
///   - Edge DOFs: shared by the 2 elements on either side
///   - Interior DOFs: unique per element
///
/// For C¹ continuity constraints (derivatives z_u, z_v, z_uv).

#include "core/types.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/cubic_bezier_basis_2d.hpp"
#include <vector>
#include <map>
#include <set>

namespace drifter {

/// @brief Constraint for a hanging node at non-conforming interface (cubic)
struct CubicHangingNodeConstraint {
    Index slave_dof;
    std::vector<Index> master_dofs;
    std::vector<Real> weights;
};

/// @brief C¹ vertex derivative constraint for cubic elements
///
/// Enforces matching of first derivatives (z_u, z_v) and mixed partial (z_uv)
/// at shared vertices. Form: sum(c1*phi1)/scale1 - sum(c2*phi2)/scale2 = 0
struct CubicVertexDerivativeConstraint {
    Index elem1, elem2;
    int corner1, corner2;
    int nu, nv;                     ///< Derivative order (0,1), (1,0), or (1,1)
    VecX coeffs1, coeffs2;          ///< Basis derivative coefficients (16 values each)
    Real scale1, scale2;            ///< Element size scale factors (dx^nu * dy^nv)
};

/// @brief C¹ edge derivative constraint for cubic elements
///
/// Enforces matching of normal derivative z_n at Gauss points along shared edges.
struct CubicEdgeDerivativeConstraint {
    Index elem1, elem2;
    int edge1, edge2;
    Real t;                         ///< Parameter position along edge (0 to 1)
    int deriv_order;                ///< Always 1 for C¹
    VecX coeffs1, coeffs2;          ///< Basis derivative coefficients (16 values each)
    Real scale1, scale2;
};

/// @brief CG DOF manager for cubic Bezier elements
///
/// Manages global DOF numbering with sharing at element interfaces.
/// Uses 3-pass algorithm: vertex DOFs → edge DOFs → interior DOFs
class CGCubicBezierDofManager {
public:
    /// @brief Construct DOF manager
    /// @param mesh 2D quadtree mesh
    CGCubicBezierDofManager(const QuadtreeAdapter& mesh);

    // =========================================================================
    // DOF queries
    // =========================================================================

    Index num_global_dofs() const { return num_global_dofs_; }
    Index num_free_dofs() const { return num_free_dofs_; }
    int num_element_dofs() const { return CubicBezierBasis2D::NDOF; }

    Index global_dof(Index elem, int local_dof) const;
    const std::vector<Index>& element_dofs(Index elem) const;
    const std::vector<std::vector<Index>>& all_element_dofs() const {
        return elem_to_global_;
    }

    bool is_boundary_dof(Index dof) const;
    const std::vector<Index>& boundary_dofs() const { return boundary_dofs_; }

    // =========================================================================
    // Constraint handling
    // =========================================================================

    const std::vector<CubicHangingNodeConstraint>& constraints() const {
        return constraints_;
    }
    Index num_constraints() const { return static_cast<Index>(constraints_.size()); }

    const std::vector<CubicVertexDerivativeConstraint>& vertex_derivative_constraints() const {
        return vertex_derivative_constraints_;
    }
    Index num_vertex_derivative_constraints() const {
        return static_cast<Index>(vertex_derivative_constraints_.size());
    }

    /// Build C¹ vertex derivative constraints
    /// Enforces: z_u, z_v, z_uv matching at shared vertices
    void build_vertex_derivative_constraints();

    const std::vector<CubicEdgeDerivativeConstraint>& edge_derivative_constraints() const {
        return edge_derivative_constraints_;
    }
    Index num_edge_derivative_constraints() const {
        return static_cast<Index>(edge_derivative_constraints_.size());
    }

    /// Build C¹ edge derivative constraints
    /// Enforces: z_n matching at Gauss points along conforming edges
    /// @param ngauss Number of Gauss points per edge (default: 4)
    void build_edge_derivative_constraints(int ngauss = 4);

    bool is_constrained(Index dof) const;
    SpMat build_constraint_matrix() const;

    Index global_to_free(Index global_dof) const;
    Index free_to_global(Index free_dof) const;

    // =========================================================================
    // Mesh and basis access
    // =========================================================================

    const QuadtreeAdapter& mesh() const { return mesh_; }
    const CubicBezierBasis2D& basis() const { return basis_; }

private:
    const QuadtreeAdapter& mesh_;
    CubicBezierBasis2D basis_;

    Index num_global_dofs_ = 0;
    Index num_free_dofs_ = 0;

    std::vector<std::vector<Index>> elem_to_global_;
    std::vector<Index> boundary_dofs_;
    std::set<Index> boundary_dof_set_;
    std::vector<CubicHangingNodeConstraint> constraints_;
    std::vector<CubicVertexDerivativeConstraint> vertex_derivative_constraints_;
    std::vector<CubicEdgeDerivativeConstraint> edge_derivative_constraints_;
    std::set<Index> constrained_dofs_;
    std::vector<Index> global_to_free_;
    std::vector<Index> free_to_global_;

    // DOF assignment methods
    void assign_vertex_dofs();
    void assign_edge_dofs();
    void assign_edge_dofs_nonconforming();
    void assign_interior_dofs();
    void identify_boundary_dofs();

    // Constraint handling
    void build_hanging_node_constraints();
    void build_dof_mappings();

    // Helpers
    Vec2 get_dof_position(Index elem, int local_dof) const;
    std::map<std::pair<int64_t, int64_t>, Index> position_to_dof_;
    std::pair<int64_t, int64_t> quantize_position(const Vec2& pos) const;
    Index find_dof_at_position(const Vec2& pos) const;

    bool is_corner_dof(int local_dof) const;
    bool is_edge_dof(int local_dof) const;
    bool is_interior_dof(int local_dof) const;
    int get_edge_for_dof(int local_dof) const;
};

}  // namespace drifter
