#pragma once

/// @file cg_cubic_bezier_dof_manager.hpp
/// @brief CG DOF management for cubic Bezier basis bathymetry smoothing
///
/// Manages Continuous Galerkin (CG) degrees of freedom for cubic Bezier
/// elements (degree 3, 4×4 = 16 DOFs per element). CG shares DOFs at
/// interfaces:
///   - Corner DOFs: shared by all elements meeting at the vertex
///   - Edge DOFs: shared by the 2 elements on either side
///   - Interior DOFs: unique per element
///
/// For C¹ continuity constraints (derivatives z_u, z_v, z_uv).

#include "bathymetry/cg_bezier_dof_manager_base.hpp"
#include "bathymetry/cubic_bezier_basis_2d.hpp"
#include <vector>

namespace drifter {

/// @brief Constraint for a hanging node at non-conforming interface (cubic)
struct CubicHangingNodeConstraint {
    Index slave_dof;
    std::vector<Index> master_dofs;
    std::vector<Real> weights;
};

/// @brief C¹ edge derivative constraint for cubic elements
///
/// Enforces matching of normal derivative z_n at Gauss points along shared
/// edges.
struct CubicEdgeDerivativeConstraint {
    Index elem1, elem2;
    int edge1, edge2;
    Real t; ///< Parameter position along edge (0 to 1)
    int deriv_order; ///< Always 1 for C¹
    VecX coeffs1, coeffs2; ///< Basis derivative coefficients (16 values each)
    Real scale1, scale2;
};

/// @brief CG DOF manager for cubic Bezier elements
///
/// Manages global DOF numbering with sharing at element interfaces.
/// Uses 3-pass algorithm: vertex DOFs → edge DOFs → interior DOFs
class CGCubicBezierDofManager : public CGBezierDofManagerBase {
public:
    /// @brief Construct DOF manager
    /// @param mesh 2D quadtree mesh
    explicit CGCubicBezierDofManager(const QuadtreeAdapter &mesh);

    // =========================================================================
    // Base class overrides
    // =========================================================================

    int num_element_dofs() const override { return CubicBezierBasis2D::NDOF; }

    // =========================================================================
    // Cubic-specific accessors
    // =========================================================================

    const std::vector<CubicHangingNodeConstraint> &constraints() const { return constraints_; }
    const CubicBezierBasis2D &basis() const { return basis_; }

    const std::vector<CubicEdgeDerivativeConstraint> &edge_derivative_constraints() const {
        return edge_derivative_constraints_;
    }
    Index num_edge_derivative_constraints() const {
        return static_cast<Index>(edge_derivative_constraints_.size());
    }

    /// Build C¹ edge derivative constraints
    /// Enforces: z_n matching at Gauss points along conforming edges
    /// @param ngauss Number of Gauss points per edge (default: 4)
    void build_edge_derivative_constraints(int ngauss = 4);

    // =========================================================================
    // DOF classification helpers
    // =========================================================================

    bool is_corner_dof(int local_dof) const;
    bool is_edge_dof(int local_dof) const;
    bool is_interior_dof(int local_dof) const;
    int get_edge_for_dof(int local_dof) const;

protected:
    // =========================================================================
    // Base class pure virtual implementations
    // =========================================================================

    std::pair<int64_t, int64_t> quantize_position(const Vec2 &pos) const override;
    size_t num_constraints_impl() const override { return constraints_.size(); }
    void get_constraint_triplets(std::vector<Eigen::Triplet<Real>> &triplets) const override;

private:
    CubicBezierBasis2D basis_;
    std::vector<CubicHangingNodeConstraint> constraints_;
    std::vector<CubicEdgeDerivativeConstraint> edge_derivative_constraints_;

    // DOF assignment (4-pass algorithm)
    void assign_vertex_dofs();
    void assign_edge_dofs();
    void assign_interior_dofs();
    void assign_edge_dofs_nonconforming();

    // Constraint handling
    void build_hanging_node_constraints();

    // Helper
    Vec2 get_dof_position(Index elem, int local_dof) const;
};

} // namespace drifter
