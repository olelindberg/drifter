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

#include "bathymetry/cg_bezier_dof_manager_base.hpp"
#include "bathymetry/linear_bezier_basis_2d.hpp"
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
class CGLinearBezierDofManager : public CGBezierDofManagerBase {
public:
    /// @brief Construct DOF manager
    /// @param mesh 2D quadtree mesh
    explicit CGLinearBezierDofManager(const QuadtreeAdapter &mesh);

    // =========================================================================
    // Base class overrides
    // =========================================================================

    int num_element_dofs() const override { return LinearBezierBasis2D::NDOF; }

    // =========================================================================
    // Linear-specific accessors
    // =========================================================================

    const std::vector<LinearHangingNodeConstraint> &constraints() const { return constraints_; }
    const LinearBezierBasis2D &basis() const { return basis_; }

    // =========================================================================
    // Quantization parameters (for consistent vertex deduplication in VTK)
    // =========================================================================

    Real xmin_domain() const { return xmin_domain_; }
    Real ymin_domain() const { return ymin_domain_; }
    Real inv_quantization_tol() const { return inv_quantization_tol_; }

protected:
    // =========================================================================
    // Base class pure virtual implementations
    // =========================================================================

    std::pair<int64_t, int64_t> quantize_position(const Vec2 &pos) const override;
    size_t num_constraints_impl() const override { return constraints_.size(); }
    void get_constraint_triplets(std::vector<Eigen::Triplet<Real>> &triplets) const override;

private:
    LinearBezierBasis2D basis_;
    std::vector<LinearHangingNodeConstraint> constraints_;

    // Quantization parameters (mesh-relative)
    Real xmin_domain_ = 0.0;
    Real ymin_domain_ = 0.0;
    Real inv_quantization_tol_ = 1e8;

    // DOF assignment (single pass - all corners)
    void assign_vertex_dofs();

    // Constraint handling
    void build_hanging_node_constraints();

    // Helper
    Vec2 get_dof_position(Index elem, int local_dof) const;
};

} // namespace drifter
