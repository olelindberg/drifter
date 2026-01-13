#pragma once

// CGDofManager - Global DOF numbering for CG bathymetry smoothing
//
// Manages Continuous Galerkin (CG) degrees of freedom for the 2D quintic
// element mesh used in bathymetry smoothing. Unlike DG where each element
// has independent DOFs, CG shares DOFs at element interfaces.
//
// DOF assignment follows wobbler's CG2DMesh pattern with 3 passes:
// 1. Vertex DOFs (shared at corners)
// 2. Edge DOFs (shared along edges)
// 3. Interior DOFs (unique per element)
//
// For C² continuity at non-conforming interfaces (hanging nodes), this class
// also manages constraint equations that relate hanging DOFs to their coarse
// neighbors via polynomial interpolation.
//
// Usage:
//   CGDofManager dofs(quadtree, basis);
//   Index n_dofs = dofs.num_global_dofs();
//   const auto& elem_dofs = dofs.element_dofs(elem_idx);

#include "core/types.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/quintic_basis_2d.hpp"
#include <vector>
#include <map>
#include <set>

namespace drifter {

/// @brief Information about a hanging edge at a non-conforming interface
struct HangingEdgeInfo {
    /// Fine (hanging) element index
    Index fine_elem;

    /// Fine element's edge ID (0-3)
    int fine_edge;

    /// Coarse neighbor element index
    Index coarse_elem;

    /// Coarse element's edge ID (opposite edge)
    int coarse_edge;

    /// Sub-edge index (0 or 1) within the coarse edge
    int subedge_index;
};

/// @brief C² constraint at a hanging node
///
/// At non-conforming interfaces, DOFs on the fine side must satisfy
/// C² continuity constraints with the coarse side:
/// - Value continuity: u_fine = interpolated(u_coarse)
/// - Gradient continuity: ∂u/∂n_fine = interpolated(∂u/∂n_coarse)
/// - Hessian continuity: ∂²u/∂n²_fine = interpolated(∂²u/∂n²_coarse)
struct C2Constraint {
    /// Index of the constrained (slave) DOF
    Index slave_dof;

    /// Indices of the master DOFs (on coarse edge)
    std::vector<Index> master_dofs;

    /// Interpolation weights for each master DOF
    std::vector<Real> weights;

    /// Constraint type (for debugging)
    enum class Type {
        Value,      // u constraint
        Gradient,   // ∂u/∂n constraint
        Hessian     // ∂²u/∂n² constraint
    };
    Type type = Type::Value;
};

/// @brief Global DOF numbering with C² constraint handling
///
/// Manages CG degree of freedom numbering and constraints for the
/// biharmonic bathymetry smoothing problem.
class CGDofManager {
public:
    /// @brief Construct DOF manager
    /// @param mesh 2D quadtree mesh
    /// @param basis Quintic basis functions
    CGDofManager(const QuadtreeAdapter& mesh, const QuinticBasis2D& basis);

    // =========================================================================
    // DOF queries
    // =========================================================================

    /// Total number of global DOFs (before constraint elimination)
    Index num_global_dofs() const { return num_global_dofs_; }

    /// Number of free DOFs (after constraint elimination)
    Index num_free_dofs() const { return num_free_dofs_; }

    /// Get global DOF indices for an element (36 DOFs for quintic)
    const std::vector<Index>& element_dofs(Index elem) const;

    /// Get local-to-global DOF mapping for all elements
    const std::vector<std::vector<Index>>& all_element_dofs() const {
        return elem_to_global_;
    }

    /// Check if a DOF is on the domain boundary
    bool is_boundary_dof(Index dof) const;

    /// Get all DOFs on domain boundary
    const std::vector<Index>& boundary_dofs() const { return boundary_dofs_; }

    // =========================================================================
    // Constraint handling
    // =========================================================================

    /// Get all C² constraints
    const std::vector<C2Constraint>& constraints() const { return constraints_; }

    /// Check if a DOF is constrained (slave)
    bool is_constrained(Index dof) const;

    /// @brief Build transformation matrix for constraint elimination
    ///
    /// Returns matrix T such that: u_global = T * u_free
    /// where u_free contains only the unconstrained DOFs.
    ///
    /// @return Sparse transformation matrix (num_global_dofs x num_free_dofs)
    SpMat build_transformation_matrix() const;

    /// @brief Transform stiffness matrix via constraint elimination
    /// @param K Global stiffness matrix (num_global_dofs x num_global_dofs)
    /// @return Reduced matrix K_red = T^T * K * T (num_free_dofs x num_free_dofs)
    SpMat transform_matrix(const SpMat& K) const;

    /// @brief Transform RHS vector via constraint elimination
    /// @param f Global RHS vector (num_global_dofs)
    /// @return Reduced vector f_red = T^T * f (num_free_dofs)
    VecX transform_rhs(const VecX& f) const;

    /// @brief Expand solution from free DOFs to global DOFs
    /// @param u_free Solution in free DOF space (num_free_dofs)
    /// @return Full solution u_global = T * u_free (num_global_dofs)
    VecX expand_solution(const VecX& u_free) const;

    /// @brief Map from global DOF to free DOF index
    /// @return -1 if DOF is constrained, otherwise the free DOF index
    Index global_to_free(Index global_dof) const;

    /// @brief Map from free DOF to global DOF index
    Index free_to_global(Index free_dof) const;

    // =========================================================================
    // Mesh access
    // =========================================================================

    /// Get the mesh
    const QuadtreeAdapter& mesh() const { return mesh_; }

    /// Get the basis
    const QuinticBasis2D& basis() const { return basis_; }

private:
    const QuadtreeAdapter& mesh_;
    const QuinticBasis2D& basis_;

    /// Number of DOFs
    Index num_global_dofs_ = 0;
    Index num_free_dofs_ = 0;

    /// Local to global DOF mapping for each element
    /// elem_to_global_[elem][local_dof] = global_dof
    std::vector<std::vector<Index>> elem_to_global_;

    /// Boundary DOFs
    std::vector<Index> boundary_dofs_;
    std::set<Index> boundary_dof_set_;

    /// C² constraints at hanging nodes
    std::vector<C2Constraint> constraints_;

    /// Constrained DOF set (for fast lookup)
    std::set<Index> constrained_dofs_;

    /// Hanging edges detected at non-conforming interfaces
    std::vector<HangingEdgeInfo> hanging_edges_;

    /// Global to free DOF mapping (-1 if constrained)
    std::vector<Index> global_to_free_;

    /// Free to global DOF mapping
    std::vector<Index> free_to_global_;

    /// Cached transformation matrix
    mutable SpMat transformation_matrix_;
    mutable bool transformation_matrix_built_ = false;

    // =========================================================================
    // DOF assignment methods (3-pass algorithm following wobbler)
    // =========================================================================

    /// Assign DOFs at element vertices (corners)
    void assign_vertex_dofs();

    /// Assign DOFs along element edges
    void assign_edge_dofs();

    /// Assign interior DOFs (unique per element)
    void assign_interior_dofs();

    /// Build boundary DOF list
    void identify_boundary_dofs();

    // =========================================================================
    // Constraint handling methods
    // =========================================================================

    /// Detect hanging nodes at non-conforming interfaces
    void detect_hanging_nodes();

    /// Build C² constraints for hanging nodes
    void build_c2_constraints();

    /// Build the global-to-free and free-to-global mappings
    void build_dof_mappings();

    // =========================================================================
    // Helper methods
    // =========================================================================

    /// Get vertex position in physical coordinates
    Vec2 get_vertex_position(Index elem, int corner_id) const;

    /// Get edge midpoint positions
    std::vector<Vec2> get_edge_dof_positions(Index elem, int edge_id) const;

    /// Find existing DOF at a position (or return -1)
    Index find_dof_at_position(const Vec2& pos, Real tol = 1e-10) const;

    /// Position lookup for DOF sharing
    std::map<std::pair<int64_t, int64_t>, Index> position_to_dof_;  // Quantized position -> DOF

    /// Convert position to quantized key
    std::pair<int64_t, int64_t> quantize_position(const Vec2& pos) const;
};

}  // namespace drifter
