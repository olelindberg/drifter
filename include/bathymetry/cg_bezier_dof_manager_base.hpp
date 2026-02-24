#pragma once

/// @file cg_bezier_dof_manager_base.hpp
/// @brief Abstract base class for CG Bezier DOF managers
///
/// Provides common functionality shared between CGLinearBezierDofManager
/// and CGCubicBezierDofManager. Uses standard inheritance with virtual
/// methods for customization points.

#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace drifter {

/// @brief Abstract base class for CG Bezier DOF managers
///
/// Manages global DOF numbering with sharing at element interfaces.
/// Derived classes implement basis-specific DOF assignment and constraints.
///
/// Common functionality:
/// - DOF queries (global_dof, element_dofs, is_boundary_dof)
/// - Constraint matrix building
/// - DOF mapping (global <-> free)
/// - Position-based DOF lookup
///
/// Derived classes must implement:
/// - num_element_dofs() - returns NDOF for the basis
/// - quantize_position() - position hashing strategy
/// - Constraint accessors (basis-specific constraint types)
/// - DOF assignment methods (basis-specific algorithm)
class CGBezierDofManagerBase {
public:
    virtual ~CGBezierDofManagerBase() = default;

    // =========================================================================
    // DOF queries - implemented in base
    // =========================================================================

    /// @brief Total number of global DOFs
    Index num_global_dofs() const { return num_global_dofs_; }

    /// @brief Number of free (unconstrained) DOFs
    Index num_free_dofs() const { return num_free_dofs_; }

    /// @brief Number of DOFs per element (basis-specific)
    virtual int num_element_dofs() const = 0;

    /// @brief Get global DOF index for element local DOF
    /// @param elem Element index
    /// @param local_dof Local DOF index (0 to num_element_dofs()-1)
    /// @throws std::out_of_range if indices invalid
    Index global_dof(Index elem, int local_dof) const;

    /// @brief Get all global DOF indices for an element
    /// @param elem Element index
    /// @throws std::out_of_range if elem invalid
    const std::vector<Index> &element_dofs(Index elem) const;

    /// @brief Get element-to-global DOF mapping for all elements
    const std::vector<std::vector<Index>> &all_element_dofs() const { return elem_to_global_; }

    /// @brief Check if DOF is on domain boundary
    bool is_boundary_dof(Index dof) const;

    /// @brief Get sorted list of boundary DOF indices
    const std::vector<Index> &boundary_dofs() const { return boundary_dofs_; }

    // =========================================================================
    // Constraint handling - implemented in base
    // =========================================================================

    /// @brief Number of hanging node constraints
    Index num_constraints() const { return static_cast<Index>(num_constraints_impl()); }

    /// @brief Check if DOF is constrained (slave of a hanging node constraint)
    bool is_constrained(Index dof) const;

    /// @brief Build sparse constraint matrix A where Ax = 0 encodes constraints
    /// Each row: slave_coeff * x_slave - sum(master_coeff * x_master) = 0
    /// @return Sparse matrix of size (num_constraints x num_global_dofs)
    SpMat build_constraint_matrix() const;

    // =========================================================================
    // DOF mappings - implemented in base
    // =========================================================================

    /// @brief Map global DOF to free DOF index
    /// @return Free DOF index, or -1 if constrained
    Index global_to_free(Index global_dof) const;

    /// @brief Map free DOF to global DOF index
    /// @return Global DOF index, or -1 if invalid
    Index free_to_global(Index free_dof) const;

    /// @brief Get global-to-free mapping vector
    const std::vector<Index> &global_to_free_map() const { return global_to_free_; }

    /// @brief Get free-to-global mapping vector
    const std::vector<Index> &free_to_global_map() const { return free_to_global_; }

    // =========================================================================
    // Mesh access - implemented in base
    // =========================================================================

    /// @brief Get underlying quadtree mesh
    const QuadtreeAdapter &mesh() const { return mesh_; }

protected:
    // =========================================================================
    // Constructor - protected, only derived classes can construct
    // =========================================================================

    /// @brief Construct DOF manager base
    /// @param mesh 2D quadtree mesh
    explicit CGBezierDofManagerBase(const QuadtreeAdapter &mesh);

    // =========================================================================
    // Shared state
    // =========================================================================

    const QuadtreeAdapter &mesh_;

    Index num_global_dofs_ = 0;
    Index num_free_dofs_ = 0;

    std::vector<std::vector<Index>> elem_to_global_;
    std::vector<Index> boundary_dofs_;
    std::set<Index> boundary_dof_set_;
    std::set<Index> constrained_dofs_;
    std::vector<Index> global_to_free_;
    std::vector<Index> free_to_global_;

    /// Position to DOF map for CG DOF sharing
    std::map<std::pair<int64_t, int64_t>, Index> position_to_dof_;

    // =========================================================================
    // Pure virtual methods - must be implemented by derived classes
    // =========================================================================

    /// @brief Quantize position to integer grid for DOF deduplication
    /// Linear uses mesh-relative dynamic scale, Cubic uses global fixed scale
    virtual std::pair<int64_t, int64_t> quantize_position(const Vec2 &pos) const = 0;

    /// @brief Get number of constraints (implementation detail)
    virtual size_t num_constraints_impl() const = 0;

    /// @brief Get constraint triplets for building constraint matrix
    /// @param triplets Output vector of (row, col, value) triplets
    virtual void get_constraint_triplets(std::vector<Eigen::Triplet<Real>> &triplets) const = 0;

    // =========================================================================
    // Helper methods - implemented in base, callable by derived
    // =========================================================================

    /// @brief Find existing DOF at position
    /// @return DOF index, or -1 if not found
    Index find_dof_at_position(const Vec2 &pos) const;

    /// @brief Register a new DOF at position
    /// @return The new DOF index
    Index register_dof_at_position(const Vec2 &pos);

    /// @brief Identify boundary DOFs using edge iteration
    /// @param get_edge_dofs Function that returns local DOF indices for an edge
    void identify_boundary_dofs_impl(const std::function<std::vector<int>(int)> &get_edge_dofs);

    /// @brief Build global-to-free and free-to-global mappings
    /// Call after constraints are built
    void build_dof_mappings();

    /// @brief Initialize elem_to_global_ with -1 values
    /// @param num_elements Number of elements
    /// @param ndof Number of DOFs per element
    void initialize_elem_to_global(Index num_elements, int ndof);
};

} // namespace drifter
