#pragma once

/// @file bezier_basis_2d_base.hpp
/// @brief Abstract base class for 2D tensor-product Bezier basis functions
///
/// Both LinearBezierBasis2D and CubicBezierBasis2D implement this interface,
/// enabling polymorphic access to common Bezier basis functionality.

#include "core/types.hpp"
#include <vector>

namespace drifter {

/// @brief Abstract base class for 2D tensor-product Bezier basis functions
///
/// Provides a common interface for evaluating Bezier basis functions and their
/// derivatives. Both LinearBezierBasis2D (degree 1) and CubicBezierBasis2D
/// (degree 3) inherit from this class.
class BezierBasis2DBase {
public:
    virtual ~BezierBasis2DBase() = default;

    // =========================================================================
    // Basic properties
    // =========================================================================

    /// @brief Polynomial degree
    virtual int degree() const = 0;

    /// @brief Number of control points in 1D (degree + 1)
    virtual int num_nodes_1d() const = 0;

    /// @brief Total number of DOFs per element
    virtual int num_dofs() const = 0;

    // =========================================================================
    // DOF indexing
    // =========================================================================

    /// @brief Get control point position for DOF index
    /// @param dof DOF index
    /// @return (u, v) in [0, 1]^2
    virtual Vec2 control_point_position(int dof) const = 0;

    // =========================================================================
    // Basis function evaluation (parameter domain [0,1]^2)
    // =========================================================================

    /// @brief Evaluate all basis functions at point (u, v)
    /// @param u First parameter in [0, 1]
    /// @param v Second parameter in [0, 1]
    /// @return Vector of basis function values
    virtual VecX evaluate(Real u, Real v) const = 0;

    /// @brief Evaluate du (partial derivative w.r.t u) of all basis functions
    /// @return Vector of d phi_k / du values
    virtual VecX evaluate_du(Real u, Real v) const = 0;

    /// @brief Evaluate dv (partial derivative w.r.t v) of all basis functions
    /// @return Vector of d phi_k / dv values
    virtual VecX evaluate_dv(Real u, Real v) const = 0;

    /// @brief Evaluate gradient of all basis functions at (u, v)
    /// @return Matrix (NDOF x 2): column 0 = d/du, column 1 = d/dv
    virtual MatX evaluate_gradient(Real u, Real v) const = 0;

    // =========================================================================
    // Scalar interpolation
    // =========================================================================

    /// @brief Evaluate scalar field at (u,v) given control point values
    /// @param coeffs Control point values
    /// @param u, v Parameters in [0, 1]^2
    /// @return Interpolated scalar value
    virtual Real evaluate_scalar(const VecX &coeffs, Real u, Real v) const = 0;

    // =========================================================================
    // Corner and edge access (for constraint building)
    // =========================================================================

    /// @brief Get DOF index for corner
    /// @param corner_id 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1) in parameter space
    virtual int corner_dof(int corner_id) const = 0;

    /// @brief Get corner ID for a DOF index (inverse of corner_dof)
    /// @param dof DOF index
    /// @return Corner ID (0-3), or -1 if not a corner DOF
    virtual int dof_to_corner(int dof) const = 0;

    /// @brief Get DOF indices along an edge
    /// @param edge_id 0: u=0 (left), 1: u=1 (right), 2: v=0 (bottom), 3: v=1 (top)
    virtual std::vector<int> edge_dofs(int edge_id) const = 0;

    /// @brief Get (u, v) parameter values at corner
    /// @param corner_id 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)
    virtual Vec2 corner_param(int corner_id) const = 0;

    // =========================================================================
    // Conversion utilities (static, non-virtual)
    // =========================================================================

    /// @brief Convert from reference [-1,1]^2 to parameter [0,1]^2
    static Vec2 ref_to_param(Real xi, Real eta) {
        return Vec2((xi + 1.0) / 2.0, (eta + 1.0) / 2.0);
    }

    /// @brief Convert from parameter [0,1]^2 to reference [-1,1]^2
    static Vec2 param_to_ref(Real u, Real v) {
        return Vec2(2.0 * u - 1.0, 2.0 * v - 1.0);
    }
};

} // namespace drifter
