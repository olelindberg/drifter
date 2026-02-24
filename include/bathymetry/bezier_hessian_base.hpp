#pragma once

/// @file bezier_hessian_base.hpp
/// @brief Abstract base class for Bezier surface energy Hessians
///
/// Both DirichletHessian (linear Bezier) and CubicThinPlateHessian (cubic Bezier)
/// implement this interface, enabling the common hessian assembly loop to be
/// shared in CGBezierSmootherBase.

#include "core/types.hpp"

namespace drifter {

/// @brief Abstract base class for Bezier surface energy Hessians
///
/// Provides a common interface for computing energy Hessian matrices used in
/// bathymetry surface regularization. Both DirichletHessian (for linear Bezier)
/// and CubicThinPlateHessian (for cubic Bezier) inherit from this class.
class BezierHessianBase {
public:
    virtual ~BezierHessianBase() = default;

    /// @brief Get number of DOFs (basis-specific)
    /// @return 4 for linear Bezier, 16 for cubic Bezier
    virtual int num_dofs() const = 0;

    /// @brief Compute scaled Hessian for a physical element
    ///
    /// Applies proper derivative scaling for element dimensions.
    ///
    /// @param dx Element width
    /// @param dy Element height
    /// @return Scaled Hessian matrix (NDOF x NDOF)
    virtual MatX scaled_hessian(Real dx, Real dy) const = 0;

    /// @brief Get the precomputed element Hessian matrix for unit element
    /// @return Reference to precomputed Hessian
    virtual const MatX &element_hessian() const = 0;
};

} // namespace drifter
