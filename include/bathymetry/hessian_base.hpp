#pragma once

/// @file hessian_base.hpp
/// @brief Abstract base class for surface energy Hessians
///
/// DirichletHessian (linear Bezier), CubicThinPlateHessian (cubic Bezier) and
/// HermiteHessian (Hermite) implement this interface, enabling the common hessian
/// assembly loop to be shared in CGSmootherBase. The interface is basis-agnostic:
/// it depends only on num_dofs() and scaled_hessian(dx, dy).

#include "core/types.hpp"

namespace drifter {

/// @brief Abstract base class for surface energy Hessians
///
/// Provides a common interface for computing energy Hessian matrices used in
/// bathymetry surface regularization.
class HessianBase {
public:
    virtual ~HessianBase() = default;

    /// @brief Get number of DOFs (basis-specific)
    /// @return 4 for linear Bezier / C0 Hermite, 16 for cubic Bezier / C1 Hermite
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
