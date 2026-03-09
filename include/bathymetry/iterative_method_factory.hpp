#pragma once

/// @file iterative_method_factory.hpp
/// @brief Factory for creating iterative methods from SmootherType enum

#include "bathymetry/iterative_method.hpp"
#include "bathymetry/smoother_types.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace drifter {

/// @brief Factory for creating iterative methods
///
/// Creates the appropriate iterative method implementation based on
/// SmootherType enum. This provides a convenient way to instantiate
/// methods when only the enum type is known.
class IterativeMethodFactory {
public:
    /// @brief Create iterative method from SmootherType enum
    /// @param type Type of smoother/method to create
    /// @param Q System matrix (reference, must outlive the method)
    /// @param element_free_dofs Per-element free DOF indices (for Schwarz)
    /// @param element_block_lu Per-element block LU factorizations (for Schwarz)
    /// @param elements_by_color Elements grouped by color (for colored Schwarz)
    /// @param omega Damping parameter (for Jacobi and additive Schwarz)
    /// @return Unique pointer to the created method
    static std::unique_ptr<IIterativeMethod>
    create(SmootherType type, const SpMat &Q,
           const std::vector<std::vector<Index>> &element_free_dofs = {},
           const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu = {},
           const std::vector<std::vector<Index>> &elements_by_color = {},
           Real omega = 0.8);
};

} // namespace drifter
