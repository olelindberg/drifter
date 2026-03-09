#pragma once

/// @file smoother_types.hpp
/// @brief Smoother type enum for iterative methods and multigrid

namespace drifter {

/// @brief Smoother type for multigrid and standalone solvers
enum class SmootherType {
    Jacobi,
    MultiplicativeSchwarz,
    AdditiveSchwarz,
    ColoredMultiplicativeSchwarz
};

} // namespace drifter
