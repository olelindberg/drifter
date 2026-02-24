#pragma once

/// @file adaptive_smoother_types.hpp
/// @brief Common types shared between adaptive CG Bezier smoothers

#include "core/types.hpp"

namespace drifter {

/// @brief Error metric type for adaptive refinement decisions
enum class ErrorMetricType {
    NormalizedError, ///< RMS: ||z_data - z_bezier||_L2 / sqrt(area) [meters]

    // Coarsening error indicators (solution change due to refinement)
    MeanDifference, ///< ∫∫|z_fine - z_coarse|dA / ∫∫dA — mean abs diff [m]
    VolumeChange    ///< ∫∫|z_fine - z_coarse|dA — total volume change [m³]
};

/// @brief Reason for adaptive convergence
enum class ConvergenceReason {
    NotConverged,       ///< Still iterating
    ErrorThreshold,     ///< max_error <= error_threshold
    MaxElements,        ///< num_elements >= max_elements
    MaxRefinementLevel, ///< All marked elements at max level
    MaxIterations       ///< Reached max_iterations
};

} // namespace drifter
