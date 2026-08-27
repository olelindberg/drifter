#pragma once

/// @file element_error_estimator.hpp
/// @brief Interface for element-wise error estimation in adaptive mesh refinement

#include "bathymetry/adaptive_smoother_types.hpp"
#include "core/types.hpp"
#include <memory>
#include <vector>

namespace drifter {

// Forward declarations
class LinearBezierSurface;
class QuadtreeAdapter;
struct BathymetryData;

/// @brief Per-element error estimate (unified for all metrics)
struct ElementError {
    Index element;      ///< Element index
    Real error;         ///< Primary error metric value [meters]
    Real area;          ///< Element area [m²]
    int sample_count;   ///< Number of samples (pixels or Gauss points)
};

/// @brief Abstract interface for element error estimation
///
/// Provides a unified interface for different error estimation strategies:
/// - Gauss quadrature-based (L2, normalized, volume)
/// - Pixel-based (RMSE at GeoTIFF pixel centers)
class ElementErrorEstimator {
public:
    virtual ~ElementErrorEstimator() = default;

    /// @brief Estimate error for a single element
    /// @param elem Element index
    /// @return Error estimate for this element
    virtual ElementError estimate_element(Index elem) const = 0;

    /// @brief Estimate error for all elements
    /// @return Vector of per-element error estimates
    virtual std::vector<ElementError> estimate_all() const;

    /// @brief Get maximum error across all elements
    /// @return Maximum error value
    virtual Real max_error() const;

    /// @brief Get mean error across all elements
    /// @return Mean error value (skips NaN values)
    virtual Real mean_error() const;

    /// @brief Get number of elements
    virtual Index num_elements() const = 0;
};

/// @brief Factory function to create appropriate error estimator
/// @param metric Error metric type
/// @param surface Fitted surface to evaluate
/// @param data Bathymetry data from GeoTIFF
/// @param mesh Quadtree mesh
/// @param ngauss Number of Gauss points (for Gauss quadrature metrics)
/// @return Unique pointer to error estimator
std::unique_ptr<ElementErrorEstimator> create_error_estimator(
    ErrorMetricType metric,
    const LinearBezierSurface& surface,
    const BathymetryData& data,
    const QuadtreeAdapter& mesh,
    int ngauss = 4);

} // namespace drifter
