#pragma once

/// @file pixel_error_estimator.hpp
/// @brief Pixel-based error estimation for Bezier surface fitting
///
/// Computes RMSE between GeoTIFF pixel values and fitted surface at pixel
/// center locations. Unlike Gauss quadrature which samples at arbitrary
/// points using bilinear interpolation, this measures error at actual data
/// locations.

#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include "mesh/geotiff_reader.hpp"
#include <cmath>
#include <vector>

namespace drifter {

/// @brief Per-element pixel-based error estimate
struct PixelElementError {
    Index element;           ///< Element index
    Real pixel_rmse;         ///< RMSE at pixel centers within element [meters]
    Real max_pixel_error;    ///< Maximum |z_data - z_surface| across pixels [meters]
    Real sum_squared_error;  ///< Sum of squared errors (for combining elements)
    int pixel_count;         ///< Number of pixels in element
    Real area;               ///< Element area for reference [m²]
};

/// @brief Pixel-based error estimator for Bezier surfaces
///
/// Computes RMSE between GeoTIFF pixel values and fitted surface at pixel
/// center locations. This provides a direct measure of how well the surface
/// fits the original measurement data.
///
/// @tparam SurfaceType Type with evaluate(Real x, Real y) method
///         (LinearBezierSurface, CGLinearBezierBathymetrySmoother,
///          CGCubicBezierBathymetrySmoother, or their adaptive variants)
template <typename SurfaceType>
class PixelErrorEstimator {
public:
    /// @brief Construct pixel error estimator
    /// @param surface Fitted surface to evaluate
    /// @param data Bathymetry data from GeoTIFF
    /// @param mesh Quadtree mesh for element bounds
    PixelErrorEstimator(const SurfaceType& surface,
                        const BathymetryData& data,
                        const QuadtreeAdapter& mesh);

    /// @brief Estimate error for all elements
    /// @return Vector of per-element pixel error estimates
    std::vector<PixelElementError> estimate_all() const;

    /// @brief Estimate error for single element
    /// @param elem Element index
    /// @return Pixel error estimate for this element
    PixelElementError estimate_element(Index elem) const;

    /// @brief Get global RMSE across entire domain
    /// @return sqrt(sum of squared errors / total pixel count)
    Real global_rmse() const;

    /// @brief Get maximum element RMSE
    /// @return Maximum pixel_rmse across all elements
    Real max_element_rmse() const;

    /// @brief Get mean element RMSE (weighted by pixel count)
    /// @return Weighted mean pixel_rmse across all elements
    Real mean_element_rmse() const;

    /// @brief Get minimum recommended element size based on GeoTIFF resolution
    /// @return Pixel spacing (smaller of dx, dy)
    Real min_recommended_element_size() const;

    /// @brief Get number of elements with fewer than min_pixels
    /// @param min_pixels Minimum pixel threshold (default 4)
    /// @return Count of under-sampled elements
    Index count_undersampled_elements(int min_pixels = 4) const;

    /// @brief Get total number of pixels processed
    /// @return Sum of pixel_count across all elements
    Index total_pixel_count() const;

private:
    const SurfaceType& surface_;
    const BathymetryData& data_;
    const QuadtreeAdapter& mesh_;

    // Cached pixel geometry
    Real pixel_dx_;  ///< Pixel width in world units
    Real pixel_dy_;  ///< Pixel height in world units

    /// @brief Get pixel range that overlaps element bounds
    /// @param bounds Element bounds
    /// @param px_min, px_max, py_min, py_max Output pixel range (inclusive)
    void get_pixel_range(const QuadBounds& bounds,
                         int& px_min, int& px_max,
                         int& py_min, int& py_max) const;
};

// =============================================================================
// Template Implementation
// =============================================================================

template <typename SurfaceType>
PixelErrorEstimator<SurfaceType>::PixelErrorEstimator(
    const SurfaceType& surface,
    const BathymetryData& data,
    const QuadtreeAdapter& mesh)
    : surface_(surface), data_(data), mesh_(mesh) {
    pixel_dx_ = data_.pixel_size_x();
    pixel_dy_ = data_.pixel_size_y();
}

template <typename SurfaceType>
void PixelErrorEstimator<SurfaceType>::get_pixel_range(
    const QuadBounds& bounds,
    int& px_min, int& px_max,
    int& py_min, int& py_max) const {

    // Convert element corners to pixel coordinates
    double px0, py0, px1, py1;
    data_.world_to_pixel(bounds.xmin, bounds.ymin, px0, py0);
    data_.world_to_pixel(bounds.xmax, bounds.ymax, px1, py1);

    // Handle potential axis inversions
    if (px0 > px1) std::swap(px0, px1);
    if (py0 > py1) std::swap(py0, py1);

    // Get integer pixel range (floor/ceil to include all overlapping pixels)
    px_min = std::max(0, static_cast<int>(std::floor(px0)));
    px_max = std::min(data_.sizex - 1, static_cast<int>(std::ceil(px1)));
    py_min = std::max(0, static_cast<int>(std::floor(py0)));
    py_max = std::min(data_.sizey - 1, static_cast<int>(std::ceil(py1)));
}

template <typename SurfaceType>
PixelElementError PixelErrorEstimator<SurfaceType>::estimate_element(Index elem) const {
    PixelElementError result;
    result.element = elem;
    result.sum_squared_error = 0.0;
    result.max_pixel_error = 0.0;
    result.pixel_count = 0;

    const auto& bounds = mesh_.element_bounds(elem);
    result.area = (bounds.xmax - bounds.xmin) * (bounds.ymax - bounds.ymin);

    // Get pixel range overlapping this element
    int px_min, px_max, py_min, py_max;
    get_pixel_range(bounds, px_min, px_max, py_min, py_max);

    // Iterate over pixels in row-major order (cache-friendly for GeoTIFF storage)
    for (int py = py_min; py <= py_max; ++py) {
        for (int px = px_min; px <= px_max; ++px) {
            // Get pixel value (skip NoData)
            float z_pixel = data_.at_pixel(px, py);
            if (std::abs(z_pixel - data_.nodata_value) < 1e-6f || z_pixel > 1e30f) {
                continue;
            }

            // Get pixel center in world coordinates
            double wx, wy;
            data_.pixel_center_to_world(px, py, wx, wy);

            // Check if pixel center is inside element bounds
            if (wx < bounds.xmin || wx > bounds.xmax ||
                wy < bounds.ymin || wy > bounds.ymax) {
                continue;
            }

            // Evaluate surface at pixel center
            Real z_surface = surface_.evaluate(static_cast<Real>(wx), static_cast<Real>(wy));

            // Skip if surface evaluation returned NaN (can happen at boundary elements
            // where DOF coefficients weren't properly initialized)
            if (std::isnan(z_surface)) {
                continue;
            }

            // Convert pixel value to surface convention
            // GeoTIFF: depth positive means values are depth (positive = water)
            // Surface: stores elevation (negative = below sea level)
            Real z_data;
            if (data_.is_depth_positive) {
                z_data = -static_cast<Real>(z_pixel);  // depth -> elevation
            } else {
                z_data = static_cast<Real>(z_pixel);   // already elevation
            }

            // Compute error
            Real error = z_data - z_surface;
            result.sum_squared_error += error * error;
            result.max_pixel_error = std::max(result.max_pixel_error, std::abs(error));
            result.pixel_count++;
        }
    }

    // Compute RMSE
    if (result.pixel_count > 0) {
        result.pixel_rmse = std::sqrt(result.sum_squared_error / result.pixel_count);
    } else {
        result.pixel_rmse = 0.0;
    }

    return result;
}

template <typename SurfaceType>
std::vector<PixelElementError> PixelErrorEstimator<SurfaceType>::estimate_all() const {
    std::vector<PixelElementError> errors;
    errors.reserve(mesh_.num_elements());

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        errors.push_back(estimate_element(elem));
    }

    return errors;
}

template <typename SurfaceType>
Real PixelErrorEstimator<SurfaceType>::global_rmse() const {
    Real total_sum_sq = 0.0;
    Index total_count = 0;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        auto err = estimate_element(elem);
        // Skip elements with NaN sum_squared_error
        if (!std::isnan(err.sum_squared_error) && err.pixel_count > 0) {
            total_sum_sq += err.sum_squared_error;
            total_count += err.pixel_count;
        }
    }

    if (total_count > 0) {
        return std::sqrt(total_sum_sq / total_count);
    }
    return 0.0;
}

template <typename SurfaceType>
Real PixelErrorEstimator<SurfaceType>::max_element_rmse() const {
    Real max_rmse = 0.0;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        auto err = estimate_element(elem);
        if (err.pixel_count > 0) {
            max_rmse = std::max(max_rmse, err.pixel_rmse);
        }
    }

    return max_rmse;
}

template <typename SurfaceType>
Real PixelErrorEstimator<SurfaceType>::mean_element_rmse() const {
    Real weighted_sum = 0.0;
    Index total_count = 0;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        auto err = estimate_element(elem);
        if (err.pixel_count > 0) {
            weighted_sum += err.pixel_rmse * err.pixel_count;
            total_count += err.pixel_count;
        }
    }

    if (total_count > 0) {
        return weighted_sum / total_count;
    }
    return 0.0;
}

template <typename SurfaceType>
Real PixelErrorEstimator<SurfaceType>::min_recommended_element_size() const {
    return data_.min_element_size();
}

template <typename SurfaceType>
Index PixelErrorEstimator<SurfaceType>::count_undersampled_elements(int min_pixels) const {
    Index count = 0;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        auto err = estimate_element(elem);
        if (err.pixel_count < min_pixels) {
            count++;
        }
    }

    return count;
}

template <typename SurfaceType>
Index PixelErrorEstimator<SurfaceType>::total_pixel_count() const {
    Index total = 0;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        auto err = estimate_element(elem);
        total += err.pixel_count;
    }

    return total;
}

} // namespace drifter
