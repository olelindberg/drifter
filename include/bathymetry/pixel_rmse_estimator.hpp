#pragma once

/// @file pixel_rmse_estimator.hpp
/// @brief Pixel-based RMSE error estimation for adaptive refinement
///
/// This implements the ElementErrorEstimator interface using pixel-based RMSE.
/// It wraps the PixelErrorEstimator template for LinearBezierSurface.

#include "bathymetry/element_error_estimator.hpp"
#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/pixel_error_estimator.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"

namespace drifter {

/// @brief Error estimator using pixel-based RMSE (implements ElementErrorEstimator)
///
/// Wraps PixelErrorEstimator<LinearBezierSurface> to provide a unified
/// ElementErrorEstimator interface for adaptive refinement.
class PixelRMSEEstimator : public ElementErrorEstimator {
public:
    /// @brief Construct pixel RMSE error estimator
    /// @param surface Fitted surface to evaluate
    /// @param data Bathymetry data from GeoTIFF
    /// @param mesh Quadtree mesh
    PixelRMSEEstimator(const LinearBezierSurface& surface,
                       const BathymetryData& data,
                       const QuadtreeAdapter& mesh)
        : impl_(surface, data, mesh), mesh_(mesh) {}

    /// @brief Estimate error for a single element
    ElementError estimate_element(Index elem) const override {
        auto pixel_err = impl_.estimate_element(elem);
        ElementError result;
        result.element = pixel_err.element;
        result.error = pixel_err.pixel_rmse;
        result.area = pixel_err.area;
        result.sample_count = pixel_err.pixel_count;
        return result;
    }

    /// @brief Get number of elements
    Index num_elements() const override {
        return mesh_.num_elements();
    }

    /// @brief Get global RMSE across entire domain
    /// @return sqrt(sum of squared errors / total pixel count)
    Real global_rmse() const {
        return impl_.global_rmse();
    }

    /// @brief Get minimum recommended element size based on GeoTIFF resolution
    /// @return Pixel spacing (smaller of dx, dy)
    Real min_recommended_element_size() const {
        return impl_.min_recommended_element_size();
    }

    /// @brief Get number of elements with fewer than min_pixels
    /// @param min_pixels Minimum pixel threshold (default 4)
    /// @return Count of under-sampled elements
    Index count_undersampled_elements(int min_pixels = 4) const {
        return impl_.count_undersampled_elements(min_pixels);
    }

    /// @brief Get total number of pixels processed
    /// @return Sum of pixel_count across all elements
    Index total_pixel_count() const {
        return impl_.total_pixel_count();
    }

private:
    PixelErrorEstimator<LinearBezierSurface> impl_;
    const QuadtreeAdapter& mesh_;
};

} // namespace drifter
