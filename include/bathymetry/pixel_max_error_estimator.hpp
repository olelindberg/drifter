#pragma once

/// @file pixel_max_error_estimator.hpp
/// @brief Pixel-based maximum error estimation for adaptive refinement
///
/// This implements the ElementErrorEstimator interface using pixel-based max error.
/// It wraps the PixelErrorEstimator template for LinearBezierSurface.

#include "bathymetry/element_error_estimator.hpp"
#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/pixel_error_estimator.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"

namespace drifter {

/// @brief Error estimator using pixel-based max error (implements ElementErrorEstimator)
///
/// Wraps PixelErrorEstimator<LinearBezierSurface> to provide a unified
/// ElementErrorEstimator interface for adaptive refinement. Reports the maximum
/// absolute error across all pixels within each element.
class PixelMaxErrorEstimator : public ElementErrorEstimator {
  public:
    /// @brief Construct pixel max error estimator
    /// @param surface Fitted surface to evaluate
    /// @param data Bathymetry data from GeoTIFF
    /// @param mesh Quadtree mesh
  PixelMaxErrorEstimator(const LinearBezierSurface &surface, const BathymetryData &data, const QuadtreeAdapter &mesh) : impl_(surface, data, mesh), mesh_(mesh) {}

    /// @brief Estimate error for a single element
    /// @return ElementError with error = max |z_data - z_surface| across pixels
  ElementError estimate_element(Index elem) const override {
    auto pixel_err = impl_.estimate_element(elem);
    ElementError result;
    result.element      = pixel_err.element;
    result.error        = pixel_err.max_pixel_error;
    result.area         = pixel_err.area;
    result.sample_count = pixel_err.pixel_count;
    return result;
  }

    /// @brief Get number of elements
  Index num_elements() const override { return mesh_.num_elements(); }

    /// @brief Get global maximum error across entire domain
    /// @return Maximum |z_data - z_surface| across all pixels
  Real global_max_error() const {
    Real max_err = 0.0;
    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
      auto pixel_err = impl_.estimate_element(elem);
      if (pixel_err.pixel_count > 0) {
        max_err = std::max(max_err, pixel_err.max_pixel_error);
      }
    }
    return max_err;
  }

    /// @brief Get minimum recommended element size based on GeoTIFF resolution
    /// @return Pixel spacing (smaller of dx, dy)
  Real min_recommended_element_size() const { return impl_.min_recommended_element_size(); }

    /// @brief Get number of elements with fewer than min_pixels
    /// @param min_pixels Minimum pixel threshold (default 4)
    /// @return Count of under-sampled elements
  Index count_undersampled_elements(int min_pixels = 4) const { return impl_.count_undersampled_elements(min_pixels); }

    /// @brief Get total number of pixels processed
    /// @return Sum of pixel_count across all elements
  Index total_pixel_count() const { return impl_.total_pixel_count(); }

  private:
  PixelErrorEstimator<LinearBezierSurface> impl_;
  const QuadtreeAdapter &mesh_;
};

} // namespace drifter
