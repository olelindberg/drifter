#pragma once

/// @file multi_source_pixel_error_estimator.hpp
/// @brief Multi-source pixel-based error estimation for adaptive refinement
///
/// Extends pixel-based error estimation to handle multiple bathymetry sources.
/// For elements in the primary source, uses exact pixel-based error estimation.
/// For elements in tile regions, samples at a grid resolution matching the
/// tile's pixel spacing and uses the multi-source depth function.

#include "bathymetry/element_error_estimator.hpp"
#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/pixel_error_estimator.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <cmath>
#include <functional>

namespace drifter {

/// @brief Error estimator using pixel-based max error with multi-source support
///
/// For elements inside the primary source bounds, uses standard pixel-based
/// error estimation with the primary's pixel grid.
/// For elements outside (in tile regions), samples at a grid matching the
/// tile's pixel resolution and compares depth_func vs surface.
class MultiSourcePixelMaxErrorEstimator : public ElementErrorEstimator {
public:
    /// @brief Construct multi-source pixel max error estimator
    /// @param surface Fitted surface to evaluate
    /// @param primary Primary bathymetry data (for pixel-based estimation)
    /// @param multi_bathy Multi-source bathymetry (for depth evaluation)
    /// @param mesh Quadtree mesh
    MultiSourcePixelMaxErrorEstimator(const LinearBezierSurface& surface,
                                      const BathymetryData& primary,
                                      const MultiSourceBathymetry& multi_bathy,
                                      const QuadtreeAdapter& mesh)
        : surface_(surface),
          primary_(primary),
          multi_bathy_(multi_bathy),
          mesh_(mesh),
          primary_estimator_(surface, primary, mesh) {}

    /// @brief Estimate error for a single element
    /// @return ElementError with error = max |z_data - z_surface| across samples
    ElementError estimate_element(Index elem) const override {
        const auto& bounds = mesh_.element_bounds(elem);
        Real cx = (bounds.xmin + bounds.xmax) / 2.0;
        Real cy = (bounds.ymin + bounds.ymax) / 2.0;

        // If element is in primary region, use standard pixel-based estimation
        if (multi_bathy_.is_in_primary(cx, cy)) {
            auto pixel_err = primary_estimator_.estimate_element(elem);
            ElementError result;
            result.element = pixel_err.element;
            result.error = pixel_err.max_pixel_error;
            result.area = pixel_err.area;
            result.sample_count = pixel_err.pixel_count;
            return result;
        }

        // Element is in a tile region - use grid sampling
        return estimate_element_grid(elem, bounds);
    }

    /// @brief Get number of elements
    Index num_elements() const override { return mesh_.num_elements(); }

private:
    const LinearBezierSurface& surface_;
    const BathymetryData& primary_;
    const MultiSourceBathymetry& multi_bathy_;
    const QuadtreeAdapter& mesh_;
    PixelErrorEstimator<LinearBezierSurface> primary_estimator_;

    /// @brief Estimate error for an element using grid sampling
    /// @param elem Element index
    /// @param bounds Element bounds
    /// @return Error estimate based on grid sampling
    ElementError estimate_element_grid(Index elem, const QuadBounds& bounds) const {
        ElementError result;
        result.element = elem;
        result.area = (bounds.xmax - bounds.xmin) * (bounds.ymax - bounds.ymin);
        result.error = 0.0;
        result.sample_count = 0;

        // Get element center
        Real cx = (bounds.xmin + bounds.xmax) / 2.0;
        Real cy = (bounds.ymin + bounds.ymax) / 2.0;

        // Get the source that covers this element
        const BathymetryData* source = multi_bathy_.get_source_for_point(cx, cy);
        if (!source) {
            return result;
        }

        // Use the source's pixel size as sampling resolution
        Real dx = source->pixel_size_x();
        Real dy = source->pixel_size_y();

        // Number of samples in each direction (at least 2x2)
        int nx = std::max(2, static_cast<int>(std::ceil((bounds.xmax - bounds.xmin) / dx)));
        int ny = std::max(2, static_cast<int>(std::ceil((bounds.ymax - bounds.ymin) / dy)));

        // Cap number of samples to avoid excessive computation
        nx = std::min(nx, 20);
        ny = std::min(ny, 20);

        Real max_error = 0.0;
        int sample_count = 0;

        // Sample on a regular grid
        for (int iy = 0; iy <= ny; ++iy) {
            for (int ix = 0; ix <= nx; ++ix) {
                Real x = bounds.xmin + (bounds.xmax - bounds.xmin) * ix / nx;
                Real y = bounds.ymin + (bounds.ymax - bounds.ymin) * iy / ny;

                // Evaluate depth from multi-source bathymetry
                Real z_data;
                try {
                    Real depth = multi_bathy_.evaluate(x, y);
                    // Convert depth (positive downward) to elevation (negative below sea level)
                    z_data = -depth;
                } catch (const std::out_of_range&) {
                    continue;  // Point outside all sources
                }

                // Skip land (depth = 0)
                if (z_data >= 0.0) {
                    continue;
                }

                // Evaluate surface
                Real z_surface = surface_.evaluate(x, y);
                if (std::isnan(z_surface)) {
                    continue;
                }

                // Compute error
                Real error = std::abs(z_data - z_surface);
                max_error = std::max(max_error, error);
                ++sample_count;
            }
        }

        result.error = max_error;
        result.sample_count = sample_count;
        return result;
    }
};

} // namespace drifter
