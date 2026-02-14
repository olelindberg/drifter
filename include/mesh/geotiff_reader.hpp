#pragma once

// GeoTIFF Bathymetry Reader for DRIFTER
//
// Reads bathymetry data from GeoTIFF files using GDAL.
// Adapted from SeaMesh's GeoTiffReader.
//
// Features:
// - Loads elevation/bathymetry data from GeoTIFF
// - Extracts geotransform (affine coordinates)
// - Supports bilinear interpolation of bathymetry values
// - Handles NoData values (land masking)

#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gdal_priv.h>

namespace drifter {

/// @brief Lightweight metadata for a GeoTIFF file (bounds only, no raster data)
///
/// Used for on-demand tile loading - read bounds at startup, load full data
/// later.
struct BathymetryBounds {
    Real xmin = 0, xmax = 0;
    Real ymin = 0, ymax = 0;
    float nodata_value = -9999.0f;
    bool is_depth_positive = false;

    /// Check if bounds are valid
    bool is_valid() const { return xmin < xmax && ymin < ymax; }

    /// Check if a point is inside the bounds
    bool contains(double x, double y) const {
        return x >= xmin && x <= xmax && y >= ymin && y <= ymax;
    }
};

/// @brief Bathymetry data loaded from GeoTIFF
struct BathymetryData {
    /// Elevation values (row-major: data[y * sizex + x])
    /// If is_depth_positive=false: Positive = above sea level, Negative = water
    /// If is_depth_positive=true: Values are depth (positive = water depth)
    std::vector<float> elevation;

    /// Raster dimensions
    int sizex = 0;
    int sizey = 0;

    /// Geotransform (affine transformation from pixel to world coordinates)
    /// [0] = top-left X, [1] = pixel width, [2] = row rotation
    /// [3] = top-left Y, [4] = column rotation, [5] = pixel height (usually
    /// negative)
    std::array<double, 6> geotransform;

    /// WKT projection string
    std::string projection;

    /// NoData value for missing/invalid data
    float nodata_value = -9999.0f;

    /// If true, values represent depth (positive = below sea level)
    /// If false, values represent elevation (negative = below sea level)
    bool is_depth_positive = false;

    /// Bounding box in world coordinates
    Real xmin, xmax, ymin, ymax;

    /// Check if data is valid
    bool is_valid() const {
        return sizex > 0 && sizey > 0 && !elevation.empty();
    }

    /// Get elevation at pixel coordinates
    float at_pixel(int x, int y) const {
        if (x < 0 || x >= sizex || y < 0 || y >= sizey)
            return nodata_value;
        return elevation[y * sizex + x];
    }

    /// Convert pixel coordinates to world coordinates
    void pixel_to_world(int px, int py, double &wx, double &wy) const {
        wx = geotransform[0] + px * geotransform[1] + py * geotransform[2];
        wy = geotransform[3] + px * geotransform[4] + py * geotransform[5];
    }

    /// Convert world coordinates to pixel coordinates (fractional)
    void world_to_pixel(double wx, double wy, double &px, double &py) const {
        // Inverse of the geotransform
        double det = geotransform[1] * geotransform[5] -
                     geotransform[2] * geotransform[4];
        if (std::abs(det) < 1e-15) {
            px = py = 0;
            return;
        }
        double dx = wx - geotransform[0];
        double dy = wy - geotransform[3];
        px = (geotransform[5] * dx - geotransform[2] * dy) / det;
        py = (-geotransform[4] * dx + geotransform[1] * dy) / det;
    }

    /// Bilinear interpolation at world coordinates
    /// Returns nodata_value if outside bounds or on land
    float interpolate(double wx, double wy) const {
        double px, py;
        world_to_pixel(wx, wy, px, py);

        // Get the four surrounding pixels
        int x0 = static_cast<int>(std::floor(px));
        int y0 = static_cast<int>(std::floor(py));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        // Check bounds
        if (x0 < 0 || x1 >= sizex || y0 < 0 || y1 >= sizey) {
            return nodata_value;
        }

        // Get elevation at corners
        float e00 = at_pixel(x0, y0);
        float e10 = at_pixel(x1, y0);
        float e01 = at_pixel(x0, y1);
        float e11 = at_pixel(x1, y1);

        // Handle NoData values
        if (e00 == nodata_value || e10 == nodata_value || e01 == nodata_value ||
            e11 == nodata_value) {
            return nodata_value;
        }

        // Bilinear interpolation weights
        double fx = px - x0;
        double fy = py - y0;

        return static_cast<float>(
            (1 - fx) * (1 - fy) * e00 + fx * (1 - fy) * e10 +
            (1 - fx) * fy * e01 + fx * fy * e11);
    }

    /// Check if a world coordinate is land or nodata
    bool is_land(double wx, double wy) const {
        float val = interpolate(wx, wy);
        // Check for NoData (using approximate comparison for float)
        if (std::abs(val - nodata_value) < 1e-6f || val > 1e30f) {
            return true; // NoData is treated as land
        }
        if (is_depth_positive) {
            // Depth format: positive values are water, zero or negative is land
            return val <= 0.0f;
        } else {
            // Elevation format: negative is water, positive is land
            return val >= 0.0f;
        }
    }

    /// Get bathymetry depth (positive downward) at world coordinates
    /// Returns 0 for land, positive for water depth
    float get_depth(double wx, double wy) const {
        float val = interpolate(wx, wy);
        // Check for NoData
        if (std::abs(val - nodata_value) < 1e-6f || val > 1e30f) {
            return 0.0f; // Land or invalid
        }
        if (is_depth_positive) {
            // Values are already depth (positive = water)
            return val > 0.0f ? val : 0.0f;
        } else {
            // Values are elevation (negative = water)
            return val < 0.0f ? -val : 0.0f;
        }
    }
};

/// @brief Reader for GeoTIFF bathymetry files
class GeoTiffReader {
public:
    GeoTiffReader();
    ~GeoTiffReader();

    /// @brief Load bathymetry from a GeoTIFF file
    /// @param filename Path to the GeoTIFF file
    /// @return Loaded bathymetry data
    BathymetryData load(const std::string &filename);

    /// @brief Load only bounds/metadata from a GeoTIFF file (no raster data)
    ///
    /// This is much faster than load() and useful for on-demand tile loading
    /// where we need to check bounds before deciding to load full data.
    ///
    /// @param filename Path to the GeoTIFF file
    /// @return Lightweight bounds metadata
    BathymetryBounds load_bounds_only(const std::string &filename);

    /// @brief Check if GDAL support is available
    static bool is_available();

    /// @brief Get last error message
    const std::string &last_error() const { return error_message_; }

private:
    std::string error_message_;
    bool gdal_initialized_ = false;
};

/// @brief Bathymetry surface for mesh generation
/// Provides an interpolation interface over the bathymetry data
class BathymetrySurface {
public:
    /// Construct from bathymetry data
    explicit BathymetrySurface(std::shared_ptr<BathymetryData> data);

    /// Evaluate bathymetry depth at world coordinates
    /// @param x, y World coordinates
    /// @return Depth (positive downward), 0 for land
    Real depth(Real x, Real y) const;

    /// Evaluate bathymetry gradient
    /// @param x, y World coordinates
    /// @param dh_dx, dh_dy Output gradients
    void gradient(Real x, Real y, Real &dh_dx, Real &dh_dy) const;

    /// Evaluate bathymetry curvature (second derivatives via central differences)
    /// @param x, y World coordinates
    /// @param d2h_dx2, d2h_dxdy, d2h_dy2 Output second derivatives
    void curvature(Real x, Real y, Real &d2h_dx2, Real &d2h_dxdy,
                   Real &d2h_dy2) const;

    /// Check if location is water (depth > min_depth)
    bool is_water(Real x, Real y, Real min_depth = 1.0) const;

    /// Check if location is land
    bool is_land(Real x, Real y) const;

    /// Get bounding box of the bathymetry data
    void get_bounds(Real &xmin, Real &xmax, Real &ymin, Real &ymax) const;

    /// Get the underlying data
    const BathymetryData &data() const { return *data_; }

private:
    std::shared_ptr<BathymetryData> data_;
};

/// @brief Create mesh from bathymetry GeoTIFF
/// Builds an adaptive octree mesh based on bathymetry and coastline features
class BathymetryMeshGenerator {
public:
    /// Configuration for mesh generation
    struct Config {
        /// Domain bounds (if not set, uses GeoTIFF bounds)
        Real xmin = 0, xmax = 0;
        Real ymin = 0, ymax = 0;

        /// Vertical bounds (sigma coordinates)
        Real zmin = -1.0; // Bottom (sigma = -1)
        Real zmax = 0.0;  // Surface (sigma = 0)

        /// Base resolution
        int base_nx = 10;
        int base_ny = 10;
        int base_nz = 5;

        /// Maximum refinement levels
        int max_level_x = 5;
        int max_level_y = 5;
        int max_level_z = 3;

        /// Refinement criteria
        Real coastline_distance =
            5000.0; // Refine within this distance of coastline
        Real bathymetry_gradient_threshold =
            0.01;                      // Refine where gradient exceeds this
        Real min_element_size = 100.0; // Minimum element size (meters)
        Real min_depth = 1.0;          // Minimum water depth to include

        /// Whether to mask land cells
        bool mask_land = true;
    };

    /// @brief Construct generator with bathymetry data
    explicit BathymetryMeshGenerator(
        std::shared_ptr<BathymetryData> bathymetry);

    /// @brief Set configuration
    void set_config(const Config &config);

    /// @brief Generate the mesh
    /// @return Vector of element bounds for water cells
    std::vector<ElementBounds> generate();

    /// @brief Get bathymetry values at element DOFs
    /// @param element_bounds Element bounds
    /// @param order Polynomial order
    /// @return Bathymetry depths at DOF positions
    std::vector<VecX> compute_element_bathymetry(
        const std::vector<ElementBounds> &elements, int order) const;

    /// @brief Get bathymetry gradients at element DOFs
    void compute_element_gradients(
        const std::vector<ElementBounds> &elements, int order,
        std::vector<VecX> &dh_dx, std::vector<VecX> &dh_dy) const;

    /// @brief Create refinement function for adaptive meshing
    std::function<bool(const ElementBounds &)>
    create_refinement_function() const;

private:
    std::shared_ptr<BathymetryData> bathymetry_;
    Config config_;

    bool should_refine(const ElementBounds &bounds) const;
    bool is_near_coastline(const ElementBounds &bounds) const;
    bool has_steep_gradient(const ElementBounds &bounds) const;
};

} // namespace drifter
