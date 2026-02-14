#pragma once

// Multi-Source Bathymetry Lookup for DRIFTER
//
// Provides cascading bathymetry lookup across multiple sources:
// 1. Primary high-resolution source (DDM at 50m, EPSG:3034)
// 2. Fallback tiles (EMODnet at ~115m, EPSG:4326)
//
// Query points are in EPSG:3034. When the primary source does not
// contain a point, coordinates are transformed to EPSG:4326 for
// tile lookup.

#include "bathymetry/biharmonic_assembler.hpp"
#include "core/types.hpp"
#include "mesh/geotiff_reader.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <ogr_spatialref.h>

namespace drifter {

/// @brief Statistics about on-demand tile loading
struct LoadingStats {
    size_t tiles_registered; ///< Total tiles available for loading
    size_t tiles_loaded;     ///< Number of tiles actually loaded
    size_t
        bytes_loaded; ///< Total bytes loaded for tiles (not including primary)
};

/// @brief Multi-source bathymetry with cascading lookup
///
/// Implements BathymetrySource interface for seamless integration with
/// existing bathymetry smoothers. Queries the primary source first,
/// then falls back to tiles if the point is outside the primary domain.
///
/// Usage:
///   MultiSourceBathymetry bathy(
///       "ddm_50m.dybde-emodnet.tif",
///       {"C4_2024.tif", "C5_2024.tif", ...}
///   );
///   Real depth = bathy.evaluate(x, y);  // x, y in EPSG:3034
class MultiSourceBathymetry : public BathymetrySource {
public:
    /// @brief Construct multi-source bathymetry
    /// @param primary_file Path to primary bathymetry (EPSG:3034)
    /// @param tile_files Paths to fallback tiles (EPSG:4326)
    /// @throws std::runtime_error if files cannot be loaded or CRS setup fails
    MultiSourceBathymetry(
        const std::string &primary_file,
        const std::vector<std::string> &tile_files);

    ~MultiSourceBathymetry();

    // Non-copyable, movable
    MultiSourceBathymetry(const MultiSourceBathymetry &) = delete;
    MultiSourceBathymetry &operator=(const MultiSourceBathymetry &) = delete;
    MultiSourceBathymetry(MultiSourceBathymetry &&) noexcept;
    MultiSourceBathymetry &operator=(MultiSourceBathymetry &&) noexcept;

    /// @brief Evaluate depth at world coordinates (EPSG:3034)
    ///
    /// Lookup algorithm:
    /// 1. Check if point is inside primary source bounds
    ///    - If nodata: return 0 (land)
    ///    - Otherwise: return bilinear interpolated depth
    /// 2. Transform to EPSG:4326 and check each tile
    ///    - If nodata: return 0 (land)
    ///    - Otherwise: return bilinear interpolated depth
    /// 3. If no source contains the point: throw std::out_of_range
    ///
    /// @param x X coordinate in EPSG:3034
    /// @param y Y coordinate in EPSG:3034
    /// @return Depth (positive downward), 0 for land
    /// @throws std::out_of_range if point is outside all sources
    Real evaluate(Real x, Real y) const override;

    /// @brief Check if a point is land (nodata or zero depth)
    /// @param x X coordinate in EPSG:3034
    /// @param y Y coordinate in EPSG:3034
    /// @return true if land, false if water
    /// @throws std::out_of_range if point is outside all sources
    bool is_land(Real x, Real y) const;

    /// @brief Check if a point is within any source domain
    /// @param x X coordinate in EPSG:3034
    /// @param y Y coordinate in EPSG:3034
    /// @return true if the point can be queried
    bool contains(Real x, Real y) const;

    /// @brief Get combined bounding box in EPSG:3034
    /// @param xmin, xmax, ymin, ymax Output bounds
    void get_bounds(Real &xmin, Real &xmax, Real &ymin, Real &ymax) const;

    /// @brief Get number of loaded sources
    /// @return Total number of sources (1 primary + N tiles)
    size_t num_sources() const;

    /// @brief Get statistics about on-demand tile loading
    ///
    /// Returns information about how many tiles have been loaded vs registered,
    /// useful for diagnosing memory usage and loading behavior.
    ///
    /// @return Loading statistics
    LoadingStats get_loading_stats() const;

    /// @brief Check if GDAL support is available
    static bool is_available();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace drifter
