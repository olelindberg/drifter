#include "mesh/multi_source_bathymetry.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <gdal_priv.h>
#include <ogr_spatialref.h>

namespace drifter {

// =============================================================================
// Implementation details
// =============================================================================

/// @brief Info for a lazily-loaded tile
struct TileInfo {
    std::string path; ///< File path for lazy loading
    BathymetryBounds bounds; ///< Pre-loaded bounds (EPSG:4326)
    mutable std::optional<BathymetryData> data; ///< Lazily loaded raster
};

struct MultiSourceBathymetry::Impl {
    BathymetryData primary;
    std::vector<TileInfo> tiles; // Changed from BathymetryData to TileInfo
    GeoTiffReader reader;

    std::unique_ptr<OGRCoordinateTransformation> to_4326;

    // Combined bounds in EPSG:3034
    Real bounds_xmin = 0, bounds_xmax = 0;
    Real bounds_ymin = 0, bounds_ymax = 0;

    // Loading statistics
    mutable size_t tiles_loaded = 0;
    mutable size_t total_bytes_loaded = 0;

    // Check if a value is nodata
    static bool is_nodata(float val, float nodata_value) {
        return std::abs(val - nodata_value) < 1e-6f || val > 1e30f;
    }

    // Check if point is inside BathymetryData bounds
    static bool is_inside_bounds(const BathymetryData &data, double x, double y) {
        return x >= data.xmin && x <= data.xmax && y >= data.ymin && y <= data.ymax;
    }

    // Check if point is inside BathymetryBounds
    static bool is_inside_bounds(const BathymetryBounds &bounds, double x, double y) {
        return x >= bounds.xmin && x <= bounds.xmax && y >= bounds.ymin && y <= bounds.ymax;
    }

    // Get depth from BathymetryData, handling nodata
    static float get_depth_or_nodata(const BathymetryData &data, double x, double y) {
        float val = data.interpolate(x, y);
        if (is_nodata(val, data.nodata_value)) {
            return 0.0f; // Land
        }
        return data.get_depth(x, y);
    }
};

// =============================================================================
// MultiSourceBathymetry implementation
// =============================================================================

MultiSourceBathymetry::MultiSourceBathymetry(const std::string &primary_file,
                                             const std::vector<std::string> &tile_files)
    : impl_(std::make_unique<Impl>()) {
    // Load primary source (always loaded eagerly)
    std::cout << "[MultiSourceBathymetry] Loading primary source: " << primary_file << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    impl_->primary = impl_->reader.load(primary_file);
    if (!impl_->primary.is_valid()) {
        throw std::runtime_error("Failed to load primary bathymetry: " + primary_file +
                                 ". Error: " + impl_->reader.last_error());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    size_t bytes = impl_->primary.elevation.size() * sizeof(float);
    std::cout << "[MultiSourceBathymetry] Primary loaded: " << impl_->primary.sizex << "x"
              << impl_->primary.sizey << " (" << bytes / (1024 * 1024) << " MB) in " << ms << " ms"
              << std::endl;

    // Read tile bounds only (deferred loading)
    std::cout << "[MultiSourceBathymetry] Reading bounds for " << tile_files.size()
              << " tiles (deferred loading)..." << std::endl;
    start = std::chrono::high_resolution_clock::now();

    impl_->tiles.reserve(tile_files.size());
    for (const auto &tile_file : tile_files) {
        TileInfo info;
        info.path = tile_file;
        info.bounds = impl_->reader.load_bounds_only(tile_file);
        if (!info.bounds.is_valid()) {
            throw std::runtime_error("Failed to read tile bounds: " + tile_file +
                                     ". Error: " + impl_->reader.last_error());
        }
        // info.data remains empty (std::nullopt) - loaded on demand
        impl_->tiles.push_back(std::move(info));
    }

    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[MultiSourceBathymetry] Tile bounds loaded in " << ms << " ms" << std::endl;
    std::cout << "[MultiSourceBathymetry] Ready: 1 primary + " << tile_files.size()
              << " tiles available for on-demand loading" << std::endl;

    // Setup CRS transformation from EPSG:3034 to EPSG:4326
    OGRSpatialReference srcSRS, dstSRS;
    srcSRS.SetFromUserInput("EPSG:3034");
    dstSRS.SetFromUserInput("EPSG:4326");

    // Use traditional GIS order (lon, lat) for EPSG:4326
    dstSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    srcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

    impl_->to_4326.reset(OGRCreateCoordinateTransformation(&srcSRS, &dstSRS));
    if (!impl_->to_4326) {
        throw std::runtime_error("Failed to create coordinate transformation from "
                                 "EPSG:3034 to EPSG:4326");
    }

    // Compute combined bounds (primary bounds in EPSG:3034)
    impl_->bounds_xmin = impl_->primary.xmin;
    impl_->bounds_xmax = impl_->primary.xmax;
    impl_->bounds_ymin = impl_->primary.ymin;
    impl_->bounds_ymax = impl_->primary.ymax;

    // Extend bounds with tile coverage transformed to EPSG:3034
    // Note: For simplicity, we don't transform tile bounds to 3034
    // The combined bounds are approximate for the primary source
}

MultiSourceBathymetry::~MultiSourceBathymetry() {
    if (impl_) {
        size_t loaded = 0;
        size_t total_bytes = 0;
        for (const auto &tile : impl_->tiles) {
            if (tile.data.has_value()) {
                loaded++;
                total_bytes += tile.data->elevation.size() * sizeof(float);
            }
        }
        std::cout << "[MultiSourceBathymetry] Summary: " << loaded << "/" << impl_->tiles.size()
                  << " tiles were actually loaded (" << total_bytes / (1024 * 1024) << " MB)"
                  << std::endl;
        if (loaded < impl_->tiles.size()) {
            std::cout << "[MultiSourceBathymetry] Memory saved by deferred loading: "
                      << (impl_->tiles.size() - loaded) << " tiles not loaded" << std::endl;
        }
    }
}

MultiSourceBathymetry::MultiSourceBathymetry(MultiSourceBathymetry &&) noexcept = default;
MultiSourceBathymetry &
MultiSourceBathymetry::operator=(MultiSourceBathymetry &&) noexcept = default;

Real MultiSourceBathymetry::evaluate(Real x, Real y) const {
    // Step 1: Check primary source (EPSG:3034)
    if (Impl::is_inside_bounds(impl_->primary, x, y)) {
        float val = impl_->primary.interpolate(x, y);
        if (!Impl::is_nodata(val, impl_->primary.nodata_value)) {
            // Water - return depth
            return static_cast<Real>(impl_->primary.get_depth(x, y));
        }
        // Nodata in primary = land
        return 0.0;
    }

    // Step 2: Transform to EPSG:4326 for tile lookup
    double lon = x, lat = y;
    if (!impl_->to_4326->Transform(1, &lon, &lat)) {
        std::ostringstream oss;
        oss << "Point outside all sources: (" << x << ", " << y << ")";
        throw std::out_of_range(oss.str());
    }

    // Step 3: Search tiles (with lazy loading)
    for (auto &tile : impl_->tiles) {
        if (Impl::is_inside_bounds(tile.bounds, lon, lat)) {
            // Load tile data if not already loaded (lazy loading)
            if (!tile.data.has_value()) {
                auto start = std::chrono::high_resolution_clock::now();
                std::cout << "[MultiSourceBathymetry] Loading tile on demand: " << tile.path
                          << std::endl;

                tile.data = impl_->reader.load(tile.path);

                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - start).count();

                if (tile.data->is_valid()) {
                    size_t bytes = tile.data->elevation.size() * sizeof(float);
                    std::cout << "[MultiSourceBathymetry] Tile loaded: " << tile.data->sizex << "x"
                              << tile.data->sizey << " (" << bytes / (1024 * 1024) << " MB) in "
                              << ms << " ms" << std::endl;

                    impl_->tiles_loaded++;
                    impl_->total_bytes_loaded += bytes;
                } else {
                    std::cout << "[MultiSourceBathymetry] Warning: Failed to "
                                 "load tile: "
                              << tile.path << std::endl;
                }
            }

            // Use the loaded data
            if (tile.data->is_valid()) {
                float val = tile.data->interpolate(lon, lat);
                if (!Impl::is_nodata(val, tile.data->nodata_value)) {
                    // Water - return depth
                    return static_cast<Real>(tile.data->get_depth(lon, lat));
                }
                // Nodata in tile = land
                return 0.0;
            }
        }
    }

    // Step 4: Point outside all sources
    std::ostringstream oss;
    oss << "Point outside all sources: EPSG:3034(" << x << ", " << y << ") = "
        << "EPSG:4326(" << lon << ", " << lat << ")";
    throw std::out_of_range(oss.str());
}

bool MultiSourceBathymetry::is_land(Real x, Real y) const {
    Real depth = evaluate(x, y);
    return depth <= 0.0;
}

bool MultiSourceBathymetry::contains(Real x, Real y) const {
    // Check primary
    if (Impl::is_inside_bounds(impl_->primary, x, y)) {
        return true;
    }

    // Transform and check tile bounds (no loading needed)
    double lon = x, lat = y;
    if (!impl_->to_4326->Transform(1, &lon, &lat)) {
        return false;
    }

    for (const auto &tile : impl_->tiles) {
        if (Impl::is_inside_bounds(tile.bounds, lon, lat)) {
            return true;
        }
    }

    return false;
}

void MultiSourceBathymetry::get_bounds(Real &xmin, Real &xmax, Real &ymin, Real &ymax) const {
    xmin = impl_->bounds_xmin;
    xmax = impl_->bounds_xmax;
    ymin = impl_->bounds_ymin;
    ymax = impl_->bounds_ymax;
}

size_t MultiSourceBathymetry::num_sources() const { return 1 + impl_->tiles.size(); }

LoadingStats MultiSourceBathymetry::get_loading_stats() const {
    LoadingStats stats;
    stats.tiles_registered = impl_->tiles.size();
    stats.tiles_loaded = impl_->tiles_loaded;
    stats.bytes_loaded = impl_->total_bytes_loaded;
    return stats;
}

bool MultiSourceBathymetry::is_available() { return true; }

} // namespace drifter
