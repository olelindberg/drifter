#include "mesh/multi_source_bathymetry.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#ifdef DRIFTER_HAS_GDAL
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#endif

namespace drifter {

// =============================================================================
// Implementation details
// =============================================================================

struct MultiSourceBathymetry::Impl {
  BathymetryData primary;
  std::vector<BathymetryData> tiles;
  GeoTiffReader reader;

#ifdef DRIFTER_HAS_GDAL
  std::unique_ptr<OGRCoordinateTransformation> to_4326;
#endif

  // Combined bounds in EPSG:3034
  Real bounds_xmin = 0, bounds_xmax = 0;
  Real bounds_ymin = 0, bounds_ymax = 0;

  // Check if a value is nodata
  static bool is_nodata(float val, float nodata_value) {
    return std::abs(val - nodata_value) < 1e-6f || val > 1e30f;
  }

  // Check if point is inside BathymetryData bounds
  static bool is_inside_bounds(const BathymetryData &data, double x, double y) {
    return x >= data.xmin && x <= data.xmax && y >= data.ymin && y <= data.ymax;
  }

  // Get depth from BathymetryData, handling nodata
  static float get_depth_or_nodata(const BathymetryData &data, double x,
                                   double y) {
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

MultiSourceBathymetry::MultiSourceBathymetry(
    const std::string &primary_file, const std::vector<std::string> &tile_files)
    : impl_(std::make_unique<Impl>()) {
#ifndef DRIFTER_HAS_GDAL
  throw std::runtime_error("MultiSourceBathymetry requires GDAL support. "
                           "Rebuild with -DDRIFTER_USE_GDAL=ON");
#else
  // Load primary source
  impl_->primary = impl_->reader.load(primary_file);
  if (!impl_->primary.is_valid()) {
    throw std::runtime_error(
        "Failed to load primary bathymetry: " + primary_file +
        ". Error: " + impl_->reader.last_error());
  }

  // Load tiles
  impl_->tiles.reserve(tile_files.size());
  for (const auto &tile_file : tile_files) {
    BathymetryData tile = impl_->reader.load(tile_file);
    if (!tile.is_valid()) {
      throw std::runtime_error("Failed to load tile: " + tile_file +
                               ". Error: " + impl_->reader.last_error());
    }
    impl_->tiles.push_back(std::move(tile));
  }

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
#endif
}

MultiSourceBathymetry::~MultiSourceBathymetry() = default;

MultiSourceBathymetry::MultiSourceBathymetry(
    MultiSourceBathymetry &&) noexcept = default;
MultiSourceBathymetry &
MultiSourceBathymetry::operator=(MultiSourceBathymetry &&) noexcept = default;

Real MultiSourceBathymetry::evaluate(Real x, Real y) const {
#ifndef DRIFTER_HAS_GDAL
  throw std::runtime_error("GDAL support not available");
#else
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

  // Step 3: Search tiles
  for (const auto &tile : impl_->tiles) {
    if (Impl::is_inside_bounds(tile, lon, lat)) {
      float val = tile.interpolate(lon, lat);
      if (!Impl::is_nodata(val, tile.nodata_value)) {
        // Water - return depth
        return static_cast<Real>(tile.get_depth(lon, lat));
      }
      // Nodata in tile = land
      return 0.0;
    }
  }

  // Step 4: Point outside all sources
  std::ostringstream oss;
  oss << "Point outside all sources: EPSG:3034(" << x << ", " << y << ") = "
      << "EPSG:4326(" << lon << ", " << lat << ")";
  throw std::out_of_range(oss.str());
#endif
}

bool MultiSourceBathymetry::is_land(Real x, Real y) const {
  Real depth = evaluate(x, y);
  return depth <= 0.0;
}

bool MultiSourceBathymetry::contains(Real x, Real y) const {
#ifndef DRIFTER_HAS_GDAL
  return false;
#else
  // Check primary
  if (Impl::is_inside_bounds(impl_->primary, x, y)) {
    return true;
  }

  // Transform and check tiles
  double lon = x, lat = y;
  if (!impl_->to_4326->Transform(1, &lon, &lat)) {
    return false;
  }

  for (const auto &tile : impl_->tiles) {
    if (Impl::is_inside_bounds(tile, lon, lat)) {
      return true;
    }
  }

  return false;
#endif
}

void MultiSourceBathymetry::get_bounds(Real &xmin, Real &xmax, Real &ymin,
                                       Real &ymax) const {
  xmin = impl_->bounds_xmin;
  xmax = impl_->bounds_xmax;
  ymin = impl_->bounds_ymin;
  ymax = impl_->bounds_ymax;
}

size_t MultiSourceBathymetry::num_sources() const {
  return 1 + impl_->tiles.size();
}

bool MultiSourceBathymetry::is_available() {
#ifdef DRIFTER_HAS_GDAL
  return true;
#else
  return false;
#endif
}

} // namespace drifter
