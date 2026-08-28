#pragma once

/// @file raster_vtk_writer.hpp
/// @brief VTK output for GeoTIFF rasters at their native resolution
///
/// Separate from bathymetry_vtk_writer.hpp, which writes quadtree surfaces.
/// This writer emits the *source* data, unsmoothed and unresampled, so a fit
/// can be compared against the pixels it was fitted to.

#include "core/types.hpp"
#include <string>

namespace drifter {

struct BathymetryData;

namespace io {

/// @brief Write a raster, cropped to a domain box, as a VTK StructuredGrid
///
/// Emits the pixels of @p data whose centers fall inside
/// [xmin, xmax] x [ymin, ymax] at the raster's own resolution - one point per
/// pixel center, no resampling.
///
/// The raster value *is* the z coordinate of the point, so the surface has real
/// 3D geometry and a ParaView Transform scales it directly - no WarpByScalar
/// needed. The same value is also written as the "elevation" point-data array,
/// there carrying NaN where the pixel is nodata (those points sit at z = 0,
/// since a NaN coordinate would corrupt the grid bounds).
///
/// @param filename Output filename (without extension, ".vts" will be added)
/// @param data The raster (typically MultiSourceBathymetry::get_primary())
/// @param xmin, xmax, ymin, ymax Domain box in the raster's own CRS
/// @throws std::invalid_argument if the geotransform is rotated (the grid would
///         be sheared) or the domain box selects no pixels
/// @throws std::runtime_error if the file cannot be opened
void write_raster_vts(const std::string &filename, const BathymetryData &data, Real xmin, Real xmax,
                      Real ymin, Real ymax);

} // namespace io
} // namespace drifter
