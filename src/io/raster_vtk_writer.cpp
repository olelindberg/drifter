/// @file raster_vtk_writer.cpp
/// @brief VTK StructuredGrid output for GeoTIFF rasters at their native resolution

#include "io/raster_vtk_writer.hpp"
#include "io/vtk_binary_utils.hpp"
#include "mesh/geotiff_reader.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

namespace drifter {
namespace io {

void write_raster_vts(const std::string &filename, const BathymetryData &data, Real xmin, Real xmax,
                      Real ymin, Real ymax) {
    if (!data.is_valid()) {
        throw std::invalid_argument("write_raster_vts: raster has no data");
    }
    // A rotated geotransform maps pixels to a sheared grid, which VTK ImageData
    // cannot represent. Resampling it onto an axis-aligned grid would no longer
    // be the input data at its own resolution, so this is an error.
    if (data.geotransform[2] != 0.0 || data.geotransform[4] != 0.0) {
        throw std::invalid_argument("write_raster_vts: rotated geotransform is not representable "
                                    "as VTK ImageData");
    }

    // Fractional pixel coordinates of the domain corners. The geotransform is
    // axis-aligned, so x maps to px and y to py independently, but either axis
    // may be flipped (pixel height is normally negative).
    double px_a, py_a, px_b, py_b;
    data.world_to_pixel(xmin, ymin, px_a, py_a);
    data.world_to_pixel(xmax, ymax, px_b, py_b);

    const double px_lo = std::min(px_a, px_b);
    const double px_hi = std::max(px_a, px_b);
    const double py_lo = std::min(py_a, py_b);
    const double py_hi = std::max(py_a, py_b);

    // Pixel index i has its center at px = i + 0.5; keep the pixels whose
    // centers fall inside the domain box.
    auto index_range = [](double lo, double hi, int size, int &first, int &last) {
        first = std::max(0, static_cast<int>(std::ceil(lo - 0.5)));
        last = std::min(size - 1, static_cast<int>(std::floor(hi - 0.5)));
    };

    int i0, i1, j0, j1;
    index_range(px_lo, px_hi, data.sizex, i0, i1);
    index_range(py_lo, py_hi, data.sizey, j0, j1);

    if (i0 > i1 || j0 > j1) {
        throw std::invalid_argument("write_raster_vts: domain box selects no raster pixels");
    }

    const int nx = i1 - i0 + 1;
    const int ny = j1 - j0 + 1;

    // Rows are emitted bottom-up in world y: with the usual negative pixel
    // height, row j1 is the southernmost.
    const bool y_flipped = data.geotransform[5] < 0.0;

    std::vector<Real> points;
    std::vector<Real> elevation;
    points.reserve(static_cast<size_t>(nx) * ny * 3);
    elevation.reserve(static_cast<size_t>(nx) * ny);

    const Real nan_value = std::numeric_limits<Real>::quiet_NaN();

    for (int row = 0; row < ny; ++row) {
        const int j = y_flipped ? (j1 - row) : (j0 + row);
        for (int i = i0; i <= i1; ++i) {
            const float value = data.at_pixel(i, j);
            const bool is_nodata = std::abs(value - data.nodata_value) < 1e-6f || value > 1e30f;

            double wx, wy;
            data.pixel_center_to_world(i, j, wx, wy);

            // The elevation is the z coordinate, so a Transform/scale in
            // ParaView exaggerates it directly. Nodata pixels sit at z = 0 -
            // a NaN coordinate would corrupt the bounds of the whole grid -
            // and are blanked instead by the NaN in the scalar array.
            points.push_back(wx);
            points.push_back(wy);
            points.push_back(is_nodata ? 0.0 : static_cast<Real>(value));

            elevation.push_back(is_nodata ? nan_value : static_cast<Real>(value));
        }
    }

    std::ofstream file(filename + ".vts");
    if (!file.is_open()) {
        throw std::runtime_error("write_raster_vts: cannot open " + filename + ".vts");
    }

    file << std::setprecision(17);
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" "
            "header_type=\"UInt64\">\n";
    file << "  <StructuredGrid WholeExtent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 0\">\n";
    file << "    <Piece Extent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 0\">\n";
    file << "      <Points>\n        ";
    vtk::write_binary_float64(file, "Points", 3, points);
    file << "      </Points>\n";
    file << "      <PointData Scalars=\"elevation\">\n        ";
    vtk::write_binary_float64(file, "elevation", 1, elevation);
    file << "      </PointData>\n";
    file << "    </Piece>\n";
    file << "  </StructuredGrid>\n";
    file << "</VTKFile>\n";
    file.close();
}

} // namespace io
} // namespace drifter
