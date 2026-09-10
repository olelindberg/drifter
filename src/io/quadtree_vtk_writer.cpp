#include "io/quadtree_vtk_writer.hpp"
#include "core/logger.hpp"
#include "io/vtk_binary_utils.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace drifter {

namespace {

// Stream buffer size for faster I/O (1MB)
constexpr size_t STREAM_BUFFER_SIZE = 1 << 20;

// Open file with large buffer for faster I/O
std::ofstream open_buffered(const std::string& path, std::vector<char>& buffer) {
    std::ofstream f(path, std::ios::binary);
    if (f.is_open()) {
        buffer.resize(STREAM_BUFFER_SIZE);
        f.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
    }
    return f;
}

// Write VTK XML header
// Note: header_type="UInt64" is required because we use 64-bit size headers in binary encoding
void write_header(std::ostream& f, Index npoints, Index ncells) {
    f << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
      << "  <UnstructuredGrid>\n"
      << "    <Piece NumberOfPoints=\"" << npoints << "\" NumberOfCells=\"" << ncells << "\">\n";
}

// Write VTK XML footer
void write_footer(std::ostream& f) {
    f << "    </Piece>\n"
      << "  </UnstructuredGrid>\n"
      << "</VTKFile>\n";
}

} // namespace

void QuadtreeVTKWriter::write(const std::string& filename,
                              const QuadtreeAdapter& mesh,
                              const LinearBezierSurface& surface) {
    std::string full_path = filename + ".vtu";
    std::vector<char> file_buffer;
    std::ofstream f = open_buffered(full_path, file_buffer);
    if (!f.is_open()) {
        LOG_ERROR("Could not open " << full_path << " for writing");
        return;
    }

    Index ncells = mesh.num_elements();
    Index npoints = ncells * 4;

    // Pre-allocate all data buffers
    std::vector<Real> points;
    points.reserve(static_cast<size_t>(npoints) * 3);

    std::vector<int64_t> connectivity;
    connectivity.reserve(static_cast<size_t>(npoints));

    std::vector<int64_t> offsets;
    offsets.reserve(static_cast<size_t>(ncells));

    std::vector<uint8_t> types;
    types.reserve(static_cast<size_t>(ncells));

    std::vector<int32_t> level;
    level.reserve(static_cast<size_t>(ncells));

    std::vector<int64_t> element_id;
    element_id.reserve(static_cast<size_t>(ncells));

    std::vector<Real> area;
    area.reserve(static_cast<size_t>(ncells));

    // Single-pass data collection
    for (Index elem = 0; elem < ncells; ++elem) {
        const auto& bounds = mesh.element_bounds(elem);
        Eigen::Vector4d z = surface.element_coefficients(elem);
        auto elem_level = mesh.element_level(elem);
        auto elem_size = mesh.element_size(elem);

        // Points: 4 corners per quad (VTK quad order)
        points.push_back(bounds.xmin); points.push_back(bounds.ymin); points.push_back(z(0));
        points.push_back(bounds.xmax); points.push_back(bounds.ymin); points.push_back(z(1));
        points.push_back(bounds.xmax); points.push_back(bounds.ymax); points.push_back(z(3));
        points.push_back(bounds.xmin); points.push_back(bounds.ymax); points.push_back(z(2));

        // Connectivity
        Index base = elem * 4;
        connectivity.push_back(base);
        connectivity.push_back(base + 1);
        connectivity.push_back(base + 2);
        connectivity.push_back(base + 3);

        // Offsets
        offsets.push_back((elem + 1) * 4);

        // Types (VTK_QUAD = 9)
        types.push_back(9);

        // Cell data
        level.push_back(elem_level.max_level());
        element_id.push_back(elem);
        area.push_back(elem_size(0) * elem_size(1));
    }

    // Write VTK file
    write_header(f, npoints, ncells);

    // Points
    f << "      <Points>\n        ";
    vtk::write_binary_points(f, points);
    f << "      </Points>\n";

    // Cells
    f << "      <Cells>\n        ";
    vtk::write_binary_int64(f, "connectivity", connectivity);
    f << "        ";
    vtk::write_binary_int64(f, "offsets", offsets);
    f << "        ";
    vtk::write_binary_uint8(f, "types", types);
    f << "      </Cells>\n";

    // Cell data
    f << "      <CellData>\n        ";
    vtk::write_binary_int32(f, "level", level);
    f << "        ";
    vtk::write_binary_int64(f, "element_id", element_id);
    f << "        ";
    vtk::write_binary_float64(f, "area", 1, area);
    f << "      </CellData>\n";

    write_footer(f);
    f.close();
}

void QuadtreeVTKWriter::write_water_only(const std::string& filename,
                                          const QuadtreeAdapter& mesh,
                                          const LinearBezierSurface& surface,
                                          std::function<Real(Real, Real)> depth_func) {
    if (!depth_func) {
        throw std::runtime_error(
            "QuadtreeVTKWriter::write_water_only: depth_func is null. "
            "Provide a valid depth function.");
    }

    // First pass: identify water elements
    std::vector<Index> water_elements;
    water_elements.reserve(static_cast<size_t>(mesh.num_elements()));

    for (Index elem = 0; elem < mesh.num_elements(); ++elem) {
        const auto& bounds = mesh.element_bounds(elem);
        Real cx = 0.5 * (bounds.xmin + bounds.xmax);
        Real cy = 0.5 * (bounds.ymin + bounds.ymax);
        if (depth_func(cx, cy) > 0.0) {
            water_elements.push_back(elem);
        }
    }

    if (water_elements.empty()) {
        throw std::runtime_error(
            "QuadtreeVTKWriter::write_water_only: no water elements found. "
            "All " + std::to_string(mesh.num_elements()) +
            " elements have depth <= 0 at their centers.");
    }

    std::string full_path = filename + ".vtu";
    std::vector<char> file_buffer;
    std::ofstream f = open_buffered(full_path, file_buffer);
    if (!f.is_open()) {
        throw std::runtime_error(
            "QuadtreeVTKWriter::write_water_only: could not open '" + full_path + "' for writing.");
    }

    Index ncells = static_cast<Index>(water_elements.size());
    Index npoints = ncells * 4;

    // Pre-allocate all data buffers
    std::vector<Real> points;
    points.reserve(static_cast<size_t>(npoints) * 3);

    std::vector<int64_t> connectivity;
    connectivity.reserve(static_cast<size_t>(npoints));

    std::vector<int64_t> offsets;
    offsets.reserve(static_cast<size_t>(ncells));

    std::vector<uint8_t> types;
    types.reserve(static_cast<size_t>(ncells));

    std::vector<Real> depth;
    depth.reserve(static_cast<size_t>(ncells));

    std::vector<int32_t> level;
    level.reserve(static_cast<size_t>(ncells));

    std::vector<int64_t> element_id;
    element_id.reserve(static_cast<size_t>(ncells));

    std::vector<Real> area;
    area.reserve(static_cast<size_t>(ncells));

    // Single-pass data collection
    for (Index i = 0; i < ncells; ++i) {
        Index elem = water_elements[static_cast<size_t>(i)];
        const auto& bounds = mesh.element_bounds(elem);
        Eigen::Vector4d z = surface.element_coefficients(elem);
        auto elem_level = mesh.element_level(elem);
        auto elem_size = mesh.element_size(elem);

        Real cx = 0.5 * (bounds.xmin + bounds.xmax);
        Real cy = 0.5 * (bounds.ymin + bounds.ymax);

        // Points
        points.push_back(bounds.xmin); points.push_back(bounds.ymin); points.push_back(z(0));
        points.push_back(bounds.xmax); points.push_back(bounds.ymin); points.push_back(z(1));
        points.push_back(bounds.xmax); points.push_back(bounds.ymax); points.push_back(z(3));
        points.push_back(bounds.xmin); points.push_back(bounds.ymax); points.push_back(z(2));

        // Connectivity
        Index base = i * 4;
        connectivity.push_back(base);
        connectivity.push_back(base + 1);
        connectivity.push_back(base + 2);
        connectivity.push_back(base + 3);

        // Offsets
        offsets.push_back((i + 1) * 4);

        // Types
        types.push_back(9);

        // Cell data
        depth.push_back(depth_func(cx, cy));
        level.push_back(elem_level.max_level());
        element_id.push_back(elem);
        area.push_back(elem_size(0) * elem_size(1));
    }

    // Write VTK file
    write_header(f, npoints, ncells);

    // Points
    f << "      <Points>\n        ";
    vtk::write_binary_points(f, points);
    f << "      </Points>\n";

    // Cells
    f << "      <Cells>\n        ";
    vtk::write_binary_int64(f, "connectivity", connectivity);
    f << "        ";
    vtk::write_binary_int64(f, "offsets", offsets);
    f << "        ";
    vtk::write_binary_uint8(f, "types", types);
    f << "      </Cells>\n";

    // Cell data
    f << "      <CellData>\n        ";
    vtk::write_binary_float64(f, "depth", 1, depth);
    f << "        ";
    vtk::write_binary_int32(f, "level", level);
    f << "        ";
    vtk::write_binary_int64(f, "element_id", element_id);
    f << "        ";
    vtk::write_binary_float64(f, "area", 1, area);
    f << "      </CellData>\n";

    write_footer(f);
    f.close();
}

void QuadtreeVTKWriter::write_mesh_only(const std::string& filename,
                                         const QuadtreeAdapter& mesh) {
    std::string full_path = filename + ".vtu";
    std::vector<char> file_buffer;
    std::ofstream f = open_buffered(full_path, file_buffer);
    if (!f.is_open()) {
        LOG_ERROR("Could not open " << full_path << " for writing");
        return;
    }

    Index ncells = mesh.num_elements();
    Index npoints = ncells * 4;

    // Pre-allocate all data buffers
    std::vector<Real> points;
    points.reserve(static_cast<size_t>(npoints) * 3);

    std::vector<int64_t> connectivity;
    connectivity.reserve(static_cast<size_t>(npoints));

    std::vector<int64_t> offsets;
    offsets.reserve(static_cast<size_t>(ncells));

    std::vector<uint8_t> types;
    types.reserve(static_cast<size_t>(ncells));

    std::vector<int32_t> level;
    level.reserve(static_cast<size_t>(ncells));

    // Single-pass data collection
    for (Index elem = 0; elem < ncells; ++elem) {
        const auto& bounds = mesh.element_bounds(elem);
        auto elem_level = mesh.element_level(elem);

        // Points (z=0 for mesh only)
        points.push_back(bounds.xmin); points.push_back(bounds.ymin); points.push_back(0.0);
        points.push_back(bounds.xmax); points.push_back(bounds.ymin); points.push_back(0.0);
        points.push_back(bounds.xmax); points.push_back(bounds.ymax); points.push_back(0.0);
        points.push_back(bounds.xmin); points.push_back(bounds.ymax); points.push_back(0.0);

        // Connectivity
        Index base = elem * 4;
        connectivity.push_back(base);
        connectivity.push_back(base + 1);
        connectivity.push_back(base + 2);
        connectivity.push_back(base + 3);

        // Offsets
        offsets.push_back((elem + 1) * 4);

        // Types
        types.push_back(9);

        // Cell data
        level.push_back(elem_level.max_level());
    }

    // Write VTK file
    write_header(f, npoints, ncells);

    // Points
    f << "      <Points>\n        ";
    vtk::write_binary_points(f, points);
    f << "      </Points>\n";

    // Cells
    f << "      <Cells>\n        ";
    vtk::write_binary_int64(f, "connectivity", connectivity);
    f << "        ";
    vtk::write_binary_int64(f, "offsets", offsets);
    f << "        ";
    vtk::write_binary_uint8(f, "types", types);
    f << "      </Cells>\n";

    // Cell data
    f << "      <CellData>\n        ";
    vtk::write_binary_int32(f, "level", level);
    f << "      </CellData>\n";

    write_footer(f);
    f.close();
}

void QuadtreeVTKWriter::write_with_errors(const std::string& filename,
                                           const QuadtreeAdapter& mesh,
                                           const LinearBezierSurface& surface,
                                           const std::vector<ElementError>& errors,
                                           std::function<Real(Real, Real)> depth_func,
                                           std::function<int(Real, Real)> source_id_func) {
    std::string full_path = filename + ".vtu";
    std::vector<char> file_buffer;
    std::ofstream f = open_buffered(full_path, file_buffer);
    if (!f.is_open()) {
        LOG_ERROR("Could not open " << full_path << " for writing");
        return;
    }

    Index ncells = mesh.num_elements();
    Index npoints = ncells * 4;

    // Build error lookup (element index -> error value)
    std::vector<Real> error_lookup(static_cast<size_t>(ncells), 0.0);
    for (const auto& err : errors) {
        if (err.element >= 0 && err.element < ncells) {
            error_lookup[static_cast<size_t>(err.element)] = err.error;
        }
    }

    // Pre-allocate all data buffers
    std::vector<Real> points;
    points.reserve(static_cast<size_t>(npoints) * 3);

    std::vector<int64_t> connectivity;
    connectivity.reserve(static_cast<size_t>(npoints));

    std::vector<int64_t> offsets;
    offsets.reserve(static_cast<size_t>(ncells));

    std::vector<uint8_t> types;
    types.reserve(static_cast<size_t>(ncells));

    std::vector<Real> error_data;
    error_data.reserve(static_cast<size_t>(ncells));

    std::vector<Real> depth_data;
    std::vector<Real> surface_z_data;
    if (depth_func) {
        depth_data.reserve(static_cast<size_t>(ncells));
        surface_z_data.reserve(static_cast<size_t>(ncells));
    }

    std::vector<int32_t> level;
    level.reserve(static_cast<size_t>(ncells));

    std::vector<int64_t> element_id;
    element_id.reserve(static_cast<size_t>(ncells));

    std::vector<Real> area;
    area.reserve(static_cast<size_t>(ncells));

    std::vector<Real> element_size_x;
    element_size_x.reserve(static_cast<size_t>(ncells));

    std::vector<Real> element_size_y;
    element_size_y.reserve(static_cast<size_t>(ncells));

    std::vector<int32_t> source_id_data;
    if (source_id_func) {
        source_id_data.reserve(static_cast<size_t>(ncells));
    }

    // Single-pass data collection
    for (Index elem = 0; elem < ncells; ++elem) {
        const auto& bounds = mesh.element_bounds(elem);
        Eigen::Vector4d z = surface.element_coefficients(elem);
        auto elem_level = mesh.element_level(elem);
        auto elem_size = mesh.element_size(elem);

        Real cx = 0.5 * (bounds.xmin + bounds.xmax);
        Real cy = 0.5 * (bounds.ymin + bounds.ymax);

        // Points
        points.push_back(bounds.xmin); points.push_back(bounds.ymin); points.push_back(z(0));
        points.push_back(bounds.xmax); points.push_back(bounds.ymin); points.push_back(z(1));
        points.push_back(bounds.xmax); points.push_back(bounds.ymax); points.push_back(z(3));
        points.push_back(bounds.xmin); points.push_back(bounds.ymax); points.push_back(z(2));

        // Connectivity
        Index base = elem * 4;
        connectivity.push_back(base);
        connectivity.push_back(base + 1);
        connectivity.push_back(base + 2);
        connectivity.push_back(base + 3);

        // Offsets
        offsets.push_back((elem + 1) * 4);

        // Types
        types.push_back(9);

        // Cell data
        error_data.push_back(error_lookup[static_cast<size_t>(elem)]);

        if (depth_func) {
            depth_data.push_back(depth_func(cx, cy));
            surface_z_data.push_back(surface.evaluate(cx, cy));
        }

        level.push_back(elem_level.max_level());
        element_id.push_back(elem);
        area.push_back(elem_size(0) * elem_size(1));
        element_size_x.push_back(elem_size(0));
        element_size_y.push_back(elem_size(1));

        if (source_id_func) {
            source_id_data.push_back(source_id_func(cx, cy));
        }
    }

    // Write VTK file
    write_header(f, npoints, ncells);

    // Points
    f << "      <Points>\n        ";
    vtk::write_binary_points(f, points);
    f << "      </Points>\n";

    // Cells
    f << "      <Cells>\n        ";
    vtk::write_binary_int64(f, "connectivity", connectivity);
    f << "        ";
    vtk::write_binary_int64(f, "offsets", offsets);
    f << "        ";
    vtk::write_binary_uint8(f, "types", types);
    f << "      </Cells>\n";

    // Cell data
    f << "      <CellData>\n        ";
    vtk::write_binary_float64(f, "error", 1, error_data);

    if (depth_func) {
        f << "        ";
        vtk::write_binary_float64(f, "depth", 1, depth_data);
        f << "        ";
        vtk::write_binary_float64(f, "surface_z", 1, surface_z_data);
    }

    f << "        ";
    vtk::write_binary_int32(f, "level", level);
    f << "        ";
    vtk::write_binary_int64(f, "element_id", element_id);
    f << "        ";
    vtk::write_binary_float64(f, "area", 1, area);
    f << "        ";
    vtk::write_binary_float64(f, "element_size_x", 1, element_size_x);
    f << "        ";
    vtk::write_binary_float64(f, "element_size_y", 1, element_size_y);

    if (source_id_func) {
        f << "        ";
        vtk::write_binary_int32(f, "source_id", source_id_data);
    }

    f << "      </CellData>\n";

    write_footer(f);
    f.close();
    LOG_INFO("Wrote error field to: " << full_path);
}

} // namespace drifter
