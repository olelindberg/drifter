#include "io/water_vtk_writer.hpp"
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace drifter {

namespace {

const char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "abcdefghijklmnopqrstuvwxyz"
                            "0123456789+/";

std::string base64_encode(const unsigned char* data, size_t len) {
    std::string result;
    result.reserve((len + 2) / 3 * 4);

    for (size_t i = 0; i < len; i += 3) {
        unsigned int n = data[i] << 16;
        if (i + 1 < len)
            n |= data[i + 1] << 8;
        if (i + 2 < len)
            n |= data[i + 2];

        result += base64_chars[(n >> 18) & 0x3F];
        result += base64_chars[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? base64_chars[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? base64_chars[n & 0x3F] : '=';
    }

    return result;
}

void write_data_array_binary(std::ostream& out, const std::string& name, int num_components,
                             const std::vector<Real>& data) {
    out << "<DataArray type=\"Float64\" Name=\"" << name << "\" NumberOfComponents=\""
        << num_components << "\" format=\"binary\">";

    uint64_t size = data.size() * sizeof(Real);
    std::vector<unsigned char> buffer(sizeof(uint64_t) + size);
    std::memcpy(buffer.data(), &size, sizeof(uint64_t));
    std::memcpy(buffer.data() + sizeof(uint64_t), data.data(), size);

    out << base64_encode(buffer.data(), buffer.size());
    out << "</DataArray>\n";
}

} // namespace

WaterVTKWriter::WaterVTKWriter(const std::string& basename, int polynomial_order)
    : basename_(basename), order_(polynomial_order) {
    // Create output directory if needed
    std::filesystem::path dir = std::filesystem::path(basename_).parent_path();
    if (!dir.empty()) {
        std::filesystem::create_directories(dir);
    }
}

void WaterVTKWriter::set_mesh(const OctreeAdapter& mesh, DepthQuery depth_query) {
    if (!depth_query) {
        throw std::runtime_error(
            "WaterVTKWriter: depth query function is null. "
            "Provide a valid depth query function, e.g., "
            "[&seabed](Real x, Real y) { return seabed.depth(x, y); }");
    }

    mesh_ = &mesh;
    depth_query_ = std::move(depth_query);

    identify_water_elements();
    build_water_geometry();
}

void WaterVTKWriter::identify_water_elements() {
    water_elements_.clear();
    mesh_to_water_.clear();
    element_depths_.clear();

    const auto& elements = mesh_->elements();
    for (Index e = 0; e < static_cast<Index>(elements.size()); ++e) {
        const auto& bounds = elements[e]->bounds;
        Real x = 0.5 * (bounds.xmin + bounds.xmax);
        Real y = 0.5 * (bounds.ymin + bounds.ymax);
        Real depth = depth_query_(x, y);

        if (depth > 0.0) {
            size_t water_idx = water_elements_.size();
            water_elements_.push_back(e);
            mesh_to_water_[e] = water_idx;
            element_depths_.push_back(depth);
        }
    }

    if (water_elements_.empty()) {
        throw std::runtime_error(
            "WaterVTKWriter: no water elements found. "
            "All " + std::to_string(elements.size()) +
            " elements have depth <= 0 at their centers.");
    }
}

void WaterVTKWriter::build_water_geometry() {
    int nodes_per_dim = order_ + 1;
    int nodes_per_elem = nodes_per_dim * nodes_per_dim * nodes_per_dim;

    points_.clear();
    cells_.clear();

    points_.reserve(water_elements_.size() * nodes_per_elem);
    cells_.reserve(water_elements_.size());

    Index point_offset = 0;

    for (Index mesh_idx : water_elements_) {
        const auto* node_ptr = mesh_->elements()[mesh_idx];
        const auto& bounds = node_ptr->bounds;
        Vec3 min_corner(bounds.xmin, bounds.ymin, bounds.zmin);
        Vec3 max_corner(bounds.xmax, bounds.ymax, bounds.zmax);
        Vec3 size = max_corner - min_corner;

        std::vector<Index> cell_connectivity;
        cell_connectivity.reserve(nodes_per_elem);

        for (int k = 0; k < nodes_per_dim; ++k) {
            Real zeta = (order_ == 1) ? static_cast<Real>(k) : -1.0 + 2.0 * k / order_;

            for (int j = 0; j < nodes_per_dim; ++j) {
                Real eta = (order_ == 1) ? static_cast<Real>(j) : -1.0 + 2.0 * j / order_;

                for (int i = 0; i < nodes_per_dim; ++i) {
                    Real xi = (order_ == 1) ? static_cast<Real>(i) : -1.0 + 2.0 * i / order_;

                    Vec3 point;
                    point(0) = min_corner(0) + 0.5 * (xi + 1.0) * size(0);
                    point(1) = min_corner(1) + 0.5 * (eta + 1.0) * size(1);
                    point(2) = min_corner(2) + 0.5 * (zeta + 1.0) * size(2);

                    points_.push_back(point);
                    cell_connectivity.push_back(point_offset++);
                }
            }
        }

        // Reorder to VTK ordering for linear hex
        if (order_ == 1) {
            std::vector<Index> vtk_order = {0, 1, 3, 2, 4, 5, 7, 6};
            std::vector<Index> reordered(8);
            Index base = cell_connectivity[0];
            for (int i = 0; i < 8; ++i) {
                reordered[i] = base + vtk_order[i];
            }
            cells_.push_back(reordered);
        } else {
            cells_.push_back(cell_connectivity);
        }
    }
}

void WaterVTKWriter::add_point_data(const std::string& name, int num_components) {
    point_fields_[name] = FieldDef{num_components, {}};
}

void WaterVTKWriter::add_cell_data(const std::string& name, int num_components) {
    cell_fields_[name] = FieldDef{num_components, {}};
}

void WaterVTKWriter::set_point_data(const std::string& name,
                                     const std::vector<VecX>& element_data) {
    auto it = point_fields_.find(name);
    if (it == point_fields_.end()) {
        throw std::runtime_error("WaterVTKWriter: unknown point field '" + name +
                                 "'. Call add_point_data() first.");
    }

    auto remapped = remap_element_data(element_data);

    // Flatten to data vector
    size_t total_size = 0;
    for (const auto& data : remapped) {
        total_size += data.size();
    }

    it->second.data.resize(total_size);
    size_t offset = 0;
    for (const auto& data : remapped) {
        for (Index i = 0; i < data.size(); ++i) {
            it->second.data[offset++] = data(i);
        }
    }
}

void WaterVTKWriter::set_cell_data(const std::string& name,
                                    const std::vector<Real>& element_values) {
    auto it = cell_fields_.find(name);
    if (it == cell_fields_.end()) {
        throw std::runtime_error("WaterVTKWriter: unknown cell field '" + name +
                                 "'. Call add_cell_data() first.");
    }

    it->second.data = remap_cell_data(element_values);
}

std::vector<VecX> WaterVTKWriter::remap_element_data(
    const std::vector<VecX>& original_data) const {

    std::vector<VecX> remapped;
    remapped.reserve(water_elements_.size());

    for (Index mesh_idx : water_elements_) {
        if (mesh_idx < static_cast<Index>(original_data.size())) {
            remapped.push_back(original_data[mesh_idx]);
        }
    }
    return remapped;
}

std::vector<Real> WaterVTKWriter::remap_cell_data(
    const std::vector<Real>& original_data) const {

    std::vector<Real> remapped;
    remapped.reserve(water_elements_.size());

    for (Index mesh_idx : water_elements_) {
        if (mesh_idx < static_cast<Index>(original_data.size())) {
            remapped.push_back(original_data[mesh_idx]);
        }
    }
    return remapped;
}

void WaterVTKWriter::write_timestep(Real time) {
    write(time_idx_++, time);
}

void WaterVTKWriter::write(size_t time_idx, Real time) {
    if (!mesh_) {
        throw std::runtime_error(
            "WaterVTKWriter: mesh not set. Call set_mesh() before writing.");
    }

    std::string filename = get_filename(time_idx);
    write_vtu(filename, time);
    timesteps_.push_back({time, filename});
}

std::string WaterVTKWriter::get_filename(size_t time_idx) const {
    std::ostringstream ss;
    ss << basename_ << "_" << std::setfill('0') << std::setw(6) << time_idx << ".vtu";
    return ss.str();
}

void WaterVTKWriter::write_vtu(const std::string& filename, Real time) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("WaterVTKWriter: failed to open file '" + filename + "'.");
    }

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
         << "byte_order=\"LittleEndian\">\n";

    file << "<UnstructuredGrid>\n";
    file << "<FieldData>\n";
    file << "<DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" "
            "format=\"ascii\">\n";
    file << std::setprecision(15) << time << "\n";
    file << "</DataArray>\n";
    file << "</FieldData>\n";

    file << "<Piece NumberOfPoints=\"" << points_.size() << "\" NumberOfCells=\""
         << cells_.size() << "\">\n";

    // Points
    file << "<Points>\n";
    std::vector<Real> point_data;
    point_data.reserve(points_.size() * 3);
    for (const auto& pt : points_) {
        point_data.push_back(pt(0));
        point_data.push_back(pt(1));
        point_data.push_back(pt(2));
    }
    write_data_array_binary(file, "Points", 3, point_data);
    file << "</Points>\n";

    // Cells
    file << "<Cells>\n";

    // Connectivity
    file << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& cell : cells_) {
        for (Index idx : cell) {
            file << idx << " ";
        }
        file << "\n";
    }
    file << "</DataArray>\n";

    // Offsets
    file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    Index offset = 0;
    for (const auto& cell : cells_) {
        offset += cell.size();
        file << offset << " ";
    }
    file << "\n</DataArray>\n";

    // Types
    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t c = 0; c < cells_.size(); ++c) {
        int cell_type = (order_ == 1) ? 12 : 72;  // VTK_HEXAHEDRON or VTK_LAGRANGE_HEXAHEDRON
        file << cell_type << " ";
    }
    file << "\n</DataArray>\n";

    file << "</Cells>\n";

    // Point data
    if (!point_fields_.empty()) {
        file << "<PointData>\n";
        for (const auto& [name, field] : point_fields_) {
            if (!field.data.empty()) {
                write_data_array_binary(file, name, field.num_components, field.data);
            }
        }
        file << "</PointData>\n";
    }

    // Cell data (always include depth)
    file << "<CellData>\n";
    // Write depth field
    write_data_array_binary(file, "depth", 1, element_depths_);
    // Write user-defined cell fields
    for (const auto& [name, field] : cell_fields_) {
        if (!field.data.empty()) {
            write_data_array_binary(file, name, field.num_components, field.data);
        }
    }
    file << "</CellData>\n";

    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";
}

void WaterVTKWriter::finalize() {
    write_pvd();
}

void WaterVTKWriter::write_pvd() {
    std::string pvd_filename = basename_ + ".pvd";
    std::ofstream file(pvd_filename);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
    file << "<Collection>\n";

    for (const auto& [time, filename] : timesteps_) {
        file << "<DataSet timestep=\"" << std::setprecision(15) << time << "\" file=\""
             << filename << "\"/>\n";
    }

    file << "</Collection>\n";
    file << "</VTKFile>\n";
}

Index WaterVTKWriter::original_element_index(size_t water_idx) const {
    if (water_idx >= water_elements_.size()) {
        return -1;
    }
    return water_elements_[water_idx];
}

Index WaterVTKWriter::water_element_index(Index mesh_idx) const {
    auto it = mesh_to_water_.find(mesh_idx);
    return (it != mesh_to_water_.end()) ? static_cast<Index>(it->second) : -1;
}

bool WaterVTKWriter::is_water_element(Index mesh_idx) const {
    return mesh_to_water_.find(mesh_idx) != mesh_to_water_.end();
}

} // namespace drifter
