#include "io/vtk_writer.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/bernstein_basis.hpp"
#include <cstring>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace drifter {

// =============================================================================
// Base64 encoding utilities
// =============================================================================

namespace {

const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const unsigned char* data, size_t len) {
    std::string result;
    result.reserve((len + 2) / 3 * 4);

    for (size_t i = 0; i < len; i += 3) {
        unsigned int n = data[i] << 16;
        if (i + 1 < len) n |= data[i + 1] << 8;
        if (i + 2 < len) n |= data[i + 2];

        result += base64_chars[(n >> 18) & 0x3F];
        result += base64_chars[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? base64_chars[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? base64_chars[n & 0x3F] : '=';
    }

    return result;
}

}  // namespace

// =============================================================================
// VTKWriter implementation
// =============================================================================

VTKWriter::VTKWriter(const std::string& basename,
                       VTKFormat format,
                       VTKEncoding encoding)
    : basename_(basename)
    , format_(format)
    , encoding_(encoding)
{
#ifdef DRIFTER_USE_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }
#endif

    // Create output directory if needed
    std::filesystem::path dir = std::filesystem::path(basename_).parent_path();
    if (!dir.empty()) {
        std::filesystem::create_directories(dir);
    }
}

void VTKWriter::set_polynomial_order(int order) {
    order_ = order;
}

void VTKWriter::set_mesh(const OctreeAdapter& mesh) {
    mesh_ = &mesh;
    build_mesh_geometry();
}

void VTKWriter::build_mesh_geometry() {
    if (!mesh_) return;

    const auto& elements = mesh_->elements();

    // For each element, generate nodes based on polynomial order
    int nodes_per_dim = order_ + 1;
    int nodes_per_elem = nodes_per_dim * nodes_per_dim * nodes_per_dim;

    points_.clear();
    cells_.clear();
    cell_types_.clear();

    // Reserve space
    points_.reserve(elements.size() * nodes_per_elem);
    cells_.reserve(elements.size());
    cell_types_.reserve(elements.size());

    Index point_offset = 0;

    for (const auto* node_ptr : elements) {
        // Get element bounding box
        const auto& node = *node_ptr;
        const auto& bounds = node.bounds;
        Vec3 min_corner(bounds.xmin, bounds.ymin, bounds.zmin);
        Vec3 max_corner(bounds.xmax, bounds.ymax, bounds.zmax);
        Vec3 size = max_corner - min_corner;

        // Generate LGL nodes within element
        std::vector<Index> cell_connectivity;
        cell_connectivity.reserve(nodes_per_elem);

        for (int k = 0; k < nodes_per_dim; ++k) {
            Real zeta = (order_ == 1) ? static_cast<Real>(k) :
                        -1.0 + 2.0 * k / order_;

            for (int j = 0; j < nodes_per_dim; ++j) {
                Real eta = (order_ == 1) ? static_cast<Real>(j) :
                           -1.0 + 2.0 * j / order_;

                for (int i = 0; i < nodes_per_dim; ++i) {
                    Real xi = (order_ == 1) ? static_cast<Real>(i) :
                              -1.0 + 2.0 * i / order_;

                    // Map from reference [-1,1]^3 to physical coordinates
                    Vec3 point;
                    point(0) = min_corner(0) + 0.5 * (xi + 1.0) * size(0);
                    point(1) = min_corner(1) + 0.5 * (eta + 1.0) * size(1);
                    point(2) = min_corner(2) + 0.5 * (zeta + 1.0) * size(2);

                    points_.push_back(point);
                    cell_connectivity.push_back(point_offset++);
                }
            }
        }

        // Reorder to VTK ordering if necessary
        if (order_ == 1) {
            // Linear hex: VTK uses different corner ordering
            std::vector<Index> vtk_order = {0, 1, 3, 2, 4, 5, 7, 6};
            std::vector<Index> reordered(8);
            Index base = cell_connectivity[0];
            for (int i = 0; i < 8; ++i) {
                reordered[i] = base + vtk_order[i];
            }
            cells_.push_back(reordered);
            cell_types_.push_back(VTKCellType::Hexahedron);
        } else {
            // High-order: use VTK Lagrange hex
            cells_.push_back(cell_connectivity);
            cell_types_.push_back(VTKCellType::LagrangeHexahedron);
        }
    }
}

void VTKWriter::add_point_data(const std::string& name, int num_components) {
    VTKField field;
    field.name = name;
    field.num_components = num_components;
    field.location = VTKDataLocation::Point;
    point_fields_[name] = field;
}

void VTKWriter::add_cell_data(const std::string& name, int num_components) {
    VTKField field;
    field.name = name;
    field.num_components = num_components;
    field.location = VTKDataLocation::Cell;
    cell_fields_[name] = field;
}

void VTKWriter::set_point_data(const std::string& name, const VecX& data) {
    auto it = point_fields_.find(name);
    if (it == point_fields_.end()) {
        throw std::runtime_error("Unknown point field: " + name);
    }

    it->second.data.resize(data.size());
    for (Index i = 0; i < data.size(); ++i) {
        it->second.data[i] = data(i);
    }
}

void VTKWriter::set_point_data(const std::string& name,
                                const std::vector<VecX>& element_data) {
    auto it = point_fields_.find(name);
    if (it == point_fields_.end()) {
        throw std::runtime_error("Unknown point field: " + name);
    }

    // Flatten element data to point data
    size_t total_size = 0;
    for (const auto& data : element_data) {
        total_size += data.size();
    }

    it->second.data.resize(total_size);
    size_t offset = 0;
    for (const auto& data : element_data) {
        for (Index i = 0; i < data.size(); ++i) {
            it->second.data[offset++] = data(i);
        }
    }
}

void VTKWriter::set_cell_data(const std::string& name, const VecX& data) {
    auto it = cell_fields_.find(name);
    if (it == cell_fields_.end()) {
        throw std::runtime_error("Unknown cell field: " + name);
    }

    it->second.data.resize(data.size());
    for (Index i = 0; i < data.size(); ++i) {
        it->second.data[i] = data(i);
    }
}

void VTKWriter::set_cell_data(const std::string& name,
                               const std::vector<Real>& element_values) {
    auto it = cell_fields_.find(name);
    if (it == cell_fields_.end()) {
        throw std::runtime_error("Unknown cell field: " + name);
    }
    it->second.data = element_values;
}

void VTKWriter::write(size_t time_idx, Real time) {
    std::string filename = get_filename(time_idx);

    switch (format_) {
        case VTKFormat::VTU:
            write_vtu(filename, time);
            break;
        case VTKFormat::PVTU:
            write_pvtu(filename, time);
            break;
        case VTKFormat::Legacy:
            write_legacy(filename, time);
            break;
    }

    timesteps_.push_back({time, filename});
}

void VTKWriter::write_timestep(Real time) {
    write(time_idx_++, time);
}

std::string VTKWriter::get_filename(size_t time_idx) const {
    std::ostringstream ss;
    ss << basename_ << "_" << std::setfill('0') << std::setw(6) << time_idx;

#ifdef DRIFTER_USE_MPI
    if (format_ == VTKFormat::PVTU && size_ > 1) {
        ss << "_" << std::setfill('0') << std::setw(4) << rank_;
    }
#endif

    switch (format_) {
        case VTKFormat::VTU:
        case VTKFormat::PVTU:
            ss << ".vtu";
            break;
        case VTKFormat::Legacy:
            ss << ".vtk";
            break;
    }

    return ss.str();
}

void VTKWriter::write_vtu(const std::string& filename, Real time) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
         << "byte_order=\"LittleEndian\"";
    if (encoding_ == VTKEncoding::Base64) {
        file << " compressor=\"vtkZLibDataCompressor\"";
    }
    file << ">\n";

    file << "<UnstructuredGrid>\n";
    file << "<FieldData>\n";
    file << "<DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">\n";
    file << std::setprecision(15) << time << "\n";
    file << "</DataArray>\n";
    file << "</FieldData>\n";

    file << "<Piece NumberOfPoints=\"" << points_.size()
         << "\" NumberOfCells=\"" << cells_.size() << "\">\n";

    // Points
    file << "<Points>\n";
    std::vector<Real> point_data;
    point_data.reserve(points_.size() * 3);
    for (const auto& pt : points_) {
        point_data.push_back(pt(0));
        point_data.push_back(pt(1));
        point_data.push_back(pt(2));
    }

    if (encoding_ == VTKEncoding::ASCII) {
        write_data_array_ascii(file, "Points", 3, point_data);
    } else if (encoding_ == VTKEncoding::Binary) {
        write_data_array_binary(file, "Points", 3, point_data);
    } else {
        write_data_array_base64(file, "Points", 3, point_data);
    }
    file << "</Points>\n";

    // Cells
    file << "<Cells>\n";

    // Connectivity
    std::vector<Real> connectivity;
    for (const auto& cell : cells_) {
        for (Index idx : cell) {
            connectivity.push_back(static_cast<Real>(idx));
        }
    }
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
    for (VTKCellType type : cell_types_) {
        file << static_cast<int>(type) << " ";
    }
    file << "\n</DataArray>\n";

    file << "</Cells>\n";

    // Point data
    if (!point_fields_.empty()) {
        file << "<PointData>\n";
        for (const auto& [name, field] : point_fields_) {
            if (encoding_ == VTKEncoding::ASCII) {
                write_data_array_ascii(file, name, field.num_components, field.data);
            } else if (encoding_ == VTKEncoding::Binary) {
                write_data_array_binary(file, name, field.num_components, field.data);
            } else {
                write_data_array_base64(file, name, field.num_components, field.data);
            }
        }
        file << "</PointData>\n";
    }

    // Cell data
    if (!cell_fields_.empty()) {
        file << "<CellData>\n";
        for (const auto& [name, field] : cell_fields_) {
            if (encoding_ == VTKEncoding::ASCII) {
                write_data_array_ascii(file, name, field.num_components, field.data);
            } else if (encoding_ == VTKEncoding::Binary) {
                write_data_array_binary(file, name, field.num_components, field.data);
            } else {
                write_data_array_base64(file, name, field.num_components, field.data);
            }
        }
        file << "</CellData>\n";
    }

    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";
}

void VTKWriter::write_pvtu(const std::string& filename, Real time) {
#ifdef DRIFTER_USE_MPI
    // Each rank writes its piece
    write_vtu(filename, time);

    // Rank 0 writes the PVTU master file
    if (rank_ == 0) {
        std::string pvtu_filename = basename_ + "_" +
            std::to_string(time_idx_) + ".pvtu";

        std::ofstream file(pvtu_filename);
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" "
             << "byte_order=\"LittleEndian\">\n";
        file << "<PUnstructuredGrid>\n";

        file << "<PPoints>\n";
        file << "<PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n";
        file << "</PPoints>\n";

        if (!point_fields_.empty()) {
            file << "<PPointData>\n";
            for (const auto& [name, field] : point_fields_) {
                file << "<PDataArray type=\"Float64\" Name=\"" << name
                     << "\" NumberOfComponents=\"" << field.num_components << "\"/>\n";
            }
            file << "</PPointData>\n";
        }

        if (!cell_fields_.empty()) {
            file << "<PCellData>\n";
            for (const auto& [name, field] : cell_fields_) {
                file << "<PDataArray type=\"Float64\" Name=\"" << name
                     << "\" NumberOfComponents=\"" << field.num_components << "\"/>\n";
            }
            file << "</PCellData>\n";
        }

        // List all pieces
        for (int r = 0; r < size_; ++r) {
            std::ostringstream piece_name;
            piece_name << basename_ << "_" << std::setfill('0') << std::setw(6) << time_idx_
                       << "_" << std::setfill('0') << std::setw(4) << r << ".vtu";
            file << "<Piece Source=\"" << piece_name.str() << "\"/>\n";
        }

        file << "</PUnstructuredGrid>\n";
        file << "</VTKFile>\n";
    }
#else
    write_vtu(filename, time);
#endif
}

void VTKWriter::write_legacy(const std::string& filename, Real time) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    file << "# vtk DataFile Version 3.0\n";
    file << "DRIFTER Ocean Model Output, t=" << time << "\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // Points
    file << "POINTS " << points_.size() << " double\n";
    for (const auto& pt : points_) {
        file << std::setprecision(15) << pt(0) << " " << pt(1) << " " << pt(2) << "\n";
    }

    // Cells
    size_t total_size = 0;
    for (const auto& cell : cells_) {
        total_size += cell.size() + 1;
    }
    file << "\nCELLS " << cells_.size() << " " << total_size << "\n";
    for (const auto& cell : cells_) {
        file << cell.size();
        for (Index idx : cell) {
            file << " " << idx;
        }
        file << "\n";
    }

    // Cell types
    file << "\nCELL_TYPES " << cell_types_.size() << "\n";
    for (VTKCellType type : cell_types_) {
        file << static_cast<int>(type) << "\n";
    }

    // Point data
    if (!point_fields_.empty()) {
        file << "\nPOINT_DATA " << points_.size() << "\n";
        for (const auto& [name, field] : point_fields_) {
            if (field.num_components == 1) {
                file << "SCALARS " << name << " double 1\n";
                file << "LOOKUP_TABLE default\n";
            } else {
                file << "VECTORS " << name << " double\n";
            }
            for (size_t i = 0; i < field.data.size(); ++i) {
                file << field.data[i];
                if (field.num_components > 1 && (i + 1) % field.num_components == 0) {
                    file << "\n";
                } else {
                    file << " ";
                }
            }
            if (field.num_components == 1) file << "\n";
        }
    }

    // Cell data
    if (!cell_fields_.empty()) {
        file << "\nCELL_DATA " << cells_.size() << "\n";
        for (const auto& [name, field] : cell_fields_) {
            if (field.num_components == 1) {
                file << "SCALARS " << name << " double 1\n";
                file << "LOOKUP_TABLE default\n";
            } else {
                file << "VECTORS " << name << " double\n";
            }
            for (size_t i = 0; i < field.data.size(); ++i) {
                file << field.data[i];
                if (field.num_components > 1 && (i + 1) % field.num_components == 0) {
                    file << "\n";
                } else {
                    file << " ";
                }
            }
            if (field.num_components == 1) file << "\n";
        }
    }
}

void VTKWriter::finalize() {
    write_pvd();
}

void VTKWriter::write_pvd() {
    std::string pvd_filename = basename_ + ".pvd";
    std::ofstream file(pvd_filename);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
    file << "<Collection>\n";

    for (const auto& [time, filename] : timesteps_) {
        file << "<DataSet timestep=\"" << std::setprecision(15) << time
             << "\" file=\"" << filename << "\"/>\n";
    }

    file << "</Collection>\n";
    file << "</VTKFile>\n";
}

void VTKWriter::write_data_array_ascii(std::ostream& out, const std::string& name,
                                         int num_components, const std::vector<Real>& data) {
    out << "<DataArray type=\"Float64\" Name=\"" << name
        << "\" NumberOfComponents=\"" << num_components
        << "\" format=\"ascii\">\n";
    for (size_t i = 0; i < data.size(); ++i) {
        out << std::setprecision(15) << data[i];
        if ((i + 1) % num_components == 0) {
            out << "\n";
        } else {
            out << " ";
        }
    }
    out << "</DataArray>\n";
}

void VTKWriter::write_data_array_binary(std::ostream& out, const std::string& name,
                                          int num_components, const std::vector<Real>& data) {
    // For binary, we still write base64 in XML
    write_data_array_base64(out, name, num_components, data);
}

void VTKWriter::write_data_array_base64(std::ostream& out, const std::string& name,
                                          int num_components, const std::vector<Real>& data) {
    out << "<DataArray type=\"Float64\" Name=\"" << name
        << "\" NumberOfComponents=\"" << num_components
        << "\" format=\"binary\">";

    // Prepend size as 64-bit unsigned integer
    uint64_t size = data.size() * sizeof(Real);
    std::vector<unsigned char> buffer(sizeof(uint64_t) + size);
    std::memcpy(buffer.data(), &size, sizeof(uint64_t));
    std::memcpy(buffer.data() + sizeof(uint64_t), data.data(), size);

    out << base64_encode(buffer.data(), buffer.size());
    out << "</DataArray>\n";
}

#ifdef DRIFTER_USE_MPI
void VTKWriter::set_communicator(MPI_Comm comm) {
    comm_ = comm;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}
#endif

// =============================================================================
// VTKLegacyWriter implementation
// =============================================================================

VTKLegacyWriter::VTKLegacyWriter(const std::string& filename)
    : file_(filename)
{
    if (!file_) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
}

void VTKLegacyWriter::write_points(const std::vector<Vec3>& points) {
    if (!header_written_) {
        file_ << "# vtk DataFile Version 3.0\n";
        file_ << "DRIFTER Ocean Model Output\n";
        file_ << "ASCII\n";
        file_ << "DATASET UNSTRUCTURED_GRID\n";
        header_written_ = true;
    }

    num_points_ = points.size();
    file_ << "POINTS " << num_points_ << " double\n";
    for (const auto& pt : points) {
        file_ << std::setprecision(15) << pt(0) << " " << pt(1) << " " << pt(2) << "\n";
    }
}

void VTKLegacyWriter::write_hexahedra(const std::vector<std::array<Index, 8>>& cells) {
    num_cells_ = cells.size();
    file_ << "\nCELLS " << num_cells_ << " " << num_cells_ * 9 << "\n";
    for (const auto& cell : cells) {
        file_ << "8";
        for (Index idx : cell) {
            file_ << " " << idx;
        }
        file_ << "\n";
    }

    file_ << "\nCELL_TYPES " << num_cells_ << "\n";
    for (size_t i = 0; i < num_cells_; ++i) {
        file_ << "12\n";  // VTK_HEXAHEDRON
    }
}

void VTKLegacyWriter::add_point_scalar(const std::string& name, const VecX& data) {
    if (!point_data_started_) {
        file_ << "\nPOINT_DATA " << num_points_ << "\n";
        point_data_started_ = true;
    }

    file_ << "SCALARS " << name << " double 1\n";
    file_ << "LOOKUP_TABLE default\n";
    for (Index i = 0; i < data.size(); ++i) {
        file_ << std::setprecision(15) << data(i) << "\n";
    }
}

void VTKLegacyWriter::add_point_vector(const std::string& name,
                                         const VecX& u, const VecX& v, const VecX& w) {
    if (!point_data_started_) {
        file_ << "\nPOINT_DATA " << num_points_ << "\n";
        point_data_started_ = true;
    }

    file_ << "VECTORS " << name << " double\n";
    for (Index i = 0; i < u.size(); ++i) {
        file_ << std::setprecision(15) << u(i) << " " << v(i) << " " << w(i) << "\n";
    }
}

void VTKLegacyWriter::add_cell_scalar(const std::string& name, const VecX& data) {
    if (!cell_data_started_) {
        file_ << "\nCELL_DATA " << num_cells_ << "\n";
        cell_data_started_ = true;
    }

    file_ << "SCALARS " << name << " double 1\n";
    file_ << "LOOKUP_TABLE default\n";
    for (Index i = 0; i < data.size(); ++i) {
        file_ << std::setprecision(15) << data(i) << "\n";
    }
}

void VTKLegacyWriter::close() {
    file_.close();
}

// =============================================================================
// OceanVTKWriter implementation
// =============================================================================

OceanVTKWriter::OceanVTKWriter(const std::string& basename,
                                 const OctreeAdapter& mesh,
                                 int polynomial_order)
{
    writer_ = std::make_unique<VTKWriter>(basename, VTKFormat::VTU, VTKEncoding::Binary);
    writer_->set_polynomial_order(polynomial_order);
    writer_->set_mesh(mesh);

    // Add standard ocean fields
    writer_->add_point_data("eta", 1);
    writer_->add_point_data("velocity", 3);
    writer_->add_point_data("temperature", 1);
    writer_->add_point_data("salinity", 1);
}

void OceanVTKWriter::write(Real time,
                            const std::vector<VecX>& eta,
                            const std::vector<VecX>& u,
                            const std::vector<VecX>& v,
                            const std::vector<VecX>& w,
                            const std::vector<VecX>& temperature,
                            const std::vector<VecX>& salinity) {
    // Flatten eta to point data
    writer_->set_point_data("eta", eta);

    // Combine velocity components
    size_t total_points = 0;
    for (const auto& data : u) {
        total_points += data.size();
    }

    std::vector<Real> velocity_data;
    velocity_data.reserve(total_points * 3);

    size_t offset = 0;
    for (size_t e = 0; e < u.size(); ++e) {
        for (Index i = 0; i < u[e].size(); ++i) {
            velocity_data.push_back(u[e](i));
            velocity_data.push_back(v[e](i));
            velocity_data.push_back(w[e](i));
        }
    }

    VTKField vel_field;
    vel_field.data = velocity_data;
    // Set directly via the data structure

    writer_->set_point_data("temperature", temperature);
    writer_->set_point_data("salinity", salinity);

    writer_->write(time_idx_++, time);
}

void OceanVTKWriter::finalize() {
    writer_->finalize();
}

// =============================================================================
// HighOrderVTKWriter implementation
// =============================================================================

HighOrderVTKWriter::HighOrderVTKWriter(const std::string& basename, int polynomial_order)
    : order_(polynomial_order)
{
    writer_ = std::make_unique<VTKWriter>(basename, VTKFormat::VTU, VTKEncoding::Binary);
    writer_->set_polynomial_order(order_);
    compute_vtk_ordering();
}

void HighOrderVTKWriter::set_mesh(const OctreeAdapter& mesh, const HexahedronBasis& basis) {
    basis_ = &basis;
    writer_->set_mesh(mesh);
}

void HighOrderVTKWriter::add_scalar_field(const std::string& name) {
    writer_->add_point_data(name, 1);
}

void HighOrderVTKWriter::add_vector_field(const std::string& name) {
    writer_->add_point_data(name, 3);
}

void HighOrderVTKWriter::set_field(const std::string& name,
                                     const std::vector<VecX>& element_data) {
    // Reorder each element's data from DG to VTK ordering
    std::vector<VecX> reordered_data;
    reordered_data.reserve(element_data.size());

    for (const auto& data : element_data) {
        VecX vtk_data;
        reorder_to_vtk(data, vtk_data);
        reordered_data.push_back(vtk_data);
    }

    writer_->set_point_data(name, reordered_data);
}

void HighOrderVTKWriter::write_timestep(Real time) {
    writer_->write_timestep(time);
}

void HighOrderVTKWriter::finalize() {
    writer_->finalize();
}

void HighOrderVTKWriter::compute_vtk_ordering() {
    // VTK Lagrange hexahedron node ordering
    // See: https://kitware.github.io/vtk-examples/site/VTKFileFormats/
    //
    // VTK order: corners, edges, faces, interior
    // DG order: typically i + j*(p+1) + k*(p+1)^2

    int n = order_ + 1;
    int total = n * n * n;
    vtk_ordering_.resize(total);

    // This is a simplified version; full implementation would handle
    // VTK's specific ordering (corners first, then edges, faces, interior)

    // For now, use identity (assumes DG ordering matches VTK for linear)
    for (int i = 0; i < total; ++i) {
        vtk_ordering_[i] = i;
    }

    // TODO: Implement proper VTK high-order ordering if needed
}

void HighOrderVTKWriter::reorder_to_vtk(const VecX& dg_data, VecX& vtk_data) const {
    vtk_data.resize(dg_data.size());
    for (size_t i = 0; i < vtk_ordering_.size(); ++i) {
        if (vtk_ordering_[i] < dg_data.size()) {
            vtk_data(i) = dg_data(vtk_ordering_[i]);
        }
    }
}

// =============================================================================
// XDMFWriter implementation (stub - requires HDF5)
// =============================================================================

XDMFWriter::XDMFWriter(const std::string& basename)
    : basename_(basename)
    , h5_filename_(basename + ".h5")
    , xdmf_filename_(basename + ".xdmf")
{
}

void XDMFWriter::set_mesh(const OctreeAdapter& mesh, int order) {
    mesh_ = &mesh;
    order_ = order;

    int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
    num_cells_ = mesh.num_elements();
    num_points_ = num_cells_ * nodes_per_elem;
}

void XDMFWriter::add_attribute(const std::string& name, int num_components,
                                 const std::string& center) {
    attributes_[name] = {num_components, center};
}

void XDMFWriter::write_timestep(Real time, const std::map<std::string, VecX>& fields) {
    size_t time_idx = timesteps_.size();
    timesteps_.push_back({time, time_idx});

    // Write fields to HDF5 (requires HDF5 library)
    // For now, this is a stub

    update_xdmf();
}

void XDMFWriter::finalize() {
    update_xdmf();
}

void XDMFWriter::update_xdmf() {
    std::ofstream file(xdmf_filename_);

    file << "<?xml version=\"1.0\" ?>\n";
    file << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    file << "<Xdmf Version=\"3.0\">\n";
    file << "<Domain>\n";

    file << "<Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

    for (const auto& [time, idx] : timesteps_) {
        file << "<Grid Name=\"mesh\" GridType=\"Uniform\">\n";
        file << "<Time Value=\"" << std::setprecision(15) << time << "\"/>\n";

        // Topology
        file << "<Topology TopologyType=\"Hexahedron\" NumberOfElements=\""
             << num_cells_ << "\">\n";
        file << "<DataItem Dimensions=\"" << num_cells_ << " 8\" Format=\"HDF\">"
             << h5_filename_ << ":/mesh/connectivity</DataItem>\n";
        file << "</Topology>\n";

        // Geometry
        file << "<Geometry GeometryType=\"XYZ\">\n";
        file << "<DataItem Dimensions=\"" << num_points_ << " 3\" Format=\"HDF\">"
             << h5_filename_ << ":/mesh/coordinates</DataItem>\n";
        file << "</Geometry>\n";

        // Attributes
        for (const auto& [name, info] : attributes_) {
            const auto& [num_components, center] = info;
            file << "<Attribute Name=\"" << name << "\" AttributeType=\"";
            if (num_components == 1) file << "Scalar";
            else if (num_components == 3) file << "Vector";
            else file << "Tensor";
            file << "\" Center=\"" << center << "\">\n";
            file << "<DataItem Dimensions=\"" << num_points_;
            if (num_components > 1) file << " " << num_components;
            file << "\" Format=\"HDF\">"
                 << h5_filename_ << ":/" << name << "/" << idx << "</DataItem>\n";
            file << "</Attribute>\n";
        }

        file << "</Grid>\n";
    }

    file << "</Grid>\n";
    file << "</Domain>\n";
    file << "</Xdmf>\n";
}

// =============================================================================
// SeabedVTKWriter implementation
// =============================================================================

SeabedVTKWriter::SeabedVTKWriter(const std::string& filename,
                                  SeabedInterpolation method)
    : filename_(filename)
    , method_(method)
{
}

void SeabedVTKWriter::set_interpolation_method(SeabedInterpolation method) {
    method_ = method;
    // Reset interpolator so it gets recreated with new method
    interpolator_.reset();
}

const SeabedInterpolator& SeabedVTKWriter::get_interpolator() const {
    if (!interpolator_ || interpolator_->order() != order_ ||
        interpolator_->method() != method_) {
        interpolator_ = std::make_unique<SeabedInterpolator>(order_, method_);
    }
    return *interpolator_;
}

void SeabedVTKWriter::set_mesh(const OctreeAdapter& mesh,
                                const std::vector<VecX>& element_coords,
                                int order) {
    mesh_ = &mesh;
    element_coords_ = element_coords;
    order_ = order;
}

void SeabedVTKWriter::set_resolution(int resolution) {
    resolution_ = resolution;
}

void SeabedVTKWriter::add_scalar_field(const std::string& name,
                                        const std::vector<VecX>& element_data) {
    scalar_fields_[name] = element_data;
}

Vec3 SeabedVTKWriter::evaluate_point(const VecX& coords, Real xi, Real eta, int order) const {
    // Delegate to the interpolator (uses Lagrange or Bernstein depending on method_)
    (void)order;  // Use member order_ via interpolator
    return get_interpolator().evaluate_point(coords, xi, eta);
}

Real SeabedVTKWriter::evaluate_scalar(const VecX& data, Real xi, Real eta, int order) const {
    // Delegate to the interpolator (uses Lagrange or Bernstein depending on method_)
    (void)order;  // Use member order_ via interpolator
    return get_interpolator().evaluate_scalar(data, xi, eta);
}

void SeabedVTKWriter::write() {
    if (!mesh_ || element_coords_.empty()) {
        throw std::runtime_error("SeabedVTKWriter: mesh and coordinates not set");
    }

    std::string vtk_filename = filename_ + ".vtk";
    std::ofstream vtk_file(vtk_filename);
    if (!vtk_file) {
        throw std::runtime_error("Failed to open file: " + vtk_filename);
    }

    const auto& elements = mesh_->elements();
    size_t num_elements = elements.size();

    // Each element's bottom face is subdivided into resolution_ x resolution_ quads
    int pts_per_face = (resolution_ + 1) * (resolution_ + 1);
    int cells_per_face = resolution_ * resolution_;

    num_points_ = num_elements * pts_per_face;
    num_cells_ = num_elements * cells_per_face;

    // Write VTK header
    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << "DRIFTER High-Resolution Seabed Surface\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";

    // Collect all points by evaluating Lagrange polynomials on each bottom face
    std::vector<Vec3> all_points;
    all_points.reserve(num_points_);

    // Also collect scalar field values if any
    std::map<std::string, std::vector<Real>> field_values;
    for (const auto& [name, _] : scalar_fields_) {
        field_values[name].reserve(num_points_);
    }

    for (size_t e = 0; e < num_elements; ++e) {
        const VecX& coords = element_coords_[e];

        // Generate points on bottom face (zeta = -1) at high resolution
        for (int j = 0; j <= resolution_; ++j) {
            Real eta = -1.0 + 2.0 * j / resolution_;
            for (int i = 0; i <= resolution_; ++i) {
                Real xi = -1.0 + 2.0 * i / resolution_;

                Vec3 pt = evaluate_point(coords, xi, eta, order_);
                all_points.push_back(pt);

                // Evaluate scalar fields
                for (const auto& [name, elem_data] : scalar_fields_) {
                    Real val = evaluate_scalar(elem_data[e], xi, eta, order_);
                    field_values[name].push_back(val);
                }
            }
        }
    }

    // Write points
    vtk_file << "POINTS " << num_points_ << " double\n";
    for (const auto& pt : all_points) {
        vtk_file << std::setprecision(15) << pt(0) << " " << pt(1) << " " << pt(2) << "\n";
    }

    // Write cells (quads)
    vtk_file << "\nCELLS " << num_cells_ << " " << num_cells_ * 5 << "\n";
    for (size_t e = 0; e < num_elements; ++e) {
        Index base = static_cast<Index>(e * pts_per_face);
        int stride = resolution_ + 1;

        for (int j = 0; j < resolution_; ++j) {
            for (int i = 0; i < resolution_; ++i) {
                // Quad corners (counter-clockwise)
                Index p0 = base + j * stride + i;
                Index p1 = base + j * stride + (i + 1);
                Index p2 = base + (j + 1) * stride + (i + 1);
                Index p3 = base + (j + 1) * stride + i;

                vtk_file << "4 " << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
        }
    }

    // Cell types: VTK_QUAD = 9
    vtk_file << "\nCELL_TYPES " << num_cells_ << "\n";
    for (size_t c = 0; c < num_cells_; ++c) {
        vtk_file << "9\n";
    }

    // Write point data
    if (!field_values.empty()) {
        vtk_file << "\nPOINT_DATA " << num_points_ << "\n";
        for (const auto& [name, values] : field_values) {
            vtk_file << "SCALARS " << name << " double 1\n";
            vtk_file << "LOOKUP_TABLE default\n";
            for (Real v : values) {
                vtk_file << std::setprecision(15) << v << "\n";
            }
        }
    }

    vtk_file.close();
}

}  // namespace drifter
