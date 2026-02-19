#pragma once

// VTK/VTU Output Writer for DG Ocean Model
//
// Provides visualization-ready output in VTK formats.
// Key features:
// - VTU (VTKUnstructuredGrid) for single timesteps
// - PVTU for parallel output (one piece per MPI rank)
// - PVD collection for time series
// - XDMF for HDF5-based output (optional)
// - High-order element support via VTK Lagrange elements
//
// Usage:
//   VTKWriter writer("output", mesh, order);
//   writer.add_cell_data("velocity", 3);  // 3-component vector
//   writer.add_point_data("temperature", 1);
//   writer.write_timestep(time, solution);

#include "core/types.hpp"
#include "dg/bernstein_basis.hpp"
#include "mesh/octree_adapter.hpp"
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Forward declaration
namespace drifter {
struct BathymetryData;
class SeabedInterpolator;
} // namespace drifter

#ifdef DRIFTER_USE_MPI
#include <mpi.h>
#endif

namespace drifter {

// Forward declarations
class HexahedronBasis;

/// @brief VTK cell types
enum class VTKCellType {
    Hexahedron = 12, // Linear hexahedron (8 nodes)
    LagrangeHexahedron = 72, // Higher-order Lagrange hexahedron
    QuadraticHexahedron = 25 // Serendipity 20-node hex
};

/// @brief VTK data encoding
enum class VTKEncoding { ASCII,
                         Binary,
                         Base64 };

/// @brief VTK output format
enum class VTKFormat {
    VTU, // VTK UnstructuredGrid XML
    PVTU // Parallel VTU (one file per rank)
};

/// @brief Field data location
enum class VTKDataLocation {
    Point, // Node-based data
    Cell // Element-based data
};

/// @brief VTK field definition
struct VTKField {
    std::string name;
    int num_components = 1;
    VTKDataLocation location = VTKDataLocation::Point;
    std::vector<Real> data;
};

/// @brief VTK/VTU writer for unstructured grids
class VTKWriter {
public:
    /// @brief Construct writer with output basename
    /// @param basename Output path without extension (e.g., "output/solution")
    /// @param format Output format
    VTKWriter(const std::string &basename, VTKFormat format = VTKFormat::VTU,
              VTKEncoding encoding = VTKEncoding::Binary);

    ~VTKWriter() = default;

    /// @brief Set polynomial order for high-order output
    void set_polynomial_order(int order);

    /// @brief Set mesh (must be called before write)
    void set_mesh(const OctreeAdapter &mesh);

    /// @brief Add point data field
    void add_point_data(const std::string &name, int num_components = 1);

    /// @brief Add cell data field
    void add_cell_data(const std::string &name, int num_components = 1);

    /// @brief Set point data values
    void set_point_data(const std::string &name, const VecX &data);
    void set_point_data(const std::string &name, const std::vector<VecX> &element_data);

    /// @brief Set cell data values
    void set_cell_data(const std::string &name, const VecX &data);
    void set_cell_data(const std::string &name, const std::vector<Real> &element_values);

    /// @brief Write current state to file
    /// @param time_idx Timestep index (used in filename)
    /// @param time Physical time (stored as field data)
    void write(size_t time_idx, Real time);

    /// @brief Write with automatic timestep indexing
    void write_timestep(Real time);

    /// @brief Finalize and write collection file (PVD)
    void finalize();

    /// @brief Get output filename for given timestep
    std::string get_filename(size_t time_idx) const;

#ifdef DRIFTER_USE_MPI
    /// @brief Set MPI communicator for parallel output
    void set_communicator(MPI_Comm comm);
#endif

private:
    std::string basename_;
    VTKFormat format_;
    VTKEncoding encoding_;
    int order_ = 1;

    const OctreeAdapter* mesh_ = nullptr;
    std::vector<Vec3> points_;
    std::vector<std::vector<Index>> cells_;
    std::vector<VTKCellType> cell_types_;

    std::map<std::string, VTKField> point_fields_;
    std::map<std::string, VTKField> cell_fields_;

    size_t time_idx_ = 0;
    std::vector<std::pair<Real, std::string>> timesteps_; // For PVD

#ifdef DRIFTER_USE_MPI
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_ = 0;
    int size_ = 1;
#endif

    void build_mesh_geometry();
    void write_vtu(const std::string &filename, Real time);
    void write_pvtu(const std::string &filename, Real time);
    void write_pvd();

    // Encoding helpers
    void write_data_array_ascii(std::ostream &out, const std::string &name, int num_components,
                                const std::vector<Real> &data);
    void write_data_array_binary(std::ostream &out, const std::string &name, int num_components,
                                 const std::vector<Real> &data);
    void write_data_array_base64(std::ostream &out, const std::string &name, int num_components,
                                 const std::vector<Real> &data);

    // VTK node ordering for high-order elements
    std::vector<int> get_vtk_node_ordering(int order) const;
};

/// @brief High-order VTK writer using Lagrange elements
class HighOrderVTKWriter {
public:
    HighOrderVTKWriter(const std::string &basename, int polynomial_order);

    void set_mesh(const OctreeAdapter &mesh, const HexahedronBasis &basis);

    void add_scalar_field(const std::string &name);
    void add_vector_field(const std::string &name);

    void set_field(const std::string &name, const std::vector<VecX> &element_data);

    void write_timestep(Real time);

    void finalize();

private:
    std::unique_ptr<VTKWriter> writer_;
    const HexahedronBasis* basis_ = nullptr;
    int order_;

    std::vector<int> vtk_ordering_;

    void compute_vtk_ordering();
    void reorder_to_vtk(const VecX &dg_data, VecX &vtk_data) const;
};

/// @brief XDMF writer for HDF5-based output (lightweight wrapper)
class XDMFWriter {
public:
    XDMFWriter(const std::string &basename);

    void set_mesh(const OctreeAdapter &mesh, int order);

    void add_attribute(const std::string &name, int num_components,
                       const std::string &center = "Node");

    void write_timestep(Real time, const std::map<std::string, VecX> &fields);

    void finalize();

private:
    std::string basename_;
    std::string h5_filename_;
    std::string xdmf_filename_;

    const OctreeAdapter* mesh_ = nullptr;
    int order_ = 1;

    std::vector<std::pair<Real, size_t>> timesteps_;
    std::map<std::string, std::pair<int, std::string>> attributes_;

    size_t num_points_ = 0;
    size_t num_cells_ = 0;

    void write_mesh_to_h5();
    void write_field_to_h5(const std::string &name, size_t time_idx, const VecX &data);
    void update_xdmf();
};

/// @brief Ocean-specific VTK output with standard variables
class OceanVTKWriter {
public:
    OceanVTKWriter(const std::string &basename, const OctreeAdapter &mesh, int polynomial_order);

    void write(Real time, const std::vector<VecX> &eta, const std::vector<VecX> &u,
               const std::vector<VecX> &v, const std::vector<VecX> &w,
               const std::vector<VecX> &temperature, const std::vector<VecX> &salinity);

    void finalize();

private:
    std::unique_ptr<VTKWriter> writer_;
    size_t time_idx_ = 0;
};

/// @brief High-resolution seabed surface VTK writer
/// Extracts bottom faces from hexahedral elements and outputs them at high
/// resolution by evaluating polynomials at many points on each face.
///
/// Supports two interpolation methods:
/// - Lagrange: Standard high-order interpolation (may overshoot/undershoot)
/// - Bernstein: Bounded interpolation with convex hull property (guaranteed
/// bounded)
///
/// For seabed visualization, Bernstein interpolation is recommended to avoid
/// spurious oscillations that can make the seabed appear above the surface.
class SeabedVTKWriter {
public:
    /// @brief Construct writer with output path
    /// @param filename Output VTK filename (without extension)
    /// @param method Interpolation method (default: Bernstein for bounded
    /// output)
    /// @param format Output format (default: VTU for modern XML format)
    SeabedVTKWriter(const std::string &filename,
                    SeabedInterpolation method = SeabedInterpolation::Bernstein,
                    VTKFormat format = VTKFormat::VTU);

    /// @brief Set the mesh and element coordinates
    /// @param mesh The octree mesh adapter
    /// @param element_coords Physical coordinates at each element's DOF nodes
    ///        (vector of size num_elements, each VecX has 3*num_dofs values:
    ///        x,y,z interleaved)
    /// @param order Polynomial order of the DG representation
    void set_mesh(const OctreeAdapter &mesh, const std::vector<VecX> &element_coords, int order);

    /// @brief Set output resolution (subdivisions per face)
    /// @param resolution Number of subdivisions per dimension on each face
    /// (default: 10)
    ///        Each bottom face will be subdivided into resolution x resolution
    ///        quads
    void set_resolution(int resolution);

    /// @brief Set interpolation method
    /// @param method Lagrange (unbounded) or Bernstein (bounded)
    void set_interpolation_method(SeabedInterpolation method);

    /// @brief Get current interpolation method
    SeabedInterpolation interpolation_method() const { return method_; }

    /// @brief Add scalar field data to output
    /// @param name Field name
    /// @param element_data Data at each element's DOF nodes (same structure as
    /// element_coords)
    void add_scalar_field(const std::string &name, const std::vector<VecX> &element_data);

    /// @brief Write the seabed surface to VTK file
    void write();

    /// @brief Get the number of points written
    size_t num_points() const { return num_points_; }

    /// @brief Get the number of cells written
    size_t num_cells() const { return num_cells_; }

private:
    std::string filename_;
    int resolution_ = 10;
    int order_ = 1;
    SeabedInterpolation method_;
    VTKFormat format_;

    const OctreeAdapter* mesh_ = nullptr;
    std::vector<VecX> element_coords_;

    std::map<std::string, std::vector<VecX>> scalar_fields_;

    size_t num_points_ = 0;
    size_t num_cells_ = 0;

    // Interpolator (created lazily when write() is called)
    mutable std::unique_ptr<SeabedInterpolator> interpolator_;

    // Get or create interpolator
    const SeabedInterpolator &get_interpolator() const;

    // Evaluate interpolated point/scalar on bottom face
    Vec3 evaluate_point(const VecX &coords, Real xi, Real eta) const;
    Real evaluate_scalar(const VecX &data, Real xi, Real eta) const;

    // Write using VTU format
    void write_vtu();
};

} // namespace drifter
