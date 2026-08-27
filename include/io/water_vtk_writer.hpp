#pragma once

// Water-Only VTK Writer for DG Ocean Model
//
// Outputs only water elements (where depth > 0) to VTK format.
// Automatically includes depth as a cell data field for visualization.
//
// Usage:
//   WaterVTKWriter writer("output/water", mesh, order);
//   writer.set_depth_query([&seabed](Real x, Real y) { return seabed.depth(x, y); });
//   writer.add_point_data("velocity", 3);
//   writer.set_point_data("velocity", element_data);  // Uses original element indexing
//   writer.write_timestep(time);

#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace drifter {

/// @brief VTK writer that outputs only water elements (depth > 0)
///
/// Filters 3D hexahedral elements based on bathymetry depth at element centers.
/// Only elements where the water depth exceeds zero are written to output.
/// Depth is automatically included as a cell data field.
///
/// Supports:
/// - VTU format with VTK Lagrange hexahedra for high-order elements
/// - Point data and cell data fields
/// - PVD time series for animation
class WaterVTKWriter {
public:
    /// @brief Depth query function type: (x, y) -> depth at that location
    using DepthQuery = std::function<Real(Real, Real)>;

    /// @brief Construct water VTK writer
    /// @param basename Output path without extension (e.g., "output/water")
    /// @param polynomial_order Polynomial order for high-order output
    WaterVTKWriter(const std::string& basename, int polynomial_order);

    ~WaterVTKWriter() = default;

    /// @brief Set mesh and depth query function
    /// @param mesh The octree mesh (must outlive this writer)
    /// @param depth_query Function (x, y) -> depth; elements with depth > 0 are water
    void set_mesh(const OctreeAdapter& mesh, DepthQuery depth_query);

    /// @brief Add point data field
    void add_point_data(const std::string& name, int num_components = 1);

    /// @brief Add cell data field
    void add_cell_data(const std::string& name, int num_components = 1);

    /// @brief Set point data values from element data
    /// @param name Field name
    /// @param element_data Data indexed by ORIGINAL mesh element index
    void set_point_data(const std::string& name, const std::vector<VecX>& element_data);

    /// @brief Set cell data values
    /// @param name Field name
    /// @param element_values Values indexed by ORIGINAL mesh element index
    void set_cell_data(const std::string& name, const std::vector<Real>& element_values);

    /// @brief Write current state to file with automatic timestep indexing
    void write_timestep(Real time);

    /// @brief Write current state to file with explicit timestep index
    void write(size_t time_idx, Real time);

    /// @brief Finalize and write PVD collection file
    void finalize();

    /// @brief Get number of water elements (after filtering)
    size_t num_water_elements() const { return water_elements_.size(); }

    /// @brief Get original mesh element index for a water element
    Index original_element_index(size_t water_idx) const;

    /// @brief Get water element index for an original mesh element (-1 if filtered out)
    Index water_element_index(Index mesh_idx) const;

    /// @brief Check if a mesh element is a water element
    bool is_water_element(Index mesh_idx) const;

    /// @brief Get all water element indices (original mesh indices)
    const std::vector<Index>& water_elements() const { return water_elements_; }

private:
    std::string basename_;
    int order_ = 1;

    const OctreeAdapter* mesh_ = nullptr;
    DepthQuery depth_query_;

    // Water element tracking
    std::vector<Index> water_elements_;                // water_idx -> mesh_idx
    std::unordered_map<Index, size_t> mesh_to_water_;  // mesh_idx -> water_idx
    std::vector<Real> element_depths_;                 // depth at each water element center

    // Geometry for water elements only
    std::vector<Vec3> points_;
    std::vector<std::vector<Index>> cells_;

    // Field definitions
    struct FieldDef {
        int num_components;
        std::vector<Real> data;
    };
    std::map<std::string, FieldDef> point_fields_;
    std::map<std::string, FieldDef> cell_fields_;

    size_t time_idx_ = 0;
    std::vector<std::pair<Real, std::string>> timesteps_;  // For PVD

    /// @brief Identify water elements based on depth query
    void identify_water_elements();

    /// @brief Build mesh geometry for water elements only
    void build_water_geometry();

    /// @brief Write VTU file
    void write_vtu(const std::string& filename, Real time);

    /// @brief Write PVD collection file
    void write_pvd();

    /// @brief Get output filename for given timestep
    std::string get_filename(size_t time_idx) const;

    /// @brief Remap element data from original to water element indexing
    std::vector<VecX> remap_element_data(const std::vector<VecX>& original_data) const;

    /// @brief Remap cell data from original to water element indexing
    std::vector<Real> remap_cell_data(const std::vector<Real>& original_data) const;
};

} // namespace drifter
