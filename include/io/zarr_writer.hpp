#pragma once

// Zarr v3 Output Writer for DG Ocean Model
//
// Provides cloud-native output via zarrs_ffi (Rust library with C bindings).
// Key features:
// - Zarr v3 format with sharding for efficient chunk access
// - Parallel-safe writes (each rank writes own chunks)
// - Morton-ordered chunks for good spatial locality
// - Compression via blosc2
// - CF-conventions compatible metadata
//
// Usage:
//   ZarrWriter writer("output.zarr", config);
//   writer.add_variable("temperature", {nt, nz, ny, nx}, Float64);
//   writer.initialize();
//   writer.write_timestep(0, mesh, solution);

#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifdef DRIFTER_USE_MPI
#include <mpi.h>
#endif

#ifdef DRIFTER_HAS_ZARR
extern "C" {
#include <zarrs.h>
}
#endif

namespace drifter {

/// @brief Data type for Zarr arrays
enum class ZarrDataType { Float32, Float64, Int32, Int64, UInt8 };

/// @brief Compression codec for Zarr
enum class ZarrCodec {
    None,
    Blosc, // Blosc2 with LZ4
    GZip,
    Zstd
};

/// @brief Chunk ordering for Zarr
enum class ZarrChunkOrder {
    RowMajor,    // C order
    ColumnMajor, // Fortran order
    Morton       // Space-filling curve (good for AMR)
};

/// @brief Configuration for Zarr output
struct ZarrConfig {
    std::string store_path; ///< Path to Zarr store (directory or S3 URL)
    std::vector<size_t> chunk_shape;    ///< Chunk dimensions
    ZarrCodec codec = ZarrCodec::Blosc; ///< Compression codec
    int compression_level = 5;          ///< Compression level (codec-dependent)
    ZarrChunkOrder chunk_order = ZarrChunkOrder::RowMajor;
    bool enable_sharding = true;     ///< Use Zarr v3 sharding
    std::vector<size_t> shard_shape; ///< Shard dimensions (if sharding enabled)

    // Metadata
    std::string title = "DRIFTER Ocean Model Output";
    std::string institution = "";
    std::string source = "DRIFTER 3D DG Coastal Ocean Model";
    std::string conventions = "CF-1.8";
    std::string history = "";
};

/// @brief Variable definition for Zarr array
struct ZarrVariable {
    std::string name;
    std::vector<std::string>
        dimensions; ///< Dimension names (e.g., ["time", "z", "y", "x"])
    std::vector<size_t> shape; ///< Array shape
    ZarrDataType dtype = ZarrDataType::Float64;

    // CF-conventions attributes
    std::string long_name;
    std::string units;
    std::string standard_name;
    Real fill_value = -9999.0;

    // Chunk configuration (overrides global if set)
    std::vector<size_t> chunks;

    // For coordinates
    bool is_coordinate = false;
    std::vector<Real> coordinate_values;
};

/// @brief Dimension definition
struct ZarrDimension {
    std::string name;
    size_t size;
    bool unlimited = false; ///< Can grow (time dimension)
};

/// @brief Zarr v3 writer using zarrs_ffi
class ZarrWriter {
public:
    /// @brief Construct Zarr writer
    /// @param config Writer configuration
    explicit ZarrWriter(const ZarrConfig &config);

    ~ZarrWriter();

    // Prevent copying (owns Zarr handles)
    ZarrWriter(const ZarrWriter &) = delete;
    ZarrWriter &operator=(const ZarrWriter &) = delete;

    // Move semantics
    ZarrWriter(ZarrWriter &&) noexcept;
    ZarrWriter &operator=(ZarrWriter &&) noexcept;

    /// @brief Add a dimension
    void
    add_dimension(const std::string &name, size_t size, bool unlimited = false);

    /// @brief Add a variable
    void add_variable(const ZarrVariable &var);

    /// @brief Add a variable with simple signature
    void add_variable(
        const std::string &name, const std::vector<std::string> &dimensions,
        ZarrDataType dtype = ZarrDataType::Float64);

    /// @brief Set variable attribute
    void set_attribute(
        const std::string &var_name, const std::string &attr_name,
        const std::string &value);

    void set_attribute(
        const std::string &var_name, const std::string &attr_name, Real value);

    /// @brief Initialize the store (create arrays)
    void initialize();

    /// @brief Write coordinate variable (1D)
    void write_coordinate(const std::string &name, const VecX &values);

    /// @brief Write a full variable at given time index
    void
    write_variable(const std::string &name, size_t time_idx, const VecX &data);

    /// @brief Write a 3D field at given time index
    void write_variable_3d(
        const std::string &name, size_t time_idx,
        const std::vector<VecX> &element_data, const OctreeAdapter &mesh);

    /// @brief Write time value
    void write_time(size_t time_idx, Real time_value);

    /// @brief Write a complete timestep (all prognostic variables)
    void write_timestep(
        size_t time_idx, Real time, const OctreeAdapter &mesh,
        const std::vector<VecX> &eta, const std::vector<VecX> &u,
        const std::vector<VecX> &v, const std::vector<VecX> &temperature,
        const std::vector<VecX> &salinity);

    /// @brief Finalize and close (flush metadata)
    void finalize();

    /// @brief Check if store is initialized
    bool is_initialized() const { return initialized_; }

    /// @brief Get current time index
    size_t current_time_index() const { return current_time_idx_; }

#ifdef DRIFTER_USE_MPI
    /// @brief Set MPI communicator for parallel writes
    void set_communicator(MPI_Comm comm);

    /// @brief Write variable in parallel (each rank writes own elements)
    void write_variable_parallel(
        const std::string &name, size_t time_idx,
        const std::vector<VecX> &local_data, const OctreeAdapter &local_mesh);
#endif

private:
    ZarrConfig config_;
    std::map<std::string, ZarrDimension> dimensions_;
    std::map<std::string, ZarrVariable> variables_;
    std::map<std::string, std::map<std::string, std::string>> attributes_;

    bool initialized_ = false;
    size_t current_time_idx_ = 0;

#ifdef DRIFTER_HAS_ZARR
    ZarrsStorage *storage_ = nullptr;
    std::map<std::string, ZarrsArray *> arrays_;
#endif

#ifdef DRIFTER_USE_MPI
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_ = 0;
    int size_ = 1;
#endif

    void create_store();
    void create_array(const ZarrVariable &var);
    void write_metadata();
    void compute_chunk_indices(
        const std::vector<size_t> &global_idx, std::vector<size_t> &chunk_idx,
        std::vector<size_t> &local_idx) const;

    // Convert element DOFs to structured grid for output
    void interpolate_to_output_grid(
        const std::vector<VecX> &element_data, const OctreeAdapter &mesh,
        std::vector<Real> &output_buffer);
};

/// @brief Simple wrapper for common ocean model output
class OceanOutputWriter {
public:
    /// @brief Construct with output path and mesh
    OceanOutputWriter(
        const std::string &path, const OctreeAdapter &mesh,
        int polynomial_order);

    /// @brief Setup standard ocean variables (eta, u, v, T, S, w)
    void setup_standard_variables();

    /// @brief Add tracer variable
    void add_tracer(
        const std::string &name, const std::string &long_name,
        const std::string &units);

    /// @brief Initialize (call after adding all variables)
    void initialize();

    /// @brief Write standard fields at given time
    void write(
        Real time, const std::vector<VecX> &eta, const std::vector<VecX> &u,
        const std::vector<VecX> &v, const std::vector<VecX> &w,
        const std::vector<VecX> &temperature,
        const std::vector<VecX> &salinity);

    /// @brief Finalize output
    void finalize();

private:
    std::unique_ptr<ZarrWriter> writer_;
    const OctreeAdapter &mesh_;
    int order_;
    size_t time_idx_ = 0;

    // Grid dimensions for output
    size_t nx_, ny_, nz_;

    void setup_dimensions();
    void setup_coordinates();
};

#ifndef DRIFTER_HAS_ZARR
// Stub implementation when zarrs_ffi is not available
class ZarrWriterStub {
public:
    explicit ZarrWriterStub(const ZarrConfig &) {
        throw std::runtime_error(
            "Zarr support not compiled (DRIFTER_HAS_ZARR not defined)");
    }
};
#endif

} // namespace drifter
