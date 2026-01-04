#include "io/zarr_writer.hpp"
#include <stdexcept>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <filesystem>

namespace drifter {

// =============================================================================
// ZarrWriter implementation
// =============================================================================

ZarrWriter::ZarrWriter(const ZarrConfig& config)
    : config_(config)
{
#ifdef DRIFTER_USE_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }
#endif
}

ZarrWriter::~ZarrWriter() {
    if (initialized_) {
        finalize();
    }

#ifdef DRIFTER_HAS_ZARR
    for (auto& [name, array] : arrays_) {
        if (array) {
            zarrs_array_free(array);
        }
    }
    if (storage_) {
        zarrs_storage_free(storage_);
    }
#endif
}

ZarrWriter::ZarrWriter(ZarrWriter&& other) noexcept
    : config_(std::move(other.config_))
    , dimensions_(std::move(other.dimensions_))
    , variables_(std::move(other.variables_))
    , attributes_(std::move(other.attributes_))
    , initialized_(other.initialized_)
    , current_time_idx_(other.current_time_idx_)
{
#ifdef DRIFTER_HAS_ZARR
    storage_ = other.storage_;
    arrays_ = std::move(other.arrays_);
    other.storage_ = nullptr;
#endif
    other.initialized_ = false;
}

ZarrWriter& ZarrWriter::operator=(ZarrWriter&& other) noexcept {
    if (this != &other) {
        if (initialized_) {
            finalize();
        }
#ifdef DRIFTER_HAS_ZARR
        for (auto& [name, array] : arrays_) {
            if (array) zarrs_array_free(array);
        }
        if (storage_) zarrs_storage_free(storage_);
#endif

        config_ = std::move(other.config_);
        dimensions_ = std::move(other.dimensions_);
        variables_ = std::move(other.variables_);
        attributes_ = std::move(other.attributes_);
        initialized_ = other.initialized_;
        current_time_idx_ = other.current_time_idx_;

#ifdef DRIFTER_HAS_ZARR
        storage_ = other.storage_;
        arrays_ = std::move(other.arrays_);
        other.storage_ = nullptr;
#endif
        other.initialized_ = false;
    }
    return *this;
}

void ZarrWriter::add_dimension(const std::string& name, size_t size, bool unlimited) {
    if (initialized_) {
        throw std::runtime_error("Cannot add dimension after initialization");
    }
    dimensions_[name] = ZarrDimension{name, size, unlimited};
}

void ZarrWriter::add_variable(const ZarrVariable& var) {
    if (initialized_) {
        throw std::runtime_error("Cannot add variable after initialization");
    }
    variables_[var.name] = var;
}

void ZarrWriter::add_variable(const std::string& name,
                               const std::vector<std::string>& dimensions,
                               ZarrDataType dtype) {
    ZarrVariable var;
    var.name = name;
    var.dimensions = dimensions;
    var.dtype = dtype;

    // Compute shape from dimensions
    var.shape.resize(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        auto it = dimensions_.find(dimensions[i]);
        if (it == dimensions_.end()) {
            throw std::runtime_error("Unknown dimension: " + dimensions[i]);
        }
        var.shape[i] = it->second.size;
    }

    add_variable(var);
}

void ZarrWriter::set_attribute(const std::string& var_name,
                                const std::string& attr_name,
                                const std::string& value) {
    attributes_[var_name][attr_name] = value;
}

void ZarrWriter::set_attribute(const std::string& var_name,
                                const std::string& attr_name,
                                Real value) {
    std::ostringstream oss;
    oss << std::setprecision(15) << value;
    attributes_[var_name][attr_name] = oss.str();
}

void ZarrWriter::initialize() {
    if (initialized_) return;

    create_store();

    for (const auto& [name, var] : variables_) {
        create_array(var);
    }

    write_metadata();

    initialized_ = true;
}

void ZarrWriter::create_store() {
#ifdef DRIFTER_HAS_ZARR
    // Create filesystem store
    ZarrsResult result = zarrs_storage_new_filesystem(
        config_.store_path.c_str(),
        &storage_
    );

    if (result != ZARRS_SUCCESS) {
        throw std::runtime_error("Failed to create Zarr store at: " + config_.store_path);
    }
#else
    // Create directory structure manually for fallback JSON metadata
    std::filesystem::create_directories(config_.store_path);
#endif
}

void ZarrWriter::create_array(const ZarrVariable& var) {
#ifdef DRIFTER_HAS_ZARR
    // Determine chunk shape
    std::vector<uint64_t> shape(var.shape.begin(), var.shape.end());
    std::vector<uint64_t> chunks;

    if (!var.chunks.empty()) {
        chunks.assign(var.chunks.begin(), var.chunks.end());
    } else if (!config_.chunk_shape.empty()) {
        chunks.assign(config_.chunk_shape.begin(), config_.chunk_shape.end());
        // Pad or truncate to match array dimensions
        chunks.resize(var.shape.size(), 1);
        // Ensure chunks don't exceed array size
        for (size_t i = 0; i < chunks.size(); ++i) {
            chunks[i] = std::min(chunks[i], shape[i]);
        }
    } else {
        // Default chunking: small in time, larger in space
        chunks.resize(var.shape.size());
        for (size_t i = 0; i < var.shape.size(); ++i) {
            if (i == 0 && var.dimensions[i] == "time") {
                chunks[i] = 1;  // One timestep per chunk
            } else {
                chunks[i] = std::min(static_cast<uint64_t>(64), shape[i]);
            }
        }
    }

    // Data type
    ZarrsDataType dtype;
    switch (var.dtype) {
        case ZarrDataType::Float32: dtype = ZARRS_FLOAT32; break;
        case ZarrDataType::Float64: dtype = ZARRS_FLOAT64; break;
        case ZarrDataType::Int32: dtype = ZARRS_INT32; break;
        case ZarrDataType::Int64: dtype = ZARRS_INT64; break;
        case ZarrDataType::UInt8: dtype = ZARRS_UINT8; break;
        default: dtype = ZARRS_FLOAT64;
    }

    // Create array
    std::string path = "/" + var.name;
    ZarrsArray* array = nullptr;

    ZarrsResult result = zarrs_array_new(
        storage_,
        path.c_str(),
        shape.data(),
        static_cast<size_t>(shape.size()),
        chunks.data(),
        dtype,
        var.fill_value,
        &array
    );

    if (result != ZARRS_SUCCESS) {
        throw std::runtime_error("Failed to create Zarr array: " + var.name);
    }

    arrays_[var.name] = array;
#else
    // Fallback: create metadata files
    std::filesystem::path array_path = std::filesystem::path(config_.store_path) / var.name;
    std::filesystem::create_directories(array_path);

    // Write zarr.json metadata
    std::ofstream meta(array_path / "zarr.json");
    meta << "{\n";
    meta << "  \"zarr_format\": 3,\n";
    meta << "  \"node_type\": \"array\",\n";
    meta << "  \"shape\": [";
    for (size_t i = 0; i < var.shape.size(); ++i) {
        meta << var.shape[i];
        if (i < var.shape.size() - 1) meta << ", ";
    }
    meta << "],\n";
    meta << "  \"data_type\": \"float64\",\n";
    meta << "  \"chunk_grid\": {\n";
    meta << "    \"name\": \"regular\",\n";
    meta << "    \"configuration\": {\"chunk_shape\": [";
    for (size_t i = 0; i < var.shape.size(); ++i) {
        size_t chunk = (i == 0 && var.dimensions[i] == "time") ? 1 :
                       std::min(static_cast<size_t>(64), var.shape[i]);
        meta << chunk;
        if (i < var.shape.size() - 1) meta << ", ";
    }
    meta << "]}\n";
    meta << "  },\n";
    meta << "  \"fill_value\": " << var.fill_value << "\n";
    meta << "}\n";
    meta.close();
#endif
}

void ZarrWriter::write_metadata() {
    // Write root group metadata
    std::filesystem::path root_meta = std::filesystem::path(config_.store_path) / "zarr.json";
    std::ofstream meta(root_meta);
    meta << "{\n";
    meta << "  \"zarr_format\": 3,\n";
    meta << "  \"node_type\": \"group\",\n";
    meta << "  \"attributes\": {\n";
    meta << "    \"title\": \"" << config_.title << "\",\n";
    meta << "    \"institution\": \"" << config_.institution << "\",\n";
    meta << "    \"source\": \"" << config_.source << "\",\n";
    meta << "    \"Conventions\": \"" << config_.conventions << "\",\n";

    // Add creation time
    auto now = std::time(nullptr);
    auto tm = *std::gmtime(&now);
    std::ostringstream time_ss;
    time_ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    meta << "    \"history\": \"Created " << time_ss.str() << " by DRIFTER\"\n";

    meta << "  }\n";
    meta << "}\n";
    meta.close();
}

void ZarrWriter::write_coordinate(const std::string& name, const VecX& values) {
    if (!initialized_) {
        throw std::runtime_error("Writer not initialized");
    }

#ifdef DRIFTER_HAS_ZARR
    auto it = arrays_.find(name);
    if (it == arrays_.end()) {
        throw std::runtime_error("Unknown variable: " + name);
    }

    std::vector<uint64_t> start = {0};
    std::vector<uint64_t> count = {static_cast<uint64_t>(values.size())};

    ZarrsResult result = zarrs_array_write(
        it->second,
        start.data(),
        count.data(),
        values.data(),
        values.size() * sizeof(Real)
    );

    if (result != ZARRS_SUCCESS) {
        throw std::runtime_error("Failed to write coordinate: " + name);
    }
#else
    // Fallback: write binary chunk file
    std::filesystem::path chunk_path =
        std::filesystem::path(config_.store_path) / name / "c" / "0";
    std::filesystem::create_directories(chunk_path.parent_path());

    std::ofstream out(chunk_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              values.size() * sizeof(Real));
#endif
}

void ZarrWriter::write_variable(const std::string& name, size_t time_idx, const VecX& data) {
    if (!initialized_) {
        throw std::runtime_error("Writer not initialized");
    }

    auto var_it = variables_.find(name);
    if (var_it == variables_.end()) {
        throw std::runtime_error("Unknown variable: " + name);
    }

#ifdef DRIFTER_HAS_ZARR
    auto arr_it = arrays_.find(name);
    if (arr_it == arrays_.end()) {
        throw std::runtime_error("Array not found: " + name);
    }

    // Determine hyperslab for writing
    std::vector<uint64_t> start(var_it->second.shape.size(), 0);
    std::vector<uint64_t> count(var_it->second.shape.size());

    if (!var_it->second.dimensions.empty() && var_it->second.dimensions[0] == "time") {
        start[0] = time_idx;
        count[0] = 1;
        size_t spatial_size = 1;
        for (size_t i = 1; i < var_it->second.shape.size(); ++i) {
            count[i] = var_it->second.shape[i];
            spatial_size *= var_it->second.shape[i];
        }
        if (static_cast<size_t>(data.size()) != spatial_size) {
            throw std::runtime_error("Data size mismatch for variable: " + name);
        }
    } else {
        for (size_t i = 0; i < var_it->second.shape.size(); ++i) {
            count[i] = var_it->second.shape[i];
        }
    }

    ZarrsResult result = zarrs_array_write(
        arr_it->second,
        start.data(),
        count.data(),
        data.data(),
        data.size() * sizeof(Real)
    );

    if (result != ZARRS_SUCCESS) {
        throw std::runtime_error("Failed to write variable: " + name);
    }
#else
    // Fallback: write binary chunk files
    std::ostringstream chunk_name;
    chunk_name << "c";
    if (!var_it->second.dimensions.empty() && var_it->second.dimensions[0] == "time") {
        chunk_name << "/" << time_idx;
    }
    chunk_name << "/0";

    std::filesystem::path chunk_path =
        std::filesystem::path(config_.store_path) / name / chunk_name.str();
    std::filesystem::create_directories(chunk_path.parent_path());

    std::ofstream out(chunk_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),
              data.size() * sizeof(Real));
#endif
}

void ZarrWriter::write_variable_3d(const std::string& name, size_t time_idx,
                                     const std::vector<VecX>& element_data,
                                     const OctreeAdapter& mesh) {
    // Interpolate element data to structured output grid
    std::vector<Real> output_buffer;
    interpolate_to_output_grid(element_data, mesh, output_buffer);

    VecX data = Eigen::Map<VecX>(output_buffer.data(), output_buffer.size());
    write_variable(name, time_idx, data);
}

void ZarrWriter::write_time(size_t time_idx, Real time_value) {
    if (!initialized_) {
        throw std::runtime_error("Writer not initialized");
    }

#ifdef DRIFTER_HAS_ZARR
    auto it = arrays_.find("time");
    if (it != arrays_.end()) {
        std::vector<uint64_t> start = {time_idx};
        std::vector<uint64_t> count = {1};

        zarrs_array_write(
            it->second,
            start.data(),
            count.data(),
            &time_value,
            sizeof(Real)
        );
    }
#else
    // Fallback: append to time chunk file
    std::filesystem::path time_path =
        std::filesystem::path(config_.store_path) / "time" / "c" / "0";
    std::filesystem::create_directories(time_path.parent_path());

    std::fstream out(time_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!out) {
        out.open(time_path, std::ios::binary | std::ios::out);
    }
    out.seekp(time_idx * sizeof(Real));
    out.write(reinterpret_cast<const char*>(&time_value), sizeof(Real));
#endif

    current_time_idx_ = time_idx + 1;
}

void ZarrWriter::write_timestep(size_t time_idx, Real time,
                                  const OctreeAdapter& mesh,
                                  const std::vector<VecX>& eta,
                                  const std::vector<VecX>& u,
                                  const std::vector<VecX>& v,
                                  const std::vector<VecX>& temperature,
                                  const std::vector<VecX>& salinity) {
    write_time(time_idx, time);

    if (variables_.count("eta")) {
        // eta is 2D (surface only)
        std::vector<Real> eta_buffer;
        // Extract surface values
        VecX eta_flat(eta.size());
        for (size_t e = 0; e < eta.size(); ++e) {
            // Assuming first element of each VecX is representative
            eta_flat(e) = eta[e].size() > 0 ? eta[e](0) : 0.0;
        }
        write_variable("eta", time_idx, eta_flat);
    }

    if (variables_.count("u")) {
        write_variable_3d("u", time_idx, u, mesh);
    }
    if (variables_.count("v")) {
        write_variable_3d("v", time_idx, v, mesh);
    }
    if (variables_.count("temperature")) {
        write_variable_3d("temperature", time_idx, temperature, mesh);
    }
    if (variables_.count("salinity")) {
        write_variable_3d("salinity", time_idx, salinity, mesh);
    }
}

void ZarrWriter::finalize() {
    if (!initialized_) return;

#ifdef DRIFTER_HAS_ZARR
    // Sync all arrays
    for (auto& [name, array] : arrays_) {
        if (array) {
            zarrs_array_sync(array);
        }
    }
#endif

    // Update time dimension in metadata if it grew
    // (For Zarr v3, this is typically handled automatically)

    initialized_ = false;
}

void ZarrWriter::interpolate_to_output_grid(const std::vector<VecX>& element_data,
                                             const OctreeAdapter& mesh,
                                             std::vector<Real>& output_buffer) {
    // Simple approach: flatten all element data
    // A full implementation would interpolate to a regular grid

    size_t total_size = 0;
    for (const auto& data : element_data) {
        total_size += data.size();
    }

    output_buffer.resize(total_size);
    size_t offset = 0;
    for (const auto& data : element_data) {
        std::copy(data.data(), data.data() + data.size(),
                  output_buffer.begin() + offset);
        offset += data.size();
    }
}

#ifdef DRIFTER_USE_MPI
void ZarrWriter::set_communicator(MPI_Comm comm) {
    comm_ = comm;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}

void ZarrWriter::write_variable_parallel(const std::string& name, size_t time_idx,
                                          const std::vector<VecX>& local_data,
                                          const OctreeAdapter& local_mesh) {
    // Each rank writes its own elements
    // No coordination needed due to Zarr's chunk-based design

    // Get global element indices from Morton codes
    const auto& elements = local_mesh.elements();

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& node = elements[i];
        // Morton code determines chunk index
        // This provides automatic spatial locality

        // Compute chunk indices from Morton code
        // (Implementation depends on chunk layout)

        // Write local data to appropriate chunks
        // zarrs handles concurrent writes to different chunks
    }
}
#endif

// =============================================================================
// OceanOutputWriter implementation
// =============================================================================

OceanOutputWriter::OceanOutputWriter(const std::string& path,
                                       const OctreeAdapter& mesh,
                                       int polynomial_order)
    : mesh_(mesh)
    , order_(polynomial_order)
{
    ZarrConfig config;
    config.store_path = path;
    config.title = "DRIFTER Ocean Model Output";
    config.source = "DRIFTER 3D DG Coastal Ocean Model";
    config.codec = ZarrCodec::Blosc;
    config.compression_level = 5;

    writer_ = std::make_unique<ZarrWriter>(config);

    setup_dimensions();
}

void OceanOutputWriter::setup_dimensions() {
    // Estimate grid size from mesh
    // For DG: each element has (order+1)^3 nodes
    size_t num_elements = mesh_.num_elements();
    size_t nodes_per_elem = (order_ + 1) * (order_ + 1) * (order_ + 1);

    // Get bounding box extents for dimension sizes
    // Use element count as proxy for grid size
    size_t n_per_dim = static_cast<size_t>(std::cbrt(num_elements)) + 1;

    nx_ = n_per_dim * (order_ + 1);
    ny_ = n_per_dim * (order_ + 1);
    nz_ = (order_ + 1);  // Typically fewer vertical levels

    writer_->add_dimension("time", 0, true);  // Unlimited
    writer_->add_dimension("z", nz_);
    writer_->add_dimension("y", ny_);
    writer_->add_dimension("x", nx_);
}

void OceanOutputWriter::setup_coordinates() {
    // Time coordinate
    ZarrVariable time_var;
    time_var.name = "time";
    time_var.dimensions = {"time"};
    time_var.shape = {0};  // Will grow
    time_var.dtype = ZarrDataType::Float64;
    time_var.long_name = "time";
    time_var.units = "seconds since simulation start";
    time_var.is_coordinate = true;
    writer_->add_variable(time_var);

    // Z coordinate (sigma levels)
    ZarrVariable z_var;
    z_var.name = "z";
    z_var.dimensions = {"z"};
    z_var.shape = {nz_};
    z_var.dtype = ZarrDataType::Float64;
    z_var.long_name = "sigma level";
    z_var.units = "1";
    z_var.standard_name = "ocean_sigma_coordinate";
    z_var.is_coordinate = true;
    writer_->add_variable(z_var);

    // Y coordinate
    ZarrVariable y_var;
    y_var.name = "y";
    y_var.dimensions = {"y"};
    y_var.shape = {ny_};
    y_var.dtype = ZarrDataType::Float64;
    y_var.long_name = "y coordinate";
    y_var.units = "m";
    y_var.is_coordinate = true;
    writer_->add_variable(y_var);

    // X coordinate
    ZarrVariable x_var;
    x_var.name = "x";
    x_var.dimensions = {"x"};
    x_var.shape = {nx_};
    x_var.dtype = ZarrDataType::Float64;
    x_var.long_name = "x coordinate";
    x_var.units = "m";
    x_var.is_coordinate = true;
    writer_->add_variable(x_var);
}

void OceanOutputWriter::setup_standard_variables() {
    setup_coordinates();

    // Free surface elevation (2D)
    ZarrVariable eta;
    eta.name = "eta";
    eta.dimensions = {"time", "y", "x"};
    eta.shape = {0, ny_, nx_};
    eta.long_name = "sea surface height";
    eta.units = "m";
    eta.standard_name = "sea_surface_height_above_geoid";
    writer_->add_variable(eta);

    // Velocity components (3D)
    ZarrVariable u;
    u.name = "u";
    u.dimensions = {"time", "z", "y", "x"};
    u.shape = {0, nz_, ny_, nx_};
    u.long_name = "eastward velocity";
    u.units = "m s-1";
    u.standard_name = "eastward_sea_water_velocity";
    writer_->add_variable(u);

    ZarrVariable v;
    v.name = "v";
    v.dimensions = {"time", "z", "y", "x"};
    v.shape = {0, nz_, ny_, nx_};
    v.long_name = "northward velocity";
    v.units = "m s-1";
    v.standard_name = "northward_sea_water_velocity";
    writer_->add_variable(v);

    ZarrVariable w;
    w.name = "w";
    w.dimensions = {"time", "z", "y", "x"};
    w.shape = {0, nz_, ny_, nx_};
    w.long_name = "upward velocity";
    w.units = "m s-1";
    w.standard_name = "upward_sea_water_velocity";
    writer_->add_variable(w);

    // Temperature and salinity (3D)
    ZarrVariable temp;
    temp.name = "temperature";
    temp.dimensions = {"time", "z", "y", "x"};
    temp.shape = {0, nz_, ny_, nx_};
    temp.long_name = "sea water temperature";
    temp.units = "degC";
    temp.standard_name = "sea_water_temperature";
    writer_->add_variable(temp);

    ZarrVariable salt;
    salt.name = "salinity";
    salt.dimensions = {"time", "z", "y", "x"};
    salt.shape = {0, nz_, ny_, nx_};
    salt.long_name = "sea water salinity";
    salt.units = "1e-3";
    salt.standard_name = "sea_water_salinity";
    writer_->add_variable(salt);
}

void OceanOutputWriter::add_tracer(const std::string& name,
                                     const std::string& long_name,
                                     const std::string& units) {
    ZarrVariable var;
    var.name = name;
    var.dimensions = {"time", "z", "y", "x"};
    var.shape = {0, nz_, ny_, nx_};
    var.long_name = long_name;
    var.units = units;
    writer_->add_variable(var);
}

void OceanOutputWriter::initialize() {
    writer_->initialize();

    // Write coordinate values
    VecX sigma_vals(nz_);
    for (size_t k = 0; k < nz_; ++k) {
        sigma_vals(k) = -1.0 + static_cast<Real>(k) / (nz_ - 1);
    }
    writer_->write_coordinate("z", sigma_vals);

    // X and Y coordinates would come from mesh bounding box
    // Placeholder: uniform spacing
    VecX x_vals(nx_), y_vals(ny_);
    for (size_t i = 0; i < nx_; ++i) {
        x_vals(i) = static_cast<Real>(i) * 1000.0;  // 1 km spacing
    }
    for (size_t j = 0; j < ny_; ++j) {
        y_vals(j) = static_cast<Real>(j) * 1000.0;
    }
    writer_->write_coordinate("x", x_vals);
    writer_->write_coordinate("y", y_vals);
}

void OceanOutputWriter::write(Real time,
                                const std::vector<VecX>& eta,
                                const std::vector<VecX>& u,
                                const std::vector<VecX>& v,
                                const std::vector<VecX>& w,
                                const std::vector<VecX>& temperature,
                                const std::vector<VecX>& salinity) {
    writer_->write_time(time_idx_, time);
    writer_->write_variable_3d("eta", time_idx_, eta, mesh_);
    writer_->write_variable_3d("u", time_idx_, u, mesh_);
    writer_->write_variable_3d("v", time_idx_, v, mesh_);
    writer_->write_variable_3d("w", time_idx_, w, mesh_);
    writer_->write_variable_3d("temperature", time_idx_, temperature, mesh_);
    writer_->write_variable_3d("salinity", time_idx_, salinity, mesh_);

    ++time_idx_;
}

void OceanOutputWriter::finalize() {
    writer_->finalize();
}

}  // namespace drifter
