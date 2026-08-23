#pragma once

/// @file lowrider_config.hpp
/// @brief Configuration structures for Lowrider adaptive mesh generator

#include "bathymetry/adaptive_smoother_types.hpp"
#include "core/types.hpp"
#include <string>
#include <vector>

namespace drifter {

/// @brief Domain configuration
struct LowriderDomainConfig {
    Real xmin = 0.0;
    Real xmax = 1000.0;
    Real ymin = 0.0;
    Real ymax = 1000.0;
    int initial_nx = 4;
    int initial_ny = 4;
};

/// @brief Refinement configuration
struct LowriderRefinementConfig {
    Real error_threshold = 0.5;      ///< Stop when max error < threshold (meters)
    int max_iterations = 10;         ///< Maximum adaptation iterations
    int max_elements = 10000;        ///< Maximum number of elements
    int max_level = 10;              ///< Maximum refinement level per axis
    Real dorfler_theta = 0.5;        ///< Fraction of total squared error to capture
    ErrorMetricType error_metric = ErrorMetricType::NormalizedError;
    int ngauss = 4;                  ///< Gauss points per direction for error integration

    // Pixel resolution limit
    bool enforce_pixel_limit = true; ///< Stop refining at GeoTIFF pixel resolution
    Real min_element_size = 0.0;     ///< Minimum element size (0 = auto from GeoTIFF)
};

/// @brief VTK writer type selection
enum class VTKWriterType {
    All,   ///< Write all elements (existing VTKWriter)
    Water  ///< Write only water elements (WaterVTKWriter)
};

/// @brief Output configuration
struct LowriderOutputConfig {
    std::string vtk_file = "lowrider_mesh";
    bool write_per_iteration = false;     ///< Write VTK after each iteration
    VTKWriterType vtk_writer_type = VTKWriterType::All;  ///< Type of VTK writer to use
};

/// @brief Data configuration (matches drifter format)
struct LowriderDataConfig {
    std::string data_dir;                      ///< Base directory for data files
    std::string primary_file;                  ///< Primary bathymetry GeoTIFF
    std::vector<std::string> tile_files;       ///< Additional high-resolution tiles
};

/// @brief Coastline refinement configuration
struct LowriderCoastlineConfig {
    std::string file;              ///< Path to shapefile/GeoPackage (empty = disabled)
    std::string layer;             ///< Layer name (optional, defaults to first)
    std::string srs;               ///< Target SRS (e.g., "EPSG:3034")
    int max_level = 10;            ///< Max refinement level near coastline
    Real min_polygon_area = 0.0;   ///< Filter small polygons (0 = no filter)

    bool enabled() const { return !file.empty(); }
};

/// @brief Main configuration structure
struct LowriderConfig {
    LowriderDataConfig data;                   ///< Multi-source bathymetry data
    LowriderDomainConfig domain;
    LowriderRefinementConfig refinement;
    LowriderCoastlineConfig coastline;         ///< Coastline refinement (optional)
    LowriderOutputConfig output;
    bool verbose = true;
};

/// @brief Result of adaptive mesh generation
struct LowriderAdaptiveResult {
    Index num_elements;       ///< Final number of elements
    Real max_error;           ///< Maximum error (meters)
    Real mean_error;          ///< Mean error (meters)
    int iterations;           ///< Number of iterations performed
    bool converged;           ///< True if error threshold reached
    std::string convergence_reason;
};

/// @brief Print configuration to stdout
void print_lowrider_config(const LowriderConfig& config);

} // namespace drifter
