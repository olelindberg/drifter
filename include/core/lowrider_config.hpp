#pragma once

/// @file lowrider_config.hpp
/// @brief Configuration structures for Lowrider adaptive mesh generator

#include "core/types.hpp"
#include <string>
#include <vector>

namespace drifter {

/// @brief Error metric type for refinement decisions
enum class ErrorMetricType {
    NormalizedError,  ///< L2 error / sqrt(area) = RMS
    MeanDifference,   ///< integral |z_data - z_surface| dA / area
    VolumeError       ///< integral |z_data - z_surface| dA
};

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
};

/// @brief Output configuration
struct LowriderOutputConfig {
    std::string vtk_file = "lowrider_mesh";
    bool write_per_iteration = false;  ///< Write VTK after each iteration
};

/// @brief Data configuration (matches drifter format)
struct LowriderDataConfig {
    std::string data_dir;                      ///< Base directory for data files
    std::string primary_file;                  ///< Primary bathymetry GeoTIFF
    std::vector<std::string> tile_files;       ///< Additional high-resolution tiles
};

/// @brief Main configuration structure
struct LowriderConfig {
    LowriderDataConfig data;                   ///< Multi-source bathymetry data
    LowriderDomainConfig domain;
    LowriderRefinementConfig refinement;
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
