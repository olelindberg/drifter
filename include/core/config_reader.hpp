#pragma once

/// @file config_reader.hpp
/// @brief JSON configuration file reader for DRIFTER bathymetry smoother

#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "core/types.hpp"
#include <string>
#include <vector>

namespace drifter {

/// @brief Top-level configuration for DRIFTER bathymetry smoother application
struct DrifterConfig {
  // =========================================================================
  // Data paths
  // =========================================================================
  std::string data_dir;                  ///< Base directory for data files
  std::string primary_file;              ///< Primary bathymetry file (relative to data_dir)
  std::vector<std::string> tile_files;   ///< Tile files (relative to data_dir)

  // =========================================================================
  // Domain definition
  // =========================================================================
  Real center_x    = 0.0;      ///< Domain center X coordinate
  Real center_y    = 0.0;      ///< Domain center Y coordinate
  Real domain_size = 1000.0; ///< Domain size (square domain)

  // =========================================================================
  // Initial grid
  // =========================================================================
  int nx = 4; ///< Initial grid elements in X
  int ny = 4; ///< Initial grid elements in Y

  // =========================================================================
  // Adaptive smoother configuration
  // =========================================================================
  AdaptiveCGCubicBezierConfig adaptive;

  // =========================================================================
  // Output
  // =========================================================================
  std::string output_file = "/tmp/drifter_bathymetry"; ///< Output file path (without extension)
  int vtk_subdivision     = 8;                              ///< VTK subdivision level for visualization
};

/// @brief Read and write DRIFTER configuration from/to JSON files
class ConfigReader {
  public:
  /// @brief Load configuration from JSON file
  /// @param filepath Path to JSON config file
  /// @return Parsed configuration with defaults for missing optional fields
  /// @throws std::runtime_error on file not found, parse errors, or missing required fields
  static DrifterConfig load(const std::string &filepath);

  /// @brief Save configuration to JSON file
  /// @param config Configuration to save
  /// @param filepath Output path
  /// @throws std::runtime_error on write errors
  static void save(const DrifterConfig &config, const std::string &filepath);
};

/// @brief Print configuration to stdout
void print_config(const DrifterConfig &config);

} // namespace drifter
