#pragma once

/// @file drifter.hpp
/// @brief Main Drifter application class

#include "core/config_reader.hpp"

namespace drifter {

/// @brief Main application class for DRIFTER bathymetry smoother
class Drifter {
public:
  /// @brief Construct Drifter with configuration
  /// @param config Application configuration
  explicit Drifter(const DrifterConfig &config);

  /// @brief Run the bathymetry smoothing pipeline
  /// @return 0 on success, non-zero on failure
  int run();

private:
  /// @brief Check if all required data files exist
  /// @return true if all files exist
  bool data_files_exist() const;

  DrifterConfig config_;
};

} // namespace drifter
