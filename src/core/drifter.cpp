/// @file drifter.cpp
/// @brief Main Drifter application class implementation

#include "core/drifter.hpp"
#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "core/logger.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <vector>

namespace drifter {

Drifter::Drifter(const DrifterConfig &config) : config_(config) {}

bool Drifter::data_files_exist() const {
    std::string primary_path = config_.data_dir + config_.primary_file;
    if (!std::filesystem::exists(primary_path)) {
        LOG_ERROR("Primary bathymetry file not found: " << primary_path);
        return false;
    }
    for (const auto& tile : config_.tile_files) {
        std::string tile_path = config_.data_dir + tile;
        if (!std::filesystem::exists(tile_path)) {
            LOG_ERROR("Tile file not found: " << tile_path);
            return false;
        }
    }

    return true;
}

int Drifter::run() {
    // Check data files
    if (!data_files_exist()) {
        LOG_ERROR("Bathymetry data not available. Exiting.");
        return 1;
    }

    // Build full paths for tile files
    std::string primary_path = config_.data_dir + config_.primary_file;
    std::vector<std::string> tile_paths;
    for (const auto& tile : config_.tile_files) {
        tile_paths.push_back(config_.data_dir + tile);
    }

    // Load multi-source bathymetry
    LOG_INFO("Loading bathymetry data...");
    MultiSourceBathymetry bathymetry(primary_path, tile_paths);

  // Create depth and land mask functions
  auto depth_func = [&bathymetry](Real x, Real y) -> Real {
    try {
      return bathymetry.evaluate(x, y);
    } catch (const std::out_of_range &) {
      return 0.0;
    }
  };

  auto land_mask = [&bathymetry](Real x, Real y) -> bool {
    try {
      return bathymetry.is_land(x, y);
    } catch (const std::out_of_range &) {
      return true;
    }
  };

  // Compute domain bounds
  Real xmin = config_.center_x - config_.domain_size / 2;
  Real xmax = config_.center_x + config_.domain_size / 2;
  Real ymin = config_.center_y - config_.domain_size / 2;
  Real ymax = config_.center_y + config_.domain_size / 2;

    LOG_INFO("=== Adaptive CG Cubic Bezier Bathymetry Smoother ===");
    LOG_INFO("Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", " << ymax << "]");

  // Create smoother
  AdaptiveCGCubicBezierSmoother smoother(xmin, xmax, ymin, ymax, config_.nx, config_.ny, config_.adaptive);
  smoother.set_bathymetry_data(depth_func);
  smoother.set_land_mask(land_mask);

  // Solve with timing
  auto start  = std::chrono::high_resolution_clock::now();
  auto result = smoother.solve_adaptive();
  auto end    = std::chrono::high_resolution_clock::now();

  double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Print results
    LOG_INFO("Final result:");
    LOG_INFO("  Elements: " << result.num_elements);
    LOG_INFO("  Max error: " << result.max_error << " m");
    LOG_INFO("  Mean error: " << result.mean_error << " m");
    LOG_INFO("  Converged: " << (result.converged ? "yes" : "no"));
    LOG_INFO("  Time: " << time_ms << " ms");

  // Compute refinement statistics
  Real max_level = 0.0;
  for (Index i = 0; i < smoother.mesh().num_elements(); ++i) {
    max_level = std::max(max_level, static_cast<Real>(smoother.mesh().element_level(i).max_level()));
  }
  auto element_size_min = config_.domain_size / std::pow(2.0, max_level);

    LOG_INFO("Number of levels         : " << max_level);
    LOG_INFO("Size of smallest element : " << element_size_min);

    // Write VTK output
    smoother.write_vtk(config_.output_file, config_.vtk_subdivision);
    LOG_INFO("Output written to        : " << config_.output_file << ".vtu");

    LOG_INFO("Simulation complete.");
    return 0;
}

} // namespace drifter
