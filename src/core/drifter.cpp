/// @file drifter.cpp
/// @brief Main Drifter application class implementation

#include "core/drifter.hpp"
#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace drifter {

Drifter::Drifter(const DrifterConfig &config) : config_(config) {}

bool Drifter::data_files_exist() const {
  std::string primary_path = config_.data_dir + config_.primary_file;
  if (!std::filesystem::exists(primary_path)) {
    std::cerr << "Primary bathymetry file not found: " << primary_path << std::endl;
    return false;
  }
  for (const auto &tile : config_.tile_files) {
    std::string tile_path = config_.data_dir + tile;
    if (!std::filesystem::exists(tile_path)) {
      std::cerr << "Tile file not found: " << tile_path << std::endl;
      return false;
    }
  }
  return true;
}

int Drifter::run() {
  // Check data files
  if (!data_files_exist()) {
    std::cerr << "\nBathymetry data not available. Exiting.\n";
    return 1;
  }

  // Build full paths for tile files
  std::string primary_path = config_.data_dir + config_.primary_file;
  std::vector<std::string> tile_paths;
  for (const auto &tile : config_.tile_files) {
    tile_paths.push_back(config_.data_dir + tile);
  }

  // Load multi-source bathymetry
  std::cout << "Loading bathymetry data..." << std::endl;
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

  std::cout << "\n=== Adaptive CG Cubic Bezier Bathymetry Smoother ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", " << ymax << "]" << std::endl;

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
  std::cout << "\nFinal result:" << std::endl;
  std::cout << "  Elements: " << result.num_elements << std::endl;
  std::cout << "  Max error: " << result.max_error << " m" << std::endl;
  std::cout << "  Mean error: " << result.mean_error << " m" << std::endl;
  std::cout << "  Converged: " << (result.converged ? "yes" : "no") << std::endl;
  std::cout << "  Time: " << time_ms << " ms" << std::endl;

  // Compute refinement statistics
  Real max_level = 0.0;
  for (Index i = 0; i < smoother.mesh().num_elements(); ++i) {
    max_level = std::max(max_level, static_cast<Real>(smoother.mesh().element_level(i).max_level()));
  }
  auto element_size_min = config_.domain_size / std::pow(2.0, max_level);

  std::cout << "Number of levels         : " << max_level << std::endl;
  std::cout << "Size of smallest element : " << element_size_min << std::endl;

  // Write VTK output
  smoother.write_vtk(config_.output_file, config_.vtk_subdivision);
  std::cout << "Output written to        : " << config_.output_file << ".vtu" << std::endl;

  std::cout << "\nSimulation complete.\n";
  return 0;
}

} // namespace drifter
