#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "core/config_reader.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace drifter;

bool data_files_exist(const DrifterConfig &config) {
  std::string primary_path = config.data_dir + config.primary_file;
  if (!std::filesystem::exists(primary_path)) {
    std::cerr << "Primary bathymetry file not found: " << primary_path << std::endl;
    return false;
  }
  for (const auto &tile : config.tile_files) {
    std::string tile_path = config.data_dir + tile;
    if (!std::filesystem::exists(tile_path)) {
      std::cerr << "Tile file not found: " << tile_path << std::endl;
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {
  std::cout << "====================================\n";
  std::cout << "  DRIFTER - Coastal Ocean Model\n";
  std::cout << "  Adaptive Bathymetry Smoother\n";
  std::cout << "====================================\n\n";

  // Parse command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config.json>\n";
    std::cerr << "\nExample:\n";
    std::cerr << "  " << argv[0] << " config/example.json\n";
    return 1;
  }

  std::string config_path = argv[1];

  // Load configuration
  DrifterConfig config;
  try {
    std::cout << "Loading configuration from: " << config_path << std::endl;
    config = ConfigReader::load(config_path);
  } catch (const std::exception &e) {
    std::cerr << "Error loading config: " << e.what() << "\n";
    return 1;
  }

  // Check data files
  if (!data_files_exist(config)) {
    std::cerr << "\nBathymetry data not available. Exiting.\n";
    return 1;
  }

  // Build full paths for tile files
  std::string primary_path = config.data_dir + config.primary_file;
  std::vector<std::string> tile_paths;
  for (const auto &tile : config.tile_files) {
    tile_paths.push_back(config.data_dir + tile);
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
  Real xmin = config.center_x - config.domain_size / 2;
  Real xmax = config.center_x + config.domain_size / 2;
  Real ymin = config.center_y - config.domain_size / 2;
  Real ymax = config.center_y + config.domain_size / 2;

  std::cout << "\n=== Adaptive CG Cubic Bezier Bathymetry Smoother ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", " << ymax << "]" << std::endl;

  // Create smoother
  AdaptiveCGCubicBezierSmoother smoother(xmin, xmax, ymin, ymax, config.nx, config.ny, config.adaptive);
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
  auto element_size_min = config.domain_size / std::pow(2.0, max_level);

  std::cout << "Number of levels         : " << max_level << std::endl;
  std::cout << "Size of smallest element : " << element_size_min << std::endl;

  // Write VTK output
  smoother.write_vtk(config.output_file, config.vtk_subdivision);
  std::cout << "Output written to        : " << config.output_file << ".vtu" << std::endl;

  std::cout << "\nSimulation complete.\n";
  return 0;
}
