#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace drifter;

// Data paths
const std::string data_dir                = "/home/ole/Projects/drifter/data/input/";
const std::string primary_file            = data_dir + "ddm_50m.dybde-emodnet.tif";
const std::vector<std::string> tile_files = {
    data_dir + "C4_2024.tif", data_dir + "C5_2024.tif", data_dir + "C6_2024.tif", data_dir + "C7_2024.tif", data_dir + "D4_2024.tif", data_dir + "D5_2024.tif", data_dir + "D6_2024.tif", data_dir + "D7_2024.tif", data_dir + "E4_2024.tif", data_dir + "E5_2024.tif", data_dir + "E6_2024.tif", data_dir + "E7_2024.tif",
};

bool data_files_exist() {
  if (!std::filesystem::exists(primary_file)) {
    std::cerr << "Primary bathymetry file not found: " << primary_file << std::endl;
    return false;
  }
  for (const auto &tile : tile_files) {
    if (!std::filesystem::exists(tile)) {
      std::cerr << "Tile file not found: " << tile << std::endl;
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

  // Check data files
  if (!data_files_exist()) {
    std::cerr << "\nBathymetry data not available. Exiting.\n";
    return 1;
  }

  // Load multi-source bathymetry
  std::cout << "Loading bathymetry data..." << std::endl;
  MultiSourceBathymetry bathymetry(primary_file, tile_files);

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

  // Kattegat test area (EPSG:3034)
  Real center_x    = 4095238.0;
  Real center_y    = 3344695.0;
  Real domain_size = 2000000.0;

  Real xmin = center_x - domain_size / 2;
  Real xmax = center_x + domain_size / 2;
  Real ymin = center_y - domain_size / 2;
  Real ymax = center_y + domain_size / 2;

  // Configure adaptive smoother
  AdaptiveCGCubicBezierConfig config;
  config.error_threshold      = 1.0;
  config.error_metric_type    = ErrorMetricType::VolumeChange;
  config.max_iterations       = 1;
  config.max_elements         = 10000;
  config.max_refinement_level = 12;
  config.verbose              = true;
  config.ngauss_error         = 6;
  config.error_output_dir     = "/tmp/adaptive_cg_cubic_errors";

  // Smoother config
  config.smoother_config.lambda               = 10.0;
  config.smoother_config.edge_ngauss          = 4;
  config.smoother_config.use_iterative_solver = true;
  config.smoother_config.use_multigrid        = true;
  config.smoother_config.schur_preconditioner =

      SchurPreconditionerType::BlockDiagApproxCG;
  config.smoother_config.verbose = true;

  // Multigrid config
  config.smoother_config.multigrid_config.smoother_type        = SmootherType::MultiplicativeSchwarz;
  config.smoother_config.multigrid_config.verbose              = true;
  config.smoother_config.multigrid_config.min_tree_level       = 2;
  config.smoother_config.multigrid_config.pre_smoothing        = 2;
  config.smoother_config.multigrid_config.post_smoothing       = 2;
  config.smoother_config.multigrid_config.transfer_strategy    = TransferOperatorStrategy::BezierSubdivision;
  config.smoother_config.multigrid_config.coarse_grid_strategy = CoarseGridStrategy::CachedRediscretization;

  std::cout << "\n=== Adaptive CG Cubic Bezier Bathymetry Smoother ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", " << ymax << "]" << std::endl;

  // Create smoother (4x4 initial grid)
  AdaptiveCGCubicBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4, config);
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
  auto element_size_min = domain_size / std::pow(2.0, max_level);

  std::cout << "Number of levels         : " << max_level << std::endl;
  std::cout << "Size of smallest element : " << element_size_min << std::endl;

  // Write VTK output
  std::string output_file = "/tmp/drifter_bathymetry";
  smoother.write_vtk(output_file, 8);
  std::cout << "Output written to        : " << output_file << ".vtu" << std::endl;

  std::cout << "\nSimulation complete.\n";
  return 0;
}
