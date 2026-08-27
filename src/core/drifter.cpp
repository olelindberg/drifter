/// @file drifter.cpp
/// @brief Main Drifter application class implementation

#include "core/drifter.hpp"
#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
<<<<<<< HEAD
#include "bathymetry/adaptive_cg_hermite_smoother.hpp"
=======
#include "core/logger.hpp"
>>>>>>> 48e2f3b (Add logger with file/line/function context)
#include "mesh/multi_source_bathymetry.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
<<<<<<< HEAD
#include <functional>
#include <iostream>
#include <string>
=======
>>>>>>> 48e2f3b (Add logger with file/line/function context)
#include <vector>

namespace drifter {

namespace {

/// @brief Build, solve and write one adaptive smoother
///
/// The Bezier and Hermite adaptive smoothers share everything used here: the
/// constructor signature, set_bathymetry_data / set_land_mask and mesh() from
/// AdaptiveCGSmootherBase, solve_adaptive(), and write_vtk(). Their result
/// structs differ in type but not in field names, so one template covers both.
template <typename Smoother, typename Config>
int run_adaptive_smoother(const DrifterConfig &config, const Config &adaptive_config,
                          const std::string &label, Real xmin, Real xmax, Real ymin, Real ymax,
                          const std::function<Real(Real, Real)> &depth_func,
                          const std::function<bool(Real, Real)> &land_mask) {
  std::cout << "\n=== " << label << " ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", " << ymax << "]" << std::endl;

  // Create smoother
  Smoother smoother(xmin, xmax, ymin, ymax, config.nx, config.ny, adaptive_config);
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

  return 0;
}

} // namespace

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

<<<<<<< HEAD
  // Dispatch on the configured smoother family
  int status = 1;
  switch (config_.smoother_kind) {
  case BathySmootherKind::CubicBezier:
    status = run_adaptive_smoother<AdaptiveCGCubicBezierSmoother>(
        config_, config_.adaptive, "Adaptive CG Cubic Bezier Bathymetry Smoother", xmin, xmax, ymin,
        ymax, depth_func, land_mask);
    break;
  case BathySmootherKind::HermiteC0:
  case BathySmootherKind::HermiteC1:
    status = run_adaptive_smoother<AdaptiveCGHermiteSmoother>(
        config_, config_.hermite_adaptive,
        "Adaptive CG " + to_string(config_.smoother_kind) + " Bathymetry Smoother", xmin, xmax,
        ymin, ymax, depth_func, land_mask);
    break;
=======
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
>>>>>>> 48e2f3b (Add logger with file/line/function context)
  }

<<<<<<< HEAD
  if (status != 0) {
    return status;
  }
=======
    LOG_INFO("Number of levels         : " << max_level);
    LOG_INFO("Size of smallest element : " << element_size_min);

    // Write VTK output
    smoother.write_vtk(config_.output_file, config_.vtk_subdivision);
    LOG_INFO("Output written to        : " << config_.output_file << ".vtu");
>>>>>>> 48e2f3b (Add logger with file/line/function context)

    LOG_INFO("Simulation complete.");
    return 0;
}

} // namespace drifter
