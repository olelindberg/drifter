/// @file drifter.cpp
/// @brief Main Drifter application class implementation

#include "core/drifter.hpp"
#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "bathymetry/adaptive_cg_hermite_smoother.hpp"
#include "core/logger.hpp"
#include "io/raster_vtk_writer.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <iostream>
#include <string>
#include <vector>

namespace drifter {

namespace {

/// @brief Build, solve and write one adaptive smoother
///
/// The Bezier and Hermite adaptive smoothers share everything used here: the
/// constructor signature, set_bathymetry_data / set_land_mask and mesh() from
/// AdaptiveCGSmootherBase, solve_adaptive(), and write_vtk(). Their result
/// structs differ in type but not in field names, so one template covers both.
///
/// write_vtk()'s second argument is the one thing that is not shared: for the
/// Bezier path it is a sample resolution, for the Hermite path the polynomial
/// degree of the emitted cells. The caller passes the right one as @p vtk_arg.
template <typename Smoother, typename Config>
int run_adaptive_smoother(const DrifterConfig &config, const Config &adaptive_config,
                          const std::string &label, Real xmin, Real xmax, Real ymin, Real ymax,
                          const std::function<Real(Real, Real)> &depth_func,
                          const std::function<bool(Real, Real)> &land_mask, int vtk_arg,
                          const std::function<Real(Real, Real)> &resolution_func) {
  std::cout << "\n=== " << label << " ===" << std::endl;
  std::cout << "Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", " << ymax << "]" << std::endl;

  // Create smoother
  Smoother smoother(xmin, xmax, ymin, ymax, config.nx, config.ny, adaptive_config);
  smoother.set_bathymetry_data(depth_func);
  smoother.set_land_mask(land_mask);

  // Only the Hermite path enforces the data-resolution refinement limits
  if constexpr (requires { smoother.set_resolution_func(resolution_func); }) {
    smoother.set_resolution_func(resolution_func);
  }

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

  // Refinement statistics, measured on the mesh rather than derived from the
  // domain size (elements may be anisotropic, and nx/ny need not be equal)
  int max_level = 0;
  Real element_size_min = std::numeric_limits<Real>::max();
  for (Index i = 0; i < smoother.mesh().num_elements(); ++i) {
    max_level = std::max(max_level, smoother.mesh().element_level(i).max_level());
    const auto &b = smoother.mesh().element_bounds(i);
    element_size_min = std::min(element_size_min, std::min(b.xmax - b.xmin, b.ymax - b.ymin));
  }

  std::cout << "Number of levels         : " << max_level << std::endl;
  std::cout << "Size of smallest element : " << element_size_min << " m" << std::endl;

  // How much data the smallest element actually sees, which is what the
  // min_element_size / min_data_points_per_element limits act on
  const Real resolution = resolution_func ? resolution_func(0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
                                          : 0.0;
  if (resolution > 0.0) {
    const Real pts = (element_size_min / resolution) * (element_size_min / resolution);
    std::cout << "Data resolution          : " << resolution << " m" << std::endl;
    std::cout << "Data points in smallest  : " << pts << std::endl;
  }

  // Write VTK output
  smoother.write_vtk(config.output_file, vtk_arg);
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

  // Write the source data at its own resolution, so the fit can be compared
  // against the pixels it was fitted to.
  if (config_.write_input_raster) {
    const std::string raster_path = config_.output_file + "_input_raster";
    io::write_raster_vts(raster_path, bathymetry.get_primary(), xmin, xmax, ymin, ymax);
    LOG_INFO("Input raster written to  : " << raster_path << ".vts");
  }

  // Data resolution at a point, so refinement can stop at the pixel limit of
  // whichever source covers the element (tiles are finer than the primary)
  auto resolution_func = [&bathymetry](Real x, Real y) -> Real {
    try {
      return bathymetry.get_min_element_size_meters(x, y);
    } catch (const std::out_of_range &) {
      return 0.0;
    }
  };

  // Dispatch on the configured smoother family
  int status = 1;
  switch (config_.smoother_kind) {
  case BathySmootherKind::CubicBezier:
    status = run_adaptive_smoother<AdaptiveCGCubicBezierSmoother>(
        config_, config_.adaptive, "Adaptive CG Cubic Bezier Bathymetry Smoother", xmin, xmax, ymin,
        ymax, depth_func, land_mask, config_.vtk_subdivision, resolution_func);
    break;
  case BathySmootherKind::HermiteC0:
  case BathySmootherKind::HermiteC1: {
    // The per-iteration writes go through the config, the final one through the
    // argument; both must use the same visual degree.
    AdaptiveCGHermiteConfig hermite_config = config_.hermite_adaptive;
    hermite_config.vtk_order               = config_.vtk_surface_degree;
    status                                 = run_adaptive_smoother<AdaptiveCGHermiteSmoother>(
        config_, hermite_config,
        "Adaptive CG " + to_string(config_.smoother_kind) + " Bathymetry Smoother", xmin, xmax,
        ymin, ymax, depth_func, land_mask, config_.vtk_surface_degree, resolution_func);
    break;
  }
  }

  if (status != 0) {
    return status;
  }

    LOG_INFO("Simulation complete.");
    return 0;
}

} // namespace drifter
