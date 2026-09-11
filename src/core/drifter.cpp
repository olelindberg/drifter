/// @file drifter.cpp
/// @brief Main Drifter application class implementation

#include "core/drifter.hpp"
#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "bathymetry/adaptive_cg_hermite_smoother.hpp"
#include "core/logger.hpp"
#include "io/quadtree_vtk_writer.hpp"
#include "io/raster_vtk_writer.hpp"
#include "mesh/coastline_refinement.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <chrono>
#include <memory>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace drifter {

namespace {

/// @brief The square domain the run covers, in the primary source's CRS
struct DomainBounds {
  Real xmin;
  Real xmax;
  Real ymin;
  Real ymax;
};

/// @brief Centre and size from the config, as the corners everything else wants
DomainBounds make_domain_bounds(const DrifterConfig &config) {
  return DomainBounds{config.center_x - config.domain_size / 2, config.center_x + config.domain_size / 2, config.center_y - config.domain_size / 2, config.center_y + config.domain_size / 2};
}

/// @brief The point queries a smoother makes against the bathymetry
///
/// @warning Every callable holds the MultiSourceBathymetry by reference, so the
/// source must outlive the struct.
struct BathymetrySamplers {
  /// Depth, positive downward. 0 for water at sea level, for land and for a
  /// NoData gap alike - which is why the two masks below exist.
  std::function<Real(Real, Real)> depth;

  /// Land or outside every source, used for the land region of the fit
  std::function<bool(Real, Real)> land_mask;

  /// Whether a measurement exists here. A gap is dropped from the fit and
  /// spanned by the smoothness term; pinning it instead would force the surface
  /// from the surrounding depth up to 0 across one element, which is what
  /// produced the spikes.
  std::function<bool(Real, Real)> has_data;

  /// Whether the surface is held at 0 here by a Dirichlet condition
  std::function<bool(Real, Real)> is_land;

  /// Data resolution at a point, so refinement can stop at the pixel limit of
  /// whichever source covers the element (tiles are finer than the primary)
  std::function<Real(Real, Real)> resolution;
};

/// @brief Wrap a bathymetry source as the queries the smoothers take
BathymetrySamplers make_bathymetry_samplers(const MultiSourceBathymetry &bathymetry) {
  BathymetrySamplers samplers;

  samplers.depth = [&bathymetry](Real x, Real y) -> Real {
    try {
      return bathymetry.evaluate(x, y);
    } catch (const std::out_of_range &) {
      return 0.0;
    }
  };

  samplers.land_mask = [&bathymetry](Real x, Real y) -> bool {
    try {
      return bathymetry.is_land(x, y);
    } catch (const std::out_of_range &) {
      return true;
    }
  };

  samplers.has_data = [&bathymetry](Real x, Real y) -> bool { return bathymetry.has_data(x, y); };
  samplers.is_land  = [&bathymetry](Real x, Real y) -> bool { return bathymetry.is_land_point(x, y); };

  samplers.resolution = [&bathymetry](Real x, Real y) -> Real {
    try {
      return bathymetry.get_min_element_size_meters(x, y);
    } catch (const std::out_of_range &) {
      return 0.0;
    }
  };

  return samplers;
}

/// @brief Write the source data at its own resolution, if the config asks for it
///
/// Exists so the fit can be compared against the pixels it was fitted to.
void write_input_raster(const DrifterConfig &config, const MultiSourceBathymetry &bathymetry, const DomainBounds &bounds) {
  if (!config.write_input_raster) {
    return;
  }

  const std::string raster_path = config.output_file + "_input_raster";
  io::write_raster_vts(raster_path, bathymetry.get_primary(), bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax);
  LOG_INFO("Input raster written to  : " << raster_path << ".vts");
}

/// @brief Load the coastline the mesh should resolve, if one was configured
///
/// The domain-filtered overload pushes the bounds into GDAL's spatial filter,
/// which is what makes a global dataset usable here.
///
/// @return the index, null when no coastline is configured; nullopt on failure
std::optional<std::shared_ptr<const CoastlineIndex>> load_coastline_index(const DrifterConfig &config, const DomainBounds &bounds) {
  if (!config.coastline.enabled()) {
    return std::shared_ptr<const CoastlineIndex>{};
  }

  LOG_INFO("Loading coastline: " << config.coastline.file);
  CoastlineReader reader;
  if (!reader.load(config.coastline.file, config.coastline.layer, config.coastline.srs, bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax)) {
    LOG_ERROR("Failed to load coastline: " << reader.last_error());
    return std::nullopt;
  }

  LOG_INFO("Coastline building index ...");
  auto index = reader.build_index(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax);
  LOG_INFO("Coastline index: " << index->num_segments() << " segments, " << index->num_circumradius_points() << " circumradius samples");
  if (index->num_circumradius_points() == 0) {
    LOG_WARNING("Coastline carries no circumradius samples in this domain; the pre-pass "
                "will refine nothing");
  }

  // Written alongside the input raster, and under the same flag: both exist to
  // be overlaid on the fitted surface when checking what the mesh followed.
  if (config.write_input_raster) {
    const std::string coast_path = config.output_file + "_coastline";
    reader.write_vtk(coast_path);
    reader.write_circumradius_comb_vtk(coast_path + "_comb");
    LOG_INFO("Coastline written to     : " << coast_path << ".vtp");
  }

  return std::shared_ptr<const CoastlineIndex>{std::move(index)};
}

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
template <typename Smoother, typename Config> int run_adaptive_smoother(const DrifterConfig &config, const Config &adaptive_config, const std::string &label, const DomainBounds &bounds, const BathymetrySamplers &samplers, int vtk_arg, const std::shared_ptr<const CoastlineIndex> &coastline_index) {
  LOG_INFO("=== " << label << " ===");
  LOG_INFO("Domain: [" << bounds.xmin << ", " << bounds.xmax << "] x [" << bounds.ymin << ", " << bounds.ymax << "]");

  // Create smoother
  Smoother smoother(bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax, config.nx, config.ny, adaptive_config);
  // Before the data: the assembly must know which quadrature points carry no
  // observation, and the Hermite DOF manager needs the land region when it is built.
  smoother.set_data_masks(samplers.has_data, samplers.is_land);
  smoother.set_bathymetry_data(samplers.depth);
  smoother.set_land_mask(samplers.land_mask);

  // Only the Hermite path enforces the data-resolution refinement limits
  if constexpr (requires { smoother.set_resolution_func(samplers.resolution); }) {
    smoother.set_resolution_func(samplers.resolution);
  }

  // Likewise the coastline pre-pass, which respects those same limits
  if constexpr (requires { smoother.set_coastline(coastline_index, 0); }) {
    if (coastline_index) {
      smoother.set_coastline(coastline_index, config.coastline.max_level);

      // Run the pre-pass here rather than leaving it to solve_adaptive(), so the
      // shoreline-driven mesh can be written before any surface is fitted - which
      // is the whole run when max_iterations is 0. The call is idempotent, so
      // solve_adaptive() will not repeat it.
      smoother.refine_coastline();
      if (config.write_input_raster) {
        const std::string mesh_path = config.output_file + "_coastline_mesh";
        QuadtreeVTKWriter().write_mesh_only(mesh_path, smoother.mesh());
        LOG_INFO("Coastline mesh written to: " << mesh_path << ".vtu");
      }
    }
  }

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

  // Refinement statistics, measured on the mesh rather than derived from the
  // domain size (elements may be anisotropic, and nx/ny need not be equal)
  int max_level         = 0;
  Real element_size_min = std::numeric_limits<Real>::max();
  for (Index i = 0; i < smoother.mesh().num_elements(); ++i) {
    max_level        = std::max(max_level, smoother.mesh().element_level(i).max_level());
    const auto &b    = smoother.mesh().element_bounds(i);
    element_size_min = std::min(element_size_min, std::min(b.xmax - b.xmin, b.ymax - b.ymin));
  }

  LOG_INFO("Number of levels         : " << max_level);
  LOG_INFO("Size of smallest element : " << element_size_min << " m");

  // How much data the smallest element actually sees, which is what the
  // min_element_size / min_data_points_per_element limits act on
  const Real resolution = samplers.resolution ? samplers.resolution(0.5 * (bounds.xmin + bounds.xmax), 0.5 * (bounds.ymin + bounds.ymax)) : 0.0;
  if (resolution > 0.0) {
    const Real pts = (element_size_min / resolution) * (element_size_min / resolution);
    LOG_INFO("Data resolution          : " << resolution << " m");
    LOG_INFO("Data points in smallest  : " << pts);
  }

  // Write VTK output. With max_iterations = 0 the run is a coastline pre-pass only:
  // there is no fitted surface to write, and the mesh it produced has already been
  // written above.
  if (smoother.is_solved()) {
    smoother.write_vtk(config.output_file, vtk_arg);
    LOG_INFO("Output written to        : " << config.output_file << ".vtu");
  } else {
    LOG_INFO("No surface was fitted (max_iterations = " << adaptive_config.max_iterations << "); no surface VTK written");
  }

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
  for (const auto &tile : config_.tile_files) {
    std::string tile_path = config_.data_dir + tile;
    if (!std::filesystem::exists(tile_path)) {
      LOG_ERROR("Tile file not found: " << tile_path);
      return false;
    }
  }

    // The coastline path is absolute, not relative to data_dir
  if (config_.coastline.enabled() && !std::filesystem::exists(config_.coastline.file)) {
    LOG_ERROR("Coastline file not found: " << config_.coastline.file);
    return false;
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
  for (const auto &tile : config_.tile_files) {
    tile_paths.push_back(config_.data_dir + tile);
  }

    // Load multi-source bathymetry
  LOG_INFO("Loading bathymetry data...");
  MultiSourceBathymetry bathymetry(primary_path, tile_paths);

  const BathymetrySamplers samplers = make_bathymetry_samplers(bathymetry);
  const DomainBounds bounds         = make_domain_bounds(config_);

  write_input_raster(config_, bathymetry, bounds);

  const auto coastline_index = load_coastline_index(config_, bounds);
  if (!coastline_index) {
    return 1;
  }

  // Dispatch on the configured smoother family
  int status = 1;
  switch (config_.smoother_kind) {
  case BathySmootherKind::CubicBezier:
    status = run_adaptive_smoother<AdaptiveCGCubicBezierSmoother>(config_, config_.adaptive, "Adaptive CG Cubic Bezier Bathymetry Smoother", bounds, samplers, config_.vtk_subdivision, *coastline_index);
    break;
  case BathySmootherKind::HermiteC0:
  case BathySmootherKind::HermiteC1: {
    // The per-iteration writes go through the config, the final one through the
    // argument; both must use the same visual degree.
    AdaptiveCGHermiteConfig hermite_config = config_.hermite_adaptive;
    hermite_config.vtk_order               = config_.vtk_surface_degree;
    status                                 = run_adaptive_smoother<AdaptiveCGHermiteSmoother>(config_, hermite_config, "Adaptive CG " + to_string(config_.smoother_kind) + " Bathymetry Smoother", bounds, samplers, config_.vtk_surface_degree, *coastline_index);
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
