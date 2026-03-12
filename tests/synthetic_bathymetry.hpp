#ifndef DRIFTER_TESTS_SYNTHETIC_BATHYMETRY_HPP
#define DRIFTER_TESTS_SYNTHETIC_BATHYMETRY_HPP

#include "core/types.hpp"
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <string>

namespace drifter::testing::synthetic_bathymetry {

constexpr Real XMIN = 0.0, XMAX = 100.0;
constexpr Real YMIN = 0.0, YMAX = 100.0;

// Background: gentle slope from shallow to deep
inline Real gentle_slope(Real x, Real y) {
  Real nx = (x - XMIN) / (XMAX - XMIN);
  Real ny = (y - YMIN) / (YMAX - YMIN);
  return 50.0 + 100.0 * nx + 50.0 * ny;
}

// Steep submarine canyon cutting across the domain
// Canyon runs roughly north-south with steep walls
inline Real canyon(Real x, Real y) {
  Real nx = (x - XMIN) / (XMAX - XMIN);
  Real ny = (y - YMIN) / (YMAX - YMIN);

  Real depth = gentle_slope(x, y);

  Real canyon_center = 0.3 + 0.05 * std::sin(4.0 * M_PI * ny);
  Real canyon_width = 0.05;
  Real canyon_depth = 150.0;

  Real dist_from_center = std::abs(nx - canyon_center);
  Real wall_steepness = 50.0;
  Real canyon_profile =
      0.5 * (1.0 - std::tanh(wall_steepness * (dist_from_center - canyon_width)));

  return depth + canyon_depth * canyon_profile;
}

// Isolated seamount (underwater mountain)
inline Real seamount(Real x, Real y) {
  Real nx = (x - XMIN) / (XMAX - XMIN);
  Real ny = (y - YMIN) / (YMAX - YMIN);

  Real depth = 200.0;

  Real cx = 0.7, cy = 0.6;
  Real dx = nx - cx, dy = ny - cy;
  Real r = std::sqrt(dx * dx + dy * dy);

  Real seamount_radius = 0.15;
  Real seamount_height = 150.0;
  if (r < seamount_radius) {
    Real profile = 1.0 - (r / seamount_radius);
    profile = std::pow(profile, 0.7);
    depth -= seamount_height * profile;
  }

  return depth;
}

// Ridge system: parallel ridges with varying heights
inline Real ridge_system(Real x, Real y) {
  Real nx = (x - XMIN) / (XMAX - XMIN);
  Real ny = (y - YMIN) / (YMAX - YMIN);

  Real depth = 150.0;

  Real ridge_axis = nx - 0.5 * ny;
  Real ridge_width = 0.08;
  Real ridge_height = 80.0;

  for (int i = -1; i <= 1; ++i) {
    Real offset = i * 0.25;
    Real dist = std::abs(ridge_axis - offset);
    if (dist < ridge_width) {
      Real profile = std::cos(M_PI * dist / (2.0 * ridge_width));
      profile = profile * profile;
      depth -= ridge_height * profile * (1.0 - 0.2 * std::abs(i));
    }
  }

  return depth;
}

// Combined: canyon + seamount (most challenging for uniform grids)
inline Real canyon_and_seamount(Real x, Real y) {
  Real nx = (x - XMIN) / (XMAX - XMIN);
  Real ny = (y - YMIN) / (YMAX - YMIN);

  Real depth = 100.0 + 50.0 * nx;

  Real canyon_center = 0.25;
  Real canyon_width = 0.04;
  Real canyon_depth_val = 120.0;
  Real dist_canyon = std::abs(nx - canyon_center);
  Real canyon_profile =
      0.5 * (1.0 - std::tanh(60.0 * (dist_canyon - canyon_width)));
  depth += canyon_depth_val * canyon_profile;

  Real cx = 0.75, cy = 0.5;
  Real dx = nx - cx, dy = ny - cy;
  Real r = std::sqrt(dx * dx + dy * dy);
  Real seamount_radius = 0.12;
  Real seamount_height = 100.0;
  if (r < seamount_radius) {
    Real profile = std::pow(1.0 - r / seamount_radius, 0.8);
    depth -= seamount_height * profile;
  }

  return depth;
}

// Step function: abrupt depth change (worst case for smoothing)
inline Real shelf_break(Real x, Real y) {
  Real nx = (x - XMIN) / (XMAX - XMIN);
  (void)y;

  Real shelf_edge = 0.4;
  Real transition_width = 0.02;

  Real shallow = 50.0;
  Real deep = 300.0;

  Real transition =
      0.5 * (1.0 + std::tanh((nx - shelf_edge) / transition_width));
  return shallow + (deep - shallow) * transition;
}

// Export a bathymetry function to CSV
inline void export_to_csv(const std::string &filename,
                          std::function<Real(Real, Real)> func, int n = 200) {
  std::ofstream file(filename);
  file << std::setprecision(10);
  file << "x,y,depth\n";

  Real dx = (XMAX - XMIN) / (n - 1);
  Real dy = (YMAX - YMIN) / (n - 1);

  for (int j = 0; j < n; ++j) {
    Real y = YMIN + j * dy;
    for (int i = 0; i < n; ++i) {
      Real x = XMIN + i * dx;
      file << x << "," << y << "," << func(x, y) << "\n";
    }
  }
}

} // namespace drifter::testing::synthetic_bathymetry

#endif // DRIFTER_TESTS_SYNTHETIC_BATHYMETRY_HPP
