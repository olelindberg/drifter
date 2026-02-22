#pragma once

// Shared fixtures and utilities for integration tests

#include "../test_utils.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <filesystem>
#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace drifter {
namespace testing {

/// @brief Base fixture for integration tests with output directory setup
class SimulationTest : public DrifterTestBase {
protected:
  void SetUp() override {
    DrifterTestBase::SetUp();
    test_output_dir_ = "/tmp/drifter_test_output";
    std::filesystem::create_directories(test_output_dir_);
  }

  void TearDown() override {
    std::filesystem::remove_all(test_output_dir_);
    DrifterTestBase::TearDown();
  }

  std::string test_output_dir_;
};

// =============================================================================
// VTK Utility Functions (for high-order element output)
// =============================================================================

/// @brief Create VTK point storage ordering for Lagrange hexahedra
/// Maps VTK point ID to tensor-product index (8 corners first, then interior)
inline std::vector<int> create_vtk_point_ordering(int order) {
  int n = order + 1;
  std::vector<int> ordering;

  auto tensor_idx = [n](int i, int j, int k) { return i + j * n + k * n * n; };

  // 8 corners first (VTK's corner ordering)
  ordering.push_back(tensor_idx(0, 0, 0));
  ordering.push_back(tensor_idx(order, 0, 0));
  ordering.push_back(tensor_idx(0, order, 0));
  ordering.push_back(tensor_idx(order, order, 0));
  ordering.push_back(tensor_idx(0, 0, order));
  ordering.push_back(tensor_idx(order, 0, order));
  ordering.push_back(tensor_idx(0, order, order));
  ordering.push_back(tensor_idx(order, order, order));

  if (order < 2)
    return ordering;

  // Remaining points in layer order (k varies slowest)
  for (int k = 0; k <= order; ++k) {
    for (int j = 0; j <= order; ++j) {
      for (int i = 0; i <= order; ++i) {
        bool is_corner = (i == 0 || i == order) && (j == 0 || j == order) &&
                         (k == 0 || k == order);
        if (!is_corner) {
          ordering.push_back(tensor_idx(i, j, k));
        }
      }
    }
  }

  return ordering;
}

/// @brief VTK Lagrange hexahedron connectivity ordering for order 3
/// Node ordering VTK expects in the CONNECTIVITY section
const std::vector<int> VTK_LAGRANGE_HEX_CONNECTIVITY_ORDER3 = {
    0,  1,  3,  2,  4,  5,  7,  6,  8,  9,  13, 17, 18, 19, 10, 14,
    52, 53, 57, 61, 62, 63, 54, 58, 20, 36, 23, 39, 35, 51, 32, 48,
    24, 28, 40, 44, 27, 31, 43, 47, 21, 22, 37, 38, 33, 34, 49, 50,
    11, 12, 15, 16, 55, 56, 59, 60, 25, 26, 29, 30, 41, 42, 45, 46};

/// @brief Get VTK Lagrange hex connectivity for given order
/// Currently only order 3 is supported
inline std::vector<int> get_vtk_lagrange_hex_connectivity(int order) {
  if (order == 3) {
    return VTK_LAGRANGE_HEX_CONNECTIVITY_ORDER3;
  }
  return {};
}

// =============================================================================
// Multi-Source Bathymetry Test Fixture
// =============================================================================

/// @brief Test fixture providing multi-source bathymetry access
///
/// This fixture lazily loads multi-source bathymetry data on first access.
/// Provides factory methods for creating depth and land mask functions.
///
/// Usage:
///   class MyTest : public BathymetryTestFixture {};
///   TEST_F(MyTest, SomeTest) {
///       if (!data_files_exist()) {
///           GTEST_SKIP() << "Bathymetry data not available";
///       }
///       auto depth = create_depth_function();
///       auto is_land = create_land_mask();
///       // ... use functions
///   }
class BathymetryTestFixture : public SimulationTest {
protected:
  // Data paths (same as MultiSourceBathymetryTest)
  static inline const std::string data_dir_ =
      "/home/ole/Projects/drifter/data/input/";
  static inline const std::string primary_file_ =
      data_dir_ + "ddm_50m.dybde-emodnet.tif";
  static inline const std::vector<std::string> tile_files_ = {
      data_dir_ + "C4_2024.tif", data_dir_ + "C5_2024.tif",
      data_dir_ + "C6_2024.tif", data_dir_ + "C7_2024.tif",
      data_dir_ + "D4_2024.tif", data_dir_ + "D5_2024.tif",
      data_dir_ + "D6_2024.tif", data_dir_ + "D7_2024.tif",
      data_dir_ + "E4_2024.tif", data_dir_ + "E5_2024.tif",
      data_dir_ + "E6_2024.tif", data_dir_ + "E7_2024.tif",
  };

  /// @brief Check if all required data files exist
  static bool data_files_exist() {
    if (!std::filesystem::exists(primary_file_))
      return false;
    for (const auto &tile : tile_files_) {
      if (!std::filesystem::exists(tile))
        return false;
    }
    return true;
  }

  /// @brief Get the multi-source bathymetry (lazy loaded)
  const MultiSourceBathymetry &bathymetry() const {
    if (!bathy_) {
      bathy_ =
          std::make_unique<MultiSourceBathymetry>(primary_file_, tile_files_);
    }
    return *bathy_;
  }

  /// @brief Create depth function (returns 0 outside domain)
  /// @return Function returning positive depth in meters, 0 for land/outside
  std::function<Real(Real, Real)> create_depth_function() const {
    return [this](Real x, Real y) -> Real {
      try {
        return bathymetry().evaluate(x, y);
      } catch (const std::out_of_range &) {
        return 0.0;
      }
    };
  }

  /// @brief Create land mask function (returns true outside domain)
  /// @return Function returning true if point is land or outside domain
  std::function<bool(Real, Real)> create_land_mask() const {
    return [this](Real x, Real y) -> bool {
      try {
        return bathymetry().is_land(x, y);
      } catch (const std::out_of_range &) {
        return true;
      }
    };
  }

  /// @brief Create gradient function using central differences
  /// @return Function returning (dz/dx, dz/dy) gradient vector
  std::function<Eigen::Vector2d(Real, Real)> create_gradient_function() const {
    return [this](Real x, Real y) -> Eigen::Vector2d {
      constexpr Real h = 1.0; // 1 meter spacing
      try {
        Real dzdx = (bathymetry().evaluate(x + h, y) -
                     bathymetry().evaluate(x - h, y)) /
                    (2 * h);
        Real dzdy = (bathymetry().evaluate(x, y + h) -
                     bathymetry().evaluate(x, y - h)) /
                    (2 * h);
        return Eigen::Vector2d(dzdx, dzdy);
      } catch (const std::out_of_range &) {
        return Eigen::Vector2d::Zero();
      }
    };
  }

  /// @brief Get bathymetry bounds in EPSG:3034 coordinates
  void get_bathymetry_bounds(Real &xmin, Real &xmax, Real &ymin,
                             Real &ymax) const {
    bathymetry().get_bounds(xmin, xmax, ymin, ymax);
  }

private:
  mutable std::unique_ptr<MultiSourceBathymetry> bathy_;
};

// =============================================================================
// Legacy Bathymetry Data Path (deprecated)
// =============================================================================

/// @brief Path to bathymetry GeoTIFF file for tests
/// @deprecated Use BathymetryTestFixture instead for multi-source bathymetry
const std::string BATHYMETRY_GEOTIFF_PATH =
    "/home/ole/Projects/SeaMesh/data/bathymetry-blender/"
    "ddm_emodnet_bathymetry.tif";

} // namespace testing
} // namespace drifter

using namespace drifter;
using namespace drifter::testing;
