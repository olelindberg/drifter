#pragma once

// Shared fixtures and utilities for integration tests

#include <gtest/gtest.h>
#include "../test_utils.hpp"
#include <filesystem>
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

    if (order < 2) return ordering;

    // Remaining points in layer order (k varies slowest)
    for (int k = 0; k <= order; ++k) {
        for (int j = 0; j <= order; ++j) {
            for (int i = 0; i <= order; ++i) {
                bool is_corner = (i == 0 || i == order) &&
                                 (j == 0 || j == order) &&
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
    0, 1, 3, 2, 4, 5, 7, 6, 8,
    9, 13, 17, 18, 19, 10, 14, 52, 53,
    57, 61, 62, 63, 54, 58, 20, 36, 23,
    39, 35, 51, 32, 48, 24, 28, 40, 44,
    27, 31, 43, 47, 21, 22, 37, 38, 33,
    34, 49, 50, 11, 12, 15, 16, 55, 56,
    59, 60, 25, 26, 29, 30, 41, 42, 45,
    46
};

/// @brief Get VTK Lagrange hex connectivity for given order
/// Currently only order 3 is supported
inline std::vector<int> get_vtk_lagrange_hex_connectivity(int order) {
    if (order == 3) {
        return VTK_LAGRANGE_HEX_CONNECTIVITY_ORDER3;
    }
    return {};
}

// =============================================================================
// Bathymetry Data Path
// =============================================================================

/// @brief Path to bathymetry GeoTIFF file for tests
const std::string BATHYMETRY_GEOTIFF_PATH =
    "/home/ole/Projects/SeaMesh/data/bathymetry-blender/ddm_emodnet_bathymetry.tif";

}  // namespace testing
}  // namespace drifter

using namespace drifter;
using namespace drifter::testing;
