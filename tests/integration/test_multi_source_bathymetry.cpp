#include "mesh/multi_source_bathymetry.hpp"
#include "test_integration_fixtures.hpp"
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

using namespace drifter;
using namespace drifter::testing;

// =============================================================================
// Test fixture
// =============================================================================

class MultiSourceBathymetryTest : public ::testing::Test {
protected:
    static constexpr Real TOLERANCE = 1e-6;

    // Data paths
    const std::string data_dir = "/home/ole/Projects/drifter/data/input/";
    const std::string primary_file = data_dir + "ddm_50m.dybde-emodnet.tif";
    const std::vector<std::string> tile_files = {
        data_dir + "C4_2024.tif",
        data_dir + "C5_2024.tif",
        data_dir + "C6_2024.tif",
        data_dir + "C7_2024.tif",
        data_dir + "D4_2024.tif",
        data_dir + "D5_2024.tif",
        data_dir + "D6_2024.tif",
        data_dir + "D7_2024.tif",
        data_dir + "E4_2024.tif",
        data_dir + "E5_2024.tif",
        data_dir + "E6_2024.tif",
        data_dir + "E7_2024.tif",
    };

    bool data_files_exist() const {
        if (!std::filesystem::exists(primary_file)) return false;
        for (const auto& tile : tile_files) {
            if (!std::filesystem::exists(tile)) return false;
        }
        return true;
    }
};

// =============================================================================
// Basic construction tests
// =============================================================================

#ifdef DRIFTER_HAS_GDAL

TEST_F(MultiSourceBathymetryTest, IsAvailable) {
    EXPECT_TRUE(MultiSourceBathymetry::is_available());
}

TEST_F(MultiSourceBathymetryTest, ConstructWithValidFiles) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Data files not available";
    }

    MultiSourceBathymetry bathy(primary_file, tile_files);
    EXPECT_EQ(bathy.num_sources(), 1 + tile_files.size());
}

TEST_F(MultiSourceBathymetryTest, ConstructWithMissingPrimaryThrows) {
    EXPECT_THROW(
        MultiSourceBathymetry("nonexistent.tif", {}),
        std::runtime_error);
}

TEST_F(MultiSourceBathymetryTest, ConstructWithMissingTileThrows) {
    if (!std::filesystem::exists(primary_file)) {
        GTEST_SKIP() << "Primary file not available";
    }

    EXPECT_THROW(
        MultiSourceBathymetry(primary_file, {"nonexistent.tif"}),
        std::runtime_error);
}

// =============================================================================
// Lookup tests
// =============================================================================

TEST_F(MultiSourceBathymetryTest, EvaluateInsidePrimaryDomain) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Data files not available";
    }

    MultiSourceBathymetry bathy(primary_file, tile_files);

    // Get bounds and test center point
    Real xmin, xmax, ymin, ymax;
    bathy.get_bounds(xmin, xmax, ymin, ymax);

    Real cx = 0.5 * (xmin + xmax);
    Real cy = 0.5 * (ymin + ymax);

    // Should not throw - point is inside primary domain
    EXPECT_NO_THROW({
        Real depth = bathy.evaluate(cx, cy);
        // Depth should be non-negative (either water or land)
        EXPECT_GE(depth, 0.0);
    });
}

TEST_F(MultiSourceBathymetryTest, ContainsInsidePrimary) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Data files not available";
    }

    MultiSourceBathymetry bathy(primary_file, tile_files);

    Real xmin, xmax, ymin, ymax;
    bathy.get_bounds(xmin, xmax, ymin, ymax);

    Real cx = 0.5 * (xmin + xmax);
    Real cy = 0.5 * (ymin + ymax);

    EXPECT_TRUE(bathy.contains(cx, cy));
}

TEST_F(MultiSourceBathymetryTest, IsLandReturnsCorrectly) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Data files not available";
    }

    MultiSourceBathymetry bathy(primary_file, tile_files);

    Real xmin, xmax, ymin, ymax;
    bathy.get_bounds(xmin, xmax, ymin, ymax);

    Real cx = 0.5 * (xmin + xmax);
    Real cy = 0.5 * (ymin + ymax);

    // is_land should return true iff depth == 0
    bool is_land = bathy.is_land(cx, cy);
    Real depth = bathy.evaluate(cx, cy);

    if (is_land) {
        EXPECT_EQ(depth, 0.0);
    } else {
        EXPECT_GT(depth, 0.0);
    }
}

TEST_F(MultiSourceBathymetryTest, OutsideAllSourcesThrows) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Data files not available";
    }

    MultiSourceBathymetry bathy(primary_file, tile_files);

    // Point far outside any reasonable domain (EPSG:3034 coordinates)
    EXPECT_THROW(bathy.evaluate(-1e9, -1e9), std::out_of_range);
}

TEST_F(MultiSourceBathymetryTest, ContainsOutsideDomain) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Data files not available";
    }

    MultiSourceBathymetry bathy(primary_file, tile_files);

    // Point far outside
    EXPECT_FALSE(bathy.contains(-1e9, -1e9));
}

// =============================================================================
// BathymetrySource interface test
// =============================================================================

TEST_F(MultiSourceBathymetryTest, WorksAsBathymetrySource) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Data files not available";
    }

    MultiSourceBathymetry bathy(primary_file, tile_files);

    // Cast to BathymetrySource interface
    const BathymetrySource& source = bathy;

    Real xmin, xmax, ymin, ymax;
    bathy.get_bounds(xmin, xmax, ymin, ymax);

    Real cx = 0.5 * (xmin + xmax);
    Real cy = 0.5 * (ymin + ymax);

    // Should work via interface
    Real depth = source.evaluate(cx, cy);
    EXPECT_GE(depth, 0.0);
}

// =============================================================================
// Bounds test
// =============================================================================

TEST_F(MultiSourceBathymetryTest, GetBoundsReturnsValidBounds) {
    if (!data_files_exist()) {
        GTEST_SKIP() << "Data files not available";
    }

    MultiSourceBathymetry bathy(primary_file, tile_files);

    Real xmin, xmax, ymin, ymax;
    bathy.get_bounds(xmin, xmax, ymin, ymax);

    EXPECT_LT(xmin, xmax);
    EXPECT_LT(ymin, ymax);
}

#else  // DRIFTER_HAS_GDAL

TEST_F(MultiSourceBathymetryTest, NotAvailableWithoutGDAL) {
    EXPECT_FALSE(MultiSourceBathymetry::is_available());
}

TEST_F(MultiSourceBathymetryTest, ThrowsWithoutGDAL) {
    EXPECT_THROW(
        MultiSourceBathymetry("any.tif", {}),
        std::runtime_error);
}

#endif  // DRIFTER_HAS_GDAL
