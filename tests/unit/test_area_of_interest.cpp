#include <gtest/gtest.h>
#include "mesh/area_of_interest.hpp"
#include "mesh/octree_adapter.hpp"
#include "../test_utils.hpp"

using namespace drifter;
using namespace drifter::testing;

class AreaOfInterestTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
        // Create a simple 4x4x1 mesh
        mesh_ = std::make_unique<OctreeAdapter>(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);
        mesh_->build_uniform(4, 4, 1);
    }

    std::unique_ptr<OctreeAdapter> mesh_;
};

// =============================================================================
// Basic Region Definition Tests
// =============================================================================

TEST_F(AreaOfInterestTest, DefineRegionWKT) {
    AreaOfInterest aoi;

    // Define a region covering the center of the domain
    int bit = aoi.define_region("center",
        "POLYGON((25 25, 75 25, 75 75, 25 75, 25 25))");

    EXPECT_EQ(bit, 0);
    EXPECT_TRUE(aoi.has_region("center"));
    EXPECT_EQ(aoi.num_regions(), 1);
}

TEST_F(AreaOfInterestTest, DefineRegionPoints) {
    AreaOfInterest aoi;

    // Define a region using point vector
    std::vector<std::pair<Real, Real>> points = {
        {25, 25}, {75, 25}, {75, 75}, {25, 75}
    };
    int bit = aoi.define_region("center", points);

    EXPECT_EQ(bit, 0);
    EXPECT_TRUE(aoi.has_region("center"));
}

TEST_F(AreaOfInterestTest, DefineMultipleRegions) {
    AreaOfInterest aoi;

    int bit1 = aoi.define_region("left",
        "POLYGON((0 0, 50 0, 50 100, 0 100, 0 0))");
    int bit2 = aoi.define_region("right",
        "POLYGON((50 0, 100 0, 100 100, 50 100, 50 0))");

    EXPECT_EQ(bit1, 0);
    EXPECT_EQ(bit2, 1);
    EXPECT_EQ(aoi.num_regions(), 2);
}

TEST_F(AreaOfInterestTest, MaxRegionsLimit) {
    AreaOfInterest aoi;

    // Define 8 regions (max allowed)
    for (int i = 0; i < 8; ++i) {
        std::string name = "region" + std::to_string(i);
        aoi.define_region(name,
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))");
    }

    EXPECT_EQ(aoi.num_regions(), 8);

    // 9th region should throw
    EXPECT_THROW(
        aoi.define_region("region8", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"),
        std::runtime_error);
}

TEST_F(AreaOfInterestTest, InvalidWKTThrows) {
    AreaOfInterest aoi;

    EXPECT_THROW(
        aoi.define_region("invalid", "NOT_A_POLYGON"),
        std::runtime_error);
}

TEST_F(AreaOfInterestTest, DuplicateRegionNameThrows) {
    AreaOfInterest aoi;

    aoi.define_region("region1", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))");

    EXPECT_THROW(
        aoi.define_region("region1", "POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))"),
        std::runtime_error);
}

// =============================================================================
// Membership Computation Tests
// =============================================================================

TEST_F(AreaOfInterestTest, RebuildComputesMembership) {
    AreaOfInterest aoi;

    // Region covering lower-left quadrant (0-50, 0-50)
    aoi.define_region("lower_left",
        "POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))");
    aoi.rebuild(*mesh_);

    // Count elements in region
    Index count = aoi.count_in_region("lower_left");
    EXPECT_GT(count, 0);
    EXPECT_LT(count, mesh_->num_elements());
}

TEST_F(AreaOfInterestTest, ElementCenterDeterminesMembership) {
    AreaOfInterest aoi;

    // Region covering exactly one element's center
    // With 4x4 grid, each element is 25x25. First element center is at (12.5, 12.5)
    aoi.define_region("single",
        "POLYGON((10 10, 15 10, 15 15, 10 15, 10 10))");
    aoi.rebuild(*mesh_);

    Index count = aoi.count_in_region("single");
    EXPECT_EQ(count, 1);
}

TEST_F(AreaOfInterestTest, NoElementsOutsideRegion) {
    AreaOfInterest aoi;

    // Region completely outside the domain
    aoi.define_region("outside",
        "POLYGON((200 200, 300 200, 300 300, 200 300, 200 200))");
    aoi.rebuild(*mesh_);

    Index count = aoi.count_in_region("outside");
    EXPECT_EQ(count, 0);
}

TEST_F(AreaOfInterestTest, AllElementsInsideLargeRegion) {
    AreaOfInterest aoi;

    // Region covering entire domain
    aoi.define_region("all",
        "POLYGON((-10 -10, 110 -10, 110 110, -10 110, -10 -10))");
    aoi.rebuild(*mesh_);

    Index count = aoi.count_in_region("all");
    EXPECT_EQ(count, mesh_->num_elements());
}

// =============================================================================
// Query Tests
// =============================================================================

TEST_F(AreaOfInterestTest, IsInRegionByName) {
    AreaOfInterest aoi;

    aoi.define_region("left_half",
        "POLYGON((0 0, 50 0, 50 100, 0 100, 0 0))");
    aoi.rebuild(*mesh_);

    // At least some elements should be in region
    bool found_in = false;
    bool found_out = false;
    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        if (aoi.is_in_region(e, "left_half")) {
            found_in = true;
        } else {
            found_out = true;
        }
    }

    EXPECT_TRUE(found_in);
    EXPECT_TRUE(found_out);
}

TEST_F(AreaOfInterestTest, IsInRegionByBit) {
    AreaOfInterest aoi;

    int bit = aoi.define_region("test",
        "POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))");
    aoi.rebuild(*mesh_);

    // Results should match between name and bit queries
    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        EXPECT_EQ(aoi.is_in_region(e, "test"), aoi.is_in_region(e, bit));
    }
}

TEST_F(AreaOfInterestTest, GetFlags) {
    AreaOfInterest aoi;

    aoi.define_region("a", "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))");
    aoi.rebuild(*mesh_);

    // All elements should have flag bit 0 set
    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        EXPECT_EQ(aoi.get_flags(e) & 1, 1);
    }
}

TEST_F(AreaOfInterestTest, GetRegionNames) {
    AreaOfInterest aoi;

    // Two overlapping regions
    aoi.define_region("all", "POLYGON((-10 -10, 110 -10, 110 110, -10 110, -10 -10))");
    aoi.define_region("left", "POLYGON((0 0, 50 0, 50 100, 0 100, 0 0))");
    aoi.rebuild(*mesh_);

    // Find an element in left half
    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        Vec3 center = mesh_->element_center(e);
        if (center(0) < 50) {
            auto names = aoi.get_region_names(e);
            EXPECT_EQ(names.size(), 2);
            break;
        }
    }
}

// =============================================================================
// Bulk Query Tests
// =============================================================================

TEST_F(AreaOfInterestTest, ElementsInRegion) {
    AreaOfInterest aoi;

    aoi.define_region("corner",
        "POLYGON((0 0, 25 0, 25 25, 0 25, 0 0))");
    aoi.rebuild(*mesh_);

    auto elements = aoi.elements_in_region("corner");

    // Verify all returned elements are actually in region
    for (Index e : elements) {
        EXPECT_TRUE(aoi.is_in_region(e, "corner"));
    }
}

TEST_F(AreaOfInterestTest, AsCellData) {
    AreaOfInterest aoi;

    aoi.define_region("half",
        "POLYGON((0 0, 50 0, 50 100, 0 100, 0 0))");
    aoi.rebuild(*mesh_);

    auto data = aoi.as_cell_data("half");

    EXPECT_EQ(data.size(), static_cast<size_t>(mesh_->num_elements()));

    // Values should be 0.0 or 1.0
    for (Real v : data) {
        EXPECT_TRUE(v == 0.0 || v == 1.0);
    }
}

TEST_F(AreaOfInterestTest, CombinedRegionIds) {
    AreaOfInterest aoi;

    aoi.define_region("a", "POLYGON((0 0, 50 0, 50 100, 0 100, 0 0))");
    aoi.define_region("b", "POLYGON((50 0, 100 0, 100 100, 50 100, 50 0))");
    aoi.rebuild(*mesh_);

    auto ids = aoi.combined_region_ids();

    EXPECT_EQ(ids.size(), static_cast<size_t>(mesh_->num_elements()));

    // Each element should have either bit 0 or bit 1 set
    for (Real id : ids) {
        int flags = static_cast<int>(id);
        EXPECT_TRUE((flags & 1) != 0 || (flags & 2) != 0);
    }
}

TEST_F(AreaOfInterestTest, ForEachInRegion) {
    AreaOfInterest aoi;

    aoi.define_region("all",
        "POLYGON((-10 -10, 110 -10, 110 110, -10 110, -10 -10))");
    aoi.rebuild(*mesh_);

    Index count = 0;
    aoi.for_each_in_region("all", [&count](Index) {
        ++count;
    });

    EXPECT_EQ(count, mesh_->num_elements());
}

// =============================================================================
// AMR Callback Tests
// =============================================================================

TEST_F(AreaOfInterestTest, OnRefineRecomputesFromGeometry) {
    AreaOfInterest aoi;

    // Region covering left half only
    aoi.define_region("left",
        "POLYGON((0 0, 50 0, 50 100, 0 100, 0 0))");
    aoi.rebuild(*mesh_);

    // Use existing mesh elements as "children" to test recomputation
    // Elements with centers in left half should be in region
    std::vector<Index> test_elements;
    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        test_elements.push_back(e);
    }

    // Clear flags and recompute via on_refine
    aoi.resize(0);
    aoi.on_refine(*mesh_, test_elements);

    // Verify membership matches element center position
    for (Index e : test_elements) {
        Vec3 center = mesh_->element_center(e);
        bool expected_in = center(0) < 50.0;
        EXPECT_EQ(aoi.is_in_region(e, "left"), expected_in);
    }
}

TEST_F(AreaOfInterestTest, OnCoarsenCombinesFlags) {
    AreaOfInterest aoi;

    // Two regions
    aoi.define_region("a", "POLYGON((0 0, 50 0, 50 100, 0 100, 0 0))");
    aoi.define_region("b", "POLYGON((50 0, 100 0, 100 100, 50 100, 50 0))");
    aoi.rebuild(*mesh_);

    // Find children in different regions
    std::vector<Index> children;
    Index child_a = -1, child_b = -1;
    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        if (aoi.is_in_region(e, "a") && child_a < 0) child_a = e;
        if (aoi.is_in_region(e, "b") && child_b < 0) child_b = e;
    }

    if (child_a >= 0 && child_b >= 0) {
        children = {child_a, child_b};
        Index parent = 100;  // New parent index

        aoi.on_coarsen(children, parent);

        // Parent should have both flags
        EXPECT_TRUE(aoi.is_in_region(parent, "a"));
        EXPECT_TRUE(aoi.is_in_region(parent, "b"));
    }
}

// =============================================================================
// Region Management Tests
// =============================================================================

TEST_F(AreaOfInterestTest, RemoveRegion) {
    AreaOfInterest aoi;

    aoi.define_region("test",
        "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))");
    aoi.rebuild(*mesh_);

    EXPECT_TRUE(aoi.has_region("test"));

    aoi.remove_region("test");

    EXPECT_FALSE(aoi.has_region("test"));
    EXPECT_EQ(aoi.num_regions(), 0);
}

TEST_F(AreaOfInterestTest, ClearRegions) {
    AreaOfInterest aoi;

    aoi.define_region("a", "POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))");
    aoi.define_region("b", "POLYGON((50 50, 100 50, 100 100, 50 100, 50 50))");
    aoi.rebuild(*mesh_);

    aoi.clear_regions();

    EXPECT_EQ(aoi.num_regions(), 0);
    EXPECT_FALSE(aoi.has_region("a"));
    EXPECT_FALSE(aoi.has_region("b"));
}

TEST_F(AreaOfInterestTest, RegionNamesReturnsAll) {
    AreaOfInterest aoi;

    aoi.define_region("alpha", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))");
    aoi.define_region("beta", "POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))");
    aoi.define_region("gamma", "POLYGON((40 40, 50 40, 50 50, 40 50, 40 40))");

    auto names = aoi.region_names();

    EXPECT_EQ(names.size(), 3);
}

TEST_F(AreaOfInterestTest, GetRegionIndex) {
    AreaOfInterest aoi;

    aoi.define_region("first", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))");
    aoi.define_region("second", "POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))");

    EXPECT_EQ(aoi.get_region_index("first"), 0);
    EXPECT_EQ(aoi.get_region_index("second"), 1);
    EXPECT_EQ(aoi.get_region_index("nonexistent"), -1);
}
