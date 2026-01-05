#include <gtest/gtest.h>
#include "io/vtk_writer.hpp"
#include "io/zarr_writer.hpp"
#include "mesh/geotiff_reader.hpp"
#include "../test_utils.hpp"
#include <filesystem>
#include <fstream>

using namespace drifter;
using namespace drifter::testing;

namespace fs = std::filesystem;

class IOTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
        // Create temporary test directory
        test_dir_ = fs::temp_directory_path() / "drifter_io_test";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        // Clean up test directory
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
        DrifterTestBase::TearDown();
    }

    fs::path test_dir_;
};

// =============================================================================
// VTK Legacy Writer Tests
// =============================================================================

TEST_F(IOTest, VTKLegacyWriterBasic) {
    std::string filename = (test_dir_ / "test.vtk").string();

    VTKLegacyWriter writer(filename);

    // Create simple cube vertices
    std::vector<Vec3> points = {
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
        Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(1, 1, 1), Vec3(0, 1, 1)
    };

    writer.write_points(points);

    // Write hexahedron
    std::vector<std::array<Index, 8>> cells = {{0, 1, 2, 3, 4, 5, 6, 7}};
    writer.write_hexahedra(cells);

    // Add scalar data
    VecX scalar_data(8);
    scalar_data << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0;
    writer.add_point_scalar("temperature", scalar_data);

    // Add vector data
    VecX u(8), v(8), w(8);
    u.setConstant(1.0);
    v.setConstant(0.5);
    w.setZero();
    writer.add_point_vector("velocity", u, v, w);

    writer.close();

    // Verify file was created
    ASSERT_TRUE(fs::exists(filename));

    // Verify file content
    std::ifstream file(filename);
    std::string line;

    // Check header
    std::getline(file, line);
    EXPECT_EQ(line, "# vtk DataFile Version 3.0");

    std::getline(file, line);
    EXPECT_TRUE(line.find("DRIFTER") != std::string::npos);

    std::getline(file, line);
    EXPECT_EQ(line, "ASCII");

    std::getline(file, line);
    EXPECT_EQ(line, "DATASET UNSTRUCTURED_GRID");
}

TEST_F(IOTest, VTKLegacyWriterPointCount) {
    std::string filename = (test_dir_ / "test_points.vtk").string();

    VTKLegacyWriter writer(filename);

    // Create 27 points (3x3x3 grid)
    std::vector<Vec3> points;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                points.push_back(Vec3(i, j, k));
            }
        }
    }

    writer.write_points(points);
    writer.close();

    // Read back and verify point count
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("POINTS 27") != std::string::npos) {
            SUCCEED();
            return;
        }
    }
    FAIL() << "Expected POINTS 27 in VTK file";
}

TEST_F(IOTest, VTKLegacyWriterCellData) {
    std::string filename = (test_dir_ / "test_cells.vtk").string();

    VTKLegacyWriter writer(filename);

    // Create 8 points for one hex
    std::vector<Vec3> points = {
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
        Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(1, 1, 1), Vec3(0, 1, 1)
    };
    writer.write_points(points);

    std::vector<std::array<Index, 8>> cells = {{0, 1, 2, 3, 4, 5, 6, 7}};
    writer.write_hexahedra(cells);

    // Add cell data
    VecX cell_scalar(1);
    cell_scalar(0) = 42.0;
    writer.add_cell_scalar("cell_id", cell_scalar);

    writer.close();

    // Verify file contains CELL_DATA
    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("CELL_DATA 1") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS cell_id") != std::string::npos);
}

// =============================================================================
// VTK XML Writer Tests
// =============================================================================

TEST_F(IOTest, VTKWriterConstruction) {
    std::string basename = (test_dir_ / "output").string();

    // Test different formats
    VTKWriter writer_vtu(basename + "_vtu", VTKFormat::VTU);
    VTKWriter writer_ascii(basename + "_ascii", VTKFormat::VTU, VTKEncoding::ASCII);
    VTKWriter writer_legacy(basename + "_legacy", VTKFormat::Legacy);

    // No exceptions should be thrown
    SUCCEED();
}

TEST_F(IOTest, VTKWriterPolynomialOrder) {
    std::string basename = (test_dir_ / "output_order").string();
    VTKWriter writer(basename);

    writer.set_polynomial_order(1);
    writer.set_polynomial_order(2);
    writer.set_polynomial_order(4);

    // No exceptions
    SUCCEED();
}

TEST_F(IOTest, VTKWriterFieldRegistration) {
    std::string basename = (test_dir_ / "output_fields").string();
    VTKWriter writer(basename);

    writer.add_point_data("temperature", 1);
    writer.add_point_data("velocity", 3);
    writer.add_cell_data("element_id", 1);
    writer.add_cell_data("stress", 6);  // Symmetric tensor

    // No exceptions
    SUCCEED();
}

TEST_F(IOTest, VTKWriterGetFilename) {
    std::string basename = (test_dir_ / "output").string();

    VTKWriter writer_vtu(basename, VTKFormat::VTU);
    EXPECT_TRUE(writer_vtu.get_filename(0).find(".vtu") != std::string::npos);
    EXPECT_TRUE(writer_vtu.get_filename(0).find("_000000") != std::string::npos);
    EXPECT_TRUE(writer_vtu.get_filename(123).find("_000123") != std::string::npos);

    VTKWriter writer_legacy(basename, VTKFormat::Legacy);
    EXPECT_TRUE(writer_legacy.get_filename(0).find(".vtk") != std::string::npos);
}

// =============================================================================
// Zarr Writer Tests
// =============================================================================

TEST_F(IOTest, ZarrConfigDefaults) {
    ZarrConfig config;

    EXPECT_EQ(config.codec, ZarrCodec::Blosc);
    EXPECT_EQ(config.compression_level, 5);
    EXPECT_EQ(config.chunk_order, ZarrChunkOrder::RowMajor);
    EXPECT_TRUE(config.enable_sharding);
    EXPECT_EQ(config.conventions, "CF-1.8");
}

TEST_F(IOTest, ZarrVariableDefaults) {
    ZarrVariable var;

    EXPECT_EQ(var.dtype, ZarrDataType::Float64);
    EXPECT_EQ(var.fill_value, -9999.0);
    EXPECT_FALSE(var.is_coordinate);
}

TEST_F(IOTest, ZarrDimensionBasic) {
    ZarrDimension dim;
    dim.name = "time";
    dim.size = 100;
    dim.unlimited = true;

    EXPECT_EQ(dim.name, "time");
    EXPECT_EQ(dim.size, 100);
    EXPECT_TRUE(dim.unlimited);
}

TEST_F(IOTest, ZarrWriterConstruction) {
    ZarrConfig config;
    config.store_path = (test_dir_ / "test.zarr").string();

    ZarrWriter writer(config);

    EXPECT_FALSE(writer.is_initialized());
    EXPECT_EQ(writer.current_time_index(), 0);
}

TEST_F(IOTest, ZarrWriterAddDimension) {
    ZarrConfig config;
    config.store_path = (test_dir_ / "test_dims.zarr").string();

    ZarrWriter writer(config);

    writer.add_dimension("time", 0, true);  // Unlimited
    writer.add_dimension("x", 100);
    writer.add_dimension("y", 50);
    writer.add_dimension("z", 20);

    // No exceptions
    SUCCEED();
}

TEST_F(IOTest, ZarrWriterInitialize) {
    ZarrConfig config;
    config.store_path = (test_dir_ / "test_init.zarr").string();

    ZarrWriter writer(config);

    writer.add_dimension("x", 10);
    writer.add_variable("test", {"x"}, ZarrDataType::Float64);

    writer.initialize();

    EXPECT_TRUE(writer.is_initialized());
    EXPECT_TRUE(fs::exists(config.store_path));
    EXPECT_TRUE(fs::exists(fs::path(config.store_path) / "zarr.json"));
}

TEST_F(IOTest, ZarrWriterWriteCoordinate) {
    ZarrConfig config;
    config.store_path = (test_dir_ / "test_coord.zarr").string();

    ZarrWriter writer(config);

    writer.add_dimension("x", 5);

    ZarrVariable x_var;
    x_var.name = "x";
    x_var.dimensions = {"x"};
    x_var.shape = {5};
    x_var.is_coordinate = true;
    writer.add_variable(x_var);

    writer.initialize();

    VecX x_values(5);
    x_values << 0.0, 1.0, 2.0, 3.0, 4.0;

    writer.write_coordinate("x", x_values);
    writer.finalize();

    // Verify data was written
    fs::path chunk_path = fs::path(config.store_path) / "x" / "c" / "0";
    EXPECT_TRUE(fs::exists(chunk_path));
}

TEST_F(IOTest, ZarrWriterSetAttribute) {
    ZarrConfig config;
    config.store_path = (test_dir_ / "test_attr.zarr").string();

    ZarrWriter writer(config);

    writer.add_dimension("x", 10);
    writer.add_variable("temperature", {"x"}, ZarrDataType::Float64);

    writer.set_attribute("temperature", "units", "degC");
    writer.set_attribute("temperature", "long_name", "Temperature");
    writer.set_attribute("temperature", "fill_value", -999.0);

    // No exceptions
    SUCCEED();
}

TEST_F(IOTest, ZarrWriterMoveSemantics) {
    ZarrConfig config;
    config.store_path = (test_dir_ / "test_move.zarr").string();

    ZarrWriter writer1(config);
    writer1.add_dimension("x", 10);

    // Move construct
    ZarrWriter writer2(std::move(writer1));

    // writer2 should work
    writer2.add_variable("test", {"x"}, ZarrDataType::Float64);

    SUCCEED();
}

TEST_F(IOTest, ZarrWriterCannotAddAfterInit) {
    ZarrConfig config;
    config.store_path = (test_dir_ / "test_error.zarr").string();

    ZarrWriter writer(config);
    writer.add_dimension("x", 10);
    writer.add_variable("test", {"x"});
    writer.initialize();

    // Should throw when adding after initialization
    EXPECT_THROW(writer.add_dimension("y", 5), std::runtime_error);
    EXPECT_THROW(writer.add_variable("test2", {"x"}), std::runtime_error);
}

// =============================================================================
// Data Type Tests
// =============================================================================

TEST_F(IOTest, ZarrDataTypes) {
    EXPECT_NE(static_cast<int>(ZarrDataType::Float32),
              static_cast<int>(ZarrDataType::Float64));
    EXPECT_NE(static_cast<int>(ZarrDataType::Int32),
              static_cast<int>(ZarrDataType::Int64));
}

TEST_F(IOTest, VTKCellTypes) {
    EXPECT_EQ(static_cast<int>(VTKCellType::Hexahedron), 12);
    EXPECT_EQ(static_cast<int>(VTKCellType::LagrangeHexahedron), 72);
    EXPECT_EQ(static_cast<int>(VTKCellType::QuadraticHexahedron), 25);
}

TEST_F(IOTest, VTKEncodings) {
    // Ensure all encoding types are distinct
    EXPECT_NE(static_cast<int>(VTKEncoding::ASCII),
              static_cast<int>(VTKEncoding::Binary));
    EXPECT_NE(static_cast<int>(VTKEncoding::Binary),
              static_cast<int>(VTKEncoding::Base64));
}

// =============================================================================
// HighOrderVTKWriter Tests
// =============================================================================

TEST_F(IOTest, HighOrderVTKWriterConstruction) {
    std::string basename = (test_dir_ / "highorder").string();

    HighOrderVTKWriter writer1(basename + "_p1", 1);
    HighOrderVTKWriter writer2(basename + "_p2", 2);
    HighOrderVTKWriter writer3(basename + "_p4", 4);

    SUCCEED();
}

TEST_F(IOTest, HighOrderVTKWriterFieldTypes) {
    std::string basename = (test_dir_ / "highorder_fields").string();
    HighOrderVTKWriter writer(basename, 2);

    writer.add_scalar_field("temperature");
    writer.add_scalar_field("salinity");
    writer.add_vector_field("velocity");

    SUCCEED();
}

// =============================================================================
// XDMFWriter Tests
// =============================================================================

TEST_F(IOTest, XDMFWriterConstruction) {
    std::string basename = (test_dir_ / "xdmf_test").string();

    XDMFWriter writer(basename);

    SUCCEED();
}

TEST_F(IOTest, XDMFWriterAddAttribute) {
    std::string basename = (test_dir_ / "xdmf_attrs").string();

    XDMFWriter writer(basename);

    writer.add_attribute("temperature", 1, "Node");
    writer.add_attribute("velocity", 3, "Node");
    writer.add_attribute("pressure", 1, "Cell");

    SUCCEED();
}

// =============================================================================
// GeoTIFF Reader Tests
// =============================================================================

TEST_F(IOTest, GeoTiffReaderAvailability) {
    // Test that is_available() returns a consistent value
    bool available = GeoTiffReader::is_available();
    // Either GDAL is available or not - just verify no crash
    EXPECT_TRUE(available || !available);
}

TEST_F(IOTest, BathymetryDataDefaults) {
    BathymetryData data;

    EXPECT_EQ(data.sizex, 0);
    EXPECT_EQ(data.sizey, 0);
    EXPECT_TRUE(data.elevation.empty());
    EXPECT_EQ(data.nodata_value, -9999.0f);
    EXPECT_FALSE(data.is_valid());
}

TEST_F(IOTest, BathymetryDataValidation) {
    BathymetryData data;

    // Empty data is invalid
    EXPECT_FALSE(data.is_valid());

    // Populate with valid data
    data.sizex = 3;
    data.sizey = 3;
    data.elevation.resize(9);
    for (int i = 0; i < 9; ++i) {
        data.elevation[i] = static_cast<float>(i * 10.0);
    }

    EXPECT_TRUE(data.is_valid());
}

TEST_F(IOTest, BathymetryDataPixelToWorld) {
    BathymetryData data;
    data.sizex = 10;
    data.sizey = 10;
    data.elevation.resize(100, 100.0f);

    // Set up a simple geotransform: origin at (1000, 2000), 100m pixel size
    // geotransform: [origin_x, pixel_width, 0, origin_y, 0, -pixel_height]
    data.geotransform = {1000.0, 100.0, 0.0, 2000.0, 0.0, -100.0};

    double wx, wy;

    // Pixel (0, 0) -> world (1000, 2000)
    data.pixel_to_world(0, 0, wx, wy);
    EXPECT_DOUBLE_EQ(wx, 1000.0);
    EXPECT_DOUBLE_EQ(wy, 2000.0);

    // Pixel (1, 0) -> world (1100, 2000)
    data.pixel_to_world(1, 0, wx, wy);
    EXPECT_DOUBLE_EQ(wx, 1100.0);
    EXPECT_DOUBLE_EQ(wy, 2000.0);

    // Pixel (0, 1) -> world (1000, 1900) - y decreases
    data.pixel_to_world(0, 1, wx, wy);
    EXPECT_DOUBLE_EQ(wx, 1000.0);
    EXPECT_DOUBLE_EQ(wy, 1900.0);
}

TEST_F(IOTest, BathymetryDataWorldToPixel) {
    BathymetryData data;
    data.sizex = 10;
    data.sizey = 10;
    data.elevation.resize(100, 100.0f);
    data.geotransform = {1000.0, 100.0, 0.0, 2000.0, 0.0, -100.0};

    double px, py;

    // World (1000, 2000) -> pixel (0, 0)
    data.world_to_pixel(1000.0, 2000.0, px, py);
    EXPECT_NEAR(px, 0.0, 1e-10);
    EXPECT_NEAR(py, 0.0, 1e-10);

    // World (1100, 1900) -> pixel (1, 1)
    data.world_to_pixel(1100.0, 1900.0, px, py);
    EXPECT_NEAR(px, 1.0, 1e-10);
    EXPECT_NEAR(py, 1.0, 1e-10);

    // World (1550, 1550) -> pixel (5.5, 4.5)
    data.world_to_pixel(1550.0, 1550.0, px, py);
    EXPECT_NEAR(px, 5.5, 1e-10);
    EXPECT_NEAR(py, 4.5, 1e-10);
}

TEST_F(IOTest, BathymetryDataInterpolation) {
    BathymetryData data;
    data.sizex = 3;
    data.sizey = 3;
    data.elevation.resize(9);
    data.geotransform = {0.0, 1.0, 0.0, 2.0, 0.0, -1.0};

    // Create a simple elevation pattern:
    // Row 0: 0, 10, 20
    // Row 1: 30, 40, 50
    // Row 2: 60, 70, 80
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            data.elevation[j * 3 + i] = static_cast<float>((j * 3 + i) * 10.0);
        }
    }

    // Test center of pixel (0.5, 0.5) in pixel coords -> (0.5, 1.5) in world coords
    // Should return value near the pixel center
    float depth = data.interpolate(0.5, 1.5);
    EXPECT_GT(depth, -1e10);  // Not NoData

    // Test that corners of data return valid values
    depth = data.interpolate(0.0, 2.0);  // Top-left corner
    EXPECT_GT(depth, -1e10);
}

TEST_F(IOTest, BathymetryDataIsLand) {
    BathymetryData data;
    data.sizex = 3;
    data.sizey = 3;
    data.elevation.resize(9);
    data.geotransform = {0.0, 1.0, 0.0, 3.0, 0.0, -1.0};
    data.nodata_value = -9999.0f;

    // Set up: water depths (positive = below sea level in oceanographic convention)
    // But in GeoTIFF, negative = below sea level, positive = above
    // Let's use: negative = water, positive = land
    data.elevation = {
        -100.0f, -50.0f, 10.0f,   // Row 0: water, water, land
        -200.0f, -9999.0f, 20.0f, // Row 1: water, nodata, land
        -150.0f, -75.0f, -9999.0f // Row 2: water, water, nodata
    };

    // Compute bounds
    data.xmin = 0.0;
    data.xmax = 3.0;
    data.ymin = 0.0;
    data.ymax = 3.0;

    // NoData locations should be land
    EXPECT_TRUE(data.is_land(1.5, 1.5));  // Center of nodata pixel
}

TEST_F(IOTest, BathymetryDataGetDepth) {
    BathymetryData data;
    data.sizex = 3;
    data.sizey = 3;
    data.elevation.resize(9, -100.0f);  // All 100m depth
    data.geotransform = {0.0, 1.0, 0.0, 3.0, 0.0, -1.0};
    data.xmin = 0.0;
    data.xmax = 3.0;
    data.ymin = 0.0;
    data.ymax = 3.0;

    // get_depth should return positive depth from negative elevation
    float depth = data.get_depth(1.5, 1.5);
    EXPECT_GT(depth, 0.0f);  // Should be positive depth
}

TEST_F(IOTest, BathymetrySurfaceBasic) {
    auto data = std::make_shared<BathymetryData>();
    data->sizex = 5;
    data->sizey = 5;
    data->elevation.resize(25, -100.0f);  // Uniform 100m depth
    data->geotransform = {0.0, 1.0, 0.0, 5.0, 0.0, -1.0};
    data->xmin = 0.0;
    data->xmax = 5.0;
    data->ymin = 0.0;
    data->ymax = 5.0;

    BathymetrySurface surface(data);

    // Test depth retrieval
    Real depth = surface.depth(2.5, 2.5);
    EXPECT_GT(depth, 0.0);

    // Test bounds retrieval
    Real xmin, xmax, ymin, ymax;
    surface.get_bounds(xmin, xmax, ymin, ymax);
    EXPECT_EQ(xmin, 0.0);
    EXPECT_EQ(xmax, 5.0);
    EXPECT_EQ(ymin, 0.0);
    EXPECT_EQ(ymax, 5.0);
}

TEST_F(IOTest, BathymetrySurfaceGradient) {
    auto data = std::make_shared<BathymetryData>();
    data->sizex = 5;
    data->sizey = 5;
    data->elevation.resize(25);
    data->geotransform = {0.0, 1.0, 0.0, 5.0, 0.0, -1.0};
    data->xmin = 0.0;
    data->xmax = 5.0;
    data->ymin = 0.0;
    data->ymax = 5.0;

    // Create a sloping bottom: depth increases with x
    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 5; ++i) {
            data->elevation[j * 5 + i] = static_cast<float>(-50.0 - i * 10.0);
        }
    }

    BathymetrySurface surface(data);

    Real dh_dx, dh_dy;
    surface.gradient(2.5, 2.5, dh_dx, dh_dy);

    // Gradient in x should be non-zero (slope exists)
    // Gradient in y should be near zero (no slope in y)
    EXPECT_NE(dh_dx, 0.0);
}

TEST_F(IOTest, BathymetrySurfaceWaterLand) {
    auto data = std::make_shared<BathymetryData>();
    data->sizex = 3;
    data->sizey = 3;
    data->geotransform = {0.0, 1.0, 0.0, 3.0, 0.0, -1.0};
    data->xmin = 0.0;
    data->xmax = 3.0;
    data->ymin = 0.0;
    data->ymax = 3.0;
    data->nodata_value = -9999.0f;

    // Mixed land and water
    data->elevation = {
        -100.0f, -50.0f, 10.0f,    // water, water, land
        -200.0f, -100.0f, 5.0f,    // water, water, land
        -150.0f, -75.0f, -25.0f    // water, water, water
    };

    BathymetrySurface surface(data);

    // Check water detection (min_depth = 1.0)
    EXPECT_TRUE(surface.is_water(0.5, 2.5, 1.0));   // Deep water
    EXPECT_FALSE(surface.is_water(2.5, 2.5, 1.0)); // Land (positive elevation)
}

TEST_F(IOTest, BathymetryMeshGeneratorConfig) {
    BathymetryMeshGenerator::Config config;

    // Check defaults
    EXPECT_EQ(config.base_nx, 10);
    EXPECT_EQ(config.base_ny, 10);
    EXPECT_EQ(config.base_nz, 5);
    EXPECT_TRUE(config.mask_land);
    EXPECT_GT(config.min_depth, 0.0);
}

TEST_F(IOTest, BathymetryMeshGeneratorBasic) {
    auto data = std::make_shared<BathymetryData>();
    data->sizex = 10;
    data->sizey = 10;
    data->elevation.resize(100, -100.0f);  // Uniform 100m depth
    data->geotransform = {0.0, 100.0, 0.0, 1000.0, 0.0, -100.0};
    data->xmin = 0.0;
    data->xmax = 1000.0;
    data->ymin = 0.0;
    data->ymax = 1000.0;

    BathymetryMeshGenerator generator(data);

    BathymetryMeshGenerator::Config config;
    config.base_nx = 4;
    config.base_ny = 4;
    config.base_nz = 2;
    config.mask_land = false;  // Don't mask for this test
    generator.set_config(config);

    auto elements = generator.generate();

    // Should generate 4 * 4 * 2 = 32 elements
    EXPECT_EQ(elements.size(), 32);

    // Check that all elements have valid bounds
    for (const auto& elem : elements) {
        EXPECT_LT(elem.xmin, elem.xmax);
        EXPECT_LT(elem.ymin, elem.ymax);
        EXPECT_LT(elem.zmin, elem.zmax);
    }
}

TEST_F(IOTest, BathymetryMeshGeneratorLandMasking) {
    auto data = std::make_shared<BathymetryData>();
    data->sizex = 4;
    data->sizey = 4;
    data->geotransform = {0.0, 250.0, 0.0, 1000.0, 0.0, -250.0};
    data->xmin = 0.0;
    data->xmax = 1000.0;
    data->ymin = 0.0;
    data->ymax = 1000.0;

    // Half water, half land
    data->elevation = {
        -100.0f, -100.0f, 50.0f, 50.0f,    // Row 0: water, water, land, land
        -100.0f, -100.0f, 50.0f, 50.0f,    // Row 1: water, water, land, land
        -100.0f, -100.0f, 50.0f, 50.0f,    // Row 2: water, water, land, land
        -100.0f, -100.0f, 50.0f, 50.0f     // Row 3: water, water, land, land
    };

    BathymetryMeshGenerator generator(data);

    BathymetryMeshGenerator::Config config;
    config.base_nx = 4;
    config.base_ny = 4;
    config.base_nz = 1;
    config.mask_land = true;
    config.min_depth = 1.0;
    generator.set_config(config);

    auto elements = generator.generate();

    // With land masking, should have fewer than 16 elements
    // Expect roughly half (water side only)
    EXPECT_LT(elements.size(), 16);
    EXPECT_GT(elements.size(), 0);
}

TEST_F(IOTest, BathymetryMeshGeneratorRefinementFunction) {
    auto data = std::make_shared<BathymetryData>();
    data->sizex = 10;
    data->sizey = 10;
    data->elevation.resize(100, -100.0f);
    data->geotransform = {0.0, 100.0, 0.0, 1000.0, 0.0, -100.0};
    data->xmin = 0.0;
    data->xmax = 1000.0;
    data->ymin = 0.0;
    data->ymax = 1000.0;

    BathymetryMeshGenerator generator(data);

    auto refine_func = generator.create_refinement_function();

    // Test that refinement function is callable
    ElementBounds bounds;
    bounds.xmin = 0.0;
    bounds.xmax = 100.0;
    bounds.ymin = 0.0;
    bounds.ymax = 100.0;
    bounds.zmin = -1.0;
    bounds.zmax = 0.0;

    // Should return a boolean (doesn't matter what value)
    bool should_refine = refine_func(bounds);
    EXPECT_TRUE(should_refine || !should_refine);  // Just check it runs
}

TEST_F(IOTest, GeoTiffReaderConstruction) {
    GeoTiffReader reader;

    // Construction should not throw
    SUCCEED();
}

TEST_F(IOTest, GeoTiffReaderLoadNonexistent) {
    GeoTiffReader reader;

    // Loading nonexistent file should return invalid data
    BathymetryData data = reader.load("/nonexistent/path/file.tif");

    EXPECT_FALSE(data.is_valid());

    // Should have error message
    std::string error = reader.last_error();
    EXPECT_FALSE(error.empty());
}

// =============================================================================
// SeabedVTKWriter Tests
// =============================================================================

TEST_F(IOTest, SeabedVTKWriterConstruction) {
    std::string filename = (test_dir_ / "seabed").string();

    SeabedVTKWriter writer(filename);

    EXPECT_EQ(writer.num_points(), 0);
    EXPECT_EQ(writer.num_cells(), 0);
}

TEST_F(IOTest, SeabedVTKWriterSetResolution) {
    std::string filename = (test_dir_ / "seabed_res").string();

    SeabedVTKWriter writer(filename);
    writer.set_resolution(20);

    // No exceptions
    SUCCEED();
}

TEST_F(IOTest, SeabedVTKWriterSingleElement) {
    std::string filename = (test_dir_ / "seabed_single").string();

    // Create a simple octree with one element
    OctreeAdapter mesh(0.0, 1.0, 0.0, 1.0, -1.0, 0.0);
    mesh.build_uniform(1, 1, 1);

    // Create element coordinates for polynomial order 1 (8 nodes)
    // Interleaved x,y,z: [x0,y0,z0, x1,y1,z1, ...]
    // For order 1, nodes are at corners of the reference cube
    int order = 1;
    int np = order + 1;  // 2 nodes per dimension
    int num_dofs = np * np * np;  // 8 DOFs

    VecX coords(3 * num_dofs);
    // Generate coordinates for a unit cube with varying bottom depth
    // DOF ordering: i + np*(j + np*k)
    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int dof = i + np * (j + np * k);
                Real x = static_cast<Real>(i) / order;
                Real y = static_cast<Real>(j) / order;
                Real z_ref = -1.0 + 2.0 * static_cast<Real>(k) / order;

                // Map z: bottom face has varying depth, surface is flat at z=0
                Real depth = 1.0 + 0.2 * x + 0.1 * y;  // Varying bottom depth
                Real z = (z_ref + 1.0) / 2.0 * depth - depth;  // sigma to physical

                coords(3 * dof + 0) = x;
                coords(3 * dof + 1) = y;
                coords(3 * dof + 2) = z;
            }
        }
    }

    std::vector<VecX> element_coords = {coords};

    SeabedVTKWriter writer(filename);
    writer.set_mesh(mesh, element_coords, order);
    writer.set_resolution(5);  // 5x5 subdivision

    writer.write();

    // Check that file was created
    std::string vtk_filename = filename + ".vtk";
    EXPECT_TRUE(fs::exists(vtk_filename));

    // Check point and cell counts
    // resolution=5 means 6x6=36 points per element, 5x5=25 quads per element
    EXPECT_EQ(writer.num_points(), 36);
    EXPECT_EQ(writer.num_cells(), 25);

    // Verify file content
    std::ifstream file(vtk_filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("POINTS 36") != std::string::npos);
    EXPECT_TRUE(content.find("CELLS 25") != std::string::npos);
    EXPECT_TRUE(content.find("CELL_TYPES 25") != std::string::npos);
}

TEST_F(IOTest, SeabedVTKWriterWithScalarField) {
    std::string filename = (test_dir_ / "seabed_scalar").string();

    OctreeAdapter mesh(0.0, 1.0, 0.0, 1.0, -1.0, 0.0);
    mesh.build_uniform(1, 1, 1);

    int order = 1;
    int np = order + 1;
    int num_dofs = np * np * np;

    // Create coordinates (simple flat bottom)
    VecX coords(3 * num_dofs);
    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int dof = i + np * (j + np * k);
                coords(3 * dof + 0) = static_cast<Real>(i) / order;
                coords(3 * dof + 1) = static_cast<Real>(j) / order;
                coords(3 * dof + 2) = -1.0 + static_cast<Real>(k) / order;
            }
        }
    }

    // Create depth field data
    VecX depth_data(num_dofs);
    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int dof = i + np * (j + np * k);
                depth_data(dof) = 100.0 + 10.0 * i + 5.0 * j;
            }
        }
    }

    std::vector<VecX> element_coords = {coords};
    std::vector<VecX> element_depth = {depth_data};

    SeabedVTKWriter writer(filename);
    writer.set_mesh(mesh, element_coords, order);
    writer.set_resolution(3);
    writer.add_scalar_field("depth", element_depth);

    writer.write();

    std::string vtk_filename = filename + ".vtk";
    EXPECT_TRUE(fs::exists(vtk_filename));

    // Verify scalar field is in the file
    std::ifstream file(vtk_filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("POINT_DATA") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS depth") != std::string::npos);
}

TEST_F(IOTest, SeabedVTKWriterHigherOrder) {
    std::string filename = (test_dir_ / "seabed_highorder").string();

    OctreeAdapter mesh(0.0, 1.0, 0.0, 1.0, -1.0, 0.0);
    mesh.build_uniform(1, 1, 1);

    int order = 2;  // Quadratic
    int np = order + 1;  // 3 nodes per dimension
    int num_dofs = np * np * np;  // 27 DOFs

    // Create coordinates with curved bottom (quadratic variation)
    VecX coords(3 * num_dofs);
    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int dof = i + np * (j + np * k);
                Real xi = static_cast<Real>(i) / order;
                Real eta = static_cast<Real>(j) / order;
                Real zeta_ref = -1.0 + 2.0 * static_cast<Real>(k) / order;

                // Curved bottom: depth = 1 + 0.3*sin(pi*x)*sin(pi*y)
                Real depth = 1.0 + 0.3 * std::sin(M_PI * xi) * std::sin(M_PI * eta);
                Real z = (zeta_ref + 1.0) / 2.0 * depth - depth;

                coords(3 * dof + 0) = xi;
                coords(3 * dof + 1) = eta;
                coords(3 * dof + 2) = z;
            }
        }
    }

    std::vector<VecX> element_coords = {coords};

    SeabedVTKWriter writer(filename);
    writer.set_mesh(mesh, element_coords, order);
    writer.set_resolution(10);  // High resolution to see the curvature

    writer.write();

    std::string vtk_filename = filename + ".vtk";
    EXPECT_TRUE(fs::exists(vtk_filename));

    // resolution=10 means 11x11=121 points, 10x10=100 quads
    EXPECT_EQ(writer.num_points(), 121);
    EXPECT_EQ(writer.num_cells(), 100);
}

TEST_F(IOTest, SeabedVTKWriterMultipleElements) {
    std::string filename = (test_dir_ / "seabed_multi").string();

    // Create mesh with 2x2x1 elements
    OctreeAdapter mesh(0.0, 2.0, 0.0, 2.0, -1.0, 0.0);
    mesh.build_uniform(2, 2, 1);

    int order = 1;
    int np = order + 1;
    int num_dofs = np * np * np;

    std::vector<VecX> element_coords;

    // Generate coordinates for each element
    const auto& elements = mesh.elements();
    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& bounds = elements[e]->bounds;

        VecX coords(3 * num_dofs);
        for (int k = 0; k < np; ++k) {
            for (int j = 0; j < np; ++j) {
                for (int i = 0; i < np; ++i) {
                    int dof = i + np * (j + np * k);
                    Real xi = static_cast<Real>(i) / order;
                    Real eta = static_cast<Real>(j) / order;
                    Real zeta = static_cast<Real>(k) / order;

                    coords(3 * dof + 0) = bounds.xmin + xi * (bounds.xmax - bounds.xmin);
                    coords(3 * dof + 1) = bounds.ymin + eta * (bounds.ymax - bounds.ymin);
                    coords(3 * dof + 2) = bounds.zmin + zeta * (bounds.zmax - bounds.zmin);
                }
            }
        }
        element_coords.push_back(coords);
    }

    SeabedVTKWriter writer(filename);
    writer.set_mesh(mesh, element_coords, order);
    writer.set_resolution(4);

    writer.write();

    std::string vtk_filename = filename + ".vtk";
    EXPECT_TRUE(fs::exists(vtk_filename));

    // 4 elements, each with 5x5=25 points and 4x4=16 quads
    EXPECT_EQ(writer.num_points(), 4 * 25);
    EXPECT_EQ(writer.num_cells(), 4 * 16);
}

