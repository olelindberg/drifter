#include "io/raster_vtk_writer.hpp"
#include "mesh/geotiff_reader.hpp"
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace drifter;

namespace {

// A small north-up raster: 10x8 pixels of 100 m, top-left at (1000, 2000),
// pixel height negative as GDAL writes it. Values are elevations (negative in
// water), with one nodata pixel.
BathymetryData make_raster() {
    BathymetryData data;
    data.sizex = 10;
    data.sizey = 8;
    data.geotransform = {1000.0, 100.0, 0.0, 2000.0, 0.0, -100.0};
    data.nodata_value = -9999.0f;
    data.is_depth_positive = false;
    data.xmin = 1000.0;
    data.xmax = 2000.0;
    data.ymin = 1200.0;
    data.ymax = 2000.0;

    data.elevation.resize(static_cast<size_t>(data.sizex) * data.sizey);
    for (int j = 0; j < data.sizey; ++j) {
        for (int i = 0; i < data.sizex; ++i) {
            data.elevation[j * data.sizex + i] = static_cast<float>(-(i + 10 * j) - 1);
        }
    }
    data.elevation[0] = data.nodata_value;
    return data;
}

std::string read_file(const std::string &path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

class RasterVtkWriterTest : public ::testing::Test {
protected:
    void SetUp() override { base_ = "/tmp/drifter_test_raster_" + std::to_string(::getpid()); }
    void TearDown() override { std::remove((base_ + ".vts").c_str()); }
    std::string base_;
};

// A box strictly inside the raster selects exactly the pixels whose centers it
// contains.
TEST_F(RasterVtkWriterTest, CropsToPixelCentersInsideTheDomain) {
    const BathymetryData data = make_raster();

    // x in [1150, 1550] covers pixel centers 1150, 1250, 1350, 1450, 1550 -> i = 1..5
    // y in [1450, 1750] covers pixel centers 1450, 1550, 1650, 1750 -> j = 5..2
    io::write_raster_vts(base_, data, 1150.0, 1550.0, 1450.0, 1750.0);

    const std::string content = read_file(base_ + ".vts");
    ASSERT_FALSE(content.empty());

    EXPECT_NE(content.find("type=\"StructuredGrid\""), std::string::npos) << content;
    EXPECT_NE(content.find("WholeExtent=\"0 4 0 3 0 0\""), std::string::npos) << content;
    // The raster value is the z coordinate, so the points are 3-component
    EXPECT_NE(content.find("Name=\"Points\" NumberOfComponents=\"3\""), std::string::npos)
        << content;
    EXPECT_NE(content.find("Name=\"elevation\""), std::string::npos);
    EXPECT_EQ(content.find("Name=\"depth\""), std::string::npos);
}

// A box larger than the raster is clamped to the full raster extent.
TEST_F(RasterVtkWriterTest, ClampsToRasterBounds) {
    const BathymetryData data = make_raster();

    io::write_raster_vts(base_, data, -1e6, 1e6, -1e6, 1e6);

    const std::string content = read_file(base_ + ".vts");
    EXPECT_NE(content.find("WholeExtent=\"0 9 0 7 0 0\""), std::string::npos) << content;
}

// No fallback resampling: a sheared geotransform is rejected outright.
TEST_F(RasterVtkWriterTest, RejectsRotatedGeotransform) {
    BathymetryData data = make_raster();
    data.geotransform[2] = 5.0;

    EXPECT_THROW(io::write_raster_vts(base_, data, 1150.0, 1550.0, 1450.0, 1750.0),
                 std::invalid_argument);
}

// A domain box that misses the raster is an error, not an empty file.
TEST_F(RasterVtkWriterTest, RejectsDomainOutsideTheRaster) {
    const BathymetryData data = make_raster();

    EXPECT_THROW(io::write_raster_vts(base_, data, 5000.0, 6000.0, 5000.0, 6000.0),
                 std::invalid_argument);
}

} // namespace
