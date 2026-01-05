// Integration tests for VTK output

#include <gtest/gtest.h>
#include "mesh/octree_adapter.hpp"
#include "io/vtk_writer.hpp"
#include "test_integration_fixtures.hpp"
#include <filesystem>
#include <fstream>

using namespace drifter;
using namespace drifter::testing;

TEST_F(SimulationTest, VTKWriterCreation) {
    std::string basename = test_output_dir_ + "/test_vtk";

    VTKWriter writer(basename, VTKFormat::VTU, VTKEncoding::ASCII);
    writer.set_polynomial_order(2);

    // Create simple mesh
    OctreeAdapter mesh(0.0, 100.0, 0.0, 100.0, -10.0, 0.0);
    mesh.build_uniform(2, 2, 1);

    writer.set_mesh(mesh);
    writer.add_point_data("test_scalar", 1);

    // Set some data
    int n_per_elem = 3 * 3 * 3;  // Order 2
    std::vector<VecX> data(mesh.num_elements());
    for (Index e = 0; e < mesh.num_elements(); ++e) {
        data[e] = VecX::Constant(n_per_elem, static_cast<Real>(e));
    }
    writer.set_point_data("test_scalar", data);

    // Write file
    writer.write(0, 0.0);

    // Check file exists
    std::string filename = writer.get_filename(0);
    EXPECT_TRUE(std::filesystem::exists(filename));

    // Check file has content
    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_GT(content.size(), 100);  // Should have substantial content
    EXPECT_NE(content.find("VTKFile"), std::string::npos);
    EXPECT_NE(content.find("UnstructuredGrid"), std::string::npos);
}

TEST_F(SimulationTest, VTKWriterLegacyFormat) {
    std::string basename = test_output_dir_ + "/test_legacy";

    // Create a simple 1x1x1 mesh
    OctreeAdapter mesh(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    mesh.build_uniform(1, 1, 1);

    VTKWriter writer(basename, VTKFormat::Legacy, VTKEncoding::ASCII);
    writer.set_polynomial_order(1);
    writer.set_mesh(mesh);

    // Add point data
    writer.add_point_data("node_id", 1);
    VecX scalar(8);
    scalar << 0, 1, 2, 3, 4, 5, 6, 7;
    writer.set_point_data("node_id", scalar);

    writer.write(0, 0.0);

    // Check file exists and has content
    std::string filename = basename + "_000000.vtk";
    EXPECT_TRUE(std::filesystem::exists(filename));

    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("POINTS 8"), std::string::npos);
    EXPECT_NE(content.find("CELLS 1"), std::string::npos);
    EXPECT_NE(content.find("POINT_DATA"), std::string::npos);
}

TEST_F(SimulationTest, VTKPVDCollection) {
    std::string basename = test_output_dir_ + "/time_series";

    VTKWriter writer(basename, VTKFormat::VTU, VTKEncoding::ASCII);
    writer.set_polynomial_order(1);

    OctreeAdapter mesh(0.0, 10.0, 0.0, 10.0, -1.0, 0.0);
    mesh.build_uniform(1, 1, 1);
    writer.set_mesh(mesh);

    writer.add_point_data("eta", 1);

    // Write multiple timesteps
    for (int t = 0; t < 3; ++t) {
        std::vector<VecX> eta(1);
        eta[0] = VecX::Constant(8, static_cast<Real>(t) * 0.1);
        writer.set_point_data("eta", eta);
        writer.write_timestep(static_cast<Real>(t));
    }

    // Finalize creates PVD file
    writer.finalize();

    // Check PVD file exists
    std::string pvd_file = basename + ".pvd";
    EXPECT_TRUE(std::filesystem::exists(pvd_file));

    std::ifstream file(pvd_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("Collection"), std::string::npos);
    EXPECT_NE(content.find("timestep"), std::string::npos);
}
