// Integration tests for VTK output

#include <gtest/gtest.h>
#include "mesh/octree_adapter.hpp"
#include "io/vtk_writer.hpp"
#include "test_integration_fixtures.hpp"
#include <filesystem>
#include <fstream>

using namespace drifter;
using namespace drifter::testing;

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
