// Integration tests for GeoTIFF bathymetry and coastline-adaptive meshes

#include <gtest/gtest.h>
#include "mesh/octree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"
#include "mesh/coastline_refinement.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/bernstein_basis.hpp"
#include "io/vtk_writer.hpp"
#include "test_integration_fixtures.hpp"
#include <filesystem>
#include <fstream>

using namespace drifter;
using namespace drifter::testing;

// Configurable land polygon GeoPackage path
const std::string LAND_POLYGON_GPKG_PATH =
    "/home/ole/Projects/SeaMesh/data/Klimadatastyrelsen/TopografiskLandpolygon/landpolygon.gpkg";
const std::string LAND_POLYGON_LAYER = "landpolygon_2500";
const std::string LAND_POLYGON_PROJECTION = "EPSG:3034";

TEST_F(SimulationTest, GeoTiffBathymetryMesh) {
    // Test loading real Danish bathymetry data and generating mesh
    // Outputs VTK file for visualization in ParaView

    // Check if GDAL is available
    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
    }

    // Use configurable bathymetry file path
    std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;

    // Check if file exists
    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    // Load bathymetry
    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);

    ASSERT_TRUE(bathy.is_valid()) << "Failed to load GeoTIFF: " << reader.last_error();

    std::cout << "Loaded bathymetry: " << bathy.sizex << " x " << bathy.sizey << " pixels\n";
    std::cout << "Bounds: x=[" << bathy.xmin << ", " << bathy.xmax << "], "
              << "y=[" << bathy.ymin << ", " << bathy.ymax << "]\n";

    // Create mesh generator with a subset of the domain for testing
    // Use a smaller region to keep the test fast
    auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));

    BathymetryMeshGenerator generator(bathy_ptr);

    BathymetryMeshGenerator::Config config;
    // Use coarse resolution for fast test
    config.base_nx = 20;
    config.base_ny = 20;
    config.base_nz = 3;
    config.mask_land = true;
    config.min_depth = 2.0;  // Only cells with at least 2m depth
    config.zmin = -1.0;      // Sigma coordinates
    config.zmax = 0.0;

    // Focus on a smaller region (e.g., around Kattegat)
    // The full Danish model is quite large
    Real center_x = (bathy_ptr->xmin + bathy_ptr->xmax) / 2;
    Real center_y = (bathy_ptr->ymin + bathy_ptr->ymax) / 2;
    Real region_size = 100000.0;  // 100 km region

    config.xmin = center_x - region_size / 2;
    config.xmax = center_x + region_size / 2;
    config.ymin = center_y - region_size / 2;
    config.ymax = center_y + region_size / 2;

    generator.set_config(config);

    // Generate mesh elements
    auto elements = generator.generate();

    std::cout << "Generated " << elements.size() << " mesh elements\n";

    // Skip if no water elements found in this region
    if (elements.empty()) {
        GTEST_SKIP() << "No water elements in selected region";
    }

    EXPECT_GT(elements.size(), 0);

    // Create high-order DG basis for node positions
    const int poly_order = 3;  // Polynomial order for visualization
    HexahedronBasis basis(poly_order);
    const auto& ref_nodes = basis.lgl_nodes();
    const int nodes_per_elem = static_cast<int>(ref_nodes.size());
    const int n1d = poly_order + 1;  // Nodes per direction

    std::cout << "Using polynomial order " << poly_order
              << " (" << nodes_per_elem << " nodes per element)\n";

    // Create VTK output for visualization
    std::string vtk_path = test_output_dir_ + "/danish_bathymetry";
    std::ofstream vtk_file(vtk_path + ".vtk");

    // Write VTK 5.1 header (required for Lagrange hexahedra)
    vtk_file << "# vtk DataFile Version 5.1\n";
    vtk_file << "DRIFTER Danish Bathymetry Mesh - High Order Elements\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";

    // Get VTK point ordering: maps VTK point ID to tensor-product index
    auto vtk_point_order = create_vtk_point_ordering(poly_order);

    // Collect all points with bathymetry-following coordinates
    // First collect in tensor-product order, then reorder for VTK
    std::vector<Vec3> all_points;
    std::vector<Real> point_depths;

    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& bounds = elements[e];

        // Element dimensions in horizontal
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real dz_sigma = bounds.zmax - bounds.zmin;

        // Collect points in tensor-product order first
        std::vector<Vec3> elem_points(nodes_per_elem);
        std::vector<Real> elem_depths(nodes_per_elem);

        for (size_t i = 0; i < ref_nodes.size(); ++i) {
            const auto& ref_node = ref_nodes[i];
            // Map reference coordinates [-1,1]^3 to physical coordinates
            // Horizontal: simple linear mapping
            Real x = bounds.xmin + 0.5 * (ref_node.x() + 1.0) * dx;
            Real y = bounds.ymin + 0.5 * (ref_node.y() + 1.0) * dy;

            // Sigma coordinate in [zmin, zmax]
            Real sigma = bounds.zmin + 0.5 * (ref_node.z() + 1.0) * dz_sigma;

            // Get local bathymetry depth at this (x,y) position
            Real local_depth = bathy_ptr->get_depth(x, y);

            // Physical z from sigma: z = sigma * H(x,y)
            // sigma=0 at surface, sigma=-1 at bottom
            Real z = sigma * local_depth;

            elem_points[i] = Vec3(x, y, z);
            elem_depths[i] = local_depth;
        }

        // Reorder to VTK point order (corners first, then remaining by layer)
        for (int vtk_id = 0; vtk_id < nodes_per_elem; ++vtk_id) {
            int tensor_idx = vtk_point_order[vtk_id];
            all_points.push_back(elem_points[tensor_idx]);
            point_depths.push_back(elem_depths[tensor_idx]);
        }
    }

    // Write points
    vtk_file << "POINTS " << all_points.size() << " double\n";
    for (const auto& p : all_points) {
        vtk_file << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    // Write cells as VTK_LAGRANGE_HEXAHEDRON (cell type 72) using VTK 5.1 format
    size_t num_cells = elements.size();

    // Get the VTK connectivity ordering
    auto vtk_connectivity = get_vtk_lagrange_hex_connectivity(poly_order);

    // VTK 5.1 format: CELLS <num_offsets> <num_connectivity_entries>
    // num_offsets = num_cells + 1, num_connectivity = num_cells * nodes_per_elem
    size_t num_offsets = num_cells + 1;
    size_t num_connectivity = num_cells * nodes_per_elem;
    vtk_file << "\nCELLS " << num_offsets << " " << num_connectivity << "\n";
    vtk_file << "OFFSETS vtktypeint64\n";
    for (size_t e = 0; e <= num_cells; ++e) {
        vtk_file << (e * nodes_per_elem) << " ";
    }
    vtk_file << "\n";
    vtk_file << "CONNECTIVITY vtktypeint64\n";
    for (size_t e = 0; e < num_cells; ++e) {
        Index base = static_cast<Index>(e * nodes_per_elem);
        for (int conn_idx : vtk_connectivity) {
            vtk_file << (base + conn_idx) << " ";
        }
        vtk_file << "\n";
    }

    // Cell types: VTK_LAGRANGE_HEXAHEDRON = 72
    vtk_file << "\nCELL_TYPES " << num_cells << "\n";
    for (size_t e = 0; e < num_cells; ++e) {
        vtk_file << "72\n";
    }

    // Point data: depth at each node
    vtk_file << "\nPOINT_DATA " << all_points.size() << "\n";
    vtk_file << "SCALARS depth double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (Real d : point_depths) {
        vtk_file << d << "\n";
    }

    // Cell data: element ID
    vtk_file << "\nCELL_DATA " << num_cells << "\n";
    vtk_file << "SCALARS element_id double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (size_t e = 0; e < num_cells; ++e) {
        vtk_file << e << "\n";
    }

    vtk_file.close();

    // Verify output file exists
    std::string vtk_filename = vtk_path + ".vtk";
    EXPECT_TRUE(std::filesystem::exists(vtk_filename));

    std::cout << "VTK output written to: " << vtk_filename << "\n";
    std::cout << "Open in ParaView to visualize the Danish bathymetry mesh\n";
    std::cout << "High-order Lagrange hexahedra with bathymetry-following bottom surface\n";

    // Copy to a persistent location for easy access
    std::string persistent_path = "/tmp/danish_bathymetry.vtk";
    std::filesystem::copy_file(vtk_filename, persistent_path,
                                std::filesystem::copy_options::overwrite_existing);
    std::cout << "Also copied to: " << persistent_path << "\n";
}

TEST_F(SimulationTest, GeoTiffFullDomainMesh) {
    // Test generating mesh for the full Danish bathymetry domain
    // This creates a coarser mesh suitable for overview visualization

    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
    }

    // Use configurable bathymetry file path
    std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;

    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);
    ASSERT_TRUE(bathy.is_valid());

    auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));

    BathymetryMeshGenerator generator(bathy_ptr);

    BathymetryMeshGenerator::Config config;
    // Coarse mesh for full domain
    config.base_nx = 50;
    config.base_ny = 50;
    config.base_nz = 1;  // Single layer for 2D overview
    config.mask_land = true;
    config.min_depth = 1.0;

    generator.set_config(config);

    auto elements = generator.generate();

    std::cout << "Full domain mesh: " << elements.size() << " elements\n";

    if (elements.empty()) {
        GTEST_SKIP() << "No water elements generated";
    }

    // Create high-order DG basis for node positions
    const int poly_order = 3;
    HexahedronBasis basis(poly_order);
    const auto& ref_nodes = basis.lgl_nodes();
    const int nodes_per_elem = static_cast<int>(ref_nodes.size());

    // Write VTK for full domain with high-order elements
    std::string vtk_path = "/tmp/danish_bathymetry_full.vtk";
    std::ofstream vtk_file(vtk_path);

    // VTK 5.1 format (required for Lagrange hexahedra)
    vtk_file << "# vtk DataFile Version 5.1\n";
    vtk_file << "DRIFTER Danish Bathymetry Full Domain - High Order Elements\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";

    // Get VTK point ordering: maps VTK point ID to tensor-product index
    auto vtk_point_order = create_vtk_point_ordering(poly_order);

    // Collect all points with bathymetry-following coordinates
    // First collect in tensor-product order, then reorder for VTK
    std::vector<Vec3> all_points;
    std::vector<Real> point_depths;

    for (const auto& bounds : elements) {
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real dz_sigma = bounds.zmax - bounds.zmin;

        // Collect points in tensor-product order first
        std::vector<Vec3> elem_points(nodes_per_elem);
        std::vector<Real> elem_depths(nodes_per_elem);

        for (size_t i = 0; i < ref_nodes.size(); ++i) {
            const auto& ref_node = ref_nodes[i];
            Real x = bounds.xmin + 0.5 * (ref_node.x() + 1.0) * dx;
            Real y = bounds.ymin + 0.5 * (ref_node.y() + 1.0) * dy;
            Real sigma = bounds.zmin + 0.5 * (ref_node.z() + 1.0) * dz_sigma;

            Real local_depth = bathy_ptr->get_depth(x, y);
            Real z = sigma * local_depth;

            elem_points[i] = Vec3(x, y, z);
            elem_depths[i] = local_depth;
        }

        // Reorder to VTK point order (corners first, then remaining by layer)
        for (int vtk_id = 0; vtk_id < nodes_per_elem; ++vtk_id) {
            int tensor_idx = vtk_point_order[vtk_id];
            all_points.push_back(elem_points[tensor_idx]);
            point_depths.push_back(elem_depths[tensor_idx]);
        }
    }

    // Write points
    vtk_file << "POINTS " << all_points.size() << " double\n";
    for (const auto& p : all_points) {
        vtk_file << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    // Write cells as VTK_LAGRANGE_HEXAHEDRON using VTK 5.1 format
    size_t num_cells = elements.size();
    auto vtk_connectivity = get_vtk_lagrange_hex_connectivity(poly_order);

    // VTK 5.1 format: CELLS <num_offsets> <num_connectivity_entries>
    size_t num_offsets = num_cells + 1;
    size_t num_connectivity = num_cells * nodes_per_elem;
    vtk_file << "\nCELLS " << num_offsets << " " << num_connectivity << "\n";
    vtk_file << "OFFSETS vtktypeint64\n";
    for (size_t e = 0; e <= num_cells; ++e) {
        vtk_file << (e * nodes_per_elem) << " ";
    }
    vtk_file << "\n";
    vtk_file << "CONNECTIVITY vtktypeint64\n";
    for (size_t e = 0; e < num_cells; ++e) {
        Index base = static_cast<Index>(e * nodes_per_elem);
        for (int conn_idx : vtk_connectivity) {
            vtk_file << (base + conn_idx) << " ";
        }
        vtk_file << "\n";
    }

    vtk_file << "\nCELL_TYPES " << num_cells << "\n";
    for (size_t e = 0; e < num_cells; ++e) {
        vtk_file << "72\n";
    }

    vtk_file << "\nPOINT_DATA " << all_points.size() << "\n";
    vtk_file << "SCALARS depth double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (Real d : point_depths) {
        vtk_file << d << "\n";
    }

    vtk_file.close();

    EXPECT_TRUE(std::filesystem::exists(vtk_path));
    std::cout << "Full domain VTK: " << vtk_path << "\n";
}

TEST_F(SimulationTest, CoastlineAdaptiveOctreeMesh) {
    // Test generating a coastline-adaptive multiresolution octree mesh
    // Elements are refined near the coastline (land polygon boundary)
    // This reproduces the SeaMesh coastline-adaptive example using land polygons

    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
    }

    // Load bathymetry for domain bounds and depth values
    std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;

    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);
    ASSERT_TRUE(bathy.is_valid());

    std::cout << "Loaded bathymetry: " << bathy.sizex << " x " << bathy.sizey << " pixels\n";
    std::cout << "Bounds: x=[" << bathy.xmin << ", " << bathy.xmax << "], "
              << "y=[" << bathy.ymin << ", " << bathy.ymax << "]\n";

    auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));

    // Load land polygon for coastline refinement
    if (!std::filesystem::exists(LAND_POLYGON_GPKG_PATH)) {
        GTEST_SKIP() << "Land polygon file not found: " << LAND_POLYGON_GPKG_PATH;
    }

    std::cout << "Loading land polygon from: " << LAND_POLYGON_GPKG_PATH << "\n";

    CoastlineReader coastline_reader;
    bool loaded = coastline_reader.load(LAND_POLYGON_GPKG_PATH,
                                         LAND_POLYGON_LAYER,
                                         LAND_POLYGON_PROJECTION);
    ASSERT_TRUE(loaded) << "Failed to load land polygon: " << coastline_reader.last_error();

    // Swap X/Y coordinates (required for EPSG:3034)
    coastline_reader.swap_xy();

    // Remove small islands (area < 1 km²)
    const double min_area = 1.0e6;  // 1 km² in m²
    coastline_reader.remove_small_polygons(min_area);

    std::cout << "Loaded " << coastline_reader.num_polygons() << " land polygons\n";
    auto bbox = coastline_reader.bounding_box();
    std::cout << "Coastline bounds: x=[" << bg::get<0>(bbox.min_corner()) << ", "
              << bg::get<0>(bbox.max_corner()) << "], y=["
              << bg::get<1>(bbox.min_corner()) << ", "
              << bg::get<1>(bbox.max_corner()) << "]\n";

    // Build R-tree index for fast coastline intersection queries
    auto coastline_index = std::make_shared<CoastlineIndex>();
    coastline_index->build(coastline_reader.polygons());
    std::cout << "Built coastline R-tree with " << coastline_index->num_segments() << " segments\n";

    // Build adaptive octree using coastline refinement criteria
    OctreeAdapter octree(bathy_ptr->xmin, bathy_ptr->xmax,
                         bathy_ptr->ymin, bathy_ptr->ymax,
                         -1.0, 0.0);  // Sigma coordinates

    // Maximum refinement levels per axis
    const int max_level_x = 6;
    const int max_level_y = 6;
    const int max_level_z = 0;  // No vertical refinement for 2D view

    // Bathymetry gradient threshold for refinement
    // Refine where |grad(h)| > threshold (dimensionless slope)
    const Real bathymetry_gradient_threshold = 0.005;  // 0.5% slope

    // Create coastline refinement criterion
    CoastlineRefinement coastline_criterion(coastline_index, max_level_x);

    // Lambda to compute bathymetry gradient magnitude in an element
    auto compute_bathymetry_gradient = [&bathy_ptr](const ElementBounds& bounds) -> Real {
        // Sample depths at corners and center
        Real h00 = bathy_ptr->get_depth(bounds.xmin, bounds.ymin);
        Real h10 = bathy_ptr->get_depth(bounds.xmax, bounds.ymin);
        Real h01 = bathy_ptr->get_depth(bounds.xmin, bounds.ymax);
        Real h11 = bathy_ptr->get_depth(bounds.xmax, bounds.ymax);

        // Skip if any corner is land (depth = 0)
        if (h00 <= 0 || h10 <= 0 || h01 <= 0 || h11 <= 0) {
            return 0.0;
        }

        // Compute gradient using finite differences
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;

        // dh/dx at bottom and top edges
        Real dhdx_bottom = (h10 - h00) / dx;
        Real dhdx_top = (h11 - h01) / dx;
        Real dhdx = 0.5 * (dhdx_bottom + dhdx_top);

        // dh/dy at left and right edges
        Real dhdy_left = (h01 - h00) / dy;
        Real dhdy_right = (h11 - h10) / dy;
        Real dhdy = 0.5 * (dhdy_left + dhdy_right);

        // Gradient magnitude (dimensionless slope)
        return std::sqrt(dhdx * dhdx + dhdy * dhdy);
    };

    // Build adaptive mesh: refine where element intersects coastline OR has steep bathymetry
    octree.build_adaptive(
        [&coastline_criterion, &compute_bathymetry_gradient, bathymetry_gradient_threshold](const ElementBounds& bounds) -> bool {
            // Criterion 1: Coastline intersection (R-tree based)
            if (coastline_criterion.should_refine(bounds, 0)) {
                return true;
            }

            // Criterion 2: Steep bathymetry gradient
            Real grad_mag = compute_bathymetry_gradient(bounds);
            if (grad_mag > bathymetry_gradient_threshold) {
                return true;
            }

            return false;
        },
        max_level_x, max_level_y, max_level_z);

    // Balance the octree to ensure 2:1 constraint
    octree.balance();

    std::cout << "Adaptive octree mesh: " << octree.num_elements() << " elements\n";
    ASSERT_GT(octree.num_elements(), 0);

    // Create high-order DG basis for node positions
    const int poly_order = 3;
    HexahedronBasis basis(poly_order);
    const auto& ref_nodes = basis.lgl_nodes();
    const int nodes_per_elem = static_cast<int>(ref_nodes.size());

    // Write VTK output for visualization
    std::string vtk_path = "/tmp/danish_bathymetry_adaptive.vtk";
    std::ofstream vtk_file(vtk_path);

    // VTK 5.1 format (required for Lagrange hexahedra)
    vtk_file << "# vtk DataFile Version 5.1\n";
    vtk_file << "DRIFTER Coastline Adaptive Octree Mesh - High Order Elements\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";

    // Get VTK point ordering
    auto vtk_point_order = create_vtk_point_ordering(poly_order);

    // First pass: collect bathymetry depths for all elements in tensor-product order
    // Store elem_depths[elem_idx][node_idx]
    std::vector<std::vector<Real>> all_elem_depths(octree.num_elements());

    octree.for_each_element([&](Index e, const OctreeNode& node) {
        const auto& bounds = node.bounds;
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;

        all_elem_depths[e].resize(nodes_per_elem);

        for (size_t i = 0; i < ref_nodes.size(); ++i) {
            const auto& ref_node = ref_nodes[i];
            Real x = bounds.xmin + 0.5 * (ref_node.x() + 1.0) * dx;
            Real y = bounds.ymin + 0.5 * (ref_node.y() + 1.0) * dy;
            all_elem_depths[e][i] = bathy_ptr->get_depth(x, y);
        }
    });

    // Second pass: project coarse cell bathymetry onto fine cell face nodes
    // at non-conforming interfaces to ensure mesh continuity
    const int n1d = poly_order + 1;

    // Extract 1D LGL points from 3D reference nodes
    std::vector<Real> lgl_pts(n1d);
    for (int i = 0; i < n1d; ++i) {
        lgl_pts[i] = ref_nodes[i].x();
    }

    // Create interpolator for non-conforming projection
    // Use Bernstein for bounded interpolation that matches the VTK writer
    SeabedInterpolator projection_interp(poly_order, SeabedInterpolation::Bernstein);

    // Wrapper to evaluate coarse bathymetry polynomial at reference coordinates
    auto interp_coarse_bathy = [&](const std::vector<Real>& coarse_depths,
                                    Real xi, Real eta) -> Real {
        // Convert std::vector to VecX for 2D bottom face data (k=0 layer)
        VecX coarse_2d(n1d * n1d);
        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                int idx_2d = i + n1d * j;
                coarse_2d(idx_2d) = coarse_depths[idx_2d];  // k=0 layer
            }
        }
        return projection_interp.evaluate_scalar_2d(coarse_2d, xi, eta);
    };

    // Find non-conforming interfaces and project coarse bathymetry to fine cells
    int num_projections = 0;
    for (Index e = 0; e < octree.num_elements(); ++e) {
        const OctreeNode& node = *octree.elements()[e];
        const auto& bounds = node.bounds;

        // Check each horizontal face for coarser neighbors
        for (int face_id = 0; face_id < 4; ++face_id) {
            NeighborInfo info = octree.get_neighbor(e, face_id);
            if (info.is_boundary() || info.neighbor_elements.empty()) continue;

            // Get the neighbor - for non-conforming, this returns the coarser element
            Index neigh_e = info.neighbor_elements[0];
            const OctreeNode& neigh_node = *octree.elements()[neigh_e];

            // Check if neighbor is coarser (larger element size)
            Real my_size = (bounds.xmax - bounds.xmin) * (bounds.ymax - bounds.ymin);
            Real neigh_size = (neigh_node.bounds.xmax - neigh_node.bounds.xmin) *
                              (neigh_node.bounds.ymax - neigh_node.bounds.ymin);

            if (neigh_size > my_size * 1.5) {  // Neighbor is coarser
                const auto& coarse_bounds = neigh_node.bounds;
                num_projections++;

                // Project coarse bathymetry onto this element's face nodes
                for (int k = 0; k < n1d; ++k) {
                    for (int j = 0; j < n1d; ++j) {
                        for (int i = 0; i < n1d; ++i) {
                            // Check if node is on the interface face
                            bool on_face = false;
                            if (face_id == 0 && i == 0) on_face = true;       // -x face
                            if (face_id == 1 && i == n1d-1) on_face = true;   // +x face
                            if (face_id == 2 && j == 0) on_face = true;       // -y face
                            if (face_id == 3 && j == n1d-1) on_face = true;   // +y face

                            if (!on_face) continue;

                            int fine_idx = i + j * n1d + k * n1d * n1d;

                            // Physical position of this fine node
                            Real x = bounds.xmin + 0.5 * (lgl_pts[i] + 1.0) * (bounds.xmax - bounds.xmin);
                            Real y = bounds.ymin + 0.5 * (lgl_pts[j] + 1.0) * (bounds.ymax - bounds.ymin);

                            // Map to coarse element's reference coordinates [-1, 1]
                            Real xi_c = 2.0 * (x - coarse_bounds.xmin) / (coarse_bounds.xmax - coarse_bounds.xmin) - 1.0;
                            Real eta_c = 2.0 * (y - coarse_bounds.ymin) / (coarse_bounds.ymax - coarse_bounds.ymin) - 1.0;

                            // Clamp to valid range
                            xi_c = std::clamp(xi_c, -1.0, 1.0);
                            eta_c = std::clamp(eta_c, -1.0, 1.0);

                            // Evaluate coarse bathymetry polynomial at this point
                            Real proj_depth = interp_coarse_bathy(all_elem_depths[neigh_e], xi_c, eta_c);

                            // Update fine element's bathymetry at this face node
                            all_elem_depths[e][fine_idx] = proj_depth;
                        }
                    }
                }
            }
        }
    }
    std::cout << "Projected coarse bathymetry at " << num_projections << " non-conforming faces\n";

    // Final pass: build VTK points with projected bathymetry
    std::vector<Vec3> all_points;
    std::vector<Real> point_depths;
    std::vector<Real> point_levels;  // Store refinement level for visualization

    octree.for_each_element([&](Index e, const OctreeNode& node) {
        const auto& bounds = node.bounds;

        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real dz_sigma = bounds.zmax - bounds.zmin;

        // Collect points in tensor-product order first
        std::vector<Vec3> elem_points(nodes_per_elem);

        for (size_t i = 0; i < ref_nodes.size(); ++i) {
            const auto& ref_node = ref_nodes[i];
            Real x = bounds.xmin + 0.5 * (ref_node.x() + 1.0) * dx;
            Real y = bounds.ymin + 0.5 * (ref_node.y() + 1.0) * dy;
            Real sigma = bounds.zmin + 0.5 * (ref_node.z() + 1.0) * dz_sigma;

            // Use the (potentially projected) depth
            Real local_depth = all_elem_depths[e][i];
            Real z = sigma * local_depth;

            elem_points[i] = Vec3(x, y, z);
        }

        // Reorder to VTK point order
        for (int vtk_id = 0; vtk_id < nodes_per_elem; ++vtk_id) {
            int tensor_idx = vtk_point_order[vtk_id];
            all_points.push_back(elem_points[tensor_idx]);
            point_depths.push_back(all_elem_depths[e][tensor_idx]);
            point_levels.push_back(static_cast<Real>(node.level.level_x));
        }
    });

    // Write points
    vtk_file << "POINTS " << all_points.size() << " double\n";
    for (const auto& p : all_points) {
        vtk_file << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    // Write cells as VTK_LAGRANGE_HEXAHEDRON using VTK 5.1 format
    size_t num_cells = octree.num_elements();
    auto vtk_connectivity = get_vtk_lagrange_hex_connectivity(poly_order);

    size_t num_offsets = num_cells + 1;
    size_t num_connectivity = num_cells * nodes_per_elem;
    vtk_file << "\nCELLS " << num_offsets << " " << num_connectivity << "\n";
    vtk_file << "OFFSETS vtktypeint64\n";
    for (size_t e = 0; e <= num_cells; ++e) {
        vtk_file << (e * nodes_per_elem) << " ";
    }
    vtk_file << "\n";
    vtk_file << "CONNECTIVITY vtktypeint64\n";
    for (size_t e = 0; e < num_cells; ++e) {
        Index base = static_cast<Index>(e * nodes_per_elem);
        for (int conn_idx : vtk_connectivity) {
            vtk_file << (base + conn_idx) << " ";
        }
        vtk_file << "\n";
    }

    vtk_file << "\nCELL_TYPES " << num_cells << "\n";
    for (size_t e = 0; e < num_cells; ++e) {
        vtk_file << "72\n";
    }

    // Point data
    vtk_file << "\nPOINT_DATA " << all_points.size() << "\n";
    vtk_file << "SCALARS depth double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (Real d : point_depths) {
        vtk_file << d << "\n";
    }
    vtk_file << "SCALARS refinement_level double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (Real lev : point_levels) {
        vtk_file << lev << "\n";
    }

    // Cell data: refinement level
    vtk_file << "\nCELL_DATA " << num_cells << "\n";
    vtk_file << "SCALARS cell_level double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    octree.for_each_element([&](Index e, const OctreeNode& node) {
        vtk_file << node.level.level_x << "\n";
    });

    vtk_file.close();

    EXPECT_TRUE(std::filesystem::exists(vtk_path));
    std::cout << "Coastline-adaptive VTK: " << vtk_path << "\n";
    std::cout << "Elements refined near coastline with max level " << max_level_x << "\n";

    // =========================================================================
    // Generate high-resolution seabed surface using SeabedVTKWriter
    // =========================================================================

    // Build element coordinates in the format expected by SeabedVTKWriter
    // (interleaved x,y,z for each DOF)
    std::vector<VecX> element_coords_interleaved;
    std::vector<VecX> element_depth_data;

    element_coords_interleaved.reserve(octree.num_elements());
    element_depth_data.reserve(octree.num_elements());

    octree.for_each_element([&](Index e, const OctreeNode& node) {
        const auto& bounds = node.bounds;
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real dz_sigma = bounds.zmax - bounds.zmin;

        VecX coords(3 * nodes_per_elem);
        VecX depths(nodes_per_elem);

        for (int i = 0; i < nodes_per_elem; ++i) {
            const auto& ref_node = ref_nodes[i];

            Real x = bounds.xmin + 0.5 * (ref_node.x() + 1.0) * dx;
            Real y = bounds.ymin + 0.5 * (ref_node.y() + 1.0) * dy;
            Real sigma = bounds.zmin + 0.5 * (ref_node.z() + 1.0) * dz_sigma;

            // Use the projected depth from earlier pass
            Real local_depth = all_elem_depths[e][i];
            Real z = sigma * local_depth;

            coords(3 * i + 0) = x;
            coords(3 * i + 1) = y;
            coords(3 * i + 2) = z;
            depths(i) = local_depth;
        }

        element_coords_interleaved.push_back(coords);
        element_depth_data.push_back(depths);
    });

    // Note: Non-conforming projection was already applied to all_elem_depths above
    // using the same Bernstein interpolation as the VTK writer

    // Write high-resolution seabed surface
    std::string seabed_path = "/tmp/danish_seabed_adaptive";

    SeabedVTKWriter seabed_writer(seabed_path);
    seabed_writer.set_mesh(octree, element_coords_interleaved, poly_order);
    seabed_writer.set_resolution(10);  // 10x10 quads per element face
    seabed_writer.add_scalar_field("depth", element_depth_data);

    seabed_writer.write();

    std::string seabed_vtk_path = seabed_path + ".vtu";
    EXPECT_TRUE(std::filesystem::exists(seabed_vtk_path));

    std::cout << "\nHigh-resolution seabed surface: " << seabed_vtk_path << "\n";
    std::cout << "  Points: " << seabed_writer.num_points() << "\n";
    std::cout << "  Cells (quads): " << seabed_writer.num_cells() << "\n";
    std::cout << "  Resolution: 10x10 subdivisions per element bottom face\n";
}

TEST_F(SimulationTest, HighResolutionSeabedVTK) {
    // Test the SeabedVTKWriter with coastline-adaptive mesh
    // This generates a high-resolution 2D surface mesh of the seabed
    // by evaluating Lagrange polynomials at many points on each element's bottom face

    if (!GeoTiffReader::is_available()) {
        GTEST_SKIP() << "GDAL not available, skipping GeoTIFF test";
    }

    std::string geotiff_path = BATHYMETRY_GEOTIFF_PATH;
    if (!std::filesystem::exists(geotiff_path)) {
        GTEST_SKIP() << "GeoTIFF file not found: " << geotiff_path;
    }

    GeoTiffReader reader;
    BathymetryData bathy = reader.load(geotiff_path);
    ASSERT_TRUE(bathy.is_valid());

    auto bathy_ptr = std::make_shared<BathymetryData>(std::move(bathy));

    // Build mesh with bathymetry generator
    BathymetryMeshGenerator generator(bathy_ptr);

    BathymetryMeshGenerator::Config config;
    config.base_nx = 20;
    config.base_ny = 20;
    config.base_nz = 3;  // 3 vertical layers
    config.mask_land = true;
    config.min_depth = 1.0;
    generator.set_config(config);

    auto element_bounds = generator.generate();

    std::cout << "Generated " << element_bounds.size() << " water elements\n";

    if (element_bounds.empty()) {
        GTEST_SKIP() << "No water elements generated";
    }

    // Build OctreeAdapter from element bounds
    OctreeAdapter octree(bathy_ptr->xmin, bathy_ptr->xmax,
                         bathy_ptr->ymin, bathy_ptr->ymax,
                         -1.0, 0.0);  // Sigma coordinates
    octree.build_uniform(config.base_nx, config.base_ny, config.base_nz);

    // Create high-order DG basis
    const int poly_order = 3;
    HexahedronBasis basis(poly_order);
    const auto& ref_nodes = basis.lgl_nodes();
    const int num_dofs = static_cast<int>(ref_nodes.size());

    // Generate element coordinates with bathymetry-following bottom
    std::vector<VecX> element_coords;
    std::vector<VecX> element_depths;

    const auto& elements = octree.elements();
    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& bounds = elements[e]->bounds;

        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real dz_sigma = bounds.zmax - bounds.zmin;

        VecX coords(3 * num_dofs);
        VecX depths(num_dofs);

        for (int i = 0; i < num_dofs; ++i) {
            const auto& ref_node = ref_nodes[i];

            // Horizontal position
            Real x = bounds.xmin + 0.5 * (ref_node.x() + 1.0) * dx;
            Real y = bounds.ymin + 0.5 * (ref_node.y() + 1.0) * dy;

            // Sigma coordinate
            Real sigma = bounds.zmin + 0.5 * (ref_node.z() + 1.0) * dz_sigma;

            // Get local bathymetry
            Real local_depth = bathy_ptr->get_depth(x, y);
            if (local_depth < 1.0) local_depth = 1.0;  // Minimum depth

            // Physical z from sigma: z = sigma * H(x,y)
            Real z = sigma * local_depth;

            coords(3 * i + 0) = x;
            coords(3 * i + 1) = y;
            coords(3 * i + 2) = z;
            depths(i) = local_depth;
        }

        element_coords.push_back(coords);
        element_depths.push_back(depths);
    }

    std::cout << "Generated coordinates for " << element_coords.size()
              << " elements with " << num_dofs << " DOFs each\n";

    // Write high-resolution seabed VTK
    std::string seabed_path = "/tmp/danish_seabed_highres";

    SeabedVTKWriter seabed_writer(seabed_path);
    seabed_writer.set_mesh(octree, element_coords, poly_order);
    seabed_writer.set_resolution(20);  // 20x20 quads per element face = smooth surface
    seabed_writer.add_scalar_field("depth", element_depths);

    seabed_writer.write();

    std::string seabed_vtk_path = seabed_path + ".vtu";
    EXPECT_TRUE(std::filesystem::exists(seabed_vtk_path));

    std::cout << "High-resolution seabed VTK: " << seabed_vtk_path << "\n";
    std::cout << "  Points: " << seabed_writer.num_points() << "\n";
    std::cout << "  Cells (quads): " << seabed_writer.num_cells() << "\n";
    std::cout << "  Resolution: 20x20 subdivisions per element bottom face\n";
    std::cout << "Open in ParaView to visualize the smooth seabed surface\n";
}
