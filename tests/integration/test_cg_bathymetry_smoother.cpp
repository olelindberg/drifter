#include <gtest/gtest.h>
#include "bathymetry/cg_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include "mesh/seabed_surface.hpp"
#include "test_integration_fixtures.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace drifter;
using namespace drifter::testing;

class CGBathymetrySmootherTest : public ::testing::Test {
protected:
    static constexpr Real TOLERANCE = 1e-10;
    static constexpr Real LOOSE_TOLERANCE = 1e-6;

    std::unique_ptr<OctreeAdapter> create_octree(int nx, int ny, int nz) {
        auto octree = std::make_unique<OctreeAdapter>(
            0.0, 100.0,   // x bounds
            0.0, 100.0,   // y bounds
            -1.0, 0.0     // z bounds
        );
        octree->build_uniform(nx, ny, nz);
        return octree;
    }
};

// =============================================================================
// Construction tests
// =============================================================================

TEST_F(CGBathymetrySmootherTest, ConstructFromOctree) {
    auto octree = create_octree(4, 4, 2);

    Real alpha = 0.01;
    Real beta = 1000.0;

    CGBathymetrySmoother smoother(*octree, alpha, beta);

    EXPECT_EQ(smoother.alpha(), alpha);
    EXPECT_EQ(smoother.beta(), beta);
    EXPECT_GT(smoother.num_dofs(), 0);
    EXPECT_FALSE(smoother.is_solved());
}

TEST_F(CGBathymetrySmootherTest, ConstructFromQuadtree) {
    QuadtreeAdapter quadtree;
    quadtree.build_uniform(0.0, 10.0, 0.0, 10.0, 4, 4);

    CGBathymetrySmoother smoother(quadtree, 0.01, 1000.0);

    EXPECT_GT(smoother.num_dofs(), 0);
    EXPECT_FALSE(smoother.is_solved());
}

// =============================================================================
// Solve tests
// =============================================================================

TEST_F(CGBathymetrySmootherTest, SolveConstantBathymetry) {
    auto octree = create_octree(2, 2, 1);

    Real alpha = 0.0;  // Pure data fitting
    Real beta = 1.0;
    Real constant_depth = 50.0;

    CGBathymetrySmoother smoother(*octree, alpha, beta);
    smoother.set_bathymetry_data([constant_depth](Real, Real) {
        return constant_depth;
    });

    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Evaluate at several points
    const auto& mesh = smoother.mesh();
    for (Index e = 0; e < mesh.num_elements(); ++e) {
        Vec2 center = mesh.element_center(e);
        Real depth = smoother.evaluate(center(0), center(1));
        EXPECT_NEAR(depth, constant_depth, LOOSE_TOLERANCE);
    }
}

TEST_F(CGBathymetrySmootherTest, SolveLinearBathymetry) {
    auto octree = create_octree(2, 2, 1);

    Real alpha = 0.0;  // Pure data fitting
    Real beta = 1.0;

    // Linear function: z = 10 + 0.5*x + 0.3*y
    auto linear_bathy = [](Real x, Real y) {
        return 10.0 + 0.5 * x + 0.3 * y;
    };

    CGBathymetrySmoother smoother(*octree, alpha, beta);
    smoother.set_bathymetry_data(linear_bathy);
    smoother.solve();

    EXPECT_TRUE(smoother.is_solved());

    // Verify at DOF positions
    const auto& solution = smoother.solution();
    const auto& dofs = smoother.dof_manager();
    const auto& mesh = smoother.mesh();

    for (Index e = 0; e < mesh.num_elements(); ++e) {
        const auto& bounds = mesh.element_bounds(e);
        Vec2 center = mesh.element_center(e);

        Real expected = linear_bathy(center(0), center(1));
        Real computed = smoother.evaluate(center(0), center(1));

        EXPECT_NEAR(computed, expected, LOOSE_TOLERANCE)
            << "At center (" << center(0) << ", " << center(1) << ")";
    }
}

TEST_F(CGBathymetrySmootherTest, SolveQuadraticBathymetry) {
    auto octree = create_octree(2, 2, 1);

    Real alpha = 0.0;  // Pure data fitting
    Real beta = 1.0;

    // Quadratic function that quintic basis can represent exactly
    auto quadratic_bathy = [](Real x, Real y) {
        return 100.0 + 0.01 * x * x + 0.01 * y * y;
    };

    CGBathymetrySmoother smoother(*octree, alpha, beta);
    smoother.set_bathymetry_data(quadratic_bathy);
    smoother.solve();

    // Check at several test points
    std::vector<Vec2> test_points = {
        Vec2(25.0, 25.0),
        Vec2(50.0, 50.0),
        Vec2(75.0, 75.0),
        Vec2(30.0, 60.0),
    };

    for (const auto& pt : test_points) {
        Real expected = quadratic_bathy(pt(0), pt(1));
        Real computed = smoother.evaluate(pt(0), pt(1));

        EXPECT_NEAR(computed, expected, LOOSE_TOLERANCE)
            << "At (" << pt(0) << ", " << pt(1) << ")";
    }
}

TEST_F(CGBathymetrySmootherTest, SolveBeforeBathymetrySetThrows) {
    auto octree = create_octree(2, 2, 1);
    CGBathymetrySmoother smoother(*octree, 0.01, 1.0);

    EXPECT_THROW(smoother.solve(), std::runtime_error);
}

TEST_F(CGBathymetrySmootherTest, EvaluateBeforeSolveThrows) {
    auto octree = create_octree(2, 2, 1);
    CGBathymetrySmoother smoother(*octree, 0.01, 1.0);
    smoother.set_bathymetry_data([](Real, Real) { return 1.0; });

    EXPECT_THROW(smoother.evaluate(50.0, 50.0), std::runtime_error);
}

// =============================================================================
// Smoothing effect tests
// =============================================================================

TEST_F(CGBathymetrySmootherTest, SmoothingReducesOscillations) {
    auto octree = create_octree(4, 4, 1);

    // High-frequency noisy data
    auto noisy_bathy = [](Real x, Real y) {
        return 50.0 + 5.0 * std::sin(0.5 * x) * std::sin(0.5 * y);
    };

    // Without smoothing
    CGBathymetrySmoother no_smooth(*octree, 0.0, 1.0);
    no_smooth.set_bathymetry_data(noisy_bathy);
    no_smooth.solve();

    // With smoothing
    CGBathymetrySmoother with_smooth(*octree, 0.1, 1.0);
    with_smooth.set_bathymetry_data(noisy_bathy);
    with_smooth.solve();

    // Compute variance of solutions
    const auto& mesh = no_smooth.mesh();
    Real var_no_smooth = 0.0;
    Real var_with_smooth = 0.0;
    Real mean_no_smooth = 0.0;
    Real mean_with_smooth = 0.0;
    int count = 0;

    // Sample at multiple points
    for (Real x = 10.0; x < 90.0; x += 10.0) {
        for (Real y = 10.0; y < 90.0; y += 10.0) {
            mean_no_smooth += no_smooth.evaluate(x, y);
            mean_with_smooth += with_smooth.evaluate(x, y);
            count++;
        }
    }
    mean_no_smooth /= count;
    mean_with_smooth /= count;

    for (Real x = 10.0; x < 90.0; x += 10.0) {
        for (Real y = 10.0; y < 90.0; y += 10.0) {
            Real v1 = no_smooth.evaluate(x, y) - mean_no_smooth;
            Real v2 = with_smooth.evaluate(x, y) - mean_with_smooth;
            var_no_smooth += v1 * v1;
            var_with_smooth += v2 * v2;
        }
    }

    // Smoothed solution should have less variance
    EXPECT_LT(var_with_smooth, var_no_smooth * 1.1);
}

// =============================================================================
// Gradient evaluation tests
// =============================================================================

TEST_F(CGBathymetrySmootherTest, GradientOfConstant) {
    auto octree = create_octree(2, 2, 1);

    CGBathymetrySmoother smoother(*octree, 0.0, 1.0);
    smoother.set_bathymetry_data([](Real, Real) { return 100.0; });
    smoother.solve();

    Vec2 grad = smoother.evaluate_gradient(50.0, 50.0);

    EXPECT_NEAR(grad(0), 0.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(grad(1), 0.0, LOOSE_TOLERANCE);
}

TEST_F(CGBathymetrySmootherTest, GradientOfLinear) {
    auto octree = create_octree(2, 2, 1);

    // Linear: z = 10 + 2*x + 3*y
    // Gradient: (2, 3)
    CGBathymetrySmoother smoother(*octree, 0.0, 1.0);
    smoother.set_bathymetry_data([](Real x, Real y) {
        return 10.0 + 2.0 * x + 3.0 * y;
    });
    smoother.solve();

    Vec2 grad = smoother.evaluate_gradient(50.0, 50.0);

    EXPECT_NEAR(grad(0), 2.0, LOOSE_TOLERANCE);
    EXPECT_NEAR(grad(1), 3.0, LOOSE_TOLERANCE);
}

// =============================================================================
// Transfer to SeabedSurface tests
// =============================================================================

TEST_F(CGBathymetrySmootherTest, TransferToSeabed) {
    auto octree = create_octree(2, 2, 2);

    // Create smoother and solve
    CGBathymetrySmoother smoother(*octree, 0.0, 1.0);
    smoother.set_bathymetry_data([](Real x, Real y) {
        return 50.0 + 0.5 * x;  // Linear slope
    });
    smoother.solve();

    // Create SeabedSurface
    int dg_order = 3;
    SeabedSurface seabed(*octree, dg_order);

    // Transfer CG solution to seabed
    smoother.transfer_to_seabed(seabed);

    // Verify transferred values at test points
    std::vector<Vec2> test_points = {
        Vec2(25.0, 25.0),
        Vec2(50.0, 50.0),
        Vec2(75.0, 75.0),
    };

    for (const auto& pt : test_points) {
        Real expected = 50.0 + 0.5 * pt(0);
        Real from_seabed = seabed.depth(pt(0), pt(1));

        // Allow larger tolerance due to interpolation
        EXPECT_NEAR(from_seabed, expected, 1.0)
            << "At (" << pt(0) << ", " << pt(1) << ")";
    }
}

// =============================================================================
// DOF access tests
// =============================================================================

TEST_F(CGBathymetrySmootherTest, SolutionAtDof) {
    auto octree = create_octree(2, 2, 1);

    CGBathymetrySmoother smoother(*octree, 0.0, 1.0);
    smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
    smoother.solve();

    // All DOFs should have approximately the same value
    for (Index d = 0; d < smoother.num_dofs(); ++d) {
        EXPECT_NEAR(smoother.solution_at_dof(d), 42.0, LOOSE_TOLERANCE);
    }
}

TEST_F(CGBathymetrySmootherTest, SolutionVectorAccess) {
    auto octree = create_octree(2, 2, 1);

    CGBathymetrySmoother smoother(*octree, 0.0, 1.0);
    smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
    smoother.solve();

    const VecX& solution = smoother.solution();
    EXPECT_EQ(solution.size(), smoother.num_dofs());
}

// =============================================================================
// Evaluation outside domain tests
// =============================================================================

TEST_F(CGBathymetrySmootherTest, EvaluateOutsideDomain) {
    auto octree = create_octree(2, 2, 1);

    CGBathymetrySmoother smoother(*octree, 0.0, 1.0);
    smoother.set_bathymetry_data([](Real, Real) { return 42.0; });
    smoother.solve();

    // Point outside domain should return NaN
    Real outside_value = smoother.evaluate(-10.0, 50.0);
    EXPECT_TRUE(std::isnan(outside_value));

    Vec2 outside_grad = smoother.evaluate_gradient(-10.0, 50.0);
    EXPECT_TRUE(std::isnan(outside_grad(0)));
    EXPECT_TRUE(std::isnan(outside_grad(1)));
}

// =============================================================================
// Parameter ratio tests
// =============================================================================

TEST_F(CGBathymetrySmootherTest, HighBetaMatchesData) {
    auto octree = create_octree(2, 2, 1);

    // Very high beta means solution should closely match data
    auto bathy_func = [](Real x, Real y) {
        return 100.0 + x * y / 1000.0;
    };

    CGBathymetrySmoother smoother(*octree, 0.001, 10000.0);
    smoother.set_bathymetry_data(bathy_func);
    smoother.solve();

    // Check at center
    Real expected = bathy_func(50.0, 50.0);
    Real computed = smoother.evaluate(50.0, 50.0);

    EXPECT_NEAR(computed, expected, 0.1);
}

TEST_F(CGBathymetrySmootherTest, HighAlphaFlattens) {
    auto octree = create_octree(2, 2, 1);

    // Curved data
    auto curved_bathy = [](Real x, Real y) {
        return 100.0 + 0.001 * x * x;
    };

    // Very high alpha should flatten the solution
    CGBathymetrySmoother smoother(*octree, 1000.0, 1.0);
    smoother.set_bathymetry_data(curved_bathy);
    smoother.solve();

    // Solution should be relatively flat (low range)
    Real min_val = std::numeric_limits<Real>::max();
    Real max_val = std::numeric_limits<Real>::lowest();

    for (Real x = 10.0; x < 90.0; x += 20.0) {
        for (Real y = 10.0; y < 90.0; y += 20.0) {
            Real val = smoother.evaluate(x, y);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }

    Real range = max_val - min_val;

    // The original data has range of about 8 (from x^2 term)
    // With high smoothing, range should be significantly reduced
    EXPECT_LT(range, 5.0);
}

// =============================================================================
// Mesh accessor tests
// =============================================================================

TEST_F(CGBathymetrySmootherTest, MeshAccessor) {
    auto octree = create_octree(2, 2, 1);
    CGBathymetrySmoother smoother(*octree, 0.01, 1.0);

    const auto& mesh = smoother.mesh();
    EXPECT_GT(mesh.num_elements(), 0);
}

TEST_F(CGBathymetrySmootherTest, DofManagerAccessor) {
    auto octree = create_octree(2, 2, 1);
    CGBathymetrySmoother smoother(*octree, 0.01, 1.0);

    const auto& dofs = smoother.dof_manager();
    EXPECT_EQ(dofs.num_global_dofs(), smoother.num_dofs());
}

// =============================================================================
// VTK Output tests (using SimulationTest fixture for output directory)
// =============================================================================

class CGBathymetryVTKTest : public SimulationTest {
protected:
    // Write VTU file showing element-wise DOF surface
    // Creates a bilinear grid surface per element using DOF positions and values
    void write_element_vtu(const std::string& filename,
                           const CGBathymetrySmoother& smoother,
                           std::function<Real(Real, Real)> raw_bathy) {
        const auto& mesh = smoother.mesh();
        const auto& dofs = smoother.dof_manager();
        const auto& basis = dofs.basis();
        const VecX& nodes = basis.nodes_1d();

        Index num_elements = mesh.num_elements();
        int n1d = QuinticBasis2D::N1D;  // 6 nodes per direction
        int points_per_elem = n1d * n1d;  // 36 points per element
        int cells_per_elem = (n1d - 1) * (n1d - 1);  // 25 quads per element

        Index total_points = num_elements * points_per_elem;
        Index total_cells = num_elements * cells_per_elem;

        std::ofstream file(filename);
        file << std::setprecision(12);
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        file << "<UnstructuredGrid>\n";
        file << "<Piece NumberOfPoints=\"" << total_points
             << "\" NumberOfCells=\"" << total_cells << "\">\n";

        // Points - DOF positions with solution as z
        file << "<Points>\n";
        file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (Index e = 0; e < num_elements; ++e) {
            const auto& bounds = mesh.element_bounds(e);
            const auto& elem_dofs = dofs.element_dofs(e);
            Real elem_dx = bounds.xmax - bounds.xmin;
            Real elem_dy = bounds.ymax - bounds.ymin;

            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    Real xi = nodes(i);
                    Real eta = nodes(j);
                    Real x = bounds.xmin + 0.5 * (xi + 1.0) * elem_dx;
                    Real y = bounds.ymin + 0.5 * (eta + 1.0) * elem_dy;

                    int local = basis.dof_index(i, j);
                    Index global = elem_dofs[local];
                    Real z = smoother.solution_at_dof(global);
                    if (std::isnan(z)) z = 0.0;

                    file << x << " " << y << " " << z << "\n";
                }
            }
        }
        file << "</DataArray>\n</Points>\n";

        // Cells - quads connecting adjacent DOFs within each element
        file << "<Cells>\n";
        file << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (Index e = 0; e < num_elements; ++e) {
            int base = e * points_per_elem;
            for (int j = 0; j < n1d - 1; ++j) {
                for (int i = 0; i < n1d - 1; ++i) {
                    int p0 = base + j * n1d + i;
                    int p1 = base + j * n1d + (i + 1);
                    int p2 = base + (j + 1) * n1d + (i + 1);
                    int p3 = base + (j + 1) * n1d + i;
                    file << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
                }
            }
        }
        file << "</DataArray>\n";
        file << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (Index c = 1; c <= total_cells; ++c) {
            file << c * 4 << "\n";
        }
        file << "</DataArray>\n";
        file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (Index c = 0; c < total_cells; ++c) {
            file << "9\n";  // VTK_QUAD
        }
        file << "</DataArray>\n</Cells>\n";

        // Point data
        file << "<PointData Scalars=\"solution\">\n";
        file << "<DataArray type=\"Float64\" Name=\"solution\" format=\"ascii\">\n";
        for (Index e = 0; e < num_elements; ++e) {
            const auto& elem_dofs = dofs.element_dofs(e);
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int local = basis.dof_index(i, j);
                    Index global = elem_dofs[local];
                    Real v = smoother.solution_at_dof(global);
                    file << (std::isnan(v) ? 0.0 : v) << "\n";
                }
            }
        }
        file << "</DataArray>\n";

        file << "<DataArray type=\"Float64\" Name=\"raw_bathy\" format=\"ascii\">\n";
        for (Index e = 0; e < num_elements; ++e) {
            const auto& bounds = mesh.element_bounds(e);
            Real elem_dx = bounds.xmax - bounds.xmin;
            Real elem_dy = bounds.ymax - bounds.ymin;
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    Real xi = nodes(i);
                    Real eta = nodes(j);
                    Real x = bounds.xmin + 0.5 * (xi + 1.0) * elem_dx;
                    Real y = bounds.ymin + 0.5 * (eta + 1.0) * elem_dy;
                    file << raw_bathy(x, y) << "\n";
                }
            }
        }
        file << "</DataArray>\n";

        file << "<DataArray type=\"Int32\" Name=\"global_dof\" format=\"ascii\">\n";
        for (Index e = 0; e < num_elements; ++e) {
            const auto& elem_dofs = dofs.element_dofs(e);
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int local = basis.dof_index(i, j);
                    file << elem_dofs[local] << "\n";
                }
            }
        }
        file << "</DataArray>\n";

        file << "<DataArray type=\"Int32\" Name=\"is_constrained\" format=\"ascii\">\n";
        for (Index e = 0; e < num_elements; ++e) {
            const auto& elem_dofs = dofs.element_dofs(e);
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int local = basis.dof_index(i, j);
                    Index global = elem_dofs[local];
                    file << (dofs.is_constrained(global) ? 1 : 0) << "\n";
                }
            }
        }
        file << "</DataArray>\n";
        file << "</PointData>\n";

        // Cell data
        file << "<CellData>\n";
        file << "<DataArray type=\"Int32\" Name=\"element_id\" format=\"ascii\">\n";
        for (Index e = 0; e < num_elements; ++e) {
            for (int c = 0; c < cells_per_elem; ++c) {
                file << e << "\n";
            }
        }
        file << "</DataArray>\n";
        file << "</CellData>\n";

        file << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
    }

    // Write VTU file with solution sampled on a regular grid
    void write_sampled_vtu(const std::string& filename,
                           const CGBathymetrySmoother& smoother,
                           std::function<Real(Real, Real)> raw_bathy,
                           int nx = 100, int ny = 100) {
        const auto& mesh = smoother.mesh();
        const auto& domain = mesh.domain_bounds();

        Real dx = (domain.xmax - domain.xmin) / nx;
        Real dy = (domain.ymax - domain.ymin) / ny;
        int total_points = (nx + 1) * (ny + 1);
        int total_cells = nx * ny;

        std::ofstream file(filename);
        file << std::setprecision(12);
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        file << "<UnstructuredGrid>\n";
        file << "<Piece NumberOfPoints=\"" << total_points
             << "\" NumberOfCells=\"" << total_cells << "\">\n";

        // Points
        file << "<Points>\n";
        file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                Real x = domain.xmin + i * dx;
                Real y = domain.ymin + j * dy;
                Real z = smoother.evaluate(x, y);
                file << x << " " << y << " " << z << "\n";
            }
        }
        file << "</DataArray>\n</Points>\n";

        // Cells
        file << "<Cells>\n";
        file << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int p0 = j * (nx + 1) + i;
                int p1 = p0 + 1;
                int p2 = p0 + (nx + 1) + 1;
                int p3 = p0 + (nx + 1);
                file << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
        }
        file << "</DataArray>\n";
        file << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int c = 1; c <= total_cells; ++c) {
            file << c * 4 << "\n";
        }
        file << "</DataArray>\n";
        file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int c = 0; c < total_cells; ++c) {
            file << "9\n";  // VTK_QUAD
        }
        file << "</DataArray>\n</Cells>\n";

        // Point data
        file << "<PointData Scalars=\"solution\">\n";
        file << "<DataArray type=\"Float64\" Name=\"solution\" format=\"ascii\">\n";
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                Real x = domain.xmin + i * dx;
                Real y = domain.ymin + j * dy;
                file << smoother.evaluate(x, y) << "\n";
            }
        }
        file << "</DataArray>\n";

        file << "<DataArray type=\"Float64\" Name=\"raw_bathy\" format=\"ascii\">\n";
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                Real x = domain.xmin + i * dx;
                Real y = domain.ymin + j * dy;
                file << raw_bathy(x, y) << "\n";
            }
        }
        file << "</DataArray>\n";

        file << "<DataArray type=\"Float64\" Name=\"difference\" format=\"ascii\">\n";
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                Real x = domain.xmin + i * dx;
                Real y = domain.ymin + j * dy;
                file << (smoother.evaluate(x, y) - raw_bathy(x, y)) << "\n";
            }
        }
        file << "</DataArray>\n";
        file << "</PointData>\n";

        file << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
    }
};

TEST_F(CGBathymetryVTKTest, UniformMeshVTKOutput) {
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 4, 4);

    auto bathy_func = [](Real x, Real y) {
        return 100.0 + 30.0 * std::sin(0.05 * x) * std::cos(0.05 * y);
    };

    CGBathymetrySmoother smoother(mesh, 0.01, 1.0);
    smoother.set_bathymetry_data(bathy_func);
    smoother.solve();

    // Write to /tmp/ directly so files persist after test
    std::string filename = "/tmp/cg_uniform_element.vtu";
    write_element_vtu(filename, smoother, bathy_func);

    EXPECT_TRUE(std::filesystem::exists(filename));
    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_GT(content.size(), 1000);
    // Type 9 is VTK_QUAD
    EXPECT_NE(content.find("<DataArray type=\"UInt8\" Name=\"types\""), std::string::npos);
}

TEST_F(CGBathymetryVTKTest, NonConformingMeshVTKOutput) {
    // Create simple non-conforming mesh: 2 fine elements meeting 1 coarse element
    // Fine elements on left, coarse on right
    //
    //  +---+---+-------+
    //  | 2 | 3 |       |
    //  +---+---+   4   |
    //  | 0 | 1 |       |
    //  +---+---+-------+
    //
    QuadtreeAdapter mesh;
    Real h = 25.0;  // fine element size

    // Fine elements (2x2, each of size h×h)
    mesh.add_element({0.0, h, 0.0, h}, {2, 2});       // elem 0: bottom-left
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});       // elem 1: bottom-right fine
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});       // elem 2: top-left
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});       // elem 3: top-right fine

    // Coarse element (size 2h×2h)
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});   // elem 4: coarse on right

    // Check what neighbor relationships were detected
    // Elements 1 and 3 (right edge at x=2h) should have FineToCoarse to elem 4
    auto info1 = mesh.get_neighbor(1, 1);  // elem 1, right edge
    auto info3 = mesh.get_neighbor(3, 1);  // elem 3, right edge
    EXPECT_EQ(info1.type, EdgeNeighborInfo::Type::FineToCoarse)
        << "Element 1 right edge should be FineToCoarse";
    EXPECT_EQ(info3.type, EdgeNeighborInfo::Type::FineToCoarse)
        << "Element 3 right edge should be FineToCoarse";

    auto bathy_func = [](Real x, Real y) {
        return 50.0 + x + 2.0 * y;  // Linear function
    };

    CGBathymetrySmoother smoother(mesh, 0.0, 1.0);  // Pure fitting, no smoothing
    smoother.set_bathymetry_data(bathy_func);
    smoother.solve();

    // Check constraints were applied
    EXPECT_GT(smoother.dof_manager().constraints().size(), 0)
        << "Expected constraints at non-conforming interface";

    // Write to /tmp/ directly so files persist after test
    // Write element-wise DOF surface
    write_element_vtu("/tmp/cg_nonconforming_element.vtu", smoother, bathy_func);
    // Write sampled grid for continuous visualization
    write_sampled_vtu("/tmp/cg_nonconforming_sampled.vtu", smoother, bathy_func, 50, 50);

    EXPECT_TRUE(std::filesystem::exists("/tmp/cg_nonconforming_element.vtu"));
    EXPECT_TRUE(std::filesystem::exists("/tmp/cg_nonconforming_sampled.vtu"));

    // Verify solution continuity at interface x=2*h (between fine and coarse)
    for (Real y = 1.0; y < 2*h - 1.0; y += 5.0) {
        Real left = smoother.evaluate(2*h - 0.01, y);
        Real right = smoother.evaluate(2*h + 0.01, y);
        EXPECT_NEAR(left, right, 1.0) << "Discontinuity at interface y=" << y;
    }
}

TEST_F(CGBathymetryVTKTest, FourPlusOneMeshVTKOutput) {
    // Simple 4-fine + 1-coarse mesh (clearest constraint visualization)
    //  +---+---+-------+
    //  | 2 | 3 |       |
    //  +---+---+   4   |
    //  | 0 | 1 |       |
    //  +---+---+-------+
    //
    // Fine elements: 25x25, Coarse element: 50x50
    QuadtreeAdapter mesh;
    Real h = 25.0;

    // Fine elements (level 2 = 4 elements per direction)
    mesh.add_element({0.0, h, 0.0, h}, {2, 2});       // elem 0
    mesh.add_element({h, 2*h, 0.0, h}, {2, 2});       // elem 1
    mesh.add_element({0.0, h, h, 2*h}, {2, 2});       // elem 2
    mesh.add_element({h, 2*h, h, 2*h}, {2, 2});       // elem 3

    // Coarse element (level 1 = 2 elements per direction)
    mesh.add_element({2*h, 4*h, 0.0, 2*h}, {1, 1});   // elem 4

    // Use smooth function to visualize smoothing behavior
    auto bathy_func = [](Real x, Real y) {
        return 100.0 + 20.0 * std::sin(0.05 * x) * std::cos(0.05 * y);
    };

    CGBathymetrySmoother smoother(mesh, 0.01, 1.0);
    smoother.set_bathymetry_data(bathy_func);
    smoother.solve();

    // Write to /tmp/ directly
    write_element_vtu("/tmp/cg_4plus1_element.vtu", smoother, bathy_func);
    write_sampled_vtu("/tmp/cg_4plus1_sampled.vtu", smoother, bathy_func, 50, 50);

    EXPECT_TRUE(std::filesystem::exists("/tmp/cg_4plus1_element.vtu"));
    EXPECT_TRUE(std::filesystem::exists("/tmp/cg_4plus1_sampled.vtu"));

    // Verify constraint count (should have constraints at interface)
    EXPECT_GT(smoother.dof_manager().constraints().size(), 0)
        << "Expected constraints at 4+1 non-conforming interface";
}

TEST_F(CGBathymetryVTKTest, NoisyBathymetrySmoothing) {
    // Test smoothing effect on noisy bathymetry data
    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 100.0, 0.0, 100.0, 4, 4);

    // Noisy bathymetry: smooth base + high-frequency noise
    auto noisy_bathy = [](Real x, Real y) {
        Real base = 100.0 + 0.5 * x + 0.3 * y;  // Linear trend
        Real noise = 5.0 * std::sin(0.5 * x) * std::sin(0.5 * y);  // High-freq noise
        return base + noise;
    };

    // No smoothing (alpha=0)
    CGBathymetrySmoother no_smooth(mesh, 0.0, 1.0);
    no_smooth.set_bathymetry_data(noisy_bathy);
    no_smooth.solve();

    // With strong smoothing (high alpha relative to beta)
    CGBathymetrySmoother with_smooth(mesh, 10.0, 1.0);
    with_smooth.set_bathymetry_data(noisy_bathy);
    with_smooth.solve();

    // Write both to compare
    write_element_vtu("/tmp/cg_noisy_raw_element.vtu", no_smooth, noisy_bathy);
    write_element_vtu("/tmp/cg_noisy_smooth_element.vtu", with_smooth, noisy_bathy);
    write_sampled_vtu("/tmp/cg_noisy_raw_sampled.vtu", no_smooth, noisy_bathy, 50, 50);
    write_sampled_vtu("/tmp/cg_noisy_smooth_sampled.vtu", with_smooth, noisy_bathy, 50, 50);

    EXPECT_TRUE(std::filesystem::exists("/tmp/cg_noisy_raw_element.vtu"));
    EXPECT_TRUE(std::filesystem::exists("/tmp/cg_noisy_smooth_element.vtu"));
}

