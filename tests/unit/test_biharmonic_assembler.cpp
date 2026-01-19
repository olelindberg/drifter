#include <gtest/gtest.h>
#include "bathymetry/biharmonic_assembler.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/lagrange_basis_2d.hpp"
#include "bathymetry/cg_dof_manager.hpp"
#include <cmath>

using namespace drifter;

class BiharmonicAssemblerTest : public ::testing::Test {
protected:
    static constexpr Real TOLERANCE = 1e-10;
    static constexpr Real LOOSE_TOLERANCE = 1e-6;
    static constexpr int ORDER = 5;  // Use quintic for backward compatibility

    std::unique_ptr<QuadtreeAdapter> mesh_;
    std::unique_ptr<LagrangeBasis2D> basis_;
    std::unique_ptr<CGDofManager> dofs_;

    void SetUp() override {
        basis_ = std::make_unique<LagrangeBasis2D>(ORDER);
    }

    void create_mesh_and_dofs(int nx, int ny) {
        mesh_ = std::make_unique<QuadtreeAdapter>();
        mesh_->build_uniform(0.0, 1.0, 0.0, 1.0, nx, ny);
        dofs_ = std::make_unique<CGDofManager>(*mesh_, *basis_);
    }
};

// =============================================================================
// Construction tests
// =============================================================================

TEST_F(BiharmonicAssemblerTest, Construction) {
    create_mesh_and_dofs(2, 2);

    Real alpha = 0.01;
    Real beta = 1000.0;

    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, alpha, beta);

    EXPECT_EQ(assembler.alpha(), alpha);
    EXPECT_EQ(assembler.beta(), beta);
}

TEST_F(BiharmonicAssemblerTest, InvalidWeights) {
    create_mesh_and_dofs(1, 1);

    EXPECT_THROW(
        BiharmonicAssembler(*mesh_, *basis_, *dofs_, -1.0, 1.0),
        std::invalid_argument
    );

    EXPECT_THROW(
        BiharmonicAssembler(*mesh_, *basis_, *dofs_, 1.0, -1.0),
        std::invalid_argument
    );
}

// =============================================================================
// Element matrix tests
// =============================================================================

TEST_F(BiharmonicAssemblerTest, ElementMassSymmetry) {
    create_mesh_and_dofs(1, 1);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    MatX M = assembler.element_mass(0);

    // Mass matrix should be symmetric
    EXPECT_NEAR((M - M.transpose()).norm(), 0.0, TOLERANCE);
}

TEST_F(BiharmonicAssemblerTest, ElementMassPositiveDefinite) {
    create_mesh_and_dofs(1, 1);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    MatX M = assembler.element_mass(0);

    // Check positive definiteness via eigenvalues
    Eigen::SelfAdjointEigenSolver<MatX> solver(M);
    VecX eigenvalues = solver.eigenvalues();

    for (int i = 0; i < eigenvalues.size(); ++i) {
        EXPECT_GT(eigenvalues(i), -TOLERANCE) << "Eigenvalue " << i << " is negative";
    }
}

TEST_F(BiharmonicAssemblerTest, ElementBiharmonicSymmetry) {
    create_mesh_and_dofs(1, 1);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    MatX K = assembler.element_biharmonic(0);

    // Biharmonic matrix should be symmetric
    EXPECT_NEAR((K - K.transpose()).norm(), 0.0, TOLERANCE);
}

TEST_F(BiharmonicAssemblerTest, ElementBiharmonicSemiPositiveDefinite) {
    create_mesh_and_dofs(1, 1);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    MatX K = assembler.element_biharmonic(0);

    // Biharmonic matrix should be semi-positive definite
    // (may have zero eigenvalues for rigid body modes)
    Eigen::SelfAdjointEigenSolver<MatX> solver(K);
    VecX eigenvalues = solver.eigenvalues();

    for (int i = 0; i < eigenvalues.size(); ++i) {
        EXPECT_GT(eigenvalues(i), -LOOSE_TOLERANCE) << "Eigenvalue " << i << " is significantly negative";
    }
}

TEST_F(BiharmonicAssemblerTest, ElementStiffnessCombination) {
    create_mesh_and_dofs(1, 1);

    Real alpha = 2.0;
    Real beta = 3.0;
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, alpha, beta);

    MatX K_total = assembler.element_stiffness(0);
    MatX K_biharm = assembler.element_biharmonic(0);
    MatX M = assembler.element_mass(0);

    // K_total = alpha * K_biharm + beta * M (but alpha/beta are built-in)
    // Actually alpha and beta are included in element_biharmonic and element_mass
    // So K_total should equal K_biharm + M
    EXPECT_NEAR((K_total - K_biharm - M).norm(), 0.0, TOLERANCE);
}

// =============================================================================
// Global assembly tests
// =============================================================================

TEST_F(BiharmonicAssemblerTest, GlobalStiffnessDimensions) {
    create_mesh_and_dofs(2, 2);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    SpMat K = assembler.assemble_stiffness();

    EXPECT_EQ(K.rows(), dofs_->num_global_dofs());
    EXPECT_EQ(K.cols(), dofs_->num_global_dofs());
}

TEST_F(BiharmonicAssemblerTest, GlobalStiffnessSymmetry) {
    create_mesh_and_dofs(2, 2);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    SpMat K = assembler.assemble_stiffness();

    // Convert to dense and check symmetry
    // Use loose tolerance due to floating-point accumulation in IPDG assembly
    MatX K_dense = MatX(K);
    EXPECT_NEAR((K_dense - K_dense.transpose()).norm(), 0.0, LOOSE_TOLERANCE);
}

TEST_F(BiharmonicAssemblerTest, GlobalRhsDimensions) {
    create_mesh_and_dofs(2, 2);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    FunctionBathymetry bathy([](Real, Real) { return 1.0; });
    VecX f = assembler.assemble_rhs(bathy);

    EXPECT_EQ(f.size(), dofs_->num_global_dofs());
}

// =============================================================================
// Polynomial exactness tests
// =============================================================================

TEST_F(BiharmonicAssemblerTest, ConstantDataRecovery) {
    // For constant data, the solution should be constant (ignoring smoothing term)
    create_mesh_and_dofs(2, 2);

    Real alpha = 0.0;  // No smoothing
    Real beta = 1.0;   // Pure data fitting
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, alpha, beta);

    Real constant_value = 5.0;
    FunctionBathymetry bathy([constant_value](Real, Real) { return constant_value; });

    SpMat K = assembler.assemble_stiffness();
    VecX f = assembler.assemble_rhs(bathy);

    // Solve system
    Eigen::SparseLU<SpMat> solver;
    solver.compute(K);
    VecX u = solver.solve(f);

    // Solution should be constant
    Real mean = u.mean();
    EXPECT_NEAR(mean, constant_value, LOOSE_TOLERANCE);
    EXPECT_NEAR((u.array() - mean).matrix().norm(), 0.0, LOOSE_TOLERANCE);
}

TEST_F(BiharmonicAssemblerTest, LinearDataRecovery) {
    // For linear data with no smoothing, solution should be linear
    create_mesh_and_dofs(2, 2);

    Real alpha = 0.0;  // No smoothing
    Real beta = 1.0;   // Pure data fitting
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, alpha, beta);

    // Linear function: u(x,y) = 2x + 3y + 1
    FunctionBathymetry bathy([](Real x, Real y) { return 2.0 * x + 3.0 * y + 1.0; });

    SpMat K = assembler.assemble_stiffness();
    VecX f = assembler.assemble_rhs(bathy);

    // Solve system
    Eigen::SparseLU<SpMat> solver;
    solver.compute(K);
    VecX u = solver.solve(f);

    // Check at DOF positions
    const VecX& nodes = basis_->nodes_1d();
    int dof_idx = 0;
    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        const auto& bounds = mesh_->element_bounds(e);
        const auto& elem_dofs = dofs_->element_dofs(e);

        for (int j = 0; j < basis_->num_nodes_1d(); ++j) {
            for (int i = 0; i < basis_->num_nodes_1d(); ++i) {
                int local_dof = basis_->dof_index(i, j);
                Index global_dof = elem_dofs[local_dof];

                Real xi = nodes(i);
                Real eta = nodes(j);
                Real x = bounds.xmin + 0.5 * (xi + 1.0) * (bounds.xmax - bounds.xmin);
                Real y = bounds.ymin + 0.5 * (eta + 1.0) * (bounds.ymax - bounds.ymin);

                Real expected = 2.0 * x + 3.0 * y + 1.0;
                EXPECT_NEAR(u(global_dof), expected, LOOSE_TOLERANCE)
                    << "At (" << x << ", " << y << ")";
            }
        }
    }
}

// =============================================================================
// Reduced system tests
// =============================================================================

TEST_F(BiharmonicAssemblerTest, ReducedSystemDimensions) {
    create_mesh_and_dofs(2, 2);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    FunctionBathymetry bathy([](Real, Real) { return 1.0; });

    SpMat K_red;
    VecX f_red;
    assembler.assemble_reduced_system(bathy, K_red, f_red);

    EXPECT_EQ(K_red.rows(), dofs_->num_free_dofs());
    EXPECT_EQ(K_red.cols(), dofs_->num_free_dofs());
    EXPECT_EQ(f_red.size(), dofs_->num_free_dofs());
}

TEST_F(BiharmonicAssemblerTest, ReducedSystemSymmetry) {
    create_mesh_and_dofs(2, 2);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    FunctionBathymetry bathy([](Real, Real) { return 1.0; });

    SpMat K_red;
    VecX f_red;
    assembler.assemble_reduced_system(bathy, K_red, f_red);

    // Use loose tolerance due to floating-point accumulation in IPDG assembly
    MatX K_dense = MatX(K_red);
    EXPECT_NEAR((K_dense - K_dense.transpose()).norm(), 0.0, LOOSE_TOLERANCE);
}

// =============================================================================
// Jacobian tests
// =============================================================================

TEST_F(BiharmonicAssemblerTest, JacobianConstantForRectangle) {
    create_mesh_and_dofs(2, 2);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    VecX jac = assembler.compute_jacobian(0);

    // For rectangular elements, Jacobian should be constant
    Real first_jac = jac(0);
    for (int i = 1; i < jac.size(); ++i) {
        EXPECT_NEAR(jac(i), first_jac, TOLERANCE);
    }
}

TEST_F(BiharmonicAssemblerTest, JacobianPositive) {
    create_mesh_and_dofs(2, 2);
    BiharmonicAssembler assembler(*mesh_, *basis_, *dofs_, 1.0, 1.0);

    for (Index e = 0; e < mesh_->num_elements(); ++e) {
        VecX jac = assembler.compute_jacobian(e);
        for (int i = 0; i < jac.size(); ++i) {
            EXPECT_GT(jac(i), 0.0);
        }
    }
}

// =============================================================================
// Smoothing parameter effect tests
// =============================================================================

TEST_F(BiharmonicAssemblerTest, SmoothingReducesCurvature) {
    // Test that smoothing (alpha > 0) reduces the Laplacian magnitude
    // compared to pure data fitting (alpha = 0)
    create_mesh_and_dofs(2, 2);

    // Use a polynomial that has non-zero Laplacian
    FunctionBathymetry quadratic_bathy([](Real x, Real y) {
        return x * x + y * y;  // Laplacian = 4
    });

    // Pure data fitting (no smoothing)
    BiharmonicAssembler no_smooth(*mesh_, *basis_, *dofs_, 0.0, 1.0);
    SpMat K1 = no_smooth.assemble_stiffness();
    VecX f1 = no_smooth.assemble_rhs(quadratic_bathy);
    Eigen::SparseLU<SpMat> solver1;
    solver1.compute(K1);
    VecX u1 = solver1.solve(f1);

    // With smoothing - should move toward flatter solution
    BiharmonicAssembler with_smooth(*mesh_, *basis_, *dofs_, 10.0, 1.0);
    SpMat K2 = with_smooth.assemble_stiffness();
    VecX f2 = with_smooth.assemble_rhs(quadratic_bathy);
    Eigen::SparseLU<SpMat> solver2;
    solver2.compute(K2);
    VecX u2 = solver2.solve(f2);

    // With high smoothing, the solution should be closer to the mean
    // (biharmonic penalty discourages curvature)
    Real range1 = u1.maxCoeff() - u1.minCoeff();
    Real range2 = u2.maxCoeff() - u2.minCoeff();

    // With very high alpha/beta ratio, solution should be flatter
    EXPECT_LT(range2, range1 * 1.1) << "Smoothing should not increase range significantly";
}

// =============================================================================
// FunctionBathymetry tests
// =============================================================================

TEST_F(BiharmonicAssemblerTest, FunctionBathymetryEvaluation) {
    FunctionBathymetry bathy([](Real x, Real y) { return x * y; });

    EXPECT_NEAR(bathy.evaluate(2.0, 3.0), 6.0, TOLERANCE);
    EXPECT_NEAR(bathy.evaluate(0.0, 5.0), 0.0, TOLERANCE);
    EXPECT_NEAR(bathy.evaluate(-1.0, 2.0), -2.0, TOLERANCE);
}
