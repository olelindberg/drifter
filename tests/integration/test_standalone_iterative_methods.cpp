#include "bathymetry/cg_cubic_bezier_dof_manager.hpp"
#include "bathymetry/iterative_method_factory.hpp"
#include "bathymetry/iterative_solver.hpp"
#include "bathymetry/jacobi_method.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/schwarz_method.hpp"
#include <Eigen/SparseLU>
#include <chrono>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <map>
#include <set>

using namespace drifter;

/// @brief Test fixture for standalone iterative method testing
///
/// Tests each iterative method (Jacobi, Schwarz variants) as a standalone
/// solver on a simple SPD system derived from the mesh structure.
class StandaloneIterativeMethodTest : public ::testing::Test {
protected:
    static constexpr Real L = 1000.0;
    static constexpr int N = 8;
    static constexpr int MAX_ITERATIONS = 1000;
    static constexpr Real TOLERANCE = 1e-6;
    static constexpr std::string_view OUTPUT_DIR = "/tmp";

    void SetUp() override {
        // Build mesh
        mesh_.build_uniform(0.0, L, 0.0, L, N, N);

        // Build DOF manager
        dof_manager_ = std::make_unique<CGCubicBezierDofManager>(mesh_);

        // Build a simple SPD system matrix
        build_simple_system();

        // Build element blocks for Schwarz methods
        build_element_blocks();

        // Build element coloring for colored Schwarz
        build_element_coloring();
    }

    void build_simple_system() {
        // Build a simple Laplacian-like matrix using the DOF structure
        // This is a simple SPD matrix suitable for testing iterative methods
        num_dofs_ = dof_manager_->num_free_dofs();

        // Use triplets to build sparse matrix
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(num_dofs_ * 17); // Estimate: 16 neighbors + diagonal

        Index num_elements = mesh_.num_elements();

        // For each element, add contributions to global matrix
        // Use a simple element stiffness pattern (identity with off-diagonals)
        for (Index e = 0; e < num_elements; ++e) {
            const auto &global_dofs = dof_manager_->element_dofs(e);

            // Get free DOF indices
            std::vector<Index> free_dofs;
            for (int local = 0; local < 16; ++local) {
                Index global = global_dofs[local];
                Index free = dof_manager_->global_to_free(global);
                if (free >= 0) {
                    free_dofs.push_back(free);
                }
            }

            // Add element contribution: simple symmetric positive definite pattern
            // Diagonal = 4, off-diagonal = -1 (within element)
            int block_size = static_cast<int>(free_dofs.size());
            for (int i = 0; i < block_size; ++i) {
                // Add to diagonal
                triplets.emplace_back(free_dofs[i], free_dofs[i], 4.0);
                // Add small off-diagonal connections within element
                for (int j = i + 1; j < block_size && j < i + 4; ++j) {
                    triplets.emplace_back(free_dofs[i], free_dofs[j], -0.5);
                    triplets.emplace_back(free_dofs[j], free_dofs[i], -0.5);
                }
            }
        }

        // Build sparse matrix
        Q_.resize(num_dofs_, num_dofs_);
        Q_.setFromTriplets(triplets.begin(), triplets.end());

        // Add small ridge for stability
        for (Index i = 0; i < num_dofs_; ++i) {
            Q_.coeffRef(i, i) += 1e-6;
        }

        // Build RHS: sinusoidal pattern
        b_.resize(num_dofs_);
        for (Index i = 0; i < num_dofs_; ++i) {
            Real t = static_cast<Real>(i) / num_dofs_;
            b_(i) = std::sin(2.0 * M_PI * t) + 0.5 * std::cos(4.0 * M_PI * t);
        }

        // Compute reference solution using SparseLU
        Eigen::SparseLU<SpMat> solver;
        solver.compute(Q_);
        x_ref_ = solver.solve(b_);

        std::cout << "System size: " << num_dofs_ << " DOFs" << std::endl;
        std::cout << "Reference solution norm: " << x_ref_.norm() << std::endl;
    }

    void build_element_blocks() {
        Index num_elements = mesh_.num_elements();

        element_free_dofs_.resize(num_elements);
        element_block_lu_.resize(num_elements);

        for (Index e = 0; e < num_elements; ++e) {
            const auto &global_dofs = dof_manager_->element_dofs(e);

            // Map to free DOF indices
            std::vector<Index> free_dofs;
            free_dofs.reserve(16);
            for (int local = 0; local < 16; ++local) {
                Index global = global_dofs[local];
                Index free = dof_manager_->global_to_free(global);
                if (free >= 0) {
                    free_dofs.push_back(free);
                }
            }
            element_free_dofs_[e] = free_dofs;

            // Extract and factorize element block
            int block_size = static_cast<int>(free_dofs.size());
            if (block_size == 0) {
                element_block_lu_[e] = Eigen::PartialPivLU<MatX>();
                continue;
            }

            MatX Q_block(block_size, block_size);
            for (int i = 0; i < block_size; ++i) {
                for (int j = 0; j < block_size; ++j) {
                    Q_block(i, j) = Q_.coeff(free_dofs[i], free_dofs[j]);
                }
            }
            element_block_lu_[e] = Q_block.partialPivLu();
        }
    }

    void build_element_coloring() {
        size_t num_elements = element_free_dofs_.size();
        if (num_elements == 0)
            return;

        // Build DOF -> elements adjacency
        std::map<Index, std::vector<Index>> dof_to_elements;
        for (size_t e = 0; e < num_elements; ++e) {
            for (Index dof : element_free_dofs_[e]) {
                dof_to_elements[dof].push_back(static_cast<Index>(e));
            }
        }

        // Greedy graph coloring
        std::vector<int> element_color(num_elements, -1);
        int max_color = -1;

        for (size_t e = 0; e < num_elements; ++e) {
            std::set<int> neighbor_colors;
            for (Index dof : element_free_dofs_[e]) {
                for (Index neighbor : dof_to_elements[dof]) {
                    if (neighbor != static_cast<Index>(e) &&
                        element_color[neighbor] >= 0) {
                        neighbor_colors.insert(element_color[neighbor]);
                    }
                }
            }

            int color = 0;
            while (neighbor_colors.count(color)) {
                color++;
            }
            element_color[e] = color;
            max_color = std::max(max_color, color);
        }

        // Build color groups
        elements_by_color_.resize(max_color + 1);
        for (size_t e = 0; e < num_elements; ++e) {
            elements_by_color_[element_color[e]].push_back(
                static_cast<Index>(e));
        }

        std::cout << "Element coloring: " << (max_color + 1) << " colors"
                  << std::endl;
    }

    // =========================================================================
    // Run solver and collect metrics
    // =========================================================================

    struct MethodMetrics {
        std::string name;
        int iterations = 0;
        Real final_residual = 0.0;
        Real relative_residual = 0.0;
        Real solution_error = 0.0;
        bool converged = false;
        double total_ms = 0.0;
        std::vector<Real> residual_history;
    };

    MethodMetrics run_jacobi(Real omega = 0.8) {
        MethodMetrics metrics;
        metrics.name = "jacobi";

        auto method = std::make_unique<JacobiMethod>(Q_, omega);
        IterativeMethodConfig config;
        config.max_iterations = MAX_ITERATIONS;
        config.tolerance = TOLERANCE;
        config.omega = omega;

        IterativeSolver solver(std::move(method), config);

        VecX x = VecX::Zero(num_dofs_);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = solver.solve(x, b_);
        auto end = std::chrono::high_resolution_clock::now();

        metrics.iterations = result.iterations;
        metrics.final_residual = result.final_residual;
        metrics.relative_residual = result.relative_residual;
        metrics.converged = result.converged;
        metrics.residual_history = result.residual_history;
        metrics.solution_error = (x - x_ref_).norm() / x_ref_.norm();
        metrics.total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        return metrics;
    }

    MethodMetrics run_multiplicative_schwarz() {
        MethodMetrics metrics;
        metrics.name = "multiplicative_schwarz";

        auto method = std::make_unique<MultiplicativeSchwarzMethod>(
            Q_, element_free_dofs_, element_block_lu_);
        IterativeMethodConfig config;
        config.max_iterations = MAX_ITERATIONS;
        config.tolerance = TOLERANCE;

        IterativeSolver solver(std::move(method), config);

        VecX x = VecX::Zero(num_dofs_);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = solver.solve(x, b_);
        auto end = std::chrono::high_resolution_clock::now();

        metrics.iterations = result.iterations;
        metrics.final_residual = result.final_residual;
        metrics.relative_residual = result.relative_residual;
        metrics.converged = result.converged;
        metrics.residual_history = result.residual_history;
        metrics.solution_error = (x - x_ref_).norm() / x_ref_.norm();
        metrics.total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        return metrics;
    }

    MethodMetrics run_additive_schwarz(Real omega = 0.5) {
        MethodMetrics metrics;
        metrics.name = "additive_schwarz";

        auto method = std::make_unique<AdditiveSchwarzMethod>(
            Q_, element_free_dofs_, element_block_lu_, omega);
        IterativeMethodConfig config;
        config.max_iterations = MAX_ITERATIONS;
        config.tolerance = TOLERANCE;

        IterativeSolver solver(std::move(method), config);

        VecX x = VecX::Zero(num_dofs_);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = solver.solve(x, b_);
        auto end = std::chrono::high_resolution_clock::now();

        metrics.iterations = result.iterations;
        metrics.final_residual = result.final_residual;
        metrics.relative_residual = result.relative_residual;
        metrics.converged = result.converged;
        metrics.residual_history = result.residual_history;
        metrics.solution_error = (x - x_ref_).norm() / x_ref_.norm();
        metrics.total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        return metrics;
    }

    MethodMetrics run_colored_schwarz() {
        MethodMetrics metrics;
        metrics.name = "colored_schwarz";

        auto method = std::make_unique<ColoredSchwarzMethod>(
            Q_, element_free_dofs_, element_block_lu_, elements_by_color_);
        IterativeMethodConfig config;
        config.max_iterations = MAX_ITERATIONS;
        config.tolerance = TOLERANCE;

        IterativeSolver solver(std::move(method), config);

        VecX x = VecX::Zero(num_dofs_);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = solver.solve(x, b_);
        auto end = std::chrono::high_resolution_clock::now();

        metrics.iterations = result.iterations;
        metrics.final_residual = result.final_residual;
        metrics.relative_residual = result.relative_residual;
        metrics.converged = result.converged;
        metrics.residual_history = result.residual_history;
        metrics.solution_error = (x - x_ref_).norm() / x_ref_.norm();
        metrics.total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        return metrics;
    }

    // =========================================================================
    // CSV output
    // =========================================================================

    void write_iteration_csv(const MethodMetrics &m) const {
        std::string filename =
            std::string(OUTPUT_DIR) + "/method_iterations_" + m.name + ".csv";
        std::ofstream ofs(filename);
        ofs << std::scientific << std::setprecision(12);
        ofs << "iteration,residual_norm\n";
        for (size_t i = 0; i < m.residual_history.size(); ++i) {
            ofs << i << "," << m.residual_history[i] << "\n";
        }
        std::cout << "Wrote: " << filename << std::endl;
    }

    void write_summary_csv(const std::vector<MethodMetrics> &all_metrics) const {
        std::string filename =
            std::string(OUTPUT_DIR) + "/method_comparison.csv";
        std::ofstream ofs(filename);
        ofs << std::scientific << std::setprecision(12);
        ofs << "method,iterations,final_residual,relative_residual,solution_"
               "error,converged,total_ms\n";
        for (const auto &m : all_metrics) {
            ofs << m.name << "," << m.iterations << "," << m.final_residual
                << "," << m.relative_residual << "," << m.solution_error << ","
                << (m.converged ? "true" : "false") << "," << m.total_ms
                << "\n";
        }
        std::cout << "Wrote: " << filename << std::endl;
    }

    void print_metrics(const MethodMetrics &m) const {
        std::cout << "\n=== " << m.name << " ===" << std::endl;
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "Iterations:        " << m.iterations << std::endl;
        std::cout << "Final residual:    " << m.final_residual << std::endl;
        std::cout << "Relative residual: " << m.relative_residual << std::endl;
        std::cout << "Solution error:    " << m.solution_error << std::endl;
        std::cout << "Converged:         " << (m.converged ? "yes" : "no")
                  << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Time (ms):         " << m.total_ms << std::endl;
    }

protected:
    QuadtreeAdapter mesh_;
    std::unique_ptr<CGCubicBezierDofManager> dof_manager_;

    SpMat Q_;
    VecX b_;
    VecX x_ref_;
    Index num_dofs_;

    std::vector<std::vector<Index>> element_free_dofs_;
    std::vector<Eigen::PartialPivLU<MatX>> element_block_lu_;
    std::vector<std::vector<Index>> elements_by_color_;
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_F(StandaloneIterativeMethodTest, JacobiConvergence) {
    auto metrics = run_jacobi(0.8);
    print_metrics(metrics);
    write_iteration_csv(metrics);

    // Jacobi should reduce residual
    EXPECT_LT(metrics.residual_history.back(), metrics.residual_history.front())
        << "Jacobi should reduce residual";
}

TEST_F(StandaloneIterativeMethodTest, MultiplicativeSchwarzConvergence) {
    auto metrics = run_multiplicative_schwarz();
    print_metrics(metrics);
    write_iteration_csv(metrics);

    // Multiplicative Schwarz should converge or make progress
    EXPECT_LT(metrics.relative_residual, 1.0)
        << "Multiplicative Schwarz should reduce relative residual";
}

TEST_F(StandaloneIterativeMethodTest, AdditiveSchwarzConvergence) {
    // Use small omega (0.1) for stability - additive Schwarz requires more damping
    auto metrics = run_additive_schwarz(0.1);
    print_metrics(metrics);
    write_iteration_csv(metrics);

    // Additive Schwarz with damping should make progress
    EXPECT_LT(metrics.residual_history.back(), metrics.residual_history.front())
        << "Additive Schwarz should reduce residual";
}

TEST_F(StandaloneIterativeMethodTest, ColoredSchwarzConvergence) {
    auto metrics = run_colored_schwarz();
    print_metrics(metrics);
    write_iteration_csv(metrics);

    // Colored Schwarz should converge or make progress
    EXPECT_LT(metrics.relative_residual, 1.0)
        << "Colored Schwarz should reduce relative residual";
}

TEST_F(StandaloneIterativeMethodTest, AllMethodsComparison) {
    std::vector<MethodMetrics> all_metrics;

    all_metrics.push_back(run_jacobi(0.8));
    all_metrics.push_back(run_multiplicative_schwarz());
    all_metrics.push_back(run_additive_schwarz(0.1)); // Small omega for stability
    all_metrics.push_back(run_colored_schwarz());

    std::cout << "\n========================================" << std::endl;
    std::cout << "    ALL METHODS COMPARISON" << std::endl;
    std::cout << "========================================" << std::endl;

    for (const auto &m : all_metrics) {
        print_metrics(m);
        write_iteration_csv(m);
    }

    write_summary_csv(all_metrics);

    // Summary table
    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << std::left << std::setw(25) << "Method" << std::setw(12)
              << "Iterations" << std::setw(15) << "Rel. Residual"
              << std::setw(15) << "Sol. Error" << std::setw(10) << "Time (ms)"
              << std::endl;
    std::cout << std::string(77, '-') << std::endl;

    for (const auto &m : all_metrics) {
        std::cout << std::left << std::setw(25) << m.name << std::setw(12)
                  << m.iterations << std::scientific << std::setprecision(3)
                  << std::setw(15) << m.relative_residual << std::setw(15)
                  << m.solution_error << std::fixed << std::setprecision(1)
                  << std::setw(10) << m.total_ms << std::endl;
    }
}

TEST_F(StandaloneIterativeMethodTest, FactoryCreation) {
    // Test that IterativeMethodFactory correctly creates each method type
    auto jacobi = IterativeMethodFactory::create(SmootherType::Jacobi, Q_);
    ASSERT_NE(jacobi, nullptr);

    auto mult_schwarz = IterativeMethodFactory::create(
        SmootherType::MultiplicativeSchwarz, Q_, element_free_dofs_,
        element_block_lu_);
    ASSERT_NE(mult_schwarz, nullptr);

    auto add_schwarz = IterativeMethodFactory::create(
        SmootherType::AdditiveSchwarz, Q_, element_free_dofs_,
        element_block_lu_, {}, 0.5);
    ASSERT_NE(add_schwarz, nullptr);

    auto colored = IterativeMethodFactory::create(
        SmootherType::ColoredMultiplicativeSchwarz, Q_, element_free_dofs_,
        element_block_lu_, elements_by_color_);
    ASSERT_NE(colored, nullptr);

    // Verify each can be applied
    VecX x = VecX::Zero(num_dofs_);
    jacobi->apply(x, b_, 1);
    EXPECT_FALSE(x.isZero()) << "Jacobi should modify x";

    x.setZero();
    mult_schwarz->apply(x, b_, 1);
    EXPECT_FALSE(x.isZero()) << "Multiplicative Schwarz should modify x";

    x.setZero();
    add_schwarz->apply(x, b_, 1);
    EXPECT_FALSE(x.isZero()) << "Additive Schwarz should modify x";

    x.setZero();
    colored->apply(x, b_, 1);
    EXPECT_FALSE(x.isZero()) << "Colored Schwarz should modify x";
}
