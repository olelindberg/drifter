#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace drifter;

// =============================================================================
// Performance Benchmark for Hierarchical Ordering and Static Condensation
// =============================================================================

struct BenchmarkResult {
    std::string config_name;
    int num_elements;
    int num_dofs;
    double setup_time_ms;
    double solve_time_ms;
    double total_time_ms;
    double max_error;
    bool static_condensation_active;
};

class HierarchicalOrderingBenchmark : public ::testing::Test {
  protected:
    static std::vector<BenchmarkResult> results_;

    static void write_results_to_markdown() {
        std::ofstream out("/home/ole/Projects/drifter/docs/hierarchical_ordering_benchmark.md");

        out << "# Hierarchical Ordering and Static Condensation Performance Benchmark\n\n";
        out << "This document presents performance measurements for the CG cubic Bezier "
            << "bathymetry smoother with different DOF ordering and condensation strategies.\n\n";

        out << "## Configuration Options\n\n";
        out << "| Option | Description |\n";
        out << "|--------|-------------|\n";
        out << "| Standard | Default Morton ordering, no static condensation |\n";
        out << "| Hierarchical | DOFs ordered by (level, constrained, type, hilbert_key) |\n";
        out << "| Static Condensation | Interior DOFs eliminated at element level (single-element only) |\n";
        out << "| Combined | Both hierarchical ordering and static condensation |\n\n";

        out << "## Important Notes\n\n";
        out << "- **Static condensation is currently only supported for single-element meshes**\n";
        out << "- For multi-element meshes, the C¹ edge constraints couple interior DOFs across "
            << "elements, making constraint transformation complex\n";
        out << "- When `use_static_condensation=true` on multi-element meshes, it automatically "
            << "falls back to standard assembly\n\n";

        // Group results by mesh type
        std::map<int, std::vector<BenchmarkResult>> by_elements;
        for (const auto &r : results_) {
            by_elements[r.num_elements].push_back(r);
        }

        out << "## Benchmark Results\n\n";

        for (const auto &[num_elem, elem_results] : by_elements) {
            out << "### " << num_elem << " Element";
            if (num_elem > 1) out << "s";
            out << " Mesh\n\n";

            out << "| Configuration | DOFs | Setup (ms) | Solve (ms) | Total (ms) | Max Error | Condensation Active |\n";
            out << "|---------------|------|------------|------------|------------|-----------|--------------------|\n";

            for (const auto &r : elem_results) {
                out << "| " << r.config_name
                    << " | " << r.num_dofs
                    << " | " << std::fixed << std::setprecision(2) << r.setup_time_ms
                    << " | " << std::fixed << std::setprecision(2) << r.solve_time_ms
                    << " | " << std::fixed << std::setprecision(2) << r.total_time_ms
                    << " | " << std::scientific << std::setprecision(2) << r.max_error
                    << " | " << (r.static_condensation_active ? "Yes" : "No")
                    << " |\n";
            }
            out << "\n";
        }

        // Summary
        out << "## Summary\n\n";

        // Find single-element results for comparison
        auto it1 = by_elements.find(1);
        if (it1 != by_elements.end() && it1->second.size() >= 2) {
            const auto &single_results = it1->second;
            double std_time = 0, cond_time = 0;
            for (const auto &r : single_results) {
                if (r.config_name == "Standard") std_time = r.total_time_ms;
                if (r.config_name == "Static Condensation") cond_time = r.total_time_ms;
            }
            if (std_time > 0 && cond_time > 0) {
                double speedup = std_time / cond_time;
                out << "### Single-Element Performance\n\n";
                out << "- Standard solve time: " << std::fixed << std::setprecision(2) << std_time << " ms\n";
                out << "- Static condensation solve time: " << std::fixed << std::setprecision(2) << cond_time << " ms\n";
                out << "- **Speedup: " << std::fixed << std::setprecision(2) << speedup << "x**\n\n";
            }
        }

        out << "### Multi-Element Behavior\n\n";
        out << "For multi-element meshes, static condensation is automatically disabled because:\n";
        out << "1. C¹ edge constraints involve interior DOFs from adjacent elements\n";
        out << "2. Proper constraint transformation requires complex cross-element coupling\n";
        out << "3. The fallback ensures correctness at the cost of the 25% DOF reduction benefit\n\n";

        out << "### Recommendations\n\n";
        out << "1. **Single-element meshes**: Enable `use_static_condensation=true` for ~25% DOF reduction\n";
        out << "2. **Multi-element meshes**: Use hierarchical ordering for better cache locality\n";
        out << "3. **Future work**: Implement proper multi-element static condensation by building "
            << "edge constraints on skeleton DOFs directly\n\n";

        out << "---\n";
        out << "*Generated by test_hierarchical_ordering_performance.cpp*\n";

        out.close();
        std::cout << "\nResults written to: docs/hierarchical_ordering_benchmark.md\n";
    }
};

std::vector<BenchmarkResult> HierarchicalOrderingBenchmark::results_;

// Helper to run benchmark with given config
BenchmarkResult run_benchmark(QuadtreeAdapter &mesh,
                              const std::string &config_name,
                              bool use_hierarchical,
                              bool use_condensation,
                              std::function<Real(Real, Real)> bathy_func) {
    BenchmarkResult result;
    result.config_name = config_name;
    result.num_elements = mesh.num_elements();

    CGCubicBezierSmootherConfig config;
    config.lambda = 10.0;
    config.use_hierarchical_ordering = use_hierarchical;
    config.use_static_condensation = use_condensation;

    auto t0 = std::chrono::high_resolution_clock::now();

    CGCubicBezierBathymetrySmoother smoother(mesh, config);

    auto t1 = std::chrono::high_resolution_clock::now();

    smoother.set_bathymetry_data(bathy_func);

    auto t2 = std::chrono::high_resolution_clock::now();

    smoother.solve();

    auto t3 = std::chrono::high_resolution_clock::now();

    result.num_dofs = smoother.num_free_dofs();
    result.setup_time_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
    result.solve_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    result.total_time_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();

    // Check if condensation was actually active
    result.static_condensation_active = use_condensation && (mesh.num_elements() == 1);

    // Compute max error at sample points
    result.max_error = 0.0;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            Real x = 0.1 + 0.8 * i / 9.0;
            Real y = 0.1 + 0.8 * j / 9.0;
            // Scale to mesh bounds
            const auto &bounds = mesh.element_bounds(0);
            Real xmin = bounds.xmin, xmax = bounds.xmax;
            Real ymin = bounds.ymin, ymax = bounds.ymax;
            for (Index e = 1; e < mesh.num_elements(); ++e) {
                const auto &b = mesh.element_bounds(e);
                xmin = std::min(xmin, b.xmin);
                xmax = std::max(xmax, b.xmax);
                ymin = std::min(ymin, b.ymin);
                ymax = std::max(ymax, b.ymax);
            }
            Real px = xmin + x * (xmax - xmin);
            Real py = ymin + y * (ymax - ymin);

            Real expected = bathy_func(px, py);
            Real actual = smoother.evaluate(px, py);
            result.max_error = std::max(result.max_error, std::abs(actual - expected));
        }
    }

    return result;
}

TEST_F(HierarchicalOrderingBenchmark, SingleElementBenchmark) {
    std::cout << "\n=== Single Element Benchmark ===\n";

    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 1, 1);

    auto bathy_func = [](Real x, Real y) {
        return 5.0 + std::sin(x * 0.3) * std::cos(y * 0.3);
    };

    // Run each config multiple times and take average
    const int num_runs = 3;

    std::vector<std::tuple<std::string, bool, bool>> configs = {
        {"Standard", false, false},
        {"Hierarchical", true, false},
        {"Static Condensation", false, true},
        {"Combined", true, true}
    };

    for (const auto &[name, hier, cond] : configs) {
        double total_setup = 0, total_solve = 0, total_total = 0;
        BenchmarkResult last_result;

        for (int i = 0; i < num_runs; ++i) {
            last_result = run_benchmark(mesh, name, hier, cond, bathy_func);
            total_setup += last_result.setup_time_ms;
            total_solve += last_result.solve_time_ms;
            total_total += last_result.total_time_ms;
        }

        last_result.setup_time_ms = total_setup / num_runs;
        last_result.solve_time_ms = total_solve / num_runs;
        last_result.total_time_ms = total_total / num_runs;

        results_.push_back(last_result);

        std::cout << name << ": " << std::fixed << std::setprecision(2)
                  << last_result.total_time_ms << " ms (DOFs: " << last_result.num_dofs
                  << ", condensation: " << (last_result.static_condensation_active ? "yes" : "no")
                  << ")\n";
    }
}

TEST_F(HierarchicalOrderingBenchmark, UniformMesh4x4Benchmark) {
    std::cout << "\n=== 4x4 Uniform Mesh Benchmark ===\n";

    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 4, 4);

    auto bathy_func = [](Real x, Real y) {
        return 5.0 + std::sin(x * 0.3) * std::cos(y * 0.3);
    };

    const int num_runs = 3;

    std::vector<std::tuple<std::string, bool, bool>> configs = {
        {"Standard", false, false},
        {"Hierarchical", true, false},
        {"Static Condensation (fallback)", false, true},
        {"Combined (fallback)", true, true}
    };

    for (const auto &[name, hier, cond] : configs) {
        double total_setup = 0, total_solve = 0, total_total = 0;
        BenchmarkResult last_result;

        for (int i = 0; i < num_runs; ++i) {
            last_result = run_benchmark(mesh, name, hier, cond, bathy_func);
            total_setup += last_result.setup_time_ms;
            total_solve += last_result.solve_time_ms;
            total_total += last_result.total_time_ms;
        }

        last_result.setup_time_ms = total_setup / num_runs;
        last_result.solve_time_ms = total_solve / num_runs;
        last_result.total_time_ms = total_total / num_runs;

        results_.push_back(last_result);

        std::cout << name << ": " << std::fixed << std::setprecision(2)
                  << last_result.total_time_ms << " ms (DOFs: " << last_result.num_dofs
                  << ", condensation: " << (last_result.static_condensation_active ? "yes" : "no")
                  << ")\n";
    }
}

TEST_F(HierarchicalOrderingBenchmark, UniformMesh8x8Benchmark) {
    std::cout << "\n=== 8x8 Uniform Mesh Benchmark ===\n";

    QuadtreeAdapter mesh;
    mesh.build_uniform(0.0, 10.0, 0.0, 10.0, 8, 8);

    auto bathy_func = [](Real x, Real y) {
        return 5.0 + std::sin(x * 0.3) * std::cos(y * 0.3);
    };

    const int num_runs = 3;

    std::vector<std::tuple<std::string, bool, bool>> configs = {
        {"Standard", false, false},
        {"Hierarchical", true, false},
    };

    for (const auto &[name, hier, cond] : configs) {
        double total_setup = 0, total_solve = 0, total_total = 0;
        BenchmarkResult last_result;

        for (int i = 0; i < num_runs; ++i) {
            last_result = run_benchmark(mesh, name, hier, cond, bathy_func);
            total_setup += last_result.setup_time_ms;
            total_solve += last_result.solve_time_ms;
            total_total += last_result.total_time_ms;
        }

        last_result.setup_time_ms = total_setup / num_runs;
        last_result.solve_time_ms = total_solve / num_runs;
        last_result.total_time_ms = total_total / num_runs;

        results_.push_back(last_result);

        std::cout << name << ": " << std::fixed << std::setprecision(2)
                  << last_result.total_time_ms << " ms (DOFs: " << last_result.num_dofs << ")\n";
    }
}

TEST_F(HierarchicalOrderingBenchmark, AdaptiveMeshBenchmark) {
    std::cout << "\n=== Adaptive Mesh Benchmark ===\n";

    // Create adaptive mesh with graded refinement toward center
    // 3 levels: 1 root element -> refined recursively toward center
    QuadtreeAdapter mesh;
    mesh.build_center_graded(3);

    std::cout << "Adaptive mesh: " << mesh.num_elements() << " elements\n";

    auto bathy_func = [](Real x, Real y) {
        return 5.0 + std::sin(x * 0.5) * std::cos(y * 0.5);
    };

    const int num_runs = 3;

    std::vector<std::tuple<std::string, bool, bool>> configs = {
        {"Standard", false, false},
        {"Hierarchical", true, false},
    };

    for (const auto &[name, hier, cond] : configs) {
        double total_setup = 0, total_solve = 0, total_total = 0;
        BenchmarkResult last_result;

        for (int i = 0; i < num_runs; ++i) {
            last_result = run_benchmark(mesh, name, hier, cond, bathy_func);
            total_setup += last_result.setup_time_ms;
            total_solve += last_result.solve_time_ms;
            total_total += last_result.total_time_ms;
        }

        last_result.setup_time_ms = total_setup / num_runs;
        last_result.solve_time_ms = total_solve / num_runs;
        last_result.total_time_ms = total_total / num_runs;

        results_.push_back(last_result);

        std::cout << name << ": " << std::fixed << std::setprecision(2)
                  << last_result.total_time_ms << " ms (DOFs: " << last_result.num_dofs << ")\n";
    }
}

TEST_F(HierarchicalOrderingBenchmark, WriteResults) {
    write_results_to_markdown();
}
