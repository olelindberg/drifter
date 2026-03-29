#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "bathymetry/adaptive_smoother_types.hpp"
#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "synthetic_bathymetry.hpp"
#include "integration/test_integration_fixtures.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>

using namespace drifter;
using namespace drifter::testing;
namespace synth = drifter::testing::synthetic_bathymetry;

// =============================================================================
// Test Fixture
// =============================================================================

class UniformVsAdaptiveConvergenceTest : public ::testing::Test {
protected:
  static constexpr Real XMIN = synth::XMIN;
  static constexpr Real XMAX = synth::XMAX;
  static constexpr Real YMIN = synth::YMIN;
  static constexpr Real YMAX = synth::YMAX;

  struct ConvergenceResult {
    std::string mesh_type;
    Index num_elements;
    Index num_dofs;
    Real max_error;
    Real mean_error;
    Real l2_error;
    double solve_time_ms;
    int iterations; // adaptive iterations or uniform level
  };

  // Compute error against reference (very fine uniform grid)
  Real compute_l2_error(const CGCubicBezierBathymetrySmoother &smoother,
                        std::function<Real(Real, Real)> reference, int ngauss) {
    Real error_sum = 0.0;
    Real domain_area = 0.0;

    const auto &mesh = smoother.mesh();

    // Gauss-Legendre points and weights
    std::vector<Real> gauss_pts, gauss_wts;
    if (ngauss == 4) {
      Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
      Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
      Real wa = (18.0 + std::sqrt(30.0)) / 36.0;
      Real wb = (18.0 - std::sqrt(30.0)) / 36.0;
      gauss_pts = {-b, -a, a, b};
      gauss_wts = {wb, wa, wa, wb};
    } else {
      // Simple 2-point rule
      Real p = 1.0 / std::sqrt(3.0);
      gauss_pts = {-p, p};
      gauss_wts = {1.0, 1.0};
    }

    for (Index e = 0; e < mesh.num_elements(); ++e) {
      auto bounds = mesh.element_bounds(e);
      Real hx = (bounds.xmax - bounds.xmin) / 2.0;
      Real hy = (bounds.ymax - bounds.ymin) / 2.0;
      Real cx = (bounds.xmax + bounds.xmin) / 2.0;
      Real cy = (bounds.ymax + bounds.ymin) / 2.0;

      for (size_t i = 0; i < gauss_pts.size(); ++i) {
        for (size_t j = 0; j < gauss_pts.size(); ++j) {
          Real x = cx + hx * gauss_pts[i];
          Real y = cy + hy * gauss_pts[j];
          Real w = gauss_wts[i] * gauss_wts[j] * hx * hy;

          Real computed = smoother.evaluate(x, y);
          Real exact = reference(x, y);
          Real diff = computed - exact;

          error_sum += w * diff * diff;
          domain_area += w;
        }
      }
    }

    return std::sqrt(error_sum / domain_area);
  }

  // Export mesh elements to CSV
  void export_mesh(const QuadtreeAdapter &mesh, const std::string &filename) {
    std::ofstream file(filename);
    file << std::setprecision(10);
    file << "xmin,xmax,ymin,ymax,level\n";

    for (Index i = 0; i < mesh.num_elements(); ++i) {
      auto bounds = mesh.element_bounds(i);
      auto level = mesh.element_level(i);
      file << bounds.xmin << "," << bounds.xmax << "," << bounds.ymin << ","
           << bounds.ymax << "," << level.max_level() << "\n";
    }
  }

  // Run uniform grid convergence study
  std::vector<ConvergenceResult>
  run_uniform_study(std::function<Real(Real, Real)> bathymetry,
                    std::function<Real(Real, Real)> reference,
                    const std::vector<int> &grid_sizes, Real lambda,
                    const std::string &output_dir = "") {
    std::vector<ConvergenceResult> results;

    CGCubicBezierSmootherConfig config;
    config.lambda = lambda;
    config.edge_ngauss = 4;
    config.use_iterative_solver = true;
    config.use_multigrid = true;

    for (int n : grid_sizes) {
      QuadtreeAdapter mesh;
      mesh.build_uniform(XMIN, XMAX, YMIN, YMAX, n, n);

      CGCubicBezierBathymetrySmoother smoother(mesh, config);
      smoother.set_bathymetry_data(bathymetry);

      auto start = std::chrono::high_resolution_clock::now();
      smoother.solve();
      auto end = std::chrono::high_resolution_clock::now();

      double time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();

      Real l2_err = compute_l2_error(smoother, reference, 4);

      // Export mesh if output directory specified
      if (!output_dir.empty()) {
        std::string mesh_file =
            output_dir + "/uniform_" + std::to_string(n) + "x" +
            std::to_string(n) + "_mesh.csv";
        export_mesh(mesh, mesh_file);
      }

      results.push_back({std::to_string(n) + "x" + std::to_string(n),
                         mesh.num_elements(), smoother.num_global_dofs(), 0.0,
                         0.0, l2_err, time_ms, n});
    }

    return results;
  }

  // Run adaptive convergence study
  std::vector<ConvergenceResult>
  run_adaptive_study(std::function<Real(Real, Real)> bathymetry,
                     std::function<Real(Real, Real)> reference,
                     const std::vector<Real> &thresholds, Real lambda,
                     const std::string &output_dir = "") {
    std::vector<ConvergenceResult> results;

    for (Real threshold : thresholds) {
      AdaptiveCGCubicBezierConfig config;
      config.error_metric_type = ErrorMetricType::NormalizedError;
      config.error_threshold = threshold;
      config.max_iterations = 15;
      config.max_elements = 5000;
      config.smoother_config.lambda = lambda;
      config.smoother_config.edge_ngauss = 4;
      config.smoother_config.use_iterative_solver = true;
      config.smoother_config.use_multigrid = true;
      config.smoother_config.multigrid_config.min_tree_level = 4;

      AdaptiveCGCubicBezierSmoother smoother(XMIN, XMAX, YMIN, YMAX, 16, 16,
                                             config);
      smoother.set_bathymetry_data(bathymetry);

      auto start = std::chrono::high_resolution_clock::now();
      auto result = smoother.solve_adaptive();
      auto end = std::chrono::high_resolution_clock::now();

      double time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();

      // Compute L2 error against reference
      auto &base_smoother = smoother.smoother();
      Real l2_err = compute_l2_error(base_smoother, reference, 4);

      // Export mesh if output directory specified
      if (!output_dir.empty()) {
        std::ostringstream mesh_filename;
        mesh_filename << output_dir << "/adaptive_thr"
                      << std::fixed << std::setprecision(1) << threshold
                      << "_mesh.csv";
        export_mesh(smoother.mesh(), mesh_filename.str());
      }

      std::ostringstream name;
      name << "thr=" << std::fixed << std::setprecision(1) << threshold;

      results.push_back({name.str(), result.num_elements,
                         smoother.smoother().num_global_dofs(), result.max_error,
                         result.mean_error, l2_err, time_ms, result.iteration});
    }

    return results;
  }

  void print_results(const std::string &title,
                     const std::vector<ConvergenceResult> &uniform,
                     const std::vector<ConvergenceResult> &adaptive) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << "| " << std::setw(14) << "Mesh"
              << " | " << std::setw(8) << "Elements"
              << " | " << std::setw(8) << "DOFs"
              << " | " << std::setw(12) << "L2 Error"
              << " | " << std::setw(10) << "Time (ms)"
              << " |\n";
    std::cout << "|" << std::string(16, '-') << "|" << std::string(10, '-')
              << "|" << std::string(10, '-') << "|" << std::string(14, '-')
              << "|" << std::string(12, '-') << "|\n";

    std::cout << "| " << std::setw(14) << "UNIFORM"
              << " | " << std::setw(8) << ""
              << " | " << std::setw(8) << ""
              << " | " << std::setw(12) << ""
              << " | " << std::setw(10) << ""
              << " |\n";

    for (const auto &r : uniform) {
      std::cout << "| " << std::setw(14) << r.mesh_type << " | " << std::setw(8)
                << r.num_elements << " | " << std::setw(8) << r.num_dofs
                << " | " << std::scientific << std::setprecision(3)
                << std::setw(12) << r.l2_error << " | " << std::fixed
                << std::setprecision(1) << std::setw(10) << r.solve_time_ms
                << " |\n";
    }

    std::cout << "| " << std::setw(14) << "ADAPTIVE"
              << " | " << std::setw(8) << ""
              << " | " << std::setw(8) << ""
              << " | " << std::setw(12) << ""
              << " | " << std::setw(10) << ""
              << " |\n";

    for (const auto &r : adaptive) {
      std::cout << "| " << std::setw(14) << r.mesh_type << " | " << std::setw(8)
                << r.num_elements << " | " << std::setw(8) << r.num_dofs
                << " | " << std::scientific << std::setprecision(3)
                << std::setw(12) << r.l2_error << " | " << std::fixed
                << std::setprecision(1) << std::setw(10) << r.solve_time_ms
                << " |\n";
    }

    std::cout << "\n";
  }

  void write_csv(const std::string &filename,
                 const std::vector<ConvergenceResult> &uniform,
                 const std::vector<ConvergenceResult> &adaptive) {
    std::ofstream file(filename);
    file << "type,mesh,elements,dofs,l2_error,time_ms\n";

    for (const auto &r : uniform) {
      file << "uniform," << r.mesh_type << "," << r.num_elements << ","
           << r.num_dofs << "," << r.l2_error << "," << r.solve_time_ms << "\n";
    }
    for (const auto &r : adaptive) {
      file << "adaptive," << r.mesh_type << "," << r.num_elements << ","
           << r.num_dofs << "," << r.l2_error << "," << r.solve_time_ms << "\n";
    }
  }
};

// =============================================================================
// Convergence Tests
// =============================================================================

TEST_F(UniformVsAdaptiveConvergenceTest, CanyonConvergence) {
  auto bathymetry = synth::canyon;
  auto reference = bathymetry;

  Real lambda = 50.0;

  auto uniform = run_uniform_study(bathymetry, reference, {4, 8, 16, 32}, lambda);
  auto adaptive = run_adaptive_study(bathymetry, reference,
                                     {10.0, 5.0, 2.0, 1.0}, lambda);

  print_results("Canyon Bathymetry: Uniform vs Adaptive", uniform, adaptive);
  write_csv("/tmp/canyon_convergence.csv", uniform, adaptive);
}

TEST_F(UniformVsAdaptiveConvergenceTest, SeamountConvergence) {
  auto bathymetry = synth::seamount;
  auto reference = bathymetry;

  Real lambda = 50.0;

  auto uniform = run_uniform_study(bathymetry, reference, {4, 8, 16, 32}, lambda);
  auto adaptive = run_adaptive_study(bathymetry, reference,
                                     {10.0, 5.0, 2.0, 1.0}, lambda);

  print_results("Seamount Bathymetry: Uniform vs Adaptive", uniform, adaptive);
  write_csv("/tmp/seamount_convergence.csv", uniform, adaptive);
}

TEST_F(UniformVsAdaptiveConvergenceTest, CanyonAndSeamountConvergence) {
  auto bathymetry = synth::canyon_and_seamount;
  auto reference = bathymetry;

  Real lambda = 50.0;

  auto uniform =
      run_uniform_study(bathymetry, reference, {4, 8, 16, 32, 64}, lambda);
  auto adaptive = run_adaptive_study(bathymetry, reference,
                                     {10.0, 5.0, 2.0, 1.0, 0.5}, lambda);

  print_results("Canyon + Seamount: Uniform vs Adaptive", uniform, adaptive);
  write_csv("/tmp/canyon_seamount_convergence.csv", uniform, adaptive);
}

TEST_F(UniformVsAdaptiveConvergenceTest, ShelfBreakConvergence) {
  auto bathymetry = synth::shelf_break;
  auto reference = bathymetry;

  Real lambda = 100.0;

  auto uniform =
      run_uniform_study(bathymetry, reference, {4, 8, 16, 32, 64}, lambda);
  auto adaptive = run_adaptive_study(bathymetry, reference,
                                     {20.0, 10.0, 5.0, 2.0, 1.0}, lambda);

  print_results("Shelf Break: Uniform vs Adaptive", uniform, adaptive);
  write_csv("/tmp/shelf_break_convergence.csv", uniform, adaptive);
}

// =============================================================================
// VTK Output for Visualization
// =============================================================================

TEST_F(UniformVsAdaptiveConvergenceTest, WriteVisualizationVTK) {
  auto bathymetry = synth::canyon_and_seamount;

  Real lambda = 50.0;

  // Uniform 16x16
  {
    QuadtreeAdapter mesh;
    mesh.build_uniform(XMIN, XMAX, YMIN, YMAX, 16, 16);

    CGCubicBezierSmootherConfig config;
    config.lambda = lambda;
    config.edge_ngauss = 4;

    CGCubicBezierBathymetrySmoother smoother(mesh, config);
    smoother.set_bathymetry_data(bathymetry);
    smoother.solve();
    smoother.write_vtk("/tmp/uniform_16x16", 8);
  }

  // Adaptive (thr=2.0 - refines both canyon and seamount)
  {
    AdaptiveCGCubicBezierConfig config;
    config.error_metric_type = ErrorMetricType::NormalizedError;
    config.error_threshold = 2.0;
    config.max_iterations = 15;
    config.max_elements = 5000;
    config.smoother_config.lambda = lambda;
    config.smoother_config.edge_ngauss = 4;
    config.smoother_config.use_iterative_solver = true;
    config.smoother_config.use_multigrid = true;
    config.smoother_config.multigrid_config.min_tree_level = 4;

    AdaptiveCGCubicBezierSmoother smoother(XMIN, XMAX, YMIN, YMAX, 16, 16, config);
    smoother.set_bathymetry_data(bathymetry);
    smoother.solve_adaptive();
    smoother.write_vtk("/tmp/adaptive_thr2.0", 8);

    std::cout << "Adaptive mesh (thr=2.0): " << smoother.mesh().num_elements()
              << " elements\n";
  }

  std::cout << "VTK files written to /tmp/uniform_16x16.vtu and "
               "/tmp/adaptive_thr2.0.vtu\n";
}

// =============================================================================
// Efficiency Comparison: Error vs DOFs
// =============================================================================

TEST_F(UniformVsAdaptiveConvergenceTest, EfficiencyComparison) {
  auto bathymetry = synth::canyon_and_seamount;

  Real lambda = 50.0;

  std::cout << "\n" << std::string(80, '=') << "\n";
  std::cout << "Efficiency Comparison: DOFs needed for target accuracy\n";
  std::cout << std::string(80, '=') << "\n\n";

  // Run both methods at multiple resolutions
  auto uniform =
      run_uniform_study(bathymetry, bathymetry, {4, 8, 16, 32, 64}, lambda);
  auto adaptive = run_adaptive_study(
      bathymetry, bathymetry, {20.0, 10.0, 5.0, 2.0, 1.0, 0.5}, lambda);

  // Find DOF counts needed for similar accuracy (L2 error targets)
  std::vector<Real> target_errors = {5.0, 2.0, 1.0, 0.5};

  std::cout << "| " << std::setw(12) << "Target Err"
            << " | " << std::setw(12) << "Uniform DOFs"
            << " | " << std::setw(14) << "Adaptive DOFs"
            << " | " << std::setw(10) << "Ratio"
            << " |\n";
  std::cout << "|" << std::string(14, '-') << "|" << std::string(14, '-')
            << "|" << std::string(16, '-') << "|" << std::string(12, '-')
            << "|\n";

  for (Real target : target_errors) {
    // Find smallest uniform DOF count achieving target
    Index uniform_dofs = 0;
    for (const auto &r : uniform) {
      if (r.l2_error <= target) {
        uniform_dofs = r.num_dofs;
        break;
      }
    }

    // Find smallest adaptive DOF count achieving target
    Index adaptive_dofs = 0;
    for (const auto &r : adaptive) {
      if (r.l2_error <= target) {
        adaptive_dofs = r.num_dofs;
        break;
      }
    }

    if (uniform_dofs > 0 && adaptive_dofs > 0) {
      Real ratio = static_cast<Real>(uniform_dofs) / adaptive_dofs;
      std::cout << "| " << std::fixed << std::setprecision(1) << std::setw(12)
                << target << " | " << std::setw(12) << uniform_dofs << " | "
                << std::setw(14) << adaptive_dofs << " | " << std::setw(10)
                << std::setprecision(2) << ratio << " |\n";
    }
  }

  std::cout << "\nRatio > 1 means adaptive needs fewer DOFs for same accuracy\n";
}

// =============================================================================
// Export Bathymetry Data for Plotting
// =============================================================================

TEST_F(UniformVsAdaptiveConvergenceTest, ExportBathymetryData) {
  std::string output_dir = "/tmp/synthetic_bathymetry";
  std::string mkdir_cmd = "mkdir -p " + output_dir;
  int ret = system(mkdir_cmd.c_str());
  (void)ret;

  std::cout << "\nExporting synthetic bathymetry functions to CSV...\n";

  synth::export_to_csv(output_dir + "/canyon.csv", synth::canyon);
  std::cout << "  Exported: canyon.csv\n";

  synth::export_to_csv(output_dir + "/seamount.csv", synth::seamount);
  std::cout << "  Exported: seamount.csv\n";

  synth::export_to_csv(output_dir + "/ridge_system.csv", synth::ridge_system);
  std::cout << "  Exported: ridge_system.csv\n";

  synth::export_to_csv(output_dir + "/canyon_and_seamount.csv",
                       synth::canyon_and_seamount);
  std::cout << "  Exported: canyon_and_seamount.csv\n";

  synth::export_to_csv(output_dir + "/shelf_break.csv", synth::shelf_break);
  std::cout << "  Exported: shelf_break.csv\n";

  std::cout << "\nAll data exported to: " << output_dir << "\n";
}

// =============================================================================
// Export Convergence Study with Meshes
// =============================================================================

TEST_F(UniformVsAdaptiveConvergenceTest, ExportConvergenceWithMeshes) {
  std::string output_dir = "/tmp/convergence_study";
  std::string mkdir_cmd = "mkdir -p " + output_dir;
  int ret = system(mkdir_cmd.c_str());
  (void)ret;

  auto bathymetry = synth::canyon_and_seamount;

  std::cout << "\nRunning convergence study with mesh export...\n";

  Real lambda = 50.0;

  // Run studies with mesh export
  auto uniform = run_uniform_study(bathymetry, bathymetry, {4, 8, 16, 32},
                                   lambda, output_dir);
  auto adaptive = run_adaptive_study(bathymetry, bathymetry,
                                     {2.0}, lambda, output_dir);

  // Export convergence results to CSV
  write_csv(output_dir + "/convergence_results.csv", uniform, adaptive);

  // Also export the bathymetry data
  synth::export_to_csv(output_dir + "/bathymetry.csv", bathymetry);

  print_results("Canyon + Seamount Convergence", uniform, adaptive);

  std::cout << "\nAll data exported to: " << output_dir << "\n";
  std::cout << "  - convergence_results.csv\n";
  std::cout << "  - bathymetry.csv\n";
  std::cout << "  - uniform_*_mesh.csv (one per grid size)\n";
  std::cout << "  - adaptive_thr*_mesh.csv (one per threshold)\n";
}
