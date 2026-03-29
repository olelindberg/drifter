#include "bathymetry/adaptive_cg_cubic_bezier_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace drifter;

/// @brief Metrics collected for adaptive solver comparison
struct AdaptiveSolverMetrics {
  std::string solver_name;
  Index num_elements = 0;
  Index num_dofs = 0;
  int adaptation_iterations = 0;
  double total_time_ms = 0.0;

  // VolumeChange statistics
  Real volume_change_min = 0.0;
  Real volume_change_max = 0.0;
  Real volume_change_rmse = 0.0;
  Real volume_change_mean = 0.0;

  // Convergence info
  Real max_error = 0.0;
  Real mean_error = 0.0;
  bool converged = false;
  ConvergenceReason reason = ConvergenceReason::NotConverged;
};

/// @brief Test fixture for adaptive solver comparison on model problem
class AdaptiveCGBezierSolverComparisonTest : public ::testing::Test {
protected:
  static constexpr Real L = 1000.0;
  static constexpr std::string_view OUTPUT_DIR = "/tmp";

  // Model problem functions
  std::function<Real(Real, Real)> bathy_func_;
  Real kx_, ky_;

  void SetUp() override {
    kx_ = 2.0 * M_PI / L;
    ky_ = 2.0 * M_PI / L;
    bathy_func_ = [this](Real x, Real y) {
      return std::exp(std::sin(kx_ * x) * std::sin(ky_ * y));
    };
  }

  // ===========================================================================
  // Model problem factories
  // ===========================================================================

  /// @brief exp(sin*sin) - smooth periodic function (uniform refinement)
  std::function<Real(Real, Real)> create_exp_sin_sin() const {
    return [this](Real x, Real y) {
      return std::exp(std::sin(kx_ * x) * std::sin(ky_ * y));
    };
  }

  /// @brief Gaussian bump - localized feature (adaptive refinement)
  /// @param cx, cy Center of bump
  /// @param sigma Width of bump
  /// @param amplitude Height of bump
  /// @param base Base depth
  std::function<Real(Real, Real)> create_gaussian_bump(Real cx, Real cy,
                                                       Real sigma,
                                                       Real amplitude = 50.0,
                                                       Real base = 10.0) const {
    return [cx, cy, sigma, amplitude, base](Real x, Real y) {
      Real dx = x - cx;
      Real dy = y - cy;
      return base +
             amplitude * std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
    };
  }

  /// @brief Single sharp bump in center - wider base with steep sides
  std::function<Real(Real, Real)> create_center_gaussian_bump() const {
    return [this](Real x, Real y) {
      // Super-Gaussian: wider base but sharper peak/edges (power 4 instead of
      // 2)
      Real dx = x - 0.5 * L;
      Real dy = y - 0.5 * L;
      Real r2 = dx * dx + dy * dy;
      Real sigma = 0.08 * L; // 80m width
      Real sigma4 = sigma * sigma * sigma * sigma;
      Real bump = 200.0 * std::exp(-(r2 * r2) / (2.0 * sigma4)); // exp(-r^4)
      return 10.0 + bump;
    };
  }

  // ===========================================================================
  // Solver configuration factories
  // ===========================================================================

  /// @brief Base adaptive config with common settings
  AdaptiveCGCubicBezierConfig create_base_adaptive_config() const {
    AdaptiveCGCubicBezierConfig config;

    // Adaptive refinement settings
    config.error_metric_type = ErrorMetricType::VolumeChange;
    config.error_threshold = 100000.0; // 0.01 m³ volume change threshold
    config.max_iterations = 10;
    config.max_elements = 5000;
    config.max_refinement_level = 5;
    config.ngauss_error = 6;
    config.verbose = true;

    // Common smoother settings
    config.smoother_config.lambda = 10.0;
    config.smoother_config.edge_ngauss = 4;
    config.smoother_config.tolerance = 1e-6;
    config.smoother_config.max_iterations = 1000;

    return config;
  }

  /// @brief Direct solver config
  AdaptiveCGCubicBezierConfig create_direct_config() const {
    auto config = create_base_adaptive_config();
    config.smoother_config.use_iterative_solver = false;
    config.smoother_config.use_multigrid = false;
    return config;
  }

  /// @brief Iterative solver with LU preconditioner
  AdaptiveCGCubicBezierConfig create_iterative_lu_config() const {
    auto config = create_base_adaptive_config();
    config.smoother_config.use_iterative_solver = true;
    config.smoother_config.use_multigrid = false;
    return config;
  }

  /// @brief Iterative solver with Multigrid preconditioner
  /// Uses BezierSubdivision + CachedRediscretization (recommended for adaptive)
  AdaptiveCGCubicBezierConfig create_iterative_mg_config() const {
    auto config = create_base_adaptive_config();
    config.smoother_config.use_iterative_solver = true;
    config.smoother_config.use_multigrid = true;

    // Multigrid configuration
    config.smoother_config.multigrid_config.min_tree_level =
        1; // Coarsen to 2x2
    config.smoother_config.multigrid_config.pre_smoothing = 2;
    config.smoother_config.multigrid_config.post_smoothing = 2;
    config.smoother_config.multigrid_config.smoother_type =
        SmootherType::ColoredMultiplicativeSchwarz;
    config.smoother_config.multigrid_config.transfer_strategy =
        TransferOperatorStrategy::BezierSubdivision;
    config.smoother_config.multigrid_config.coarse_grid_strategy =
        CoarseGridStrategy::CachedRediscretization;

    return config;
  }

  // ===========================================================================
  // Statistics computation
  // ===========================================================================

  /// @brief Compute VolumeChange statistics from error estimates
  void compute_volume_change_stats(
      const std::vector<CGCubicElementErrorEstimate> &errors, Real &min,
      Real &max, Real &rmse, Real &mean) const {
    if (errors.empty()) {
      min = max = rmse = mean = 0.0;
      return;
    }

    min = std::numeric_limits<Real>::max();
    max = 0.0;
    Real sum = 0.0;
    Real sum_sq = 0.0;

    for (const auto &e : errors) {
      min = std::min(min, e.volume_change);
      max = std::max(max, e.volume_change);
      sum += e.volume_change;
      sum_sq += e.volume_change * e.volume_change;
    }

    Index n = static_cast<Index>(errors.size());
    mean = sum / static_cast<Real>(n);
    rmse = std::sqrt(sum_sq / static_cast<Real>(n));
  }

  // ===========================================================================
  // Run solver and collect metrics
  // ===========================================================================

  AdaptiveSolverMetrics
  run_adaptive_solver(const AdaptiveCGCubicBezierConfig &config,
                      const std::string &solver_name,
                      const std::function<Real(Real, Real)> &bathy_func,
                      const std::string &vtk_prefix = "") {
    AdaptiveSolverMetrics metrics;
    metrics.solver_name = solver_name;

    std::cout << "\n=== Running " << solver_name << " ===" << std::endl;

    // Create smoother with 1x1 initial mesh
    AdaptiveCGCubicBezierSmoother smoother(0.0, L, 0.0, L, 1, 1, config);
    smoother.set_bathymetry_data(bathy_func);

    // Run adaptive solve with timing
    auto start = std::chrono::high_resolution_clock::now();
    auto result = smoother.solve_adaptive();
    auto end = std::chrono::high_resolution_clock::now();

    metrics.total_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    metrics.num_elements = result.num_elements;
    metrics.num_dofs = smoother.smoother().num_global_dofs();
    metrics.adaptation_iterations = result.iteration + 1;
    metrics.max_error = result.max_error;
    metrics.mean_error = result.mean_error;
    metrics.converged = result.converged;
    metrics.reason = result.convergence_reason;

    // Compute VolumeChange statistics
    auto errors = smoother.estimate_errors();
    compute_volume_change_stats(
        errors, metrics.volume_change_min, metrics.volume_change_max,
        metrics.volume_change_rmse, metrics.volume_change_mean);

    // Write VTK output if prefix provided
    if (!vtk_prefix.empty()) {
      std::string vtk_file = vtk_prefix + "_" + solver_name;
      smoother.write_vtk(vtk_file, 8);
      std::cout << "  VTK: " << vtk_file << ".vtu" << std::endl;
    }

    // Print summary
    std::cout << "  Elements: " << metrics.num_elements << std::endl;
    std::cout << "  DOFs: " << metrics.num_dofs << std::endl;
    std::cout << "  Iterations: " << metrics.adaptation_iterations << std::endl;
    std::cout << "  Time: " << metrics.total_time_ms << " ms" << std::endl;
    std::cout << "  Converged: " << (metrics.converged ? "yes" : "no")
              << std::endl;
    std::cout << "  VolumeChange - min: " << metrics.volume_change_min
              << ", max: " << metrics.volume_change_max
              << ", rmse: " << metrics.volume_change_rmse
              << ", mean: " << metrics.volume_change_mean << std::endl;

    return metrics;
  }

  // ===========================================================================
  // CSV output
  // ===========================================================================

  void write_comparison_csv(
      const std::vector<AdaptiveSolverMetrics> &all_metrics) const {
    std::string filename =
        std::string(OUTPUT_DIR) + "/adaptive_solver_comparison.csv";
    std::ofstream ofs(filename);
    ofs << std::scientific << std::setprecision(12);

    // Header
    ofs << "solver,num_elements,num_dofs,iterations,total_time_ms,"
        << "volume_change_min,volume_change_max,volume_change_rmse,"
        << "volume_change_mean,max_error,mean_error,converged\n";

    // Data rows
    for (const auto &m : all_metrics) {
      ofs << m.solver_name << "," << m.num_elements << "," << m.num_dofs << ","
          << m.adaptation_iterations << "," << m.total_time_ms << ","
          << m.volume_change_min << "," << m.volume_change_max << ","
          << m.volume_change_rmse << "," << m.volume_change_mean << ","
          << m.max_error << "," << m.mean_error << ","
          << (m.converged ? "true" : "false") << "\n";
    }

    std::cout << "\nWrote: " << filename << std::endl;
  }

  // Storage for collected metrics
  std::vector<AdaptiveSolverMetrics> all_metrics_;
};

// =============================================================================
// Tests
// =============================================================================

TEST_F(AdaptiveCGBezierSolverComparisonTest,
       AdaptiveSolverComparisonVolumeChange) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Adaptive Solver Comparison (VolumeChange)" << std::endl;
  std::cout << "Model: exp(sin(kx*x) * sin(ky*y))" << std::endl;
  std::cout << "Domain: " << L << " x " << L << " m" << std::endl;
  std::cout << "Initial mesh: 1x1, max refinement level: 5" << std::endl;
  std::cout << "========================================" << std::endl;

  auto bathy = create_exp_sin_sin();
  std::string vtk_prefix = std::string(OUTPUT_DIR) + "/adaptive_exp_sin_sin";

  // Run all three solvers
  all_metrics_.push_back(
      run_adaptive_solver(create_direct_config(), "direct", bathy, vtk_prefix));
  all_metrics_.push_back(run_adaptive_solver(
      create_iterative_lu_config(), "iterative_lu", bathy, vtk_prefix));
  all_metrics_.push_back(run_adaptive_solver(
      create_iterative_mg_config(), "iterative_mg", bathy, vtk_prefix));

  // Write comparison CSV
  write_comparison_csv(all_metrics_);

  // Verify all solvers completed
  for (const auto &m : all_metrics_) {
    EXPECT_GT(m.num_elements, 1)
        << "Solver " << m.solver_name << " should have refined";
    EXPECT_GT(m.num_dofs, 0)
        << "Solver " << m.solver_name << " should have DOFs";
  }

  // Print summary table
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << std::left << std::setw(15) << "Solver" << std::setw(10)
            << "Elements" << std::setw(10) << "DOFs" << std::setw(10) << "Iters"
            << std::setw(12) << "Time(ms)" << std::setw(15) << "VC_RMSE"
            << std::setw(10) << "Converged" << std::endl;
  std::cout << std::string(82, '-') << std::endl;

  for (const auto &m : all_metrics_) {
    std::cout << std::left << std::setw(15) << m.solver_name << std::setw(10)
              << m.num_elements << std::setw(10) << m.num_dofs << std::setw(10)
              << m.adaptation_iterations << std::setw(12) << std::fixed
              << std::setprecision(1) << m.total_time_ms << std::setw(15)
              << std::scientific << std::setprecision(3) << m.volume_change_rmse
              << std::setw(10) << (m.converged ? "yes" : "no") << std::endl;
  }
}

TEST_F(AdaptiveCGBezierSolverComparisonTest,
       AdaptiveSolverComparisonGaussianBump) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Adaptive Solver Comparison (Gaussian Bump)" << std::endl;
  std::cout << "Model: Single Gaussian bump at center (500,500)" << std::endl;
  std::cout << "Domain: " << L << " x " << L << " m" << std::endl;
  std::cout << "Initial mesh: 1x1, max refinement level: 5" << std::endl;
  std::cout << "========================================" << std::endl;

  auto bathy = create_center_gaussian_bump();
  std::string vtk_prefix = std::string(OUTPUT_DIR) + "/adaptive_gaussian_bump";

  // Run all three solvers
  all_metrics_.push_back(
      run_adaptive_solver(create_direct_config(), "direct", bathy, vtk_prefix));
  all_metrics_.push_back(run_adaptive_solver(
      create_iterative_lu_config(), "iterative_lu", bathy, vtk_prefix));
  all_metrics_.push_back(run_adaptive_solver(
      create_iterative_mg_config(), "iterative_mg", bathy, vtk_prefix));

  // Write comparison CSV
  {
    std::string filename =
        std::string(OUTPUT_DIR) + "/adaptive_solver_comparison_gaussian.csv";
    std::ofstream ofs(filename);
    ofs << std::scientific << std::setprecision(12);

    ofs << "solver,num_elements,num_dofs,iterations,total_time_ms,"
        << "volume_change_min,volume_change_max,volume_change_rmse,"
        << "volume_change_mean,max_error,mean_error,converged\n";

    for (const auto &m : all_metrics_) {
      ofs << m.solver_name << "," << m.num_elements << "," << m.num_dofs << ","
          << m.adaptation_iterations << "," << m.total_time_ms << ","
          << m.volume_change_min << "," << m.volume_change_max << ","
          << m.volume_change_rmse << "," << m.volume_change_mean << ","
          << m.max_error << "," << m.mean_error << ","
          << (m.converged ? "true" : "false") << "\n";
    }
    std::cout << "\nWrote: " << filename << std::endl;
  }

  // Verify all solvers completed
  for (const auto &m : all_metrics_) {
    EXPECT_GT(m.num_elements, 1)
        << "Solver " << m.solver_name << " should have refined";
    EXPECT_GT(m.num_dofs, 0)
        << "Solver " << m.solver_name << " should have DOFs";
  }

  // Print summary table
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << std::left << std::setw(15) << "Solver" << std::setw(10)
            << "Elements" << std::setw(10) << "DOFs" << std::setw(10) << "Iters"
            << std::setw(12) << "Time(ms)" << std::setw(15) << "VC_RMSE"
            << std::setw(10) << "Converged" << std::endl;
  std::cout << std::string(82, '-') << std::endl;

  for (const auto &m : all_metrics_) {
    std::cout << std::left << std::setw(15) << m.solver_name << std::setw(10)
              << m.num_elements << std::setw(10) << m.num_dofs << std::setw(10)
              << m.adaptation_iterations << std::setw(12) << std::fixed
              << std::setprecision(1) << m.total_time_ms << std::setw(15)
              << std::scientific << std::setprecision(3) << m.volume_change_rmse
              << std::setw(10) << (m.converged ? "yes" : "no") << std::endl;
  }
}
