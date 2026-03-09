#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/cubic_bezier_basis_2d.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <map>
#include <memory>
#include <set>

using namespace drifter;

/// @brief Final metrics collected after solve
struct SolverMetrics {
  Real solution_l2_norm = 0.0;
  int schur_cg_iterations = 0;
  Real data_residual = 0.0;
  Real regularization_energy = 0.0;
  Real constraint_violation = 0.0;
  Real objective_value = 0.0;
  int qinv_apply_calls = 0;
  double total_solve_ms = 0.0;
};

/// @brief Test fixture for solver comparison on model problem
class CGBezierSolverComparisonTest : public ::testing::Test {
protected:
  static constexpr Real L = 1000.0;
  static constexpr int N = 16;
  static constexpr Real SOLUTION_TOLERANCE = 1e-6;
  static constexpr Real CONSTRAINT_TOLERANCE = 1e-7;
  static constexpr std::string_view OUTPUT_DIR = "/tmp";

  void SetUp() override {
    mesh_.build_uniform(0.0, L, 0.0, L, N, N);
    kx_ = 2.0 * M_PI / L;
    ky_ = 2.0 * M_PI / L;
    bathy_func_ = [this](Real x, Real y) {
      return std::exp(std::sin(kx_ * x) * std::sin(ky_ * y));
    };
  }

  // =========================================================================
  // Solver configuration factories
  // =========================================================================

  CGCubicBezierSmootherConfig create_direct_config() const {
    CGCubicBezierSmootherConfig config;
    config.lambda = 1.0;
    config.use_iterative_solver = false;
    config.use_multigrid = false;
    config.edge_ngauss = 4;
    return config;
  }

  CGCubicBezierSmootherConfig create_iterative_lu_config() const {
    CGCubicBezierSmootherConfig config;
    config.lambda = 1.0;
    config.use_iterative_solver = true;
    config.use_multigrid = false;
    config.tolerance = 1e-6;
    config.max_iterations = 1000;
    config.edge_ngauss = 4;
    return config;
  }

  /// @brief Base multigrid config with common settings
  CGCubicBezierSmootherConfig create_iterative_mg_base_config() const {
    CGCubicBezierSmootherConfig config;
    config.lambda = 1.0;
    config.use_iterative_solver = true;
    config.use_multigrid = true;
    config.tolerance = 1e-6;
    config.max_iterations = 1000;
    config.edge_ngauss = 4;

    // 2-level multigrid: 16x16 mesh has depth 4, so level 4 and 3
    config.multigrid_config.num_levels = 2;
    config.multigrid_config.min_tree_level = 3;
    config.multigrid_config.pre_smoothing = 2;
    config.multigrid_config.post_smoothing = 2;
    config.multigrid_config.smoother_type =
        SmootherType::ColoredMultiplicativeSchwarz;

    return config;
  }

  /// @brief MG with L2 projection transfer + Galerkin coarse grid
  CGCubicBezierSmootherConfig create_iterative_mg_l2_galerkin_config() const {
    auto config = create_iterative_mg_base_config();
    config.multigrid_config.transfer_strategy =
        TransferOperatorStrategy::L2Projection;
    config.multigrid_config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    return config;
  }

  /// @brief MG with L2 projection transfer + cached rediscretization
  CGCubicBezierSmootherConfig create_iterative_mg_l2_cached_config() const {
    auto config = create_iterative_mg_base_config();
    config.multigrid_config.transfer_strategy =
        TransferOperatorStrategy::L2Projection;
    config.multigrid_config.coarse_grid_strategy =
        CoarseGridStrategy::CachedRediscretization;
    return config;
  }

  /// @brief MG with Bezier subdivision transfer + Galerkin coarse grid
  CGCubicBezierSmootherConfig
  create_iterative_mg_bezier_galerkin_config() const {
    auto config = create_iterative_mg_base_config();
    config.multigrid_config.transfer_strategy =
        TransferOperatorStrategy::BezierSubdivision;
    config.multigrid_config.coarse_grid_strategy = CoarseGridStrategy::Galerkin;
    return config;
  }

  /// @brief MG with Bezier subdivision transfer + cached rediscretization
  CGCubicBezierSmootherConfig create_iterative_mg_bezier_cached_config() const {
    auto config = create_iterative_mg_base_config();
    config.multigrid_config.transfer_strategy =
        TransferOperatorStrategy::BezierSubdivision;
    config.multigrid_config.coarse_grid_strategy =
        CoarseGridStrategy::CachedRediscretization;
    return config;
  }

  /// @brief Legacy config for backward compatibility with individual tests
  CGCubicBezierSmootherConfig create_iterative_mg_config() const {
    return create_iterative_mg_l2_galerkin_config();
  }

  // =========================================================================
  // Run solver and collect metrics
  // =========================================================================

  SolverMetrics run_solver(
      const CGCubicBezierSmootherConfig &config, const std::string &solver_name,
      std::vector<CGIterationMetrics> *iteration_history = nullptr,
      std::unique_ptr<CGCubicBezierBathymetrySmoother> *out_smoother = nullptr) {
    SolverMetrics metrics;

    auto smoother =
        std::make_unique<CGCubicBezierBathymetrySmoother>(mesh_, config);

    CGCubicSolveProfile solve_profile;
    smoother->set_solve_profile(&solve_profile);

    smoother->set_bathymetry_data(bathy_func_);

    auto start = std::chrono::high_resolution_clock::now();
    smoother->solve();
    auto end = std::chrono::high_resolution_clock::now();

    metrics.total_solve_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    const VecX &solution = smoother->solution();
    metrics.solution_l2_norm = solution.norm();
    metrics.data_residual = smoother->data_residual();
    metrics.regularization_energy = smoother->regularization_energy();
    metrics.constraint_violation = smoother->constraint_violation();
    metrics.objective_value = smoother->objective_value();
    metrics.schur_cg_iterations = solve_profile.outer_cg_iterations;
    metrics.qinv_apply_calls = solve_profile.qinv_apply_calls;

    if (iteration_history) {
      *iteration_history = std::move(solve_profile.iteration_history);
    }

    solutions_[solver_name] = solution;

    // Optionally return the smoother for surface output
    if (out_smoother) {
      *out_smoother = std::move(smoother);
    }

    return metrics;
  }

  // =========================================================================
  // CSV output
  // =========================================================================

  void write_final_metrics_csv(const std::string &solver_name,
                               const SolverMetrics &m) const {
    std::string filename =
        std::string(OUTPUT_DIR) + "/solver_metrics_" + solver_name + ".csv";
    std::ofstream ofs(filename);
    ofs << std::scientific << std::setprecision(12);
    ofs << "metric,value\n";
    ofs << "solution_l2_norm," << m.solution_l2_norm << "\n";
    ofs << "schur_cg_iterations," << m.schur_cg_iterations << "\n";
    ofs << "data_residual," << m.data_residual << "\n";
    ofs << "regularization_energy," << m.regularization_energy << "\n";
    ofs << "constraint_violation," << m.constraint_violation << "\n";
    ofs << "objective_value," << m.objective_value << "\n";
    ofs << "qinv_apply_calls," << m.qinv_apply_calls << "\n";
    ofs << "total_solve_ms," << m.total_solve_ms << "\n";
    std::cout << "Wrote: " << filename << std::endl;
  }

  void write_iteration_history_csv(
      const std::string &solver_name,
      const std::vector<CGIterationMetrics> &history) const {
    std::string filename =
        std::string(OUTPUT_DIR) + "/solver_iterations_" + solver_name + ".csv";
    std::ofstream ofs(filename);
    ofs << std::scientific << std::setprecision(12);
    ofs << "iteration,schur_residual_norm,relative_residual,alpha,pSp\n";
    for (const auto &m : history) {
      ofs << m.iteration << "," << m.schur_residual_norm << ","
          << m.relative_residual << "," << m.alpha << "," << m.pSp << "\n";
    }
    std::cout << "Wrote: " << filename << std::endl;
  }

  void write_comparison_csv() const {
    std::string filename = std::string(OUTPUT_DIR) + "/solver_comparison.csv";
    std::ofstream ofs(filename);
    ofs << std::scientific << std::setprecision(12);
    ofs << "solver1,solver2,l2_diff,relative_diff,max_pointwise_diff\n";

    std::vector<std::string> names;
    for (const auto &[name, _] : solutions_) {
      names.push_back(name);
    }

    for (size_t i = 0; i < names.size(); ++i) {
      for (size_t j = i + 1; j < names.size(); ++j) {
        const VecX &sol1 = solutions_.at(names[i]);
        const VecX &sol2 = solutions_.at(names[j]);

        Real l2_diff = (sol1 - sol2).norm();
        Real ref_norm = std::max(sol1.norm(), sol2.norm());
        Real relative_diff = (ref_norm > 1e-14) ? l2_diff / ref_norm : l2_diff;
        Real max_diff = (sol1 - sol2).cwiseAbs().maxCoeff();

        ofs << names[i] << "," << names[j] << "," << l2_diff << ","
            << relative_diff << "," << max_diff << "\n";
      }
    }
    std::cout << "Wrote: " << filename << std::endl;
  }

  /// @brief Write control point surface data (x, y, z_solution, z_analytical)
  void write_surface_csv(const std::string &solver_name,
                         const CGCubicBezierBathymetrySmoother &smoother) const {
    std::string filename =
        std::string(OUTPUT_DIR) + "/solver_surface_" + solver_name + ".csv";
    std::ofstream ofs(filename);
    ofs << std::scientific << std::setprecision(12);
    ofs << "x,y,z_solution,z_analytical,element,local_dof,global_dof\n";

    const auto &dof_manager = smoother.dof_manager();
    const auto &basis = smoother.get_basis();
    const VecX &solution = smoother.solution();

    // Track which global DOFs we've written (avoid duplicates from CG sharing)
    std::set<Index> written_dofs;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
      const auto &bounds = mesh_.element_bounds(elem);
      const auto &global_dofs = dof_manager.element_dofs(elem);

      for (int local_dof = 0; local_dof < CubicBezierBasis2D::NDOF;
           ++local_dof) {
        Index global_dof = global_dofs[local_dof];

        // Skip if already written (shared CG DOF)
        if (written_dofs.count(global_dof) > 0) {
          continue;
        }
        written_dofs.insert(global_dof);

        // Get parametric position and map to physical
        Vec2 param = basis.control_point_position(local_dof);
        Real x = bounds.xmin + param(0) * (bounds.xmax - bounds.xmin);
        Real y = bounds.ymin + param(1) * (bounds.ymax - bounds.ymin);

        Real z_solution = solution(global_dof);
        Real z_analytical = bathy_func_(x, y);

        ofs << x << "," << y << "," << z_solution << "," << z_analytical << ","
            << elem << "," << local_dof << "," << global_dof << "\n";
      }
    }
    std::cout << "Wrote: " << filename << std::endl;
  }

  /// @brief Write analytical solution on a regular grid for comparison
  void write_analytical_surface_csv() const {
    std::string filename =
        std::string(OUTPUT_DIR) + "/solver_surface_analytical.csv";
    std::ofstream ofs(filename);
    ofs << std::scientific << std::setprecision(12);
    ofs << "x,y,z_analytical\n";

    // Use same resolution as mesh (N+1 points in each direction)
    int n_points = N + 1;
    Real dx = L / N;
    for (int j = 0; j <= N; ++j) {
      for (int i = 0; i <= N; ++i) {
        Real x = i * dx;
        Real y = j * dx;
        Real z = bathy_func_(x, y);
        ofs << x << "," << y << "," << z << "\n";
      }
    }
    std::cout << "Wrote: " << filename << std::endl;
  }

  // =========================================================================
  // Output helpers
  // =========================================================================

  void print_metrics(const std::string &name, const SolverMetrics &m) const {
    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Solution L2 norm:        " << m.solution_l2_norm << std::endl;
    std::cout << "Data residual:           " << m.data_residual << std::endl;
    std::cout << "Regularization energy:   " << m.regularization_energy
              << std::endl;
    std::cout << "Constraint violation:    " << m.constraint_violation
              << std::endl;
    std::cout << "Objective value:         " << m.objective_value << std::endl;
    std::cout << "Schur CG iterations:     " << m.schur_cg_iterations
              << std::endl;
    std::cout << "Q^{-1} apply calls:      " << m.qinv_apply_calls << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total solve time (ms):   " << m.total_solve_ms << std::endl;
  }

protected:
  QuadtreeAdapter mesh_;
  Real kx_, ky_;
  std::function<Real(Real, Real)> bathy_func_;
  std::map<std::string, VecX> solutions_;
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_F(CGBezierSolverComparisonTest, DirectSolverBaseline) {
  auto config = create_direct_config();
  auto metrics = run_solver(config, "direct");

  print_metrics("Direct Solver (SparseLU)", metrics);
  write_final_metrics_csv("direct", metrics);

  EXPECT_LT(metrics.constraint_violation, 1e-6)
      << "Direct solver should satisfy constraints reasonably";
  EXPECT_GT(metrics.objective_value, 0.0);
  EXPECT_TRUE(std::isfinite(metrics.objective_value));
  EXPECT_EQ(metrics.schur_cg_iterations, 0);
}

TEST_F(CGBezierSolverComparisonTest, IterativeLUPreconditioner) {
  auto config = create_iterative_lu_config();
  std::vector<CGIterationMetrics> history;
  auto metrics = run_solver(config, "iterative_lu", &history);

  print_metrics("Iterative Solver (LU Preconditioner)", metrics);
  write_final_metrics_csv("iterative_lu", metrics);
  write_iteration_history_csv("iterative_lu", history);

  EXPECT_LT(metrics.constraint_violation, CONSTRAINT_TOLERANCE)
      << "Iterative+LU should satisfy constraints";
  EXPECT_LE(metrics.schur_cg_iterations, 50)
      << "LU-preconditioned CG should converge within 50 iterations";
}

TEST_F(CGBezierSolverComparisonTest, IterativeMultigrid) {
  auto config = create_iterative_mg_config();
  std::vector<CGIterationMetrics> history;
  auto metrics = run_solver(config, "iterative_mg", &history);

  print_metrics("Iterative Solver (2-level Multigrid)", metrics);
  write_final_metrics_csv("iterative_mg", metrics);
  write_iteration_history_csv("iterative_mg", history);

  EXPECT_LT(metrics.constraint_violation, 1e-6)
      << "Multigrid should satisfy constraints reasonably";
  EXPECT_LE(metrics.schur_cg_iterations, 200)
      << "MG-preconditioned CG should converge within 200 iterations";
  EXPECT_GT(metrics.qinv_apply_calls, 0);
}

TEST_F(CGBezierSolverComparisonTest, AllSolversComparison) {
  // MG strategy variants to test
  struct MGVariant {
    std::string name;
    std::string display_name;
    std::function<CGCubicBezierSmootherConfig()> create_config;
  };

  std::vector<MGVariant> mg_variants = {
      {"iterative_mg_l2_galerkin", "MG (L2+Galerkin)",
       [this]() { return create_iterative_mg_l2_galerkin_config(); }},
      {"iterative_mg_l2_cached", "MG (L2+Cached)",
       [this]() { return create_iterative_mg_l2_cached_config(); }},
      {"iterative_mg_bezier_galerkin", "MG (Bezier+Galerkin)",
       [this]() { return create_iterative_mg_bezier_galerkin_config(); }},
      {"iterative_mg_bezier_cached", "MG (Bezier+Cached)",
       [this]() { return create_iterative_mg_bezier_cached_config(); }},
  };

  // Run baseline solvers
  std::vector<CGIterationMetrics> history_lu;
  std::unique_ptr<CGCubicBezierBathymetrySmoother> smoother_direct;
  std::unique_ptr<CGCubicBezierBathymetrySmoother> smoother_lu;

  auto metrics_direct = run_solver(create_direct_config(), "direct", nullptr,
                                   &smoother_direct);
  auto metrics_lu = run_solver(create_iterative_lu_config(), "iterative_lu",
                               &history_lu, &smoother_lu);

  // Run all MG variants
  std::map<std::string, SolverMetrics> mg_metrics;
  std::map<std::string, std::vector<CGIterationMetrics>> mg_histories;
  std::map<std::string, std::unique_ptr<CGCubicBezierBathymetrySmoother>>
      mg_smoothers;

  for (const auto &variant : mg_variants) {
    std::vector<CGIterationMetrics> history;
    std::unique_ptr<CGCubicBezierBathymetrySmoother> smoother;
    mg_metrics[variant.name] =
        run_solver(variant.create_config(), variant.name, &history, &smoother);
    mg_histories[variant.name] = std::move(history);
    mg_smoothers[variant.name] = std::move(smoother);
  }

  // Print summary
  std::cout << "\n========================================" << std::endl;
  std::cout << "       ALL SOLVERS COMPARISON" << std::endl;
  std::cout << "========================================" << std::endl;

  print_metrics("1. Direct (SparseLU)", metrics_direct);
  print_metrics("2. Iterative (LU precond)", metrics_lu);

  int idx = 3;
  for (const auto &variant : mg_variants) {
    print_metrics(std::to_string(idx) + ". " + variant.display_name,
                  mg_metrics[variant.name]);
    ++idx;
  }

  // Solution differences vs direct baseline
  std::cout << "\n=== SOLUTION DIFFERENCES (vs Direct) ===" << std::endl;
  std::cout << std::scientific << std::setprecision(6);

  const VecX &sol_direct = solutions_["direct"];
  const VecX &sol_lu = solutions_["iterative_lu"];

  Real diff_direct_lu = (sol_direct - sol_lu).norm();
  std::cout << "||x_direct - x_lu||:              " << diff_direct_lu
            << std::endl;

  for (const auto &variant : mg_variants) {
    const VecX &sol_mg = solutions_[variant.name];
    Real diff = (sol_direct - sol_mg).norm();
    std::cout << "||x_direct - x_" << variant.name << "||: " << diff
              << std::endl;
  }

  // Timing summary
  std::cout << "\n=== TIMING SUMMARY ===" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Direct solve:             " << metrics_direct.total_solve_ms
            << " ms" << std::endl;
  std::cout << "Iterative+LU:             " << metrics_lu.total_solve_ms
            << " ms" << std::endl;

  for (const auto &variant : mg_variants) {
    std::cout << variant.display_name << ": "
              << std::setw(20 - variant.display_name.size()) << " "
              << mg_metrics[variant.name].total_solve_ms << " ms" << std::endl;
  }

  // Iteration count summary
  std::cout << "\n=== ITERATION COUNTS ===" << std::endl;
  std::cout << "Iterative+LU:             " << metrics_lu.schur_cg_iterations
            << " iterations" << std::endl;
  for (const auto &variant : mg_variants) {
    std::cout << variant.display_name << ": "
              << std::setw(20 - variant.display_name.size()) << " "
              << mg_metrics[variant.name].schur_cg_iterations << " iterations"
              << std::endl;
  }

  // Write all CSVs
  write_final_metrics_csv("direct", metrics_direct);
  write_final_metrics_csv("iterative_lu", metrics_lu);
  for (const auto &variant : mg_variants) {
    write_final_metrics_csv(variant.name, mg_metrics[variant.name]);
  }

  write_iteration_history_csv("iterative_lu", history_lu);
  for (const auto &variant : mg_variants) {
    write_iteration_history_csv(variant.name, mg_histories[variant.name]);
  }

  write_comparison_csv();

  // Write surface data for plotting
  write_surface_csv("direct", *smoother_direct);
  write_surface_csv("iterative_lu", *smoother_lu);
  for (const auto &variant : mg_variants) {
    write_surface_csv(variant.name, *mg_smoothers[variant.name]);
  }
  write_analytical_surface_csv();

  // Verify constraints
  EXPECT_LT(metrics_direct.constraint_violation, 1e-6);
  EXPECT_LT(metrics_lu.constraint_violation, CONSTRAINT_TOLERANCE);

  for (const auto &variant : mg_variants) {
    EXPECT_LT(mg_metrics[variant.name].constraint_violation, 1e-6)
        << "MG variant " << variant.name << " should satisfy constraints";
  }

  // Direct and LU should match closely
  Real rel_diff_lu = diff_direct_lu / std::max(sol_direct.norm(), Real(1e-14));
  EXPECT_LT(rel_diff_lu, SOLUTION_TOLERANCE)
      << "Direct and Iterative+LU should produce equivalent solutions";
}
