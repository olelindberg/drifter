#pragma once

/// @file adaptive_cg_linear_bezier_smoother.hpp
/// @brief Adaptive mesh refinement for CG linear Bezier bathymetry smoothing
///
/// Implements coupled adaptive refinement + CG linear Bezier smoothing where
/// the Bezier solution error drives mesh refinement. The algorithm iteratively:
/// 1. Solves CG linear Bezier smoothing on current mesh
/// 2. Estimates L2 error per element: ||z_data - z_bezier||_L2
/// 3. Refines elements where error exceeds threshold
/// 4. Re-solves until convergence
///
/// Uses the CG (Continuous Galerkin) linear Bezier smoother which shares DOFs
/// at element boundaries, providing natural cross-element coupling through the
/// energy functional. Linear elements (degree 1) have 4 DOFs per element and
/// support C0 continuity.

#include "bathymetry/cg_linear_bezier_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace drifter {

// Forward declarations
class BathymetrySource;

/// @brief Timing profile for one adaptive iteration (all times in milliseconds)
struct CGLinearIterationProfile {
  // Top-level phases
  double rebuild_ms = 0.0;          ///< Total rebuild_smoother() time
  double solve_ms = 0.0;            ///< Total smoother_->solve() time
  double error_estimation_ms = 0.0; ///< Total estimate_errors() time
  double marking_ms = 0.0;          ///< select_elements_for_refinement() time
  double refinement_ms = 0.0; ///< refine_elements() time (including re-rebuild)

  // Rebuild breakdown
  double quadtree_build_ms = 0.0;   ///< QuadtreeAdapter construction
  double smoother_init_ms = 0.0;    ///< CGLinearBezierBathymetrySmoother init
  double hessian_assembly_ms = 0.0; ///< assemble_dirichlet_hessian()
  double data_fitting_ms = 0.0;     ///< assemble_data_fitting()

  // Solve breakdown
  double matrix_build_ms = 0.0;        ///< Q matrix construction
  double sparse_lu_compute_ms = 0.0;   ///< SparseLU factorization
  double sparse_lu_solve_ms = 0.0;     ///< SparseLU back-substitution
  double constraint_condense_ms = 0.0; ///< Constraint elimination

  // Context
  Index num_elements = 0;
  Index num_dofs = 0;
  Index num_free_dofs = 0;
  Index num_constraints = 0;

  double total_ms() const {
    return rebuild_ms + solve_ms + error_estimation_ms + marking_ms +
           refinement_ms;
  }
};

/// @brief Strategy for selecting elements to refine
enum class MarkingStrategy {
  FixedFraction, ///< Legacy: top refine_fraction elements (may break symmetry)
  Dorfler,       ///< Bulk criterion: squared errors >= theta * total
  DorflerSymmetric, ///< Dorfler + include all elements at cutoff error
  RelativeThreshold ///< All elements with error > alpha * max_error
};

/// @brief Per-element error estimate for CG linear adaptive smoother
struct CGLinearElementErrorEstimate {
  Index element;         ///< Element index
  Real l2_error;         ///< L2 error ||z_data - z_bezier||_L2
  Real normalized_error; ///< Error normalized by sqrt(element area)
  bool should_refine;    ///< Marked for refinement
};

/// @brief Configuration for adaptive CG linear Bezier smoother
struct AdaptiveCGLinearBezierConfig {
  // Stopping criteria
  Real error_threshold = 0.1;    ///< Stop when max error < threshold (meters)
  int max_iterations = 10;       ///< Maximum adaptation iterations
  int max_elements = 10000;      ///< Maximum number of elements
  int max_refinement_level = 10; ///< Maximum refinement level per axis

  // Marking strategy selection
  MarkingStrategy marking_strategy = MarkingStrategy::DorflerSymmetric;

  // Dorfler parameter: fraction of total squared error to capture
  // theta = 0.5 means refine elements contributing >= 50% of total error
  Real dorfler_theta = 0.5;

  // Relative threshold: refine if error > alpha * max_error
  Real relative_alpha = 0.3;

  // Tolerance for grouping equal errors (symmetry preservation)
  Real symmetry_tolerance = 1e-12;

  // Legacy: fraction of elements to refine (only for FixedFraction strategy)
  Real refine_fraction = 0.2;

  // Error estimation
  int ngauss_error = 4; ///< Gauss points per direction for error integration

  // Smoother configuration (passed to CGLinearBezierBathymetrySmoother)
  CGLinearBezierSmootherConfig smoother_config;

  // Progress reporting
  bool verbose = false;
};

/// @brief Result of a single adaptation iteration for CG linear smoother
struct CGLinearAdaptationResult {
  int iteration;          ///< Iteration number (0-indexed)
  Index num_elements;     ///< Number of elements after this iteration
  Real max_error;         ///< Maximum normalized error across all elements
  Real mean_error;        ///< Mean normalized error across all elements
  Index elements_refined; ///< Number of elements refined in this iteration
  bool converged;         ///< True if stopping criteria met
};

/// @brief Adaptive CG linear Bezier bathymetry smoother with error-driven
/// refinement
///
/// Iteratively solves the CG linear Bezier smoothing problem and refines
/// elements where the fitting error exceeds a threshold. Uses Continuous
/// Galerkin assembly where DOFs at element boundaries are shared, providing
/// natural smoothing across element boundaries.
///
/// @par Algorithm:
/// @code
/// REPEAT:
///   1. Solve CG linear Bezier smoothing on current mesh
///   2. Estimate L2 error per element: ||z_data - z_bezier||_L2
///   3. Mark elements where normalized_error > threshold
///   4. Refine marked elements (with 2:1 balancing)
///   5. Re-solve on refined mesh
/// UNTIL error < tolerance everywhere OR max_iterations OR max_elements
/// @endcode
///
/// @par Example:
/// @code
/// AdaptiveCGLinearBezierConfig config;
/// config.error_threshold = 0.5;  // 0.5 meter threshold
/// config.max_iterations = 5;
/// config.smoother_config.lambda = 1.0;
///
/// AdaptiveCGLinearBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4,
/// config); smoother.set_bathymetry_data(geotiff_source); auto result =
/// smoother.solve_adaptive();
///
/// std::cout << "Converged: " << result.converged
///           << ", Elements: " << result.num_elements
///           << ", Max error: " << result.max_error << " m\n";
///
/// smoother.write_vtk("/tmp/adaptive_cg_linear_bathymetry", 10);
/// @endcode
class AdaptiveCGLinearBezierSmoother {
public:
  /// @brief Construct from domain bounds with initial uniform mesh
  /// @param xmin, xmax X domain bounds
  /// @param ymin, ymax Y domain bounds
  /// @param nx, ny Initial mesh size (nx x ny elements)
  /// @param config Configuration parameters
  AdaptiveCGLinearBezierSmoother(
      Real xmin, Real xmax, Real ymin, Real ymax, int nx, int ny,
      const AdaptiveCGLinearBezierConfig &config = {});

  /// @brief Construct from existing OctreeAdapter
  /// @param octree Initial mesh (will be refined in-place)
  /// @param config Configuration parameters
  /// @note The octree is modified during adaptive refinement
  explicit AdaptiveCGLinearBezierSmoother(
      OctreeAdapter &octree, const AdaptiveCGLinearBezierConfig &config = {});

  // =========================================================================
  // Data input (persistent across refinement iterations)
  // =========================================================================

  /// @brief Set bathymetry from BathymetrySource (e.g., GeoTIFF)
  /// @param source Bathymetry data source
  void set_bathymetry_data(const BathymetrySource &source);

  /// @brief Set bathymetry from function
  /// @param bathy_func Function (x, y) -> depth
  void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);

  // =========================================================================
  // Adaptive solve
  // =========================================================================

  /// @brief Run adaptive refinement loop until convergence
  /// @return Final adaptation result
  CGLinearAdaptationResult solve_adaptive();

  /// @brief Perform single adaptation iteration
  /// @return Result of this iteration
  CGLinearAdaptationResult adapt_once();

  /// @brief Get adaptation history (all iterations)
  const std::vector<CGLinearAdaptationResult> &history() const {
    return history_;
  }

  /// @brief Get per-iteration profiling data (populated when verbose=true)
  const std::vector<CGLinearIterationProfile> &profiles() const {
    return profiles_;
  }

  // =========================================================================
  // Error estimation
  // =========================================================================

  /// @brief Estimate error for all elements
  /// @return Per-element error estimates
  /// @pre Must have called solve_adaptive() or adapt_once() at least once
  std::vector<CGLinearElementErrorEstimate> estimate_errors() const;

  /// @brief Estimate error for single element
  /// @param elem Element index
  /// @return Error estimate for this element
  /// @pre Must have called solve_adaptive() or adapt_once() at least once
  CGLinearElementErrorEstimate estimate_element_error(Index elem) const;

  /// @brief Get maximum normalized error across all elements
  /// @pre Must have called solve_adaptive() or adapt_once() at least once
  Real max_error() const;

  /// @brief Get mean normalized error across all elements
  /// @pre Must have called solve_adaptive() or adapt_once() at least once
  Real mean_error() const;

  // =========================================================================
  // Access to current state
  // =========================================================================

  /// @brief Check if solved (at least one iteration complete)
  bool is_solved() const { return smoother_ && smoother_->is_solved(); }

  /// @brief Get current smoother (valid after solve)
  /// @throws std::runtime_error if not solved
  const CGLinearBezierBathymetrySmoother &smoother() const;

  /// @brief Get current mesh
  const QuadtreeAdapter &mesh() const { return *quadtree_; }

  /// @brief Get current octree (underlying 3D mesh)
  const OctreeAdapter &octree() const { return *octree_; }

  /// @brief Evaluate smoothed bathymetry at point
  /// @param x, y Physical coordinates
  /// @return Smoothed depth at (x, y)
  /// @throws std::runtime_error if not solved or point outside domain
  Real evaluate(Real x, Real y) const;

  /// @brief Write VTK output
  /// @param filename Output filename (without extension)
  /// @param resolution Subdivisions per element edge for visualization
  void write_vtk(const std::string &filename, int resolution = 6) const;

private:
  // Configuration
  AdaptiveCGLinearBezierConfig config_;

  // Mesh (owned or external reference)
  std::unique_ptr<OctreeAdapter>
      octree_owned_;      ///< Owned octree (if constructed from bounds)
  OctreeAdapter *octree_; ///< Pointer to active octree
  std::unique_ptr<QuadtreeAdapter> quadtree_; ///< 2D mesh extracted from octree

  // Persistent bathymetry source
  std::function<Real(Real, Real)> bathy_func_;

  // Current smoother (recreated after each refinement)
  std::unique_ptr<CGLinearBezierBathymetrySmoother> smoother_;

  // Adaptation history
  std::vector<CGLinearAdaptationResult> history_;

  // Profiling
  std::vector<CGLinearIterationProfile> profiles_;
  CGLinearIterationProfile *current_profile_ = nullptr;

  // Gauss quadrature nodes and weights on [0, 1]
  VecX gauss_nodes_;
  VecX gauss_weights_;

  // =========================================================================
  // Internal methods
  // =========================================================================

  /// @brief Initialize Gauss-Legendre quadrature nodes and weights
  void init_gauss_quadrature();

  /// @brief Create/recreate smoother for current mesh
  void rebuild_smoother();

  /// @brief Set bathymetry data on current smoother
  void apply_bathymetry_to_smoother();

  /// @brief Select elements for refinement using configured marking strategy
  /// @param errors Per-element error estimates
  /// @return Element indices selected for refinement
  std::vector<Index> select_elements_for_refinement(
      const std::vector<CGLinearElementErrorEstimate> &errors) const;

  /// @brief Refine marked elements and update mesh
  /// @param elements_to_refine Element indices to refine
  void refine_elements(const std::vector<Index> &elements_to_refine);

  /// @brief Compute L2 error for element via Gauss quadrature
  /// @param elem Element index
  /// @return L2 error ||z_data - z_bezier||_L2 over element
  Real compute_element_l2_error(Index elem) const;

  /// @brief Print profiling report to stdout
  void print_profile_report() const;
};

} // namespace drifter
