#pragma once

/// @file adaptive_cg_hermite_smoother.hpp
/// @brief Error-driven adaptive refinement around the CG Hermite smoother
///
/// Mirrors AdaptiveCGLinearBezierSmoother / AdaptiveCGCubicBezierSmoother: solve,
/// estimate per-element error, Dorfler-mark, refine, repeat. The only difference
/// is the inner smoother, which is CGHermiteBathymetrySmoother - so the refined
/// mesh's 2:1 T-junctions are handled by Hermite master/slave substitutions
/// rather than by KKT constraint rows, and the inner solve stays SPD throughout.

#include "bathymetry/adaptive_cg_smoother_base.hpp"
#include "bathymetry/adaptive_smoother_types.hpp"
#include "bathymetry/cg_hermite_bathymetry_smoother.hpp"
#include "core/types.hpp"
#include <functional>
#include <memory>
#include <utility>
#include <string>
#include <vector>

namespace drifter {

class BathymetrySource;

/// @brief Timing profile for one adaptive iteration (milliseconds)
struct HermiteIterationProfile {
    double rebuild_ms = 0.0;
    double solve_ms = 0.0;
    double error_estimation_ms = 0.0;
    double marking_ms = 0.0;
    double refinement_ms = 0.0;

    double quadtree_build_ms = 0.0;
    double smoother_init_ms = 0.0;
    double hessian_assembly_ms = 0.0;
    double data_fitting_ms = 0.0;

    double matrix_build_ms = 0.0;
    double constraint_condense_ms = 0.0;
    double ldlt_compute_ms = 0.0;
    double ldlt_solve_ms = 0.0;

    Index num_elements = 0;
    Index num_dofs = 0;
    Index num_free_dofs = 0;
    Index num_constraints = 0;

    double total_ms() const {
        return rebuild_ms + solve_ms + error_estimation_ms + marking_ms + refinement_ms;
    }
};

/// @brief Per-element error estimate
struct HermiteElementErrorEstimate {
    Index element = -1;
    Real l2_error = 0.0;          ///< ||z_data - z_h||_L2 over the element
    Real normalized_error = 0.0;  ///< l2_error / sqrt(area), an RMS in metres
    bool should_refine = false;

    /// Coarsening indicators: how much the solution changed under refinement
    Real mean_difference = 0.0; ///< int |z_fine - z_coarse| dA / area  [m]
    Real volume_change = 0.0;   ///< int |z_fine - z_coarse| dA         [m^3]
};

/// @brief Configuration for the adaptive CG Hermite smoother
struct AdaptiveCGHermiteConfig {
    // Stopping criteria
    Real error_threshold = 0.1;
    int max_iterations = 10;
    int max_elements = 10000;
    int max_refinement_level = 10;

    /// Stop refining once the data resolution is reached
    bool enforce_pixel_limit = true;
    /// Minimum element size in world units (0 = auto from the data resolution)
    Real min_element_size = 0.0;
    /// Minimum number of raster data points a child element must cover (0 = off)
    int min_data_points_per_element = 4;

    ErrorMetricType error_metric_type = ErrorMetricType::NormalizedError;

    // Dorfler marking
    Real dorfler_theta = 0.5;
    Real symmetry_tolerance = 1e-12;

    // Error estimation
    int ngauss_error = 4;

    /// Inner smoother configuration
    CGHermiteSmootherConfig smoother_config;

    bool verbose = false;

    /// If non-empty, per-element error CSVs are written here each iteration
    std::string error_output_dir = "";

    /// If non-empty, VTK is written to {prefix}_iter_{N}.vtu each iteration
    std::string vtk_output_prefix = "";

    /// Degree of the VTK_LAGRANGE_QUAD cells written for the fitted surface.
    /// <= 0 uses the element surface degree, which reproduces the surface
    /// exactly but gives ParaView few nodes to tessellate; a higher degree
    /// resamples the same polynomial on more nodes for better visual inspection.
    int vtk_order = 0;
};

/// @brief Result of a single adaptation iteration
struct HermiteAdaptationResult {
    int iteration = 0;
    Index num_elements = 0;
    Real max_error = 0.0;
    Real mean_error = 0.0;
    Index elements_refined = 0;
    bool converged = false;
    ConvergenceReason convergence_reason = ConvergenceReason::NotConverged;
};

/// @brief Adaptive CG Hermite bathymetry smoother
class AdaptiveCGHermiteSmoother : public AdaptiveCGSmootherBase {
public:
    /// @brief Construct from domain bounds with an initial uniform mesh
    AdaptiveCGHermiteSmoother(Real xmin, Real xmax, Real ymin, Real ymax, int nx, int ny,
                              const AdaptiveCGHermiteConfig &config = {});

    /// @brief Construct from an existing octree (refined in place)
    explicit AdaptiveCGHermiteSmoother(OctreeAdapter &octree,
                                       const AdaptiveCGHermiteConfig &config = {});

    // =========================================================================
    // Adaptive solve
    // =========================================================================

    /// @brief Run the adaptive refinement loop until a stopping criterion is met
    HermiteAdaptationResult solve_adaptive();

    /// @brief Perform a single adaptation iteration
    HermiteAdaptationResult adapt_once();

    const std::vector<HermiteAdaptationResult> &history() const { return history_; }
    const std::vector<HermiteIterationProfile> &profiles() const { return profiles_; }

    // =========================================================================
    // Error estimation
    // =========================================================================

    std::vector<HermiteElementErrorEstimate> estimate_errors() const;
    HermiteElementErrorEstimate estimate_element_error(Index elem) const;
    Real max_error() const;
    Real mean_error() const;

    // =========================================================================
    // Access
    // =========================================================================

    bool is_solved() const { return smoother_ && smoother_->is_solved(); }

    /// @brief The inner smoother
    /// @throws std::runtime_error if not yet built
    const CGHermiteBathymetrySmoother &smoother() const;

    const AdaptiveCGHermiteConfig &config() const { return config_; }

    /// @brief Set the data resolution (pixel size, world units) as a function of position
    ///
    /// Wired from MultiSourceBathymetry::get_min_element_size_meters so that
    /// high-resolution tiles allow finer refinement than the primary raster.
    /// A return value <= 0 means "unknown here" and falls back to the config.
    void set_resolution_func(std::function<Real(Real, Real)> f) {
        resolution_func_ = std::move(f);
    }

    /// @brief Write the fitted surface as per-element VTK_LAGRANGE_QUAD cells
    ///
    /// @param order Degree of the emitted cells; <= 0 falls back to
    ///              config().vtk_order, and then to the exact degree of the
    ///              Hermite element
    void write_vtk(const std::string &filename, int order = 0) const;

protected:
    bool is_solved_impl() const override { return smoother_ && smoother_->is_solved(); }
    Real smoother_evaluate(Real x, Real y) const override { return smoother_->evaluate(x, y); }
    void rebuild_smoother() override;
    void apply_bathymetry_to_smoother() override;

    VecX get_element_coefficients_impl(Index elem) const override {
        return smoother_->element_coefficients(elem);
    }
    const Basis2DBase &get_basis_impl() const override;

private:
    std::vector<Index>
    select_elements_for_refinement(const std::vector<HermiteElementErrorEstimate> &errors) const;
    void refine_elements(const std::vector<Index> &elements_to_refine);
    void compute_element_error_statistics(Index elem, Real &l2_error) const;

    /// @brief Per-element VTK cell data: the refinement error metric, its
    ///        components, and the refinement level
    std::vector<std::pair<std::string, std::vector<Real>>>
    element_cell_data(const std::vector<HermiteElementErrorEstimate> &errors) const;
    bool is_element_on_land(Index elem) const;

    /// @brief Data resolution (pixel size) at the centre of an element, or 0 if unknown
    Real element_resolution(Index elem) const;

    /// @brief Whether refining this element is allowed by the resolution limits
    bool refinement_allowed(Index elem) const;

    void print_profile_report() const;

    Real error_metric(const HermiteElementErrorEstimate &err) const {
        switch (config_.error_metric_type) {
        case ErrorMetricType::MeanDifference:
            return err.mean_difference;
        case ErrorMetricType::VolumeChange:
            return err.volume_change;
        default:
            return err.normalized_error;
        }
    }

    AdaptiveCGHermiteConfig config_;
    std::function<Real(Real, Real)> resolution_func_;
    std::unique_ptr<CGHermiteBathymetrySmoother> smoother_;
    std::vector<HermiteAdaptationResult> history_;
    std::vector<HermiteIterationProfile> profiles_;
    HermiteIterationProfile *current_profile_ = nullptr;
};

} // namespace drifter
