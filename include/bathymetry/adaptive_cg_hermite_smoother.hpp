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
#include "mesh/coastline_refinement.hpp"
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
    /// Named for the phases rather than for a factorisation: which backend runs
    /// is CGHermiteSmootherConfig::solver, and not all of them are an LDL^T.
    double factorize_ms = 0.0;
    double substitute_ms = 0.0;

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

/// @brief Why an element may not be refined
///
/// The refinement level cap is deliberately not part of this: it is a user
/// budget, not a property of the data, and is tested separately.
enum class RefinementBlock {
    None,            ///< Refinement is allowed
    Pinned,          ///< Land / NoData element, pinned out of the fit
    PixelResolution, ///< Children would fall below the raster pixel size
    DataDensity      ///< Children would cover fewer than min_data_points_per_element
};

/// @brief Result of a single adaptation iteration
struct HermiteAdaptationResult {
    int iteration = 0;
    Index num_elements = 0;
    Real max_error = 0.0;
    /// Largest error among elements that may still be refined. The stopping
    /// test uses this rather than max_error, which elements parked at their
    /// resolution floor could otherwise hold above the threshold forever.
    Real max_refinable_error = 0.0;
    /// How many elements may still be refined. Below num_elements when part of
    /// the mesh has reached a data-resolution floor, the level cap, or is pinned.
    Index num_refinable = 0;
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

    // =========================================================================
    // Coastline pre-pass
    // =========================================================================

    /// @brief Supply the coastline to refine toward before the error-driven loop
    ///
    /// The index is a pure geometry query object; loading the vector file is the
    /// caller's job. A null or empty index leaves the pre-pass a no-op.
    ///
    /// @param index Segment / circumradius R-tree, from CoastlineReader::build_index()
    /// @param max_level Level cap for the pre-pass, independent of
    ///        config().max_refinement_level, which bounds the error-driven loop
    void set_coastline(std::shared_ptr<const CoastlineIndex> index, int max_level);

    /// @brief Refine while an element is larger than the tightest coastline
    ///        feature it contains
    ///
    /// Runs automatically at the start of solve_adaptive(), once. Refines the mesh
    /// only - no surface is fitted, so this is cheap relative to an adaptive
    /// iteration and needs no bathymetry data.
    ///
    /// @return Number of refinement sweeps performed (0 if no coastline was set)
    int refine_coastline();

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

    /// @brief Refine the mesh without rebuilding the smoother
    ///
    /// Used by the coastline pre-pass, which runs before the first solve: there
    /// is no solution to carry over, so building a Hermite system per sweep only
    /// to discard it would be wasted work. adapt_once() builds it lazily.
    void refine_octree_only(const std::vector<Index> &elements_to_refine);

    /// @brief Basis values at the error-quadrature points, one row per point
    ///
    /// The grid is parametric and fixed, so this is evaluated once and reused
    /// for every element of every iteration.
    const MatX &error_basis() const;

    /// @brief L2 error over the element, ignoring pinned (land / NoData) points
    /// @param elem Element index
    /// @param l2_error Output error, renormalised by the weight actually sampled
    /// @param valid_weight Output quadrature weight that was not pinned; 0 means the
    ///        element carries no data and must not be marked for refinement
    void compute_element_error_statistics(Index elem, Real &l2_error,
                                          Real &valid_weight) const;

    /// @brief Per-element VTK cell data: the refinement error metric, its
    ///        components, and the refinement level
    std::vector<std::pair<std::string, std::vector<Real>>>
    element_cell_data(const std::vector<HermiteElementErrorEstimate> &errors) const;
    bool is_element_on_land(Index elem) const;

    /// @brief Data resolution (pixel size) at the centre of an element, or 0 if unknown
    Real element_resolution(Index elem) const;

    /// @brief Why refining this element is not allowed, or None if it is
    ///
    /// Covers only the data-driven limits; the refinement level cap is a
    /// separate test, because a level cap is a user budget rather than a
    /// property of the data. See can_refine() for the combined predicate.
    RefinementBlock classify_refinement(Index elem) const;

    /// @brief Whether refining this element is allowed by the resolution limits
    bool refinement_allowed(Index elem) const {
        return classify_refinement(elem) == RefinementBlock::None;
    }

    /// @brief Whether this element may be refined at all: below the level cap
    ///        and admissible under the data-resolution limits
    ///
    /// This is the single predicate that gates marking. The limits are
    /// per-element floors, so an element failing it is skipped while the rest
    /// of the mesh keeps adapting.
    bool can_refine(Index elem) const;

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

    std::shared_ptr<const CoastlineIndex> coastline_index_;
    int coastline_max_level_ = 10;
    bool coastline_refined_ = false;

    std::unique_ptr<CGHermiteBathymetrySmoother> smoother_;
    std::vector<HermiteAdaptationResult> history_;
    std::vector<HermiteIterationProfile> profiles_;
    HermiteIterationProfile *current_profile_ = nullptr;

    /// Lazily filled by error_basis()
    mutable MatX error_basis_;

    /// The last iteration's per-element errors, so write_vtk() need not repeat
    /// the whole estimation pass the adaptive loop has just done
    mutable std::vector<HermiteElementErrorEstimate> last_errors_;
};

} // namespace drifter
