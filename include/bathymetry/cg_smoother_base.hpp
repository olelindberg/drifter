#pragma once

/// @file cg_smoother_base.hpp
/// @brief Abstract base class for CG Bezier bathymetry smoothers
///
/// Provides common functionality shared between CGLinearBezierBathymetrySmoother
/// and CGCubicBezierBathymetrySmoother. Uses standard inheritance with virtual
/// methods for customization points. Virtual dispatch overhead is acceptable
/// since these methods are not in performance-critical inner loops.

#include "bathymetry/element_data_mask.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include "mesh/seabed_surface.hpp"
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace drifter {

// Forward declarations
class OctreeAdapter;
class BathymetrySource;
struct BathymetryPoint;
class HessianBase;
class Basis2DBase;

/// @brief Configuration for boundary relaxation zone
///
/// Gradually reduces data fitting weight near domain boundaries to eliminate
/// oscillations. Uses smoothstep interpolation for C1 smooth transition.
struct BoundaryRelaxationConfig {
    /// Enable boundary relaxation
    bool enabled = false;

    /// Relaxation zone width in physical units
    Real width = 0.0;

    /// Minimum relaxation factor at boundary (0 = full relaxation, no data fitting)
    Real min_factor = 0.0;

    /// Enable relaxation per edge: [left, right, bottom, top]
    /// Edge IDs: 0=left (x=xmin), 1=right (x=xmax), 2=bottom (y=ymin), 3=top (y=ymax)
    std::array<bool, 4> edge_enabled = {true, true, true, true};
};

/// @brief Abstract base class for CG Bezier bathymetry smoothers
///
/// Implements common functionality for both linear and cubic Bezier smoothers:
/// - Data input (set_bathymetry_data, set_scattered_points)
/// - Element lookup and evaluation
/// - Seabed transfer
/// - Diagnostic methods (data_residual, regularization_energy)
///
/// Derived classes must implement:
/// - set_bathymetry_data_impl() - assembles hessian and data fitting matrices
/// - element_coefficients() - extracts DOF values for an element
/// - evaluate_scalar() - evaluates Bezier surface at parametric coords
/// - evaluate_gradient_uv() - evaluates gradient at parametric coords
/// - dof_manager accessors
class CGSmootherBase {
public:
    virtual ~CGSmootherBase() = default;

    // =========================================================================
    // Data input - implemented in base
    // =========================================================================

    /// @brief Set bathymetry from BathymetrySource (e.g., GeoTIFF)
    ///
    /// Also adopts the source's pinned region (land / NoData / outside coverage), so
    /// those points are excluded from the least-squares term rather than fitted as
    /// depth-0 observations. See set_pin_predicate().
    void set_bathymetry_data(const BathymetrySource &source);

    /// @brief Set bathymetry from function
    /// @note Calls derived class set_bathymetry_data_impl()
    /// @note Leaves the pin predicate untouched; an analytic function has no gaps
    ///       unless the caller sets one explicitly.
    void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);

    /// @brief Tell the smoother where there is no data, and where there is land
    ///
    /// Both predicates drop their points from the least-squares term: a gap has no
    /// observation to fit, and land is a known value imposed strongly instead of
    /// fitted as weak data the smoothness term would fight.
    ///
    /// Beyond that, the two are pooled into a single "not water" region, which
    /// ElementDataMask then classifies per element - Water is solved for, Beach
    /// (the rim) is pinned to depth 0, and Inland (the interior) leaves the system
    /// altogether. Smoothers that cannot express a Dirichlet condition
    /// (the Bezier family) use only the least-squares exclusion.
    ///
    /// Must be set before set_bathymetry_data() to affect that assembly. Empty
    /// predicates (the default) mean "water everywhere", i.e. the analytic-function
    /// path, which is unaffected by any of this.
    void set_data_masks(std::function<bool(Real, Real)> has_data,
                        std::function<bool(Real, Real)> is_land) {
        has_data_func_ = std::move(has_data);
        is_land_func_ = std::move(is_land);
    }

    /// @brief The land predicate, or an empty function if none is set
    const std::function<bool(Real, Real)> &land_predicate() const { return is_land_func_; }

    /// @brief Per-element water / beach / inland classification, or null if unset
    ///
    /// Built by smoothers that act on it; the Bezier family leaves it null.
    const ElementDataMask *element_mask() const { return element_mask_.get(); }

    /// @brief Number of data-fitting quadrature points dropped by the last assembly
    Index num_excluded_quadrature_points() const { return num_excluded_quad_points_; }

    /// @brief Set bathymetry from scattered points (Vec3)
    void set_scattered_points(const std::vector<Vec3> &points);

    /// @brief Set bathymetry from scattered BathymetryPoints
    void set_scattered_points(const std::vector<BathymetryPoint> &points);

    // =========================================================================
    // Solution evaluation - implemented in base
    // =========================================================================

    /// @brief Evaluate smoothed bathymetry at point
    /// @throws std::runtime_error if not solved
    Real evaluate(Real x, Real y) const;

    /// @brief Evaluate gradient at point
    /// @throws std::runtime_error if not solved
    Vec2 evaluate_gradient(Real x, Real y) const;

    /// @brief Get solution vector
    const VecX &solution() const { return solution_; }

    // =========================================================================
    // Transfer and output - implemented in base
    // =========================================================================

    /// @brief Transfer solution to SeabedSurface
    /// @throws std::runtime_error if not solved
    void transfer_to_seabed(SeabedSurface &seabed) const;

    // =========================================================================
    // Diagnostics - implemented in base
    // =========================================================================

    /// @brief Compute data fitting residual ||Bx - d||²_W
    Real data_residual() const;

    /// @brief Compute regularization energy x'Hx
    Real regularization_energy() const;

    /// @brief Compute total objective value (alpha*regularization + lambda*data_residual)
    Real objective_value() const;

    // =========================================================================
    // Accessors - implemented in base
    // =========================================================================

    bool is_solved() const { return solved_; }
    const QuadtreeAdapter &mesh() const { return *quadtree_; }

    // DOF manager accessors - delegate to derived class
    Index num_global_dofs() const { return dof_manager_num_global_dofs(); }
    Index num_free_dofs() const { return dof_manager_num_free_dofs(); }
    Index num_constraints() const { return dof_manager_num_constraints(); }

    /// @brief Set external cache for element matrices (for multigrid reuse)
    /// @param cache Pointer to map owned by adaptive smoother (persists across refinement)
    /// @note Cache is populated during assemble_hessian_global() and assemble_data_fitting_global()
    void set_element_matrix_cache(
        std::map<std::tuple<uint64_t, int, int>, MatX>* cache) {
        element_matrix_cache_ = cache;
    }

    /// @brief Get element coefficients in the *parametric* basis
    ///
    /// For a Bernstein basis these are the control point values verbatim. For a
    /// Hermite basis, whose DOFs are physical derivatives, the raw DOF values are
    /// pre-multiplied by element_dof_scaling() so that the result can be paired
    /// with the parametric basis evaluation - i.e. these are always coefficients
    /// of basis().evaluate(u, v).
    ///
    /// @return Vector of coefficients for this element
    /// @note Public so adaptive smoothers can access coefficients
    VecX element_coefficients(Index elem) const;

    /// @brief Get element coefficients in the Bernstein control-point basis
    ///
    /// Identical to element_coefficients() for the Bezier smoothers. The Hermite
    /// smoother overrides this with the change of basis c_e = M_e q_e, so that
    /// consumers requiring Bernstein control values (SeabedSurface, the control
    /// point VTK writer) keep working. See docs/hermite_bathymetry_system.md S5.
    virtual VecX element_bernstein_coefficients(Index elem) const;

protected:
    // =========================================================================
    // Shared state
    // =========================================================================

    std::unique_ptr<QuadtreeAdapter> quadtree_owned_;
    const QuadtreeAdapter* quadtree_ = nullptr;

    VecX solution_;
    bool solved_ = false;
    bool data_set_ = false;

    SpMat H_global_;       ///< Smoothness hessian (Dirichlet or thin plate)
    SpMat BtWB_global_;    ///< Data fitting matrix
    VecX BtWd_global_;     ///< Data fitting RHS
    Real dTWd_global_ = 0; ///< Data norm for residual computation
    Real alpha_ = 0;       ///< Scale normalization factor (norm_BtWB / norm_H)

    /// Where a measurement exists; empty = data everywhere
    std::function<bool(Real, Real)> has_data_func_;

    /// Where the surface is held at depth 0 by a Dirichlet condition; empty = nowhere
    std::function<bool(Real, Real)> is_land_func_;

    /// Quadrature points dropped by the last assemble_data_fitting_global()
    Index num_excluded_quad_points_ = 0;

    /// Per-element classification; null until a derived smoother builds one
    std::shared_ptr<const ElementDataMask> element_mask_;

    /// @brief Classify the mesh from the current data masks
    ///
    /// Called by derived smoothers that act on the classification, after the mesh
    /// and the masks are both known. Leaves element_mask_ null when there are no
    /// masks, so the analytic path allocates nothing.
    void build_element_mask();

    /// @brief Whether an element is dropped from assembly
    ///
    /// Every non-water element, Beach as well as Inland: the DOF manager pins all
    /// of their DOFs, so assembling them would only build rows that condensation
    /// removes again. Beach and Inland differ in output, not in the solve - a Beach
    /// element is drawn as the flat zero it is, an Inland element is not drawn.
    bool is_element_excluded(Index elem) const {
        return element_mask_ && element_mask_->is_pinned(elem);
    }

    /// @brief Whether (x, y) is land, and so pinned to depth 0
    bool is_land(Real x, Real y) const { return is_land_func_ && is_land_func_(x, y); }

    /// @brief Whether (x, y) contributes no least-squares observation
    ///
    /// Both a gap (nothing measured) and land (known, imposed strongly elsewhere).
    bool is_excluded_from_fit(Real x, Real y) const {
        return (has_data_func_ && !has_data_func_(x, y)) || is_land(x, y);
    }

    /// External cache for element matrices (owned by adaptive smoother)
    /// If set, element matrices are stored during assembly for multigrid reuse
    std::map<std::tuple<uint64_t, int, int>, MatX>* element_matrix_cache_ = nullptr;

    /// Temporary storage for element matrices during assembly
    /// Populated in assemble_hessian_global(), completed in assemble_data_fitting_global()
    std::vector<MatX> element_matrix_cache_temp_;

    /// Boundary relaxation zone configuration
    BoundaryRelaxationConfig relaxation_config_;

    // =========================================================================
    // Pure virtual methods - must be implemented by derived classes
    // =========================================================================

    /// @brief Assemble hessian and data fitting matrices
    /// @param bathy_func Bathymetry function (x, y) -> depth
    virtual void set_bathymetry_data_impl(std::function<Real(Real, Real)> bathy_func) = 0;

    /// @brief Evaluate Bezier surface at parametric coordinates
    /// @param coeffs Control point values
    /// @param u, v Parametric coordinates in [0, 1]
    /// @return Surface value
    Real evaluate_scalar(const VecX &coeffs, Real u, Real v) const;

    /// @brief Evaluate gradient in parametric coordinates
    /// @param coeffs Control point values
    /// @param u, v Parametric coordinates in [0, 1]
    /// @return Gradient (dz/du, dz/dv) in parametric space
    Vec2 evaluate_gradient_uv(const VecX &coeffs, Real u, Real v) const;

    /// @brief Get number of global DOFs from derived class DOF manager
    virtual Index dof_manager_num_global_dofs() const = 0;

    /// @brief Get number of free DOFs from derived class DOF manager
    virtual Index dof_manager_num_free_dofs() const = 0;

    /// @brief Get number of constraints from derived class DOF manager
    virtual Index dof_manager_num_constraints() const = 0;

    /// @brief Get global DOF indices for an element
    /// @param elem Element index
    /// @return Reference to vector of global DOF indices
    virtual const std::vector<Index> &element_global_dofs(Index elem) const = 0;

    /// @brief Get reference to the basis object
    /// @return Reference to Basis2DBase (LinearBezierBasis2D or CubicBezierBasis2D)
    virtual const Basis2DBase &basis() const = 0;

    /// @brief Get number of Gauss points for data fitting
    virtual int ngauss_data() const = 0;

    /// @brief Get smoothing weight (lambda) from config
    virtual Real lambda() const = 0;

    /// @brief Get ridge regularization parameter from config
    virtual Real ridge_epsilon() const = 0;

    // =========================================================================
    // Basis-dependent hooks - defaulted so Bernstein bases need not override
    // =========================================================================

    /// @brief Per-element diagonal DOF scaling (Lambda_e)
    ///
    /// A Bernstein basis is element-size independent, so the default is all ones
    /// and the Bezier path is unaffected. A Hermite basis whose DOFs are *physical*
    /// derivatives is not: its shape functions are
    /// N_I(u,v) = Nhat_I(u,v) * h_x^a * h_y^b, and this hook supplies the diagonal
    /// h_x^a h_y^b factors. See docs/hermite_bathymetry_system.md S4.
    ///
    /// @param dx, dy Element dimensions
    /// @return Diagonal scaling of length basis().num_dofs()
    virtual VecX element_dof_scaling(Real dx, Real dy) const;

    /// @brief Diagonal of the ridge term added to Q
    ///
    /// Default is a uniform lambda*epsilon on every DOF. The Hermite smoother
    /// overrides this because its DOFs carry different physical units, which makes
    /// a uniform ridge dimensionally inconsistent - it penalises z and z_xy with
    /// the same weight. See docs/hermite_bathymetry_system.md S11.
    ///
    /// @return Ridge diagonal of length dof_manager_num_global_dofs()
    virtual VecX ridge_diagonal() const;

    // =========================================================================
    // Helper methods - implemented in base
    // =========================================================================

    /// @brief Assemble global smoothness hessian matrix H_global_
    ///
    /// Uses the hessian's scaled_hessian() method and element_global_dofs()
    /// to build the sparse global hessian matrix.
    ///
    /// @param hessian The hessian object (DirichletHessian or CubicThinPlateHessian)
    void assemble_hessian_global(const HessianBase &hessian);

    /// @brief Assemble data fitting matrices BtWB_global_, BtWd_global_, dTWd_global_
    ///
    /// Uses Gauss quadrature to integrate basis functions against bathymetry data.
    /// Uses basis() and ngauss_data() from derived class.
    ///
    /// @param bathy_func Bathymetry function (x, y) -> depth
    void assemble_data_fitting_global(std::function<Real(Real, Real)> bathy_func);

    /// @brief Assemble the system matrix Q = alpha*H + lambda*(BtWB + epsilon*I)
    /// @return Sparse Q matrix of size (num_global_dofs × num_global_dofs)
    SpMat assemble_Q() const;

    /// @brief Assemble the RHS vector b = lambda * BtWd
    /// @return Vector b of size num_global_dofs
    VecX assemble_b() const;

    /// @brief Solve unconstrained system using SparseLU
    ///
    /// Uses assemble_Q() and assemble_b(), then solves Qx = b
    void solve_unconstrained();

    /// @brief Compute Gauss-Legendre quadrature points and weights on [0, 1]
    /// @param n Number of quadrature points (1-4)
    /// @param pts Output quadrature points
    /// @param wts Output quadrature weights
    static void gauss_legendre_01(int n, std::vector<Real> &pts, std::vector<Real> &wts);

    /// @brief Find element containing point
    /// @return Element index, or -1 if not found
    Index find_element(Real x, Real y) const;

    /// @brief Find element containing point, with fallback to closest element
    /// @return Element index (always valid for points near the domain)
    Index find_element_with_fallback(Real x, Real y) const;

public:
    /// @brief Evaluate in a specific element (skips element lookup)
    ///
    /// Parametric coordinates are clamped to [0,1], so evaluating exactly on a
    /// shared edge yields that element's one-sided trace. This is how interface
    /// continuity is measured without probing at two nearby points, which would
    /// pick up the surface's own variation across the gap.
    ///
    /// @param elem Element index
    /// @param x, y World coordinates within element
    /// @return Surface value at (x, y)
    Real evaluate_in_element(Index elem, Real x, Real y) const;

    /// @brief Evaluate gradient in a specific element
    /// @see evaluate_in_element for the one-sided trace semantics
    Vec2 evaluate_gradient_in_element(Index elem, Real x, Real y) const;

protected:

    /// @brief Store element matrix in external cache
    /// @param elem Element index
    /// @param Q_local Element matrix to cache
    void cache_element_matrix(Index elem, const MatX &Q_local);

    /// @brief Compute boundary relaxation factor at physical point
    ///
    /// Returns a factor in [min_factor, 1.0] based on distance to enabled boundaries.
    /// Uses smoothstep interpolation for C1 continuity.
    ///
    /// @param x, y Physical coordinates
    /// @return Factor where 1.0 = full data fitting, min_factor = relaxed (at boundary)
    Real compute_relaxation_factor(Real x, Real y) const;
};

} // namespace drifter
