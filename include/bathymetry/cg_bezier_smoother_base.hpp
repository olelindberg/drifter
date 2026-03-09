#pragma once

/// @file cg_bezier_smoother_base.hpp
/// @brief Abstract base class for CG Bezier bathymetry smoothers
///
/// Provides common functionality shared between CGLinearBezierBathymetrySmoother
/// and CGCubicBezierBathymetrySmoother. Uses standard inheritance with virtual
/// methods for customization points. Virtual dispatch overhead is acceptable
/// since these methods are not in performance-critical inner loops.

#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include "mesh/seabed_surface.hpp"
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
class BezierHessianBase;
class BezierBasis2DBase;

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
class CGBezierSmootherBase {
public:
    virtual ~CGBezierSmootherBase() = default;

    // =========================================================================
    // Data input - implemented in base
    // =========================================================================

    /// @brief Set bathymetry from BathymetrySource (e.g., GeoTIFF)
    void set_bathymetry_data(const BathymetrySource &source);

    /// @brief Set bathymetry from function
    /// @note Calls derived class set_bathymetry_data_impl()
    void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);

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

    /// @brief Get element control point values
    /// @return Vector of DOF values for this element
    /// @note Public so adaptive smoothers can access coefficients
    VecX element_coefficients(Index elem) const;

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

    /// External cache for element matrices (owned by adaptive smoother)
    /// If set, element matrices are stored during assembly for multigrid reuse
    std::map<std::tuple<uint64_t, int, int>, MatX>* element_matrix_cache_ = nullptr;

    /// Temporary storage for element matrices during assembly
    /// Populated in assemble_hessian_global(), completed in assemble_data_fitting_global()
    std::vector<MatX> element_matrix_cache_temp_;

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
    /// @return Reference to BezierBasis2DBase (LinearBezierBasis2D or CubicBezierBasis2D)
    virtual const BezierBasis2DBase &basis() const = 0;

    /// @brief Get number of Gauss points for data fitting
    virtual int ngauss_data() const = 0;

    /// @brief Get smoothing weight (lambda) from config
    virtual Real lambda() const = 0;

    /// @brief Get ridge regularization parameter from config
    virtual Real ridge_epsilon() const = 0;

    // =========================================================================
    // Helper methods - implemented in base
    // =========================================================================

    /// @brief Assemble global smoothness hessian matrix H_global_
    ///
    /// Uses the hessian's scaled_hessian() method and element_global_dofs()
    /// to build the sparse global hessian matrix.
    ///
    /// @param hessian The hessian object (DirichletHessian or CubicThinPlateHessian)
    void assemble_hessian_global(const BezierHessianBase &hessian);

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

    /// @brief Evaluate in a specific element
    Real evaluate_in_element(Index elem, Real x, Real y) const;

    /// @brief Evaluate gradient in a specific element
    Vec2 evaluate_gradient_in_element(Index elem, Real x, Real y) const;

    /// @brief Store element matrix in external cache
    /// @param elem Element index
    /// @param Q_local Element matrix to cache
    void cache_element_matrix(Index elem, const MatX &Q_local);
};

} // namespace drifter
