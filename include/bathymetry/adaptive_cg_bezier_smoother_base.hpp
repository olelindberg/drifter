#pragma once

/// @file adaptive_cg_bezier_smoother_base.hpp
/// @brief Abstract base class for adaptive CG Bezier bathymetry smoothers
///
/// Provides common functionality shared between AdaptiveCGLinearBezierSmoother
/// and AdaptiveCGCubicBezierSmoother. Uses standard inheritance with virtual
/// methods for customization points.

#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"
#include "mesh/refine_mask.hpp"
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace drifter {

// Forward declarations
class BathymetrySource;
class BezierBasis2DBase;

/// @brief Abstract base class for adaptive CG Bezier bathymetry smoothers
///
/// Implements common functionality for both linear and cubic adaptive smoothers:
/// - Gauss quadrature initialization
/// - Bathymetry data management
/// - Land mask handling
/// - Mesh refinement
///
/// Derived classes must implement:
/// - is_solved() - check if smoother has solution
/// - smoother_evaluate() - evaluate at a point
/// - rebuild_smoother() - recreate smoother for current mesh
class AdaptiveCGBezierSmootherBase {
public:
    virtual ~AdaptiveCGBezierSmootherBase() = default;

    // =========================================================================
    // Data input - implemented in base
    // =========================================================================

    /// @brief Set bathymetry from BathymetrySource (e.g., GeoTIFF)
    /// @param source Bathymetry data source
    /// @note The source must outlive this smoother
    void set_bathymetry_data(const BathymetrySource &source);

    /// @brief Set bathymetry from function
    /// @param bathy_func Function (x, y) -> depth
    void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);

    /// @brief Set optional land mask function
    /// @param is_land_func Returns true if (x,y) is on land
    /// Elements entirely on land will not be refined
    void set_land_mask(std::function<bool(Real, Real)> is_land_func);

    // =========================================================================
    // Evaluation - implemented in base
    // =========================================================================

    /// @brief Evaluate smoothed bathymetry at point
    /// @param x, y Physical coordinates
    /// @return Smoothed depth at (x, y)
    /// @throws std::runtime_error if not solved or point outside domain
    Real evaluate(Real x, Real y) const;

    // =========================================================================
    // Accessors - implemented in base
    // =========================================================================

    /// @brief Get current 2D mesh
    const QuadtreeAdapter &mesh() const { return *quadtree_; }

    /// @brief Get current octree (underlying 3D mesh)
    const OctreeAdapter &octree() const { return *octree_; }

    /// @brief Check if bathymetry data has been set
    bool has_bathymetry_data() const { return static_cast<bool>(bathy_func_); }

protected:
    // =========================================================================
    // Shared state
    // =========================================================================

    /// Owned octree (if constructed from bounds)
    std::unique_ptr<OctreeAdapter> octree_owned_;

    /// Pointer to active octree (owned or external reference)
    OctreeAdapter* octree_ = nullptr;

    /// 2D mesh extracted from octree
    std::unique_ptr<QuadtreeAdapter> quadtree_;

    /// Bathymetry function
    std::function<Real(Real, Real)> bathy_func_;

    /// Optional land mask function
    std::function<bool(Real, Real)> land_mask_func_;

    /// Gauss-Legendre quadrature nodes on [0, 1]
    VecX gauss_nodes_;

    /// Gauss-Legendre quadrature weights on [0, 1]
    VecX gauss_weights_;

    /// Previous solutions keyed by (Morton code, level_x, level_y)
    /// Morton alone doesn't uniquely identify an element - same Morton can exist
    /// at different levels with different positions/sizes.
    std::map<std::tuple<uint64_t, int, int>, VecX> prev_solutions_;

    // =========================================================================
    // Pure virtual methods - must be implemented by derived classes
    // =========================================================================

    /// @brief Check if smoother has solution
    virtual bool is_solved_impl() const = 0;

    /// @brief Evaluate smoother at a point (assumes solved)
    virtual Real smoother_evaluate(Real x, Real y) const = 0;

    /// @brief Recreate smoother for current mesh
    virtual void rebuild_smoother() = 0;

    // =========================================================================
    // Helper methods - implemented in base
    // =========================================================================

    /// @brief Initialize Gauss-Legendre quadrature nodes and weights
    /// @param ngauss Number of quadrature points (1-6)
    /// @throws std::invalid_argument if ngauss is out of range
    void init_gauss_quadrature(int ngauss);

    /// @brief Refine marked elements and update mesh
    /// @param elements_to_refine Element indices to refine
    void refine_elements_impl(const std::vector<Index> &elements_to_refine);

    /// @brief Apply bathymetry data to current smoother
    /// @throws std::runtime_error if smoother not initialized or no bathymetry data
    virtual void apply_bathymetry_to_smoother() = 0;

    /// @brief Get element coefficients from the current smoother
    /// @param elem Element index
    /// @return Vector of control point values for this element
    virtual VecX get_element_coefficients_impl(Index elem) const = 0;

    /// @brief Get reference to the basis object for evaluating stored solutions
    /// @return Reference to BezierBasis2DBase (LinearBezierBasis2D or CubicBezierBasis2D)
    virtual const BezierBasis2DBase &get_basis_impl() const = 0;

    // =========================================================================
    // Coarsening metrics - implemented in base
    // =========================================================================

    /// @brief Store current solution coefficients keyed by Morton code
    /// Must be called before refinement to enable coarsening metrics computation
    void store_current_solution();

    /// @brief Evaluate previous solution at a point using stored coefficients
    /// @param x, y Physical coordinates
    /// @return Previous solution depth at (x, y), or 0 if not found
    Real evaluate_prev_solution(Real x, Real y) const;

    /// @brief Compute coarsening error metrics for an element
    /// @param elem Element index
    /// @param[out] mean_difference mean |z_fine - z_coarse| over element [m]
    /// @param[out] volume_change integral |z_fine - z_coarse| dA [m^3]
    void compute_coarsening_metrics(Index elem, Real &mean_difference,
                                    Real &volume_change) const;
};

} // namespace drifter
