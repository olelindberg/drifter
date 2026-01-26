#pragma once

#include "bathymetry/biharmonic_assembler.hpp"  // For BathymetrySource
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/thb_spline/thb_data_fitting.hpp"
#include "bathymetry/thb_spline/thb_hierarchy.hpp"
#include "bathymetry/thb_spline/thb_refinement_mask.hpp"
#include "bathymetry/thb_spline/thb_truncation.hpp"
#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"
#include <functional>
#include <memory>
#include <string>

namespace drifter {

/**
 * @brief Configuration for THB-spline surface fitting
 */
struct THBSurfaceConfig {
    /// Smoothing weight (optional thin-plate energy)
    Real smoothing_weight = 0.0;

    /// Number of Gauss points per direction for data sampling
    int ngauss = 4;

    /// Enable verbose logging
    bool verbose = false;
};

/**
 * @brief THB-spline surface for adaptive multiresolution bathymetry
 *
 * Fits a single C² surface using Truncated Hierarchical B-splines.
 * The surface automatically adapts resolution to match the underlying
 * octree/quadtree mesh - coarse regions use coarse basis functions
 * (naturally suppressing high-frequency content), fine regions use
 * fine basis functions.
 *
 * Key properties:
 * - Single coherent C² surface (bicubic B-splines are analytically C²)
 * - No constraint system needed (unlike Bezier approach)
 * - Partition of unity maintained by truncation mechanism
 * - Resolution naturally matches octree element sizes
 *
 * Usage:
 * @code
 *   THBSurface surface(octree);
 *   surface.set_bathymetry_data(source);
 *   surface.solve();
 *   Real depth = surface.evaluate(x, y);
 * @endcode
 */
class THBSurface {
  public:
    /**
     * @brief Construct from octree (uses bottom face as 2D domain)
     * @param octree 3D adaptive mesh
     * @param config Surface configuration
     */
    THBSurface(const OctreeAdapter& octree, const THBSurfaceConfig& config = {});

    /**
     * @brief Construct from quadtree
     * @param quadtree 2D adaptive mesh
     * @param config Surface configuration
     */
    THBSurface(const QuadtreeAdapter& quadtree, const THBSurfaceConfig& config = {});

    /// Destructor
    ~THBSurface();

    // =========================================================================
    // Data input
    // =========================================================================

    /**
     * @brief Set bathymetry data from source
     * @param source Bathymetry data source (e.g., GeoTIFF)
     */
    void set_bathymetry_data(const BathymetrySource& source);

    /**
     * @brief Set bathymetry data from function
     * @param bathy_func Function taking (x, y) -> depth
     */
    void set_bathymetry_data(std::function<Real(Real, Real)> bathy_func);

    // =========================================================================
    // Solving
    // =========================================================================

    /**
     * @brief Solve the least-squares fitting problem
     *
     * Builds and solves the system:
     *   (B^T W B + λ L) c = B^T W d
     * where:
     *   B = truncated basis evaluation matrix
     *   W = weight matrix
     *   L = smoothing matrix (optional)
     *   c = control point coefficients
     *   d = bathymetry data values
     */
    void solve();

    /// Check if solution is available
    bool is_solved() const { return solved_; }

    // =========================================================================
    // Evaluation
    // =========================================================================

    /**
     * @brief Evaluate surface at physical coordinates
     * @param x Physical x-coordinate
     * @param y Physical y-coordinate
     * @return Surface value (depth/elevation)
     */
    Real evaluate(Real x, Real y) const;

    /**
     * @brief Evaluate surface gradient at physical coordinates
     * @param x Physical x-coordinate
     * @param y Physical y-coordinate
     * @return Gradient (dz/dx, dz/dy)
     */
    Vec2 evaluate_gradient(Real x, Real y) const;

    // =========================================================================
    // Access
    // =========================================================================

    /// Access THB hierarchy
    const THBHierarchy& hierarchy() const { return *hierarchy_; }

    /// Access refinement mask
    const THBRefinementMask& mask() const { return *mask_; }

    /// Access truncation data
    const THBTruncation& truncation() const { return *truncation_; }

    /// Access active DOF coefficients (after solve)
    const VecX& active_coefficients() const { return active_coeffs_; }

    /// Number of active DOFs
    Index num_active_dofs() const { return mask_->num_active(); }

    /// Maximum refinement level in hierarchy
    int max_level() const { return hierarchy_->max_level(); }

    /// Domain bounds
    Real domain_min_x() const { return hierarchy_->domain_min_u(); }
    Real domain_max_x() const { return hierarchy_->domain_max_u(); }
    Real domain_min_y() const { return hierarchy_->domain_min_v(); }
    Real domain_max_y() const { return hierarchy_->domain_max_v(); }

    // =========================================================================
    // Output
    // =========================================================================

    /**
     * @brief Write VTK output for visualization
     * @param filename Base filename (without extension)
     * @param resolution Points per element per direction
     */
    void write_vtk(const std::string& filename, int resolution = 10) const;

  private:
    THBSurfaceConfig config_;

    // Internal quadtree (owned if constructed from octree)
    std::unique_ptr<QuadtreeAdapter> owned_quadtree_;
    const QuadtreeAdapter* quadtree_ = nullptr;

    // THB components
    std::unique_ptr<THBHierarchy> hierarchy_;
    std::unique_ptr<THBRefinementMask> mask_;
    std::unique_ptr<THBTruncation> truncation_;
    std::unique_ptr<THBDataFitting> data_fitting_;

    // Solution
    VecX active_coeffs_;
    bool data_set_ = false;
    bool solved_ = false;

    /// Initialize THB components from quadtree
    void initialize_from_quadtree(const QuadtreeAdapter& quadtree);
};

}  // namespace drifter
