#pragma once

// SeabedSurface - 2D bathymetry representation for bottom-layer octree elements
//
// Stores per-element Bernstein coefficients for the seabed depth field,
// connected to the 3D octree mesh. Handles non-conforming projection
// automatically to ensure continuity across refined interfaces.
//
// Only bottom-layer elements (where zmin is at the seabed) store coefficients.

#include "core/types.hpp"
#include "dg/bernstein_basis.hpp"
#include "mesh/geotiff_reader.hpp"
#include "mesh/octree_adapter.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace drifter {

/// @brief 2D seabed surface representation for bottom-layer octree elements
///
/// Stores bathymetry depth as Bernstein polynomial coefficients on each
/// bottom-layer element's bottom face. Provides:
/// - Point evaluation at any (x, y) location
/// - Gradient computation
/// - VTK output for visualization
/// - Dynamic updates for AMR (refinement/coarsening)
class SeabedSurface {
public:
    /// @brief Construct seabed surface connected to an octree mesh
    /// @param mesh The octree mesh (must outlive this object)
    /// @param order Polynomial order for Bernstein representation
    /// @param method Interpolation method (Bernstein recommended for
    /// boundedness)
    SeabedSurface(const OctreeAdapter &mesh, int order,
                  SeabedInterpolation method = SeabedInterpolation::Bernstein);

    /// @brief Initialize from raw bathymetry data
    /// Samples bathymetry at LGL nodes on each bottom element's bottom face,
    /// converts to Bernstein coefficients, and applies non-conforming
    /// projection.
    /// @param bathy Bathymetry data (e.g., from GeoTIFF)
    void set_from_bathymetry(const BathymetryData &bathy);

    /// @brief Initialize from bathymetry with scale-dependent box filter
    /// smoothing Samples bathymetry at LGL nodes using local averaging with no
    /// pixel cap, allowing large smoothing radii for coarse elements.
    /// @param bathy Bathymetry data (e.g., from GeoTIFF)
    /// @param smoothing_factor Filter radius = factor * min(element_size), e.g.
    /// 0.5
    void set_from_bathymetry_smoothed(const BathymetryData &bathy, Real smoothing_factor);

    /// @brief Set coefficients directly for a specific element
    /// @param seabed_elem_idx Index in bottom_elements_ (not mesh element
    /// index)
    /// @param coeffs Bernstein coefficients (order+1)^2 values
    void set_element_coefficients(size_t seabed_elem_idx, const VecX &coeffs);

    /// @brief Apply non-conforming projection to ensure interface continuity
    /// Called automatically by set_from_bathymetry(), but can be called
    /// manually after set_element_coefficients() if needed.
    /// For Lagrange basis: directly overwrites interface DOFs.
    /// For Bernstein basis: uses constrained L2 re-projection.
    void apply_nonconforming_projection();

    /// @brief Apply non-conforming projection for Bernstein basis
    /// Uses constrained least-squares to match coarse element at interface
    /// while minimizing change to the polynomial in the interior.
    void apply_bernstein_nonconforming_projection();

    /// @brief Evaluate depth at a point
    /// @param x, y World coordinates
    /// @return Bathymetry depth (positive downward), or 0 if point not found
    Real depth(Real x, Real y) const;

    /// @brief Evaluate depth gradient at a point
    /// @param x, y World coordinates
    /// @param dh_dx, dh_dy Output: gradient components
    /// @return true if point found, false otherwise
    bool gradient(Real x, Real y, Real &dh_dx, Real &dh_dy) const;

    /// @brief Write seabed surface to VTK file
    /// @param filename Output path (without extension)
    /// @param resolution Subdivisions per element face (default: 10)
    void write_vtk(const std::string &filename, int resolution = 10) const;

    // =========================================================================
    // AMR Dynamic Updates
    // =========================================================================

    /// @brief Called when a bottom-layer element is refined
    /// Interpolates parent's coefficients to children.
    /// @param parent_mesh_idx Mesh index of parent element
    /// @param child_mesh_indices Mesh indices of new child elements
    void on_refine(Index parent_mesh_idx, const std::vector<Index> &child_mesh_indices);

    /// @brief Called when bottom-layer elements are coarsened
    /// Projects children's coefficients to new parent.
    /// @param child_mesh_indices Mesh indices of child elements being removed
    /// @param new_parent_mesh_idx Mesh index of new parent element
    void on_coarsen(const std::vector<Index> &child_mesh_indices, Index new_parent_mesh_idx);

    /// @brief Rebuild seabed surface from current mesh state
    /// Use after major mesh changes or when callbacks weren't used.
    /// Requires re-setting bathymetry data afterward.
    void rebuild_from_mesh();

    // =========================================================================
    // Accessors
    // =========================================================================

    /// @brief Get the connected mesh
    const OctreeAdapter &mesh() const { return *mesh_; }

    /// @brief Get polynomial order
    int order() const { return order_; }

    /// @brief Get interpolation method
    SeabedInterpolation method() const { return method_; }

    /// @brief Get number of bottom-layer elements
    size_t num_elements() const { return bottom_elements_.size(); }

    /// @brief Get mesh element index for a seabed element
    Index mesh_element_index(size_t seabed_idx) const { return bottom_elements_[seabed_idx]; }

    /// @brief Get seabed element index for a mesh element (-1 if not bottom
    /// layer)
    Index seabed_element_index(Index mesh_idx) const;

    /// @brief Check if a mesh element is in the bottom layer
    bool is_bottom_element(Index mesh_idx) const;

    /// @brief Get depth coefficients for a seabed element
    const VecX &coefficients(size_t seabed_idx) const { return depth_coeffs_[seabed_idx]; }

    /// @brief Get all depth coefficients (for VTK output, etc.)
    const std::vector<VecX> &all_coefficients() const { return depth_coeffs_; }

    /// @brief Get coordinates for a seabed element (3*(order+1)^2 interleaved
    /// x,y,z)
    const VecX &coordinates(size_t seabed_idx) const { return coordinates_[seabed_idx]; }

private:
    const OctreeAdapter* mesh_;
    int order_;
    SeabedInterpolation method_;

    // Bottom-layer element tracking
    std::vector<Index> bottom_elements_; // Seabed idx -> mesh element idx
    std::unordered_map<Index, size_t> mesh_to_seabed_; // Mesh idx -> seabed idx

    // Per-element data (indexed by seabed element index)
    std::vector<VecX> depth_coeffs_; // (order+1)^2 Bernstein coefficients
    std::vector<VecX> coordinates_; // 3*(order+1)^2 interleaved (x,y,z) at DOFs

    // Lazy-initialized interpolator
    mutable std::unique_ptr<SeabedInterpolator> interpolator_;

    // Minimum z value in mesh (to identify bottom layer)
    Real mesh_zmin_;

    // Internal helpers
    void identify_bottom_elements();
    void allocate_storage();
    const SeabedInterpolator &get_interpolator() const;

    // Update z-coordinates from depth coefficients (call after projection)
    void update_coordinates_from_coefficients();

    // Find which seabed element contains point (x, y)
    // Returns -1 if not found
    Index find_seabed_element(Real x, Real y) const;

    // Transform world coords to reference coords for an element
    void world_to_reference(size_t seabed_idx, Real x, Real y, Real &xi, Real &eta) const;

    // Sample bathymetry with box filter averaging (no pixel cap)
    Real sample_smoothed(const BathymetryData &bathy, Real x, Real y, Real filter_radius) const;
};

} // namespace drifter
