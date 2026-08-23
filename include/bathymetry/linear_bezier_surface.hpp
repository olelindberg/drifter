#pragma once

/// @file linear_bezier_surface.hpp
/// @brief Bilinear surface interpolation with 4 corner DOFs per element

#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include "mesh/geotiff_reader.hpp"
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace drifter {

/// @brief Bilinear surface with 4 corner DOFs per quadrilateral element
///
/// DOFs at vertices: (0,0), (1,0), (0,1), (1,1) in reference space
/// Basis functions:
///   N_00 = (1-xi)(1-eta)
///   N_10 = xi(1-eta)
///   N_01 = (1-xi)eta
///   N_11 = xi*eta
///
/// DOFs are shared at edges for C0 continuity (continuous Galerkin).
class LinearBezierSurface {
public:
    /// @brief Construct from quadtree mesh
    explicit LinearBezierSurface(const QuadtreeAdapter& mesh);

    /// @brief Fit surface to bathymetry data
    /// @param data Bathymetry data from GeoTIFF
    /// @note Simply samples the bathymetry at corner positions
    void fit(const BathymetryData& data);

    /// @brief Update mesh reference after refinement
    /// @param mesh New mesh reference
    /// @return Index of first new DOF (DOFs >= this are new and need fitting)
    Index update_mesh(const QuadtreeAdapter& mesh);

    /// @brief Incrementally fit only new elements
    /// @param data Bathymetry data from GeoTIFF
    /// @param new_elements Indices of newly created elements
    /// @param first_new_dof Index of first new DOF (from update_mesh return value)
    /// @note Reuses existing DOF values, only samples new DOF positions
    void fit_incremental(const BathymetryData& data, const std::vector<Index>& new_elements,
                         Index first_new_dof);

    /// @brief Fit surface using a depth function (supports multi-source bathymetry)
    /// @param depth_func Function that returns depth at (x, y) coordinates
    void fit(std::function<Real(Real, Real)> depth_func);

    /// @brief Incrementally fit only new elements using depth function
    /// @param depth_func Function that returns depth at (x, y) coordinates
    /// @param new_elements Indices of newly created elements
    /// @param first_new_dof Index of first new DOF (from update_mesh return value)
    void fit_incremental(std::function<Real(Real, Real)> depth_func,
                         const std::vector<Index>& new_elements, Index first_new_dof);

    /// @brief Evaluate surface at a point
    /// @param x, y World coordinates
    /// @return Surface height at (x, y)
    Real evaluate(Real x, Real y) const;

    /// @brief Evaluate surface at a point within a known element (skips element lookup)
    /// @param elem Known containing element index
    /// @param x, y World coordinates
    /// @return Surface height at (x, y)
    /// @note Caller must ensure (x, y) is within element bounds
    Real evaluate_in_element(Index elem, Real x, Real y) const;

    /// @brief Get corner coefficients for an element
    /// @param elem Element index
    /// @return 4-vector of corner heights: [z00, z10, z01, z11]
    Eigen::Vector4d element_coefficients(Index elem) const;

    /// @brief Get the quadtree mesh
    const QuadtreeAdapter& mesh() const { return *mesh_; }

    /// @brief Get total number of global DOFs
    Index num_dofs() const { return num_global_dofs_; }

    /// @brief Check if surface has been fitted
    bool is_fitted() const { return is_fitted_; }

    /// @brief Evaluate bilinear basis at reference coordinates
    /// @param xi, eta Reference coordinates in [0,1]
    /// @return 4-vector of basis values: [N00, N10, N01, N11]
    static Eigen::Vector4d basis(Real xi, Real eta);

private:
    const QuadtreeAdapter* mesh_;                ///< Pointer to allow update
    VecX coefficients_;                          ///< Global DOF vector
    std::vector<std::array<Index, 4>> dof_map_;  ///< Element corners -> global DOF indices
    Index num_global_dofs_ = 0;
    bool is_fitted_ = false;

    /// Persistent corner position to DOF index map (for incremental fitting)
    std::map<std::pair<int64_t, int64_t>, Index> corner_to_dof_;

    /// @brief Build DOF connectivity (shared corners)
    void build_dof_map();

    /// @brief Map physical coordinates to reference element
    /// @param x, y World coordinates
    /// @param elem Element index
    /// @param xi, eta Output reference coordinates in [0,1]
    void world_to_reference(Real x, Real y, Index elem, Real& xi, Real& eta) const;
};

} // namespace drifter
