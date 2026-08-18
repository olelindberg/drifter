#pragma once

/// @file linear_bezier_surface.hpp
/// @brief Bilinear surface interpolation with 4 corner DOFs per element

#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include "mesh/geotiff_reader.hpp"
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

    /// @brief Evaluate surface at a point
    /// @param x, y World coordinates
    /// @return Surface height at (x, y)
    Real evaluate(Real x, Real y) const;

    /// @brief Get corner coefficients for an element
    /// @param elem Element index
    /// @return 4-vector of corner heights: [z00, z10, z01, z11]
    Eigen::Vector4d element_coefficients(Index elem) const;

    /// @brief Get the quadtree mesh
    const QuadtreeAdapter& mesh() const { return mesh_; }

    /// @brief Get total number of global DOFs
    Index num_dofs() const { return num_global_dofs_; }

    /// @brief Check if surface has been fitted
    bool is_fitted() const { return is_fitted_; }

    /// @brief Evaluate bilinear basis at reference coordinates
    /// @param xi, eta Reference coordinates in [0,1]
    /// @return 4-vector of basis values: [N00, N10, N01, N11]
    static Eigen::Vector4d basis(Real xi, Real eta);

private:
    const QuadtreeAdapter& mesh_;
    VecX coefficients_;                          ///< Global DOF vector
    std::vector<std::array<Index, 4>> dof_map_;  ///< Element corners -> global DOF indices
    Index num_global_dofs_ = 0;
    bool is_fitted_ = false;

    /// @brief Build DOF connectivity (shared corners)
    void build_dof_map();

    /// @brief Map physical coordinates to reference element
    /// @param x, y World coordinates
    /// @param elem Element index
    /// @param xi, eta Output reference coordinates in [0,1]
    void world_to_reference(Real x, Real y, Index elem, Real& xi, Real& eta) const;
};

} // namespace drifter
