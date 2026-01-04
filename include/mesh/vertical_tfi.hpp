#pragma once

// Vertical Transfinite Interpolation (TFI) for terrain-following coordinates
// Handles mesh motion as free surface moves and provides Jacobian updates.
//
// In sigma coordinates, the physical z position is:
//   z(x, y, sigma, t) = eta(x, y, t) + sigma * H(x, y, t)
// where H = eta + h is the total water depth.
//
// This class manages:
// 1. Computing z positions from sigma and water depth fields
// 2. Computing mesh velocity w_mesh = dz/dt at fixed sigma
// 3. Updating geometric factors (Jacobians) after mesh motion
//
// Adapted from wobbler's TFI2D pattern for 2D curved elements.

#include "core/types.hpp"
#include "mesh/sigma_coordinate.hpp"
#include "dg/basis_hexahedron.hpp"
#include <vector>

namespace drifter {

// Forward declarations
struct GeometricFactors;

/// @brief Vertical mesh motion via transfinite interpolation
/// @details Manages terrain-following coordinate mesh updates
class VerticalTFI {
public:
    /// @brief Construct TFI manager
    /// @param basis Hexahedron basis for DOF locations
    /// @param stretch_type Vertical stretching type
    /// @param stretch_params Stretching parameters
    VerticalTFI(const HexahedronBasis& basis,
                SigmaStretchType stretch_type = SigmaStretchType::Uniform,
                const SigmaStretchParams& stretch_params = SigmaStretchParams());

    /// @brief Get the basis
    const HexahedronBasis& basis() const { return basis_; }

    /// @brief Get number of vertical levels (LGL nodes)
    int num_vertical_levels() const { return n_vert_; }

    /// @brief Get number of horizontal nodes per level
    int num_horizontal_nodes() const { return n_horiz_; }

    /// @brief Get sigma values at vertical LGL nodes
    const VecX& sigma_at_lgl() const { return sigma_at_lgl_; }

    // =========================================================================
    // Node position updates
    // =========================================================================

    /// @brief Update vertical node positions when free surface moves
    /// @details z(x,y,sigma) = eta(x,y) + sigma * H(x,y) where H = eta + h
    /// @param eta Free surface elevation at horizontal nodes (n_horiz)
    /// @param h Bathymetry at horizontal nodes (n_horiz, positive downward)
    /// @param[in,out] nodes Physical node positions (3D array, modified in place)
    void update_node_positions(const VecX& eta, const VecX& h,
                               std::vector<Vec3>& nodes) const;

    /// @brief Update only z-coordinates of nodes (x,y unchanged)
    /// @param eta Free surface at horizontal nodes
    /// @param h Bathymetry at horizontal nodes
    /// @param[in,out] z Z-coordinates at all nodes (n_horiz * n_vert)
    void update_z_coordinates(const VecX& eta, const VecX& h, VecX& z) const;

    /// @brief Compute z-coordinates from eta, h, and reference x,y positions
    /// @param x_horiz X coordinates at horizontal nodes
    /// @param y_horiz Y coordinates at horizontal nodes
    /// @param eta Free surface at horizontal nodes
    /// @param h Bathymetry at horizontal nodes
    /// @param[out] nodes Output node positions
    void compute_node_positions(const VecX& x_horiz, const VecX& y_horiz,
                                 const VecX& eta, const VecX& h,
                                 std::vector<Vec3>& nodes) const;

    // =========================================================================
    // Mesh velocity (ALE formulation)
    // =========================================================================

    /// @brief Compute mesh velocity for ALE formulation
    /// @details w_mesh = dz/dt|_sigma = deta/dt * (1 + sigma) for uniform sigma
    ///          More generally: w_mesh = deta/dt + sigma * dH/dt
    /// @param deta_dt Time derivative of free surface at horizontal nodes
    /// @param sigma_values Sigma at each 3D node
    /// @param[out] w_mesh Mesh velocity at all nodes
    void compute_mesh_velocity(const VecX& deta_dt, const VecX& sigma_values,
                                VecX& w_mesh) const;

    /// @brief Compute mesh velocity (using internal sigma values)
    /// @param deta_dt Time derivative of free surface at horizontal nodes
    /// @param[out] w_mesh Mesh velocity at all nodes
    void compute_mesh_velocity(const VecX& deta_dt, VecX& w_mesh) const;

    /// @brief Compute mesh velocity including depth change
    /// @param deta_dt Free surface time derivative
    /// @param dh_dt Bathymetry time derivative (usually 0 for fixed bathymetry)
    /// @param eta Free surface
    /// @param h Bathymetry
    /// @param[out] w_mesh Mesh velocity at all nodes
    void compute_mesh_velocity_full(const VecX& deta_dt, const VecX& dh_dt,
                                     const VecX& eta, const VecX& h,
                                     VecX& w_mesh) const;

    // =========================================================================
    // Geometric factors
    // =========================================================================

    /// @brief Compute Jacobian matrix at each quadrature point
    /// @details For terrain-following coordinates:
    ///          J = [dx/dxi,  dx/deta,  dx/dzeta ]
    ///              [dy/dxi,  dy/deta,  dy/dzeta ]
    ///              [dz/dxi,  dz/deta,  dz/dzeta ]
    ///          The z-row depends on bathymetry and free surface.
    /// @param nodes Physical node positions
    /// @param[out] jacobians Jacobian matrices at quad points
    /// @param[out] det_J Jacobian determinants
    void compute_jacobians(const std::vector<Vec3>& nodes,
                           std::vector<Mat3>& jacobians,
                           VecX& det_J) const;

    /// @brief Compute inverse Jacobians
    void compute_jacobian_inverses(const std::vector<Mat3>& jacobians,
                                    std::vector<Mat3>& jacobian_inv) const;

    /// @brief Update full geometric factors structure
    /// @param nodes Physical node positions
    /// @param[out] gf Geometric factors (filled)
    void update_geometric_factors(const std::vector<Vec3>& nodes,
                                   GeometricFactors& gf) const;

    // =========================================================================
    // Coordinate derivatives for sigma metrics
    // =========================================================================

    /// @brief Compute dz/dx at constant sigma for all nodes
    /// @param deta_dx Free surface x-gradient at horizontal nodes
    /// @param dh_dx Bathymetry x-gradient at horizontal nodes
    /// @param[out] dz_dx Output: dz/dx at all 3D nodes
    void compute_dz_dx(const VecX& deta_dx, const VecX& dh_dx, VecX& dz_dx) const;

    /// @brief Compute dz/dy at constant sigma
    void compute_dz_dy(const VecX& deta_dy, const VecX& dh_dy, VecX& dz_dy) const;

    // =========================================================================
    // Index mapping between horizontal and 3D
    // =========================================================================

    /// @brief Get 3D node index from horizontal index and vertical level
    /// @param i_horiz Horizontal node index [0, n_horiz)
    /// @param k_vert Vertical level index [0, n_vert)
    /// @return 3D node index
    Index node_index_3d(int i_horiz, int k_vert) const {
        return static_cast<Index>(i_horiz * n_vert_ + k_vert);
    }

    /// @brief Get horizontal index from 3D node index
    int horizontal_index(Index idx_3d) const {
        return static_cast<int>(idx_3d) / n_vert_;
    }

    /// @brief Get vertical level from 3D node index
    int vertical_level(Index idx_3d) const {
        return static_cast<int>(idx_3d) % n_vert_;
    }

private:
    const HexahedronBasis& basis_;
    SigmaStretchType stretch_type_;
    SigmaStretchParams stretch_params_;

    int n_vert_;   // Number of vertical levels (order + 1)
    int n_horiz_;  // Number of horizontal nodes per level

    // Sigma values at vertical LGL nodes (from -1 at bottom to 0 at surface)
    VecX sigma_at_lgl_;

    // Precomputed differentiation matrices for geometry
    // These map from nodal values to derivatives at quadrature points
};

/// @brief Geometric factors for a single element
/// @details Stores Jacobians and metric terms needed for integration
struct GeometricFactors {
    std::vector<Mat3> jacobian;      ///< Jacobian at each quad point
    std::vector<Mat3> jacobian_inv;  ///< Inverse Jacobian
    VecX det_J;                      ///< Jacobian determinant

    // Face geometric factors
    std::array<VecX, 6> face_det_J;      ///< Face Jacobian determinants
    std::array<std::vector<Vec3>, 6> face_normals;  ///< Outward normals at face quad pts

    /// @brief Resize storage for given number of quad points
    void resize(int num_vol_quad, int num_face_quad) {
        jacobian.resize(num_vol_quad);
        jacobian_inv.resize(num_vol_quad);
        det_J.resize(num_vol_quad);

        for (int f = 0; f < 6; ++f) {
            face_det_J[f].resize(num_face_quad);
            face_normals[f].resize(num_face_quad);
        }
    }

    /// @brief Compute geometric factors from physical node positions
    /// @details Static factory method
    /// @param basis Hexahedron basis functions
    /// @param num_quad_1d Number of 1D quadrature points per direction
    /// @param physical_nodes Physical node positions
    static GeometricFactors compute(const HexahedronBasis& basis,
                                    int num_quad_1d,
                                    const std::vector<Vec3>& physical_nodes);
};

}  // namespace drifter
