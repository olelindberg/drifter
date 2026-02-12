#pragma once

// Mortar element method for non-conforming interfaces
// Provides conservative flux computation at coarse-fine boundaries
//
// Theory: The mortar approach uses L2 projection to ensure conservation:
// 1. Define a mortar space on the coarse face
// 2. Project both coarse and fine solutions to mortar via L2 projection
// 3. Compute numerical flux in mortar space
// 4. Project flux back, preserving conservation: ∫_Γ F*·n (coarse) = -∫_Γ F*·n
// (fine)

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/face_connection.hpp"
#include "dg/quadrature_3d.hpp"
#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace drifter {

/// @brief Numerical flux function signature
/// @details F*(U_L, U_R, n) where U_L, U_R are left/right states, n is outward
/// normal
using NumericalFluxFunc =
    std::function<VecX(const VecX &U_L, const VecX &U_R, const Vec3 &n)>;

/// @brief Mortar space definition for a single face
/// @details The mortar space is defined on the coarse face geometry with
///          polynomial degree = max(p_coarse, p_fine)
class MortarSpace {
public:
    /// @brief Construct mortar space for a face connection
    /// @param conn Face connection describing the interface
    /// @param basis_coarse Basis for coarse element
    /// @param basis_fine Basis for fine elements
    /// @param face_id Face ID on coarse element
    MortarSpace(
        const FaceConnection &conn, const HexahedronBasis &basis_coarse,
        const HexahedronBasis &basis_fine, int face_id);

    /// Mortar polynomial order
    int order() const { return mortar_order_; }

    /// Number of mortar DOFs
    int num_dofs() const { return num_mortar_dofs_; }

    /// Number of mortar quadrature points
    int num_quad_points() const { return mortar_quad_.size(); }

    /// Get mortar mass matrix
    const MatX &mass_matrix() const { return mass_; }

    /// Get inverse mortar mass matrix
    const MatX &mass_matrix_inv() const { return mass_inv_; }

    /// Get mortar quadrature
    const GaussQuadrature2D &quadrature() const { return mortar_quad_; }

    /// Face connection type
    FaceConnectionType connection_type() const { return conn_type_; }

    /// Number of fine elements
    int num_fine_elements() const { return num_fine_; }

    // =========================================================================
    // Projection operators
    // =========================================================================

    /// L2 projection: coarse face DOFs -> mortar DOFs
    /// P_coarse * U_coarse_face = U_mortar
    const MatX &projection_coarse() const { return P_coarse_; }

    /// L2 projection: fine face DOFs -> mortar DOFs (one per fine element)
    /// For each fine element i: P_fine[i] * U_fine_face[i] = contribution to
    /// U_mortar
    const std::vector<MatX> &projection_fine() const { return P_fine_; }

    /// Lift operator: mortar DOFs -> coarse face DOFs
    /// L_coarse * F_mortar = contribution to coarse RHS
    const MatX &lift_coarse() const { return L_coarse_; }

    /// Lift operators: mortar DOFs -> fine face DOFs (one per fine element)
    /// L_fine[i] * F_mortar = contribution to fine element i RHS
    const std::vector<MatX> &lift_fine() const { return L_fine_; }

    // =========================================================================
    // Mortar flux computation
    // =========================================================================

    /// @brief Compute mortar flux contribution to element RHS
    /// @details Uses L2 projection for conservation. The key property is:
    ///          ∫_Γ F*·n dS (coarse) + Σ_i ∫_Γ_i F*·n dS (fine_i) = 0
    ///
    /// @param flux_func Numerical flux function F*(U_L, U_R, n)
    /// @param U_coarse_face Solution on coarse face (num_face_dofs)
    /// @param U_fine_faces Solutions on fine faces (vector of num_face_dofs
    /// each)
    /// @param normal Outward normal from coarse to fine
    /// @param[out] rhs_coarse_face Flux contribution to coarse element face
    /// DOFs
    /// @param[out] rhs_fine_faces Flux contributions to fine element face DOFs
    void compute_mortar_flux(
        const NumericalFluxFunc &flux_func, const VecX &U_coarse_face,
        const std::vector<VecX> &U_fine_faces, const Vec3 &normal,
        VecX &rhs_coarse_face, std::vector<VecX> &rhs_fine_faces) const;

private:
    FaceConnectionType conn_type_;
    int face_id_;
    int mortar_order_;
    int num_mortar_dofs_;
    int num_fine_;

    // Mortar quadrature (on coarse face reference coordinates)
    GaussQuadrature2D mortar_quad_;

    // Mass matrix in mortar space
    MatX mass_;
    MatX mass_inv_;

    // Projection operators
    MatX P_coarse_;            // Coarse face -> mortar
    std::vector<MatX> P_fine_; // Fine faces -> mortar (one per fine element)

    // Lift operators (adjoint of projection * inverse mass)
    MatX L_coarse_;
    std::vector<MatX> L_fine_;

    // Sub-mortar regions for each fine element (in coarse face reference
    // coords)
    std::vector<std::pair<Vec2, Vec2>>
        subface_bounds_; // (min, max) in [-1,1]^2

    void build_projection_operators(
        const HexahedronBasis &basis_coarse, const HexahedronBasis &basis_fine);
    void compute_subface_bounds();
};

/// @brief Mortar interface handler for all non-conforming faces
class MortarInterfaceManager {
public:
    /// @brief Construct with basis information
    /// @param order Polynomial order (same for all elements in this simple
    /// version)
    explicit MortarInterfaceManager(int order);

    /// @brief Register a non-conforming face connection
    /// @param conn Face connection information
    void register_interface(const FaceConnection &conn);

    /// @brief Build all mortar operators (call after registering all
    /// interfaces)
    void build_operators();

    /// @brief Get mortar space for a specific face connection
    /// @param coarse_elem Coarse element index
    /// @param face_id Face ID on coarse element
    /// @return Pointer to mortar space, or nullptr if not found
    const MortarSpace *get_mortar(Index coarse_elem, int face_id) const;

    /// @brief Check if a face requires mortar treatment
    bool has_mortar(Index coarse_elem, int face_id) const;

    /// Number of registered non-conforming interfaces
    size_t num_interfaces() const { return mortars_.size(); }

    /// Iterate over all mortar interfaces
    template <typename Func> void for_each_interface(Func &&f) const {
        for (const auto &[key, mortar] : mortars_) {
            f(key.first, key.second, *mortar);
        }
    }

private:
    int order_;
    std::unique_ptr<HexahedronBasis> basis_;

    // Map from (coarse_elem, face_id) to mortar space
    std::map<std::pair<Index, int>, std::unique_ptr<MortarSpace>> mortars_;

    // Registered face connections
    std::vector<FaceConnection> connections_;
};

/// @brief Direct averaging interface for conforming (1:1) faces
/// @details Simpler and faster than mortar for same-level interfaces
class ConformingInterface {
public:
    /// @brief Construct conforming interface
    /// @param basis Basis for both elements
    /// @param face_id_left Face ID on left element
    /// @param face_id_right Face ID on right element
    ConformingInterface(
        const HexahedronBasis &basis, int face_id_left, int face_id_right);

    /// @brief Compute direct averaging flux
    /// @param flux_func Numerical flux function
    /// @param U_left_face Solution on left face
    /// @param U_right_face Solution on right face
    /// @param normal Outward normal from left to right
    /// @param[out] rhs_left_face Flux contribution to left element
    /// @param[out] rhs_right_face Flux contribution to right element
    void compute_flux(
        const NumericalFluxFunc &flux_func, const VecX &U_left_face,
        const VecX &U_right_face, const Vec3 &normal, VecX &rhs_left_face,
        VecX &rhs_right_face) const;

    /// Face quadrature weights
    const VecX &quad_weights() const { return quad_weights_; }

    /// Number of face quadrature points
    int num_quad_points() const { return num_quad_; }

private:
    int num_quad_;
    VecX quad_weights_;
    MatX interp_left_;  // Volume -> face quadrature for left element
    MatX interp_right_; // Volume -> face quadrature for right element
    MatX lift_left_;    // Face quadrature -> volume for left element
    MatX lift_right_;
    bool needs_orientation_flip_;
};

// =============================================================================
// Common numerical flux functions
// =============================================================================

namespace flux {

/// @brief Local Lax-Friedrichs (Rusanov) flux
/// @param U_L Left state
/// @param U_R Right state
/// @param n Outward normal
/// @param max_speed Maximum wave speed estimate
/// @param physical_flux Function to compute physical flux F(U) in direction n
VecX lax_friedrichs(
    const VecX &U_L, const VecX &U_R, const Vec3 &n, Real max_speed,
    const std::function<VecX(const VecX &, const Vec3 &)> &physical_flux);

/// @brief Central flux (unstable for hyperbolic, used for diffusion)
VecX central(
    const VecX &U_L, const VecX &U_R, const Vec3 &n,
    const std::function<VecX(const VecX &, const Vec3 &)> &physical_flux);

/// @brief Upwind flux (for scalar advection)
VecX upwind(
    const VecX &U_L, const VecX &U_R, const Vec3 &n, const Vec3 &velocity);

} // namespace flux

// =============================================================================
// Conservation verification utilities
// =============================================================================

/// @brief Verify flux conservation across a mortar interface
/// @details Checks that ∫_Γ F*·n (coarse) + Σ_i ∫_Γ_i F*·n (fine_i) ≈ 0
/// @return Relative conservation error
Real verify_mortar_conservation(
    const MortarSpace &mortar, const VecX &rhs_coarse,
    const std::vector<VecX> &rhs_fine, const VecX &quad_weights_coarse,
    const std::vector<VecX> &quad_weights_fine);

} // namespace drifter
