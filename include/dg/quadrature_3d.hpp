#pragma once

// 3D Gauss quadrature for hexahedral elements
// Adapted from wobbler: GaussQuadrature1D, GaussQuadrature2D

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include <memory>
#include <vector>

namespace drifter {

/// @brief Quadrature node type
enum class QuadratureType : uint8_t {
    GaussLegendre, ///< Interior Gauss-Legendre nodes
    GaussLobatto   ///< Gauss-Lobatto (includes boundary points)
};

/// @brief 1D Gauss quadrature rule
class GaussQuadrature1D {
public:
    /// @brief Construct 1D quadrature rule
    /// @param order Polynomial order to integrate exactly (uses (order+1)/2 + 1
    /// points for GL)
    /// @param type Quadrature type (GaussLegendre or GaussLobatto)
    GaussQuadrature1D(
        int order, QuadratureType type = QuadratureType::GaussLegendre);

    /// Number of quadrature points
    int size() const { return static_cast<int>(nodes_.size()); }

    /// Quadrature nodes in [-1, 1]
    const VecX &nodes() const { return nodes_; }

    /// Quadrature weights
    const VecX &weights() const { return weights_; }

    /// Get minimum node spacing (for CFL estimate)
    Real min_spacing() const;

    /// Quadrature type
    QuadratureType type() const { return type_; }

    /// Polynomial order that can be integrated exactly
    int order() const { return order_; }

private:
    int order_;
    QuadratureType type_;
    VecX nodes_;
    VecX weights_;
};

/// @brief 2D Gauss quadrature rule on reference quadrilateral [-1,1]^2
class GaussQuadrature2D {
public:
    /// @brief Construct 2D quadrature rule using tensor product
    /// @param order Polynomial order to integrate exactly
    /// @param type Quadrature type
    GaussQuadrature2D(
        int order, QuadratureType type = QuadratureType::GaussLegendre);

    /// @brief Construct from two 1D rules (may have different types/orders)
    GaussQuadrature2D(
        const GaussQuadrature1D &quad_xi, const GaussQuadrature1D &quad_eta);

    /// Number of quadrature points
    int size() const { return static_cast<int>(nodes_.size()); }

    /// Quadrature nodes in [-1,1]^2 (as Vec2)
    const std::vector<Vec2> &nodes() const { return nodes_; }

    /// Quadrature weights
    const VecX &weights() const { return weights_; }

    /// Get 1D quadrature in xi direction
    const GaussQuadrature1D &quad_xi() const { return quad_xi_; }

    /// Get 1D quadrature in eta direction
    const GaussQuadrature1D &quad_eta() const { return quad_eta_; }

    /// Number of points in xi direction
    int size_xi() const { return quad_xi_.size(); }

    /// Number of points in eta direction
    int size_eta() const { return quad_eta_.size(); }

private:
    GaussQuadrature1D quad_xi_;
    GaussQuadrature1D quad_eta_;
    std::vector<Vec2> nodes_;
    VecX weights_;

    void build_tensor_product();
};

/// @brief 3D Gauss quadrature rule on reference hexahedron [-1,1]^3
class GaussQuadrature3D {
public:
    /// @brief Construct 3D quadrature rule using tensor product
    /// @param order Polynomial order to integrate exactly
    /// @param type Quadrature type (same in all directions)
    GaussQuadrature3D(
        int order, QuadratureType type = QuadratureType::GaussLegendre);

    /// @brief Construct from three 1D rules (for anisotropic quadrature)
    GaussQuadrature3D(
        const GaussQuadrature1D &quad_xi, const GaussQuadrature1D &quad_eta,
        const GaussQuadrature1D &quad_zeta);

    /// Number of quadrature points
    int size() const { return static_cast<int>(nodes_.size()); }

    /// Quadrature nodes in [-1,1]^3
    const std::vector<Vec3> &nodes() const { return nodes_; }

    /// Quadrature weights
    const VecX &weights() const { return weights_; }

    /// Get 1D quadrature in xi direction
    const GaussQuadrature1D &quad_xi() const { return quad_xi_; }

    /// Get 1D quadrature in eta direction
    const GaussQuadrature1D &quad_eta() const { return quad_eta_; }

    /// Get 1D quadrature in zeta direction
    const GaussQuadrature1D &quad_zeta() const { return quad_zeta_; }

    /// Number of points in xi direction
    int size_xi() const { return quad_xi_.size(); }

    /// Number of points in eta direction
    int size_eta() const { return quad_eta_.size(); }

    /// Number of points in zeta direction
    int size_zeta() const { return quad_zeta_.size(); }

    /// Convert (i, j, k) indices to linear index
    int index(int i, int j, int k) const {
        return i + size_xi() * (j + size_eta() * k);
    }

private:
    GaussQuadrature1D quad_xi_;
    GaussQuadrature1D quad_eta_;
    GaussQuadrature1D quad_zeta_;
    std::vector<Vec3> nodes_;
    VecX weights_;

    void build_tensor_product();
};

/// @brief Face quadrature rule for surface integrals
class FaceQuadrature {
public:
    /// @brief Construct face quadrature for a specific face
    /// @param face_id Face ID (0-5)
    /// @param order Polynomial order to integrate exactly
    /// @param type Quadrature type
    FaceQuadrature(
        int face_id, int order,
        QuadratureType type = QuadratureType::GaussLegendre);

    /// Face ID
    int face_id() const { return face_id_; }

    /// Number of quadrature points
    int size() const { return quad_2d_.size(); }

    /// Quadrature nodes in face reference coordinates (tangent directions)
    const std::vector<Vec2> &face_nodes() const { return quad_2d_.nodes(); }

    /// Quadrature nodes in volume reference coordinates
    const std::vector<Vec3> &volume_nodes() const { return volume_nodes_; }

    /// Quadrature weights (for face integration, need to multiply by face
    /// Jacobian)
    const VecX &weights() const { return quad_2d_.weights(); }

    /// Get underlying 2D quadrature
    const GaussQuadrature2D &quad_2d() const { return quad_2d_; }

    /// Normal direction (outward) for this face
    Vec3 normal() const;

    /// Get tangent axes for this face
    std::pair<int, int> tangent_axes() const {
        return get_face_tangent_axes(face_id_);
    }

private:
    int face_id_;
    GaussQuadrature2D quad_2d_;
    std::vector<Vec3> volume_nodes_;

    void build_volume_nodes();
};

/// @brief Quadrature factory for common configurations
class QuadratureFactory {
public:
    /// Create volume quadrature matching LGL basis nodes (for mass-lumping)
    static GaussQuadrature3D create_lgl_collocated(int order) {
        return GaussQuadrature3D(order, QuadratureType::GaussLobatto);
    }

    /// Create volume quadrature matching GL basis nodes
    static GaussQuadrature3D create_gl_collocated(int order) {
        return GaussQuadrature3D(order, QuadratureType::GaussLegendre);
    }

    /// Create over-integrated volume quadrature for nonlinear terms
    /// @param basis_order Polynomial order of the basis
    /// @param nonlinearity_degree Degree of nonlinearity (2 for quadratic, 3
    /// for cubic)
    static GaussQuadrature3D
    create_over_integrated(int basis_order, int nonlinearity_degree) {
        int quad_order = basis_order * nonlinearity_degree;
        return GaussQuadrature3D(quad_order, QuadratureType::GaussLegendre);
    }

    /// Create face quadrature for all 6 faces
    static std::array<FaceQuadrature, 6> create_face_quadratures(
        int order, QuadratureType type = QuadratureType::GaussLegendre);

    /// Create face quadrature matching LGL face nodes (collocated)
    static std::array<FaceQuadrature, 6>
    create_lgl_face_quadratures(int order) {
        return create_face_quadratures(order, QuadratureType::GaussLobatto);
    }
};

// =============================================================================
// Integration utilities
// =============================================================================

/// @brief Integrate a function over the reference hexahedron
/// @tparam Func Callable with signature Real(const Vec3&)
/// @param quad Quadrature rule
/// @param f Function to integrate
/// @return Integral value
template <typename Func>
Real integrate_volume(const GaussQuadrature3D &quad, Func &&f) {
    Real result = 0.0;
    for (int i = 0; i < quad.size(); ++i) {
        result += quad.weights()(i) * f(quad.nodes()[i]);
    }
    return result;
}

/// @brief Integrate a function over a face in reference coordinates
/// @tparam Func Callable with signature Real(const Vec3&) - evaluated at volume
/// coords
/// @param fquad Face quadrature rule
/// @param f Function to integrate
/// @return Integral value (still needs physical Jacobian scaling)
template <typename Func>
Real integrate_face(const FaceQuadrature &fquad, Func &&f) {
    Real result = 0.0;
    for (int i = 0; i < fquad.size(); ++i) {
        result += fquad.weights()(i) * f(fquad.volume_nodes()[i]);
    }
    return result;
}

/// @brief Compute L2 inner product of two functions on reference element
template <typename Func1, typename Func2>
Real inner_product_l2(const GaussQuadrature3D &quad, Func1 &&f, Func2 &&g) {
    Real result = 0.0;
    for (int i = 0; i < quad.size(); ++i) {
        result += quad.weights()(i) * f(quad.nodes()[i]) * g(quad.nodes()[i]);
    }
    return result;
}

/// @brief Compute L2 norm of a function on reference element
template <typename Func> Real norm_l2(const GaussQuadrature3D &quad, Func &&f) {
    return std::sqrt(inner_product_l2(quad, f, f));
}

/// @brief Compute element mass matrix using quadrature
/// @param basis Basis functions
/// @param quad Quadrature rule
/// @param use_lgl Use LGL basis (true) or GL basis (false)
/// @return Mass matrix (num_dofs x num_dofs)
MatX compute_mass_matrix(
    const HexahedronBasis &basis, const GaussQuadrature3D &quad,
    bool use_lgl = true);

/// @brief Compute weak gradient operator matrices
/// @details Returns S_xi, S_eta, S_zeta where S = M^{-1} * D^T * M_quad
/// For DG: integral of v * du/dxi becomes -integral of dv/dxi * u + boundary
/// terms
/// @param basis Basis functions
/// @param quad Quadrature rule
/// @param use_lgl Use LGL basis
/// @return Tuple of (S_xi, S_eta, S_zeta)
std::tuple<MatX, MatX, MatX> compute_stiffness_matrices(
    const HexahedronBasis &basis, const GaussQuadrature3D &quad,
    bool use_lgl = true);

} // namespace drifter
