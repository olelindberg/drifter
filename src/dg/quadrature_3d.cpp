// 3D Gauss quadrature implementation

#include "dg/quadrature_3d.hpp"
#include <algorithm>
#include <cmath>

namespace drifter {

// =============================================================================
// GaussQuadrature1D
// =============================================================================

GaussQuadrature1D::GaussQuadrature1D(int order, QuadratureType type)
    : order_(order), type_(type) {

    // 'order' specifies the number of quadrature points directly
    // For GL: n points integrate exactly polynomials up to degree 2n-1
    // For LGL: n points integrate exactly polynomials up to degree 2n-3
    int n = order;
    if (n < 1)
        n = 1;

    if (type == QuadratureType::GaussLobatto && n < 2) {
        n = 2; // LGL requires at least 2 points
    }

    if (type == QuadratureType::GaussLegendre) {
        compute_gauss_legendre_nodes(n, nodes_, weights_);
    } else {
        compute_gauss_lobatto_nodes(n, nodes_, weights_);
    }
}

Real GaussQuadrature1D::min_spacing() const {
    if (nodes_.size() < 2)
        return 2.0;

    Real min_dx = std::numeric_limits<Real>::max();
    for (int i = 1; i < nodes_.size(); ++i) {
        min_dx = std::min(min_dx, nodes_(i) - nodes_(i - 1));
    }
    return min_dx;
}

// =============================================================================
// GaussQuadrature2D
// =============================================================================

GaussQuadrature2D::GaussQuadrature2D(int order, QuadratureType type)
    : quad_xi_(order, type), quad_eta_(order, type) {
    build_tensor_product();
}

GaussQuadrature2D::GaussQuadrature2D(
    const GaussQuadrature1D &quad_xi, const GaussQuadrature1D &quad_eta)
    : quad_xi_(quad_xi), quad_eta_(quad_eta) {
    build_tensor_product();
}

void GaussQuadrature2D::build_tensor_product() {
    int n_xi = quad_xi_.size();
    int n_eta = quad_eta_.size();
    int n_total = n_xi * n_eta;

    nodes_.resize(n_total);
    weights_.resize(n_total);

    for (int j = 0; j < n_eta; ++j) {
        for (int i = 0; i < n_xi; ++i) {
            int idx = i + n_xi * j;
            nodes_[idx] = Vec2(quad_xi_.nodes()(i), quad_eta_.nodes()(j));
            weights_(idx) = quad_xi_.weights()(i) * quad_eta_.weights()(j);
        }
    }
}

// =============================================================================
// GaussQuadrature3D
// =============================================================================

GaussQuadrature3D::GaussQuadrature3D(int order, QuadratureType type)
    : quad_xi_(order, type), quad_eta_(order, type), quad_zeta_(order, type) {
    build_tensor_product();
}

GaussQuadrature3D::GaussQuadrature3D(
    const GaussQuadrature1D &quad_xi, const GaussQuadrature1D &quad_eta,
    const GaussQuadrature1D &quad_zeta)
    : quad_xi_(quad_xi), quad_eta_(quad_eta), quad_zeta_(quad_zeta) {
    build_tensor_product();
}

void GaussQuadrature3D::build_tensor_product() {
    int n_xi = quad_xi_.size();
    int n_eta = quad_eta_.size();
    int n_zeta = quad_zeta_.size();
    int n_total = n_xi * n_eta * n_zeta;

    nodes_.resize(n_total);
    weights_.resize(n_total);

    for (int k = 0; k < n_zeta; ++k) {
        for (int j = 0; j < n_eta; ++j) {
            for (int i = 0; i < n_xi; ++i) {
                int idx = index(i, j, k);
                nodes_[idx] = Vec3(
                    quad_xi_.nodes()(i), quad_eta_.nodes()(j),
                    quad_zeta_.nodes()(k));
                weights_(idx) = quad_xi_.weights()(i) * quad_eta_.weights()(j) *
                                quad_zeta_.weights()(k);
            }
        }
    }
}

// =============================================================================
// FaceQuadrature
// =============================================================================

FaceQuadrature::FaceQuadrature(int face_id, int order, QuadratureType type)
    : face_id_(face_id), quad_2d_(order, type) {
    build_volume_nodes();
}

void FaceQuadrature::build_volume_nodes() {
    volume_nodes_.resize(quad_2d_.size());

    auto [t1_axis, t2_axis] = get_face_tangent_axes(face_id_);
    int normal_axis = get_face_normal_axis(face_id_);
    Real normal_val = is_positive_face(face_id_) ? 1.0 : -1.0;

    for (int i = 0; i < quad_2d_.size(); ++i) {
        Vec3 xi;
        xi(normal_axis) = normal_val;
        xi(t1_axis) = quad_2d_.nodes()[i](0);
        xi(t2_axis) = quad_2d_.nodes()[i](1);
        volume_nodes_[i] = xi;
    }
}

Vec3 FaceQuadrature::normal() const {
    Vec3 n = Vec3::Zero();
    int normal_axis = get_face_normal_axis(face_id_);
    n(normal_axis) = is_positive_face(face_id_) ? 1.0 : -1.0;
    return n;
}

// =============================================================================
// QuadratureFactory
// =============================================================================

std::array<FaceQuadrature, 6>
QuadratureFactory::create_face_quadratures(int order, QuadratureType type) {
    return {
        {FaceQuadrature(0, order, type), FaceQuadrature(1, order, type),
         FaceQuadrature(2, order, type), FaceQuadrature(3, order, type),
         FaceQuadrature(4, order, type), FaceQuadrature(5, order, type)}};
}

// =============================================================================
// Integration utilities
// =============================================================================

MatX compute_mass_matrix(
    const HexahedronBasis &basis, const GaussQuadrature3D &quad, bool use_lgl) {
    int ndof = use_lgl ? basis.num_dofs_velocity() : basis.num_dofs_tracer();
    MatX M = MatX::Zero(ndof, ndof);

    for (int q = 0; q < quad.size(); ++q) {
        VecX phi = use_lgl ? basis.evaluate_lgl(quad.nodes()[q])
                           : basis.evaluate_gl(quad.nodes()[q]);
        Real w = quad.weights()(q);

        // M_ij += w * phi_i * phi_j
        for (int i = 0; i < ndof; ++i) {
            for (int j = 0; j < ndof; ++j) {
                M(i, j) += w * phi(i) * phi(j);
            }
        }
    }

    return M;
}

std::tuple<MatX, MatX, MatX> compute_stiffness_matrices(
    const HexahedronBasis &basis, const GaussQuadrature3D &quad, bool use_lgl) {

    int ndof = use_lgl ? basis.num_dofs_velocity() : basis.num_dofs_tracer();
    MatX S_xi = MatX::Zero(ndof, ndof);
    MatX S_eta = MatX::Zero(ndof, ndof);
    MatX S_zeta = MatX::Zero(ndof, ndof);

    for (int q = 0; q < quad.size(); ++q) {
        VecX phi = use_lgl ? basis.evaluate_lgl(quad.nodes()[q])
                           : basis.evaluate_gl(quad.nodes()[q]);
        MatX grad_phi = use_lgl ? basis.evaluate_gradient_lgl(quad.nodes()[q])
                                : basis.evaluate_gradient_gl(quad.nodes()[q]);
        Real w = quad.weights()(q);

        // S_xi_ij += w * phi_i * d(phi_j)/d(xi)
        for (int i = 0; i < ndof; ++i) {
            for (int j = 0; j < ndof; ++j) {
                S_xi(i, j) += w * phi(i) * grad_phi(j, 0);
                S_eta(i, j) += w * phi(i) * grad_phi(j, 1);
                S_zeta(i, j) += w * phi(i) * grad_phi(j, 2);
            }
        }
    }

    return {S_xi, S_eta, S_zeta};
}

} // namespace drifter
