#include "mesh/vertical_tfi.hpp"
#include <stdexcept>

namespace drifter {

VerticalTFI::VerticalTFI(const HexahedronBasis& basis,
                         SigmaStretchType stretch_type,
                         const SigmaStretchParams& stretch_params)
    : basis_(basis)
    , stretch_type_(stretch_type)
    , stretch_params_(stretch_params)
    , n_vert_(basis.order() + 1)
    , n_horiz_((basis.order() + 1) * (basis.order() + 1))
{
    // Generate sigma values at vertical LGL nodes
    // LGL nodes go from -1 to 1 in reference space
    // We map this to sigma from -1 (bottom) to 0 (surface)
    const VecX& zeta_lgl = basis.lgl_basis_1d().nodes;
    sigma_at_lgl_.resize(n_vert_);

    for (int k = 0; k < n_vert_; ++k) {
        // Map zeta in [-1, 1] to sigma_uniform in [-1, 0]
        Real sigma_uniform = 0.5 * (zeta_lgl(k) - 1.0);
        // Apply stretching
        sigma_at_lgl_(k) = SigmaCoordinate::apply_stretching(
            sigma_uniform, stretch_type_, stretch_params_);
    }
}

void VerticalTFI::update_node_positions(const VecX& eta, const VecX& h,
                                         std::vector<Vec3>& nodes) const {
    if (static_cast<int>(eta.size()) != n_horiz_) {
        throw std::invalid_argument("eta size mismatch");
    }

    const int n_total = n_horiz_ * n_vert_;
    if (static_cast<int>(nodes.size()) != n_total) {
        nodes.resize(n_total);
    }

    // Update only z-coordinates (x, y preserved from initial setup)
    for (int i = 0; i < n_horiz_; ++i) {
        const Real H = eta(i) + h(i);
        for (int k = 0; k < n_vert_; ++k) {
            const Index idx = node_index_3d(i, k);
            // z = eta + sigma * H
            nodes[idx](2) = eta(i) + sigma_at_lgl_(k) * H;
        }
    }
}

void VerticalTFI::update_z_coordinates(const VecX& eta, const VecX& h,
                                        VecX& z) const {
    if (static_cast<int>(eta.size()) != n_horiz_) {
        throw std::invalid_argument("eta size mismatch");
    }

    const int n_total = n_horiz_ * n_vert_;
    if (z.size() != n_total) {
        z.resize(n_total);
    }

    for (int i = 0; i < n_horiz_; ++i) {
        const Real H = eta(i) + h(i);
        for (int k = 0; k < n_vert_; ++k) {
            z(node_index_3d(i, k)) = eta(i) + sigma_at_lgl_(k) * H;
        }
    }
}

void VerticalTFI::compute_node_positions(const VecX& x_horiz, const VecX& y_horiz,
                                          const VecX& eta, const VecX& h,
                                          std::vector<Vec3>& nodes) const {
    if (static_cast<int>(eta.size()) != n_horiz_) {
        throw std::invalid_argument("eta size mismatch");
    }

    const int n_total = n_horiz_ * n_vert_;
    nodes.resize(n_total);

    for (int i = 0; i < n_horiz_; ++i) {
        const Real H = eta(i) + h(i);
        for (int k = 0; k < n_vert_; ++k) {
            const Index idx = node_index_3d(i, k);
            nodes[idx](0) = x_horiz(i);
            nodes[idx](1) = y_horiz(i);
            nodes[idx](2) = eta(i) + sigma_at_lgl_(k) * H;
        }
    }
}

void VerticalTFI::compute_mesh_velocity(const VecX& deta_dt,
                                         const VecX& sigma_values,
                                         VecX& w_mesh) const {
    const int n = static_cast<int>(sigma_values.size());
    w_mesh.resize(n);

    // For fixed bathymetry: w_mesh = deta/dt * (1 + sigma)
    // This is dz/dt|_sigma = d(eta + sigma*H)/dt = deta/dt + sigma*dH/dt
    // With H = eta + h and dh/dt = 0: dH/dt = deta/dt
    // So w_mesh = deta/dt * (1 + sigma)

    // Determine which horizontal point each 3D point belongs to
    for (int idx = 0; idx < n; ++idx) {
        const int i_horiz = horizontal_index(idx);
        const Real sigma = sigma_values(idx);
        w_mesh(idx) = deta_dt(i_horiz) * (1.0 + sigma);
    }
}

void VerticalTFI::compute_mesh_velocity(const VecX& deta_dt, VecX& w_mesh) const {
    const int n_total = n_horiz_ * n_vert_;
    w_mesh.resize(n_total);

    for (int i = 0; i < n_horiz_; ++i) {
        for (int k = 0; k < n_vert_; ++k) {
            const Index idx = node_index_3d(i, k);
            // w_mesh = deta/dt * (1 + sigma)
            w_mesh(idx) = deta_dt(i) * (1.0 + sigma_at_lgl_(k));
        }
    }
}

void VerticalTFI::compute_mesh_velocity_full(const VecX& deta_dt, const VecX& dh_dt,
                                              const VecX& eta, const VecX& h,
                                              VecX& w_mesh) const {
    const int n_total = n_horiz_ * n_vert_;
    w_mesh.resize(n_total);

    for (int i = 0; i < n_horiz_; ++i) {
        // dH/dt = deta/dt + dh/dt
        const Real dH_dt = deta_dt(i) + dh_dt(i);

        for (int k = 0; k < n_vert_; ++k) {
            const Index idx = node_index_3d(i, k);
            // w_mesh = deta/dt + sigma * dH/dt
            w_mesh(idx) = deta_dt(i) + sigma_at_lgl_(k) * dH_dt;
        }
    }
}

void VerticalTFI::compute_jacobians(const std::vector<Vec3>& nodes,
                                     std::vector<Mat3>& jacobians,
                                     VecX& det_J) const {
    // Get differentiation matrices from basis
    const MatX& D_xi = basis_.D_xi_lgl();
    const MatX& D_eta = basis_.D_eta_lgl();
    const MatX& D_zeta = basis_.D_zeta_lgl();

    const int n_dofs = basis_.num_dofs_velocity();
    jacobians.resize(n_dofs);
    det_J.resize(n_dofs);

    // For collocated scheme, quadrature points = nodal points
    // Jacobian at each node
    for (int q = 0; q < n_dofs; ++q) {
        Mat3& J = jacobians[q];
        J.setZero();

        // J_ij = sum_k D_{qk} * x_k^j
        // where i is ref coord (xi, eta, zeta) and j is physical (x, y, z)
        for (int k = 0; k < n_dofs; ++k) {
            J(0, 0) += D_xi(q, k) * nodes[k](0);   // dx/dxi
            J(0, 1) += D_xi(q, k) * nodes[k](1);   // dy/dxi
            J(0, 2) += D_xi(q, k) * nodes[k](2);   // dz/dxi

            J(1, 0) += D_eta(q, k) * nodes[k](0);  // dx/deta
            J(1, 1) += D_eta(q, k) * nodes[k](1);  // dy/deta
            J(1, 2) += D_eta(q, k) * nodes[k](2);  // dz/deta

            J(2, 0) += D_zeta(q, k) * nodes[k](0); // dx/dzeta
            J(2, 1) += D_zeta(q, k) * nodes[k](1); // dy/dzeta
            J(2, 2) += D_zeta(q, k) * nodes[k](2); // dz/dzeta
        }

        det_J(q) = J.determinant();
    }
}

void VerticalTFI::compute_jacobian_inverses(const std::vector<Mat3>& jacobians,
                                             std::vector<Mat3>& jacobian_inv) const {
    const int n = static_cast<int>(jacobians.size());
    jacobian_inv.resize(n);

    for (int i = 0; i < n; ++i) {
        jacobian_inv[i] = jacobians[i].inverse();
    }
}

void VerticalTFI::update_geometric_factors(const std::vector<Vec3>& nodes,
                                            GeometricFactors& gf) const {
    // Compute volume Jacobians
    compute_jacobians(nodes, gf.jacobian, gf.det_J);
    compute_jacobian_inverses(gf.jacobian, gf.jacobian_inv);

    // Face geometric factors would require additional computation
    // based on face quadrature points and surface metrics
    // For now, we leave them for the caller to compute if needed
}

void VerticalTFI::compute_dz_dx(const VecX& deta_dx, const VecX& dh_dx,
                                 VecX& dz_dx) const {
    const int n_total = n_horiz_ * n_vert_;
    dz_dx.resize(n_total);

    for (int i = 0; i < n_horiz_; ++i) {
        // dz/dx|_sigma = deta/dx + sigma * dH/dx = deta/dx + sigma*(deta/dx + dh/dx)
        //              = deta/dx * (1 + sigma) + sigma * dh/dx
        const Real dH_dx = deta_dx(i) + dh_dx(i);
        for (int k = 0; k < n_vert_; ++k) {
            const Index idx = node_index_3d(i, k);
            dz_dx(idx) = deta_dx(i) + sigma_at_lgl_(k) * dH_dx;
        }
    }
}

void VerticalTFI::compute_dz_dy(const VecX& deta_dy, const VecX& dh_dy,
                                 VecX& dz_dy) const {
    const int n_total = n_horiz_ * n_vert_;
    dz_dy.resize(n_total);

    for (int i = 0; i < n_horiz_; ++i) {
        const Real dH_dy = deta_dy(i) + dh_dy(i);
        for (int k = 0; k < n_vert_; ++k) {
            const Index idx = node_index_3d(i, k);
            dz_dy(idx) = deta_dy(i) + sigma_at_lgl_(k) * dH_dy;
        }
    }
}

// GeometricFactors static factory method
GeometricFactors GeometricFactors::compute(const HexahedronBasis& basis,
                                           int num_quad_1d,
                                           const std::vector<Vec3>& physical_nodes) {
    GeometricFactors gf;

    const int num_vol_quad = num_quad_1d * num_quad_1d * num_quad_1d;
    const int num_face_quad = num_quad_1d * num_quad_1d;

    gf.resize(num_vol_quad, num_face_quad);

    // Get basis values at quadrature points
    // For LGL-collocated scheme, quad points = nodal points
    const int n_dofs = basis.num_dofs_velocity();

    const MatX& D_xi = basis.D_xi_lgl();
    const MatX& D_eta = basis.D_eta_lgl();
    const MatX& D_zeta = basis.D_zeta_lgl();

    // Compute Jacobian at each quadrature point
    // For collocated, quadrature point q corresponds to node q
    const int n_points = std::min(num_vol_quad, n_dofs);
    for (int q = 0; q < n_points; ++q) {
        Mat3& J = gf.jacobian[q];
        J.setZero();

        // For collocated, the q-th quad point uses the q-th row of D
        for (int k = 0; k < n_dofs; ++k) {
            J(0, 0) += D_xi(q, k) * physical_nodes[k](0);
            J(0, 1) += D_xi(q, k) * physical_nodes[k](1);
            J(0, 2) += D_xi(q, k) * physical_nodes[k](2);

            J(1, 0) += D_eta(q, k) * physical_nodes[k](0);
            J(1, 1) += D_eta(q, k) * physical_nodes[k](1);
            J(1, 2) += D_eta(q, k) * physical_nodes[k](2);

            J(2, 0) += D_zeta(q, k) * physical_nodes[k](0);
            J(2, 1) += D_zeta(q, k) * physical_nodes[k](1);
            J(2, 2) += D_zeta(q, k) * physical_nodes[k](2);
        }

        gf.det_J(q) = J.determinant();
        gf.jacobian_inv[q] = J.inverse();
    }

    // Face geometric factors
    // For each face, compute normal and surface Jacobian
    // Face ordering: 0 = -xi, 1 = +xi, 2 = -eta, 3 = +eta, 4 = -zeta, 5 = +zeta

    // This is a simplified version - full implementation would require
    // computing tangent vectors and cross products at face quad points

    return gf;
}

}  // namespace drifter
