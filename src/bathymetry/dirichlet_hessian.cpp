#include "bathymetry/dirichlet_hessian.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

DirichletHessian::DirichletHessian(int ngauss) : ngauss_(ngauss) {
    if (ngauss < 1) {
        throw std::invalid_argument("DirichletHessian: need at least 1 Gauss point");
    }

    compute_gauss_quadrature();
    build_derivative_matrices();
    build_hessian();
}

void DirichletHessian::compute_gauss_quadrature() {
    // Compute Gauss-Legendre nodes and weights on [-1, 1], then map to [0, 1]
    gauss_nodes_.resize(ngauss_);
    VecX weights_ref(ngauss_); // Weights on [-1, 1]

    // Gauss-Legendre nodes and weights for common orders
    if (ngauss_ == 1) {
        gauss_nodes_ << 0.5;
        weights_ref << 2.0;
    } else if (ngauss_ == 2) {
        Real a = 1.0 / std::sqrt(3.0);
        gauss_nodes_ << (1.0 - a) / 2.0, (1.0 + a) / 2.0;
        weights_ref << 1.0, 1.0;
    } else if (ngauss_ == 3) {
        gauss_nodes_ << 0.5 - std::sqrt(3.0 / 5.0) / 2.0, 0.5, 0.5 + std::sqrt(3.0 / 5.0) / 2.0;
        weights_ref << 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0;
    } else if (ngauss_ == 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real wa = (18.0 + std::sqrt(30.0)) / 36.0;
        Real wb = (18.0 - std::sqrt(30.0)) / 36.0;
        gauss_nodes_ << (1.0 - b) / 2.0, (1.0 - a) / 2.0, (1.0 + a) / 2.0, (1.0 + b) / 2.0;
        weights_ref << wb, wa, wa, wb;
    } else {
        throw std::invalid_argument("DirichletHessian: only ngauss 1-4 supported");
    }

    // Convert weights from [-1,1] to [0,1]: multiply by 0.5 (Jacobian of mapping)
    gauss_weights_.resize(ngauss_ * ngauss_);
    for (int j = 0; j < ngauss_; ++j) {
        for (int i = 0; i < ngauss_; ++i) {
            int idx = i + ngauss_ * j;
            gauss_weights_(idx) = 0.5 * weights_ref(i) * 0.5 * weights_ref(j);
        }
    }
}

void DirichletHessian::build_derivative_matrices() {
    int nquad = ngauss_ * ngauss_;
    int ndof = LinearBezierBasis2D::NDOF;

    D1U_.resize(nquad, ndof);
    D1V_.resize(nquad, ndof);

    for (int qj = 0; qj < ngauss_; ++qj) {
        for (int qi = 0; qi < ngauss_; ++qi) {
            int qidx = qi + ngauss_ * qj;
            Real u = gauss_nodes_(qi);
            Real v = gauss_nodes_(qj);

            // First derivatives
            VecX du = basis_.evaluate_du(u, v);
            VecX dv = basis_.evaluate_dv(u, v);

            for (int k = 0; k < ndof; ++k) {
                D1U_(qidx, k) = du(k);
                D1V_(qidx, k) = dv(k);
            }
        }
    }
}

void DirichletHessian::build_hessian() {
    // Dirichlet energy on [0,1]^2:
    //   E = integral [z_u^2 + z_v^2] du dv
    //
    // For z(u,v) = sum_k c_k * B_k(u,v):
    //   E = c^T * H * c
    // where:
    //   H = D1U^T * W * D1U + D1V^T * W * D1V

    MatX WD1U = gauss_weights_.asDiagonal() * D1U_;
    MatX WD1V = gauss_weights_.asDiagonal() * D1V_;

    H_u_u_ = D1U_.transpose() * WD1U;
    H_v_v_ = D1V_.transpose() * WD1V;

    H_ = H_u_u_ + H_v_v_;

    // Symmetrize (numerical precision)
    H_ = 0.5 * (H_ + H_.transpose());
    H_u_u_ = 0.5 * (H_u_u_ + H_u_u_.transpose());
    H_v_v_ = 0.5 * (H_v_v_ + H_v_v_.transpose());
}

Real DirichletHessian::energy(const VecX &coeffs) const {
    if (coeffs.size() != LinearBezierBasis2D::NDOF) {
        throw std::invalid_argument("DirichletHessian::energy: coeffs must have 4 elements");
    }
    return coeffs.transpose() * H_ * coeffs;
}

VecX DirichletHessian::gradient(const VecX &coeffs) const {
    if (coeffs.size() != LinearBezierBasis2D::NDOF) {
        throw std::invalid_argument("DirichletHessian::gradient: coeffs must have 4 elements");
    }
    return 2.0 * H_ * coeffs;
}

MatX DirichletHessian::scaled_hessian(Real dx, Real dy) const {
    // For a physical element with dimensions (dx, dy), the Dirichlet energy is:
    //
    // E_phys = integral_{physical} [z_x^2 + z_y^2] dx dy
    //
    // Using z_x = z_u / dx, z_y = z_v / dy, Jacobian = dx * dy:
    //
    // E_phys = integral [z_u^2/dx^2 + z_v^2/dy^2] * dx * dy du dv
    //        = (dy/dx) * H_u_u + (dx/dy) * H_v_v

    Real scale_u_u = dy / dx;
    Real scale_v_v = dx / dy;

    MatX H_scaled = scale_u_u * H_u_u_ + scale_v_v * H_v_v_;

    // Symmetrize
    return 0.5 * (H_scaled + H_scaled.transpose());
}

} // namespace drifter
