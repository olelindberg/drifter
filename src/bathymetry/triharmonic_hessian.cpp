#include "bathymetry/triharmonic_hessian.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

TriharmonicHessian::TriharmonicHessian(int ngauss, Real gradient_weight)
    : ngauss_(ngauss), gradient_weight_(gradient_weight),
      basis_(std::make_unique<BezierBasis2D>()) {
    if (ngauss < 3) {
        throw std::invalid_argument("TriharmonicHessian: need at least 3 Gauss "
                                    "points for third derivatives");
    }

    compute_gauss_quadrature();
    build_derivative_matrices();
    build_hessian();
}

void TriharmonicHessian::compute_gauss_quadrature() {
    // Compute Gauss-Legendre nodes and weights on [-1, 1], then map to [0, 1]
    gauss_nodes_.resize(ngauss_);
    VecX weights_ref(ngauss_); // Weights on [-1, 1]

    if (ngauss_ == 3) {
        gauss_nodes_ << 0.5 - std::sqrt(3.0 / 5.0) / 2.0, 0.5,
            0.5 + std::sqrt(3.0 / 5.0) / 2.0;
        weights_ref << 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0;
    } else if (ngauss_ == 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real wa = (18.0 + std::sqrt(30.0)) / 36.0;
        Real wb = (18.0 - std::sqrt(30.0)) / 36.0;
        gauss_nodes_ << (1.0 - b) / 2.0, (1.0 - a) / 2.0, (1.0 + a) / 2.0,
            (1.0 + b) / 2.0;
        weights_ref << wb, wa, wa, wb;
    } else if (ngauss_ == 5) {
        Real a = std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 3.0;
        Real b = std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 3.0;
        Real wa = (322.0 + 13.0 * std::sqrt(70.0)) / 900.0;
        Real wb = (322.0 - 13.0 * std::sqrt(70.0)) / 900.0;
        gauss_nodes_ << (1.0 - b) / 2.0, (1.0 - a) / 2.0, 0.5, (1.0 + a) / 2.0,
            (1.0 + b) / 2.0;
        weights_ref << wb, wa, 128.0 / 225.0, wa, wb;
    } else if (ngauss_ == 6) {
        // 6-point Gauss-Legendre on [-1, 1]
        Real x1 = 0.6612093864662645;
        Real x2 = 0.2386191860831969;
        Real x3 = 0.9324695142031521;
        Real w1 = 0.3607615730481386;
        Real w2 = 0.4679139345726910;
        Real w3 = 0.1713244923791704;

        gauss_nodes_ << (1.0 - x3) / 2.0, (1.0 - x1) / 2.0, (1.0 - x2) / 2.0,
            (1.0 + x2) / 2.0, (1.0 + x1) / 2.0, (1.0 + x3) / 2.0;
        weights_ref << w3, w1, w2, w2, w1, w3;
    } else {
        throw std::invalid_argument(
            "TriharmonicHessian: only ngauss 3-6 supported");
    }

    // Convert weights from [-1,1] to [0,1]: multiply by 0.5 (Jacobian of
    // mapping)
    gauss_weights_.resize(ngauss_ * ngauss_);
    for (int j = 0; j < ngauss_; ++j) {
        for (int i = 0; i < ngauss_; ++i) {
            int idx = i + ngauss_ * j;
            gauss_weights_(idx) = 0.5 * weights_ref(i) * 0.5 * weights_ref(j);
        }
    }
}

void TriharmonicHessian::build_derivative_matrices() {
    // Build matrices D3UUU, D3UUV, D3UVV, D3VVV, D1U, D1V where:
    // D3UUU[m, k] = d^3 B_k / du^3 evaluated at Gauss point m
    // m = i + ngauss * j indexes the 2D Gauss points
    // k = 0..35 indexes the basis functions

    int nquad = ngauss_ * ngauss_;
    int ndof = BezierBasis2D::NDOF;

    D3UUU_.resize(nquad, ndof);
    D3UUV_.resize(nquad, ndof);
    D3UVV_.resize(nquad, ndof);
    D3VVV_.resize(nquad, ndof);
    D1U_.resize(nquad, ndof);
    D1V_.resize(nquad, ndof);

    for (int qj = 0; qj < ngauss_; ++qj) {
        for (int qi = 0; qi < ngauss_; ++qi) {
            int qidx = qi + ngauss_ * qj;
            Real u = gauss_nodes_(qi);
            Real v = gauss_nodes_(qj);

            // Third derivatives
            VecX d3uuu = basis_->evaluate_d3uuu(u, v);
            VecX d3uuv = basis_->evaluate_d3uuv(u, v);
            VecX d3uvv = basis_->evaluate_d3uvv(u, v);
            VecX d3vvv = basis_->evaluate_d3vvv(u, v);

            // First derivatives (for optional gradient penalty)
            VecX du = basis_->evaluate_du(u, v);
            VecX dv = basis_->evaluate_dv(u, v);

            for (int k = 0; k < ndof; ++k) {
                D3UUU_(qidx, k) = d3uuu(k);
                D3UUV_(qidx, k) = d3uuv(k);
                D3UVV_(qidx, k) = d3uvv(k);
                D3VVV_(qidx, k) = d3vvv(k);
                D1U_(qidx, k) = du(k);
                D1V_(qidx, k) = dv(k);
            }
        }
    }
}

void TriharmonicHessian::build_hessian() {
    // Triharmonic energy on [0,1]^2:
    //   E = integral [(z_uuu + z_uvv)^2 + (z_uuv + z_vvv)^2] du dv
    //
    // For z(u,v) = sum_k c_k * B_k(u,v):
    //   z_uuu = D3UUU * c, z_uuv = D3UUV * c
    //   z_uvv = D3UVV * c, z_vvv = D3VVV * c
    //
    // Let G1 = D3UUU + D3UVV (grad_x of Laplacian)
    // Let G2 = D3UUV + D3VVV (grad_y of Laplacian)
    //
    // H = G1^T * W * G1 + G2^T * W * G2

    MatX G1 = D3UUU_ + D3UVV_;
    MatX G2 = D3UUV_ + D3VVV_;

    MatX WG1 = gauss_weights_.asDiagonal() * G1;
    MatX WG2 = gauss_weights_.asDiagonal() * G2;

    H_ = G1.transpose() * WG1 + G2.transpose() * WG2;

    // Build component Hessians for scaled_hessian
    MatX WD3UUU = gauss_weights_.asDiagonal() * D3UUU_;
    MatX WD3UUV = gauss_weights_.asDiagonal() * D3UUV_;
    MatX WD3UVV = gauss_weights_.asDiagonal() * D3UVV_;
    MatX WD3VVV = gauss_weights_.asDiagonal() * D3VVV_;

    H_uuu_uuu_ = D3UUU_.transpose() * WD3UUU;
    H_uuv_uuv_ = D3UUV_.transpose() * WD3UUV;
    H_uvv_uvv_ = D3UVV_.transpose() * WD3UVV;
    H_vvv_vvv_ = D3VVV_.transpose() * WD3VVV;
    H_uuu_uvv_ = D3UUU_.transpose() * WD3UVV;
    H_uuv_vvv_ = D3UUV_.transpose() * WD3VVV;

    // Gradient penalty components
    MatX WD1U = gauss_weights_.asDiagonal() * D1U_;
    MatX WD1V = gauss_weights_.asDiagonal() * D1V_;
    H_u_u_ = D1U_.transpose() * WD1U;
    H_v_v_ = D1V_.transpose() * WD1V;

    // Symmetrize
    H_ = 0.5 * (H_ + H_.transpose());
    H_uuu_uuu_ = 0.5 * (H_uuu_uuu_ + H_uuu_uuu_.transpose());
    H_uuv_uuv_ = 0.5 * (H_uuv_uuv_ + H_uuv_uuv_.transpose());
    H_uvv_uvv_ = 0.5 * (H_uvv_uvv_ + H_uvv_uvv_.transpose());
    H_vvv_vvv_ = 0.5 * (H_vvv_vvv_ + H_vvv_vvv_.transpose());
    H_u_u_ = 0.5 * (H_u_u_ + H_u_u_.transpose());
    H_v_v_ = 0.5 * (H_v_v_ + H_v_v_.transpose());
}

Real TriharmonicHessian::energy(const VecX &coeffs) const {
    if (coeffs.size() != BezierBasis2D::NDOF) {
        throw std::invalid_argument(
            "TriharmonicHessian::energy: coeffs must have 36 elements");
    }
    return coeffs.transpose() * H_ * coeffs;
}

VecX TriharmonicHessian::gradient(const VecX &coeffs) const {
    if (coeffs.size() != BezierBasis2D::NDOF) {
        throw std::invalid_argument(
            "TriharmonicHessian::gradient: coeffs must have 36 elements");
    }
    return 2.0 * H_ * coeffs;
}

MatX TriharmonicHessian::scaled_hessian(Real dx, Real dy) const {
    // Physical scaling for triharmonic energy:
    //
    // E_phys = integral [(z_xxx + z_xyy)^2 + (z_xxy + z_yyy)^2] dx dy
    //
    // With x = dx * u, y = dy * v:
    //   z_xxx = z_uuu / dx^3
    //   z_xxy = z_uuv / (dx^2 * dy)
    //   z_xyy = z_uvv / (dx * dy^2)
    //   z_yyy = z_vvv / dy^3
    //   Jacobian = dx * dy
    //
    // Expanding (z_xxx + z_xyy)^2:
    //   = z_xxx^2 + 2*z_xxx*z_xyy + z_xyy^2
    //   = z_uuu^2/dx^6 + 2*z_uuu*z_uvv/(dx^4*dy^2) + z_uvv^2/(dx^2*dy^4)
    //
    // With Jacobian dx*dy:
    //   z_uuu^2 term: dx*dy/dx^6 = dy/dx^5
    //   z_uuu*z_uvv cross term: 2*dx*dy/(dx^4*dy^2) = 2/(dx^3*dy)
    //   z_uvv^2 term: dx*dy/(dx^2*dy^4) = 1/(dx*dy^3)
    //
    // Expanding (z_xxy + z_yyy)^2:
    //   = z_xxy^2 + 2*z_xxy*z_yyy + z_yyy^2
    //   = z_uuv^2/(dx^4*dy^2) + 2*z_uuv*z_vvv/(dx^2*dy^4) + z_vvv^2/dy^6
    //
    // With Jacobian dx*dy:
    //   z_uuv^2 term: dx*dy/(dx^4*dy^2) = 1/(dx^3*dy)
    //   z_uuv*z_vvv cross term: 2*dx*dy/(dx^2*dy^4) = 2/(dx*dy^3)
    //   z_vvv^2 term: dx*dy/dy^6 = dx/dy^5

    Real dx3 = dx * dx * dx;
    Real dx5 = dx3 * dx * dx;
    Real dy3 = dy * dy * dy;
    Real dy5 = dy3 * dy * dy;

    Real scale_uuu_uuu = dy / dx5;
    Real scale_uvv_uvv = 1.0 / (dx * dy3);
    Real scale_uuu_uvv = 2.0 / (dx3 * dy); // Factor of 2 from expansion

    Real scale_uuv_uuv = 1.0 / (dx3 * dy);
    Real scale_vvv_vvv = dx / dy5;
    Real scale_uuv_vvv = 2.0 / (dx * dy3); // Factor of 2 from expansion

    MatX H_scaled = scale_uuu_uuu * H_uuu_uuu_ + scale_uvv_uvv * H_uvv_uvv_ +
                    scale_uuu_uvv * (H_uuu_uvv_ + H_uuu_uvv_.transpose()) +
                    scale_uuv_uuv * H_uuv_uuv_ + scale_vvv_vvv * H_vvv_vvv_ +
                    scale_uuv_vvv * (H_uuv_vvv_ + H_uuv_vvv_.transpose());

    // Add gradient penalty with proper physical scaling
    // Gradient energy: E_grad = integral[z_x^2 + z_y^2] dx dy
    // With z_x = z_u/dx, z_y = z_v/dy, Jacobian = dx*dy:
    //   E_grad = (dy/dx) * integral[z_u^2] + (dx/dy) * integral[z_v^2]
    if (gradient_weight_ > 0.0) {
        Real scale_u_u = gradient_weight_ * dy / dx;
        Real scale_v_v = gradient_weight_ * dx / dy;
        H_scaled += scale_u_u * H_u_u_ + scale_v_v * H_v_v_;
    }

    // Symmetrize
    return 0.5 * (H_scaled + H_scaled.transpose());
}

} // namespace drifter
