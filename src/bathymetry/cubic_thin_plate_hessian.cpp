#include "bathymetry/cubic_thin_plate_hessian.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

CubicThinPlateHessian::CubicThinPlateHessian(int ngauss)
    : ngauss_(ngauss), basis_(std::make_unique<CubicBezierBasis2D>()) {
    if (ngauss < 2) {
        throw std::invalid_argument("CubicThinPlateHessian: need at least 2 Gauss points");
    }

    compute_gauss_quadrature();
    build_derivative_matrices();
    build_hessian();
}

void CubicThinPlateHessian::compute_gauss_quadrature() {
    // Compute Gauss-Legendre nodes and weights on [-1, 1], then map to [0, 1]
    gauss_nodes_.resize(ngauss_);
    VecX weights_ref(ngauss_); // Weights on [-1, 1]

    // Gauss-Legendre nodes and weights for common orders
    if (ngauss_ == 2) {
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
    } else if (ngauss_ == 5) {
        Real a = std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 3.0;
        Real b = std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 3.0;
        Real wa = (322.0 + 13.0 * std::sqrt(70.0)) / 900.0;
        Real wb = (322.0 - 13.0 * std::sqrt(70.0)) / 900.0;
        gauss_nodes_ << (1.0 - b) / 2.0, (1.0 - a) / 2.0, 0.5, (1.0 + a) / 2.0, (1.0 + b) / 2.0;
        weights_ref << wb, wa, 128.0 / 225.0, wa, wb;
    } else if (ngauss_ == 6) {
        Real x1 = 0.6612093864662645;
        Real x2 = 0.2386191860831969;
        Real x3 = 0.9324695142031521;
        Real w1 = 0.3607615730481386;
        Real w2 = 0.4679139345726910;
        Real w3 = 0.1713244923791704;

        gauss_nodes_ << (1.0 - x3) / 2.0, (1.0 - x1) / 2.0, (1.0 - x2) / 2.0, (1.0 + x2) / 2.0,
            (1.0 + x1) / 2.0, (1.0 + x3) / 2.0;
        weights_ref << w3, w1, w2, w2, w1, w3;
    } else {
        throw std::invalid_argument("CubicThinPlateHessian: only ngauss 2-6 supported");
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

void CubicThinPlateHessian::build_derivative_matrices() {
    int nquad = ngauss_ * ngauss_;
    int ndof = CubicBezierBasis2D::NDOF;

    D2U_.resize(nquad, ndof);
    D2V_.resize(nquad, ndof);
    D2UV_.resize(nquad, ndof);

    for (int qj = 0; qj < ngauss_; ++qj) {
        for (int qi = 0; qi < ngauss_; ++qi) {
            int qidx = qi + ngauss_ * qj;
            Real u = gauss_nodes_(qi);
            Real v = gauss_nodes_(qj);

            // Second derivatives
            VecX d2u, d2v, d2uv;
            basis_->evaluate_second_derivatives(u, v, d2u, d2v, d2uv);

            for (int k = 0; k < ndof; ++k) {
                D2U_(qidx, k) = d2u(k);
                D2V_(qidx, k) = d2v(k);
                D2UV_(qidx, k) = d2uv(k);
            }
        }
    }
}

void CubicThinPlateHessian::build_hessian() {
    // Thin plate energy on [0,1]^2:
    //   E = integral [(z_uu + z_vv)^2 + 2*z_uv^2] du dv
    //
    // Thin plate Hessian:
    //   Let S = D2U + D2V (Laplacian)
    //   H = S^T * W * S + 2 * D2UV^T * W * D2UV

    // Thin plate energy
    MatX S = D2U_ + D2V_;
    MatX WS = gauss_weights_.asDiagonal() * S;
    MatX WDUV = gauss_weights_.asDiagonal() * D2UV_;
    H_ = S.transpose() * WS + 2.0 * D2UV_.transpose() * WDUV;

    // Symmetrize (numerical precision)
    H_ = 0.5 * (H_ + H_.transpose());
}

Real CubicThinPlateHessian::energy(const VecX &coeffs) const {
    if (coeffs.size() != CubicBezierBasis2D::NDOF) {
        throw std::invalid_argument("CubicThinPlateHessian::energy: coeffs must have 16 elements");
    }
    return coeffs.transpose() * H_ * coeffs;
}

VecX CubicThinPlateHessian::gradient(const VecX &coeffs) const {
    if (coeffs.size() != CubicBezierBasis2D::NDOF) {
        throw std::invalid_argument(
            "CubicThinPlateHessian::gradient: coeffs must have 16 elements");
    }
    return 2.0 * H_ * coeffs;
}

MatX CubicThinPlateHessian::scaled_hessian(Real dx, Real dy) const {
    // For a physical element with dimensions (dx, dy), the thin plate energy
    // is:
    //
    // E_phys = integral_{physical} [(z_xx + z_yy)^2 + 2*z_xy^2] dx dy
    //
    // Component Hessians with proper scaling:
    // H_uu_uu: scale by dy/dx^3
    // H_vv_vv: scale by dx/dy^3
    // H_uu_vv: scale by 1/(dx*dy) (cross term)
    // H_uv_uv: scale by 2/(dx*dy)

    // Build weighted component matrices
    MatX WD2U = gauss_weights_.asDiagonal() * D2U_;
    MatX WD2V = gauss_weights_.asDiagonal() * D2V_;
    MatX WD2UV = gauss_weights_.asDiagonal() * D2UV_;

    MatX H_uu_uu = D2U_.transpose() * WD2U;
    MatX H_vv_vv = D2V_.transpose() * WD2V;
    MatX H_uu_vv = D2U_.transpose() * WD2V;
    MatX H_uv_uv = D2UV_.transpose() * WD2UV;

    Real dx3 = dx * dx * dx;
    Real dy3 = dy * dy * dy;

    Real scale_uu_uu = dy / dx3;
    Real scale_vv_vv = dx / dy3;
    Real scale_uu_vv = 1.0 / (dx * dy);
    Real scale_uv_uv = 2.0 / (dx * dy);

    MatX H_scaled = scale_uu_uu * H_uu_uu + scale_vv_vv * H_vv_vv +
                    scale_uu_vv * (H_uu_vv + H_uu_vv.transpose()) + scale_uv_uv * H_uv_uv;

    // Symmetrize
    return 0.5 * (H_scaled + H_scaled.transpose());
}

} // namespace drifter
