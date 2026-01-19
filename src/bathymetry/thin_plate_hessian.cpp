#include "bathymetry/thin_plate_hessian.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

ThinPlateHessian::ThinPlateHessian(int ngauss, Real gradient_weight)
    : ngauss_(ngauss)
    , gradient_weight_(gradient_weight)
    , basis_(std::make_unique<BezierBasis2D>())
{
    if (ngauss < 3) {
        throw std::invalid_argument("ThinPlateHessian: need at least 3 Gauss points");
    }

    compute_gauss_quadrature();
    build_derivative_matrices();
    build_hessian();
}

void ThinPlateHessian::compute_gauss_quadrature() {
    // Compute Gauss-Legendre nodes and weights on [-1, 1], then map to [0, 1]
    //
    // For common orders, use precomputed values for accuracy
    // Otherwise compute using Newton's method on Legendre polynomials

    gauss_nodes_.resize(ngauss_);
    VecX weights_ref(ngauss_);  // Weights on [-1, 1]

    // Gauss-Legendre nodes and weights for common orders
    if (ngauss_ == 3) {
        gauss_nodes_ << 0.5 - std::sqrt(3.0/5.0)/2.0,
                        0.5,
                        0.5 + std::sqrt(3.0/5.0)/2.0;
        weights_ref << 5.0/9.0, 8.0/9.0, 5.0/9.0;
    } else if (ngauss_ == 4) {
        Real a = std::sqrt(3.0/7.0 - 2.0/7.0 * std::sqrt(6.0/5.0));
        Real b = std::sqrt(3.0/7.0 + 2.0/7.0 * std::sqrt(6.0/5.0));
        Real wa = (18.0 + std::sqrt(30.0)) / 36.0;
        Real wb = (18.0 - std::sqrt(30.0)) / 36.0;
        gauss_nodes_ << (1.0 - b)/2.0, (1.0 - a)/2.0, (1.0 + a)/2.0, (1.0 + b)/2.0;
        weights_ref << wb, wa, wa, wb;
    } else if (ngauss_ == 5) {
        Real a = std::sqrt(5.0 - 2.0*std::sqrt(10.0/7.0)) / 3.0;
        Real b = std::sqrt(5.0 + 2.0*std::sqrt(10.0/7.0)) / 3.0;
        Real wa = (322.0 + 13.0*std::sqrt(70.0)) / 900.0;
        Real wb = (322.0 - 13.0*std::sqrt(70.0)) / 900.0;
        gauss_nodes_ << (1.0 - b)/2.0, (1.0 - a)/2.0, 0.5, (1.0 + a)/2.0, (1.0 + b)/2.0;
        weights_ref << wb, wa, 128.0/225.0, wa, wb;
    } else if (ngauss_ == 6) {
        // 6-point Gauss-Legendre on [-1, 1]
        Real x1 = 0.6612093864662645;
        Real x2 = 0.2386191860831969;
        Real x3 = 0.9324695142031521;
        Real w1 = 0.3607615730481386;
        Real w2 = 0.4679139345726910;
        Real w3 = 0.1713244923791704;

        // Map to [0, 1]: u = (x + 1) / 2
        gauss_nodes_ << (1.0 - x3)/2.0, (1.0 - x1)/2.0, (1.0 - x2)/2.0,
                        (1.0 + x2)/2.0, (1.0 + x1)/2.0, (1.0 + x3)/2.0;
        weights_ref << w3, w1, w2, w2, w1, w3;
    } else {
        // Generic computation using Newton iteration on Legendre polynomials
        // (For production, would use a more robust library routine)
        throw std::invalid_argument("ThinPlateHessian: only ngauss 3-6 supported");
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

void ThinPlateHessian::build_derivative_matrices() {
    // Build matrices D1U, D1V, D2U, D2V, D2UV where:
    // D1U[m, k] = d B_k / du evaluated at Gauss point m
    // D2U[m, k] = d^2 B_k / du^2 evaluated at Gauss point m
    // m = i + ngauss * j indexes the 2D Gauss points
    // k = 0..35 indexes the basis functions

    int nquad = ngauss_ * ngauss_;
    int ndof = BezierBasis2D::NDOF;

    D1U_.resize(nquad, ndof);
    D1V_.resize(nquad, ndof);
    D2U_.resize(nquad, ndof);
    D2V_.resize(nquad, ndof);
    D2UV_.resize(nquad, ndof);

    for (int qj = 0; qj < ngauss_; ++qj) {
        for (int qi = 0; qi < ngauss_; ++qi) {
            int qidx = qi + ngauss_ * qj;
            Real u = gauss_nodes_(qi);
            Real v = gauss_nodes_(qj);

            // First derivatives
            VecX du = basis_->evaluate_du(u, v);
            VecX dv = basis_->evaluate_dv(u, v);

            // Second derivatives
            VecX d2u, d2v, d2uv;
            basis_->evaluate_second_derivatives(u, v, d2u, d2v, d2uv);

            for (int k = 0; k < ndof; ++k) {
                D1U_(qidx, k) = du(k);
                D1V_(qidx, k) = dv(k);
                D2U_(qidx, k) = d2u(k);
                D2V_(qidx, k) = d2v(k);
                D2UV_(qidx, k) = d2uv(k);
            }
        }
    }
}

void ThinPlateHessian::build_hessian() {
    // Combined thin plate + gradient energy on [0,1]^2:
    //   E = integral [(z_uu + z_vv)^2 + 2*z_uv^2] du dv   (thin plate)
    //     + alpha * integral [z_u^2 + z_v^2] du dv        (gradient penalty)
    //
    // For z(u,v) = sum_k c_k * B_k(u,v):
    //   z_u = D1U * c, z_v = D1V * c   (first derivatives)
    //   z_uu = D2U * c, z_vv = D2V * c, z_uv = D2UV * c  (second derivatives)
    //
    // Thin plate Hessian:
    //   Let S = D2U + D2V (Laplacian)
    //   H_tp = S^T * W * S + 2 * D2UV^T * W * D2UV
    //
    // Gradient Hessian components (for scaled_hessian):
    //   H_u_u = D1U^T * W * D1U
    //   H_v_v = D1V^T * W * D1V

    int ndof = BezierBasis2D::NDOF;

    // Thin plate energy
    MatX S = D2U_ + D2V_;
    MatX WS = gauss_weights_.asDiagonal() * S;
    MatX WDUV = gauss_weights_.asDiagonal() * D2UV_;
    H_ = S.transpose() * WS + 2.0 * D2UV_.transpose() * WDUV;

    // Gradient energy components (precompute for scaled_hessian)
    MatX WD1U = gauss_weights_.asDiagonal() * D1U_;
    MatX WD1V = gauss_weights_.asDiagonal() * D1V_;
    H_u_u_ = D1U_.transpose() * WD1U;
    H_v_v_ = D1V_.transpose() * WD1V;

    // Symmetrize (numerical precision)
    H_ = 0.5 * (H_ + H_.transpose());
    H_u_u_ = 0.5 * (H_u_u_ + H_u_u_.transpose());
    H_v_v_ = 0.5 * (H_v_v_ + H_v_v_.transpose());
}

Real ThinPlateHessian::energy(const VecX& coeffs) const {
    if (coeffs.size() != BezierBasis2D::NDOF) {
        throw std::invalid_argument("ThinPlateHessian::energy: coeffs must have 36 elements");
    }
    return coeffs.transpose() * H_ * coeffs;
}

VecX ThinPlateHessian::gradient(const VecX& coeffs) const {
    if (coeffs.size() != BezierBasis2D::NDOF) {
        throw std::invalid_argument("ThinPlateHessian::gradient: coeffs must have 36 elements");
    }
    // Gradient of x^T H x is 2 H x (since H is symmetric)
    return 2.0 * H_ * coeffs;
}

MatX ThinPlateHessian::scaled_hessian(Real dx, Real dy) const {
    // For a physical element with dimensions (dx, dy), the thin plate energy is:
    //
    // E_phys = integral_{physical} [(z_xx + z_yy)^2 + 2*z_xy^2] dx dy
    //
    // With x = dx * u, y = dy * v (mapping from [0,1]^2 to physical):
    //   z_x = z_u / dx
    //   z_xx = z_uu / dx^2
    //   z_yy = z_vv / dy^2
    //   z_xy = z_uv / (dx * dy)
    //   dx dy = dx * dy (Jacobian)
    //
    // E_phys = integral_{[0,1]^2} [(z_uu/dx^2 + z_vv/dy^2)^2 + 2*(z_uv/(dx*dy))^2] * dx*dy du dv
    //        = dx*dy * integral [(z_uu/dx^2 + z_vv/dy^2)^2 + 2*(z_uv)^2/(dx*dy)^2] du dv
    //
    // Let's expand:
    //   (z_uu/dx^2 + z_vv/dy^2)^2 = z_uu^2/dx^4 + 2*z_uu*z_vv/(dx^2*dy^2) + z_vv^2/dy^4
    //
    // E_phys = dx*dy * [ (1/dx^4)*integral[z_uu^2]
    //                  + 2/(dx^2*dy^2)*integral[z_uu*z_vv]
    //                  + (1/dy^4)*integral[z_vv^2]
    //                  + 2/(dx^2*dy^2)*integral[z_uv^2] ]
    //
    // The reference Hessian H is built as:
    //   H_ref = S^T * W * S + 2 * D2UV^T * W * D2UV   where S = D2U + D2V
    //         = H_uu_uu + H_vv_vv + (H_uu_vv + H_uu_vv^T) + 2*H_uv_uv
    //
    // Note: The cross term (H_uu_vv + H_uu_vv^T) has coefficient 1, not 2.
    // The factor of 2 in "2*z_uu*z_vv" from expanding (z_uu + z_vv)^2 is
    // absorbed into the symmetrization: 2*c^T*H_uu_vv*c = c^T*(H_uu_vv+H_uu_vv^T)*c
    //
    // For physical scaling, we need:
    //   H_phys = (dy/dx^3) * H_uu_uu + (dx/dy^3) * H_vv_vv
    //          + (1/(dx*dy)) * (H_uu_vv + H_uu_vv^T) + (2/(dx*dy)) * H_uv_uv
    //
    // To implement this, we need separate component matrices.
    // Let's build them directly:

    int nquad = ngauss_ * ngauss_;
    int ndof = BezierBasis2D::NDOF;

    // Build weighted component matrices
    MatX WD2U = gauss_weights_.asDiagonal() * D2U_;
    MatX WD2V = gauss_weights_.asDiagonal() * D2V_;
    MatX WD2UV = gauss_weights_.asDiagonal() * D2UV_;

    // Component Hessians:
    // H_uu_uu = D2U^T * W * D2U  (integral of z_uu^2)
    // H_vv_vv = D2V^T * W * D2V  (integral of z_vv^2)
    // H_uu_vv = D2U^T * W * D2V  (integral of z_uu * z_vv)
    // H_uv_uv = D2UV^T * W * D2UV (integral of z_uv^2)

    MatX H_uu_uu = D2U_.transpose() * WD2U;
    MatX H_vv_vv = D2V_.transpose() * WD2V;
    MatX H_uu_vv = D2U_.transpose() * WD2V;
    MatX H_uv_uv = D2UV_.transpose() * WD2UV;

    // Scale factors (the factor of 2 for cross terms is in the symmetrization):
    // For z_uu^2: scale by dx*dy/dx^4 = dy/dx^3
    // For z_vv^2: scale by dx*dy/dy^4 = dx/dy^3
    // For z_uu*z_vv: scale by dx*dy/(dx^2*dy^2) = 1/(dx*dy)
    // For z_uv^2: scale by 2*dx*dy/(dx*dy)^2 = 2/(dx*dy)

    Real dx3 = dx * dx * dx;
    Real dy3 = dy * dy * dy;

    Real scale_uu_uu = dy / dx3;
    Real scale_vv_vv = dx / dy3;
    Real scale_uu_vv = 1.0 / (dx * dy);
    Real scale_uv_uv = 2.0 / (dx * dy);

    MatX H_scaled = scale_uu_uu * H_uu_uu
                  + scale_vv_vv * H_vv_vv
                  + scale_uu_vv * (H_uu_vv + H_uu_vv.transpose())
                  + scale_uv_uv * H_uv_uv;

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

}  // namespace drifter
