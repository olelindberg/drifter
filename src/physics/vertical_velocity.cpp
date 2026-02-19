#include "physics/vertical_velocity.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

VerticalVelocityDiagnosis::VerticalVelocityDiagnosis(const HexahedronBasis &basis,
                                                     const GaussQuadrature3D &quad)
    : basis_(basis), quad_(quad), n_vert_(basis.order() + 1),
      n_horiz_((basis.order() + 1) * (basis.order() + 1)) {
    build_vertical_integration();
}

void VerticalVelocityDiagnosis::build_vertical_integration() {
    // Get 1D LGL nodes and weights for vertical direction
    const VecX &zeta_nodes = basis_.lgl_basis_1d().nodes;
    const VecX &zeta_weights = basis_.lgl_basis_1d().weights;

    // Map from reference [-1, 1] to sigma [-1, 0]
    // sigma = 0.5 * (zeta - 1), so dsigma = 0.5 * dzeta
    // Weights are scaled by 0.5

    vertical_integration_weights_.resize(n_vert_);
    for (int k = 0; k < n_vert_; ++k) {
        vertical_integration_weights_(k) = 0.5 * zeta_weights(k);
    }

    // Build integration matrix I(k, j) = integral from sigma(-1) to sigma(k) of
    // L_j(sigma) dsigma where L_j is the j-th Lagrange basis function at LGL
    // nodes
    vertical_integration_matrix_.resize(n_vert_, n_vert_);
    vertical_integration_matrix_.setZero();

    // For each target level k, integrate from bottom (level 0) to level k
    // Using Lagrange interpolation and quadrature
    for (int k = 0; k < n_vert_; ++k) {
        // Contribution from each source level j
        for (int j = 0; j <= k; ++j) {
            // Approximate integral using quadrature
            // For simplicity, use the fact that integral of L_j from bottom to
            // sigma_k can be computed analytically or via accumulating weights

            // Simple approach: cumulative sum of weighted basis evaluations
            // I(k, j) = sum_{l=0}^{k} w_l * L_j(sigma_l)
            // For Lagrange basis: L_j(sigma_l) = delta_{jl}
            // So I(k, j) = sum_{l=0}^{k} w_l * delta_{jl} = w_j if j <= k, else 0

            if (j <= k) {
                vertical_integration_matrix_(k, j) = vertical_integration_weights_(j);
            }
        }
    }
}

void VerticalVelocityDiagnosis::diagnose_omega(const VecX &div_Hu, const VecX &H,
                                               VecX &omega) const {
    const int n_total = n_horiz_ * n_vert_;
    if (div_Hu.size() != n_total) {
        throw std::invalid_argument("div_Hu size mismatch");
    }

    omega.resize(n_total);

    // For each horizontal column, integrate vertically
    for (int i = 0; i < n_horiz_; ++i) {
        // Bottom BC: omega(-1) = 0
        omega(node_index_3d(i, 0)) = 0.0;

        // Integrate upward
        // omega(k) = omega(k-1) - (1/H) * integral_{sigma(k-1)}^{sigma(k)} div_Hu
        // dsigma
        Real integral = 0.0;
        for (int k = 1; k < n_vert_; ++k) {
            // Accumulate integral using quadrature weight
            // This is the contribution from level k-1 to k
            const Index idx_prev = node_index_3d(i, k - 1);
            const Index idx_curr = node_index_3d(i, k);

            // Average H for this layer
            const Real H_avg = 0.5 * (H(idx_prev) + H(idx_curr));

            // Trapezoidal rule for this layer
            const Real dsigma =
                (basis_.lgl_basis_1d().nodes(k) - basis_.lgl_basis_1d().nodes(k - 1)) * 0.5;
            const Real div_avg = 0.5 * (div_Hu(idx_prev) + div_Hu(idx_curr));

            integral += div_avg * dsigma;

            // omega = -(1/H) * integral
            if (H_avg > 1e-10) {
                omega(idx_curr) = -integral / H_avg;
            } else {
                omega(idx_curr) = 0.0;
            }
        }
    }
}

void VerticalVelocityDiagnosis::diagnose_omega_from_velocity(const VecX &Hu, const VecX &Hv,
                                                             const VecX &H, const VecX &dHu_dx,
                                                             const VecX &dHv_dy,
                                                             VecX &omega) const {
    // Compute horizontal divergence
    VecX div_Hu = dHu_dx + dHv_dy;

    // Diagnose omega from divergence
    diagnose_omega(div_Hu, H, omega);
}

void VerticalVelocityDiagnosis::omega_to_physical_w(const VecX &omega, const VecX &u, const VecX &v,
                                                    const VecX &H, const VecX &dz_dx,
                                                    const VecX &dz_dy, const VecX &dz_dt,
                                                    VecX &w) const {
    const int n = static_cast<int>(omega.size());
    w.resize(n);

    // w = H*omega + u*dz/dx|_sigma + v*dz/dy|_sigma + dz/dt|_sigma
    for (int i = 0; i < n; ++i) {
        w(i) = H(i) * omega(i) + u(i) * dz_dx(i) + v(i) * dz_dy(i) + dz_dt(i);
    }
}

void VerticalVelocityDiagnosis::omega_to_w_simple(const VecX &omega, const VecX &u, const VecX &v,
                                                  const VecX &eta, const VecX &h, const VecX &sigma,
                                                  const VecX &deta_dx, const VecX &deta_dy,
                                                  const VecX &deta_dt, VecX &w) const {
    const int n = static_cast<int>(omega.size());
    w.resize(n);

    for (int i = 0; i < n; ++i) {
        const Real H = eta(i) + h(i);

        // dz/dx|_sigma = deta/dx + sigma * dH/dx
        // For fixed bathymetry: dH/dx = deta/dx, so dz/dx = deta/dx * (1 + sigma)
        const Real dz_dx = deta_dx(i) * (1.0 + sigma(i));
        const Real dz_dy = deta_dy(i) * (1.0 + sigma(i));

        // dz/dt|_sigma = deta/dt * (1 + sigma)
        const Real dz_dt = deta_dt(i) * (1.0 + sigma(i));

        w(i) = H * omega(i) + u(i) * dz_dx + v(i) * dz_dy + dz_dt;
    }
}

Real VerticalVelocityDiagnosis::surface_omega(Real deta_dt, Real u_surf, Real v_surf, Real deta_dx,
                                              Real deta_dy, Real H) {
    if (H < 1e-10)
        return 0.0;
    // D(eta)/Dt = deta/dt + u*deta/dx + v*deta/dy
    // omega(0) = D(eta)/Dt / H
    return (deta_dt + u_surf * deta_dx + v_surf * deta_dy) / H;
}

Real VerticalVelocityDiagnosis::bottom_omega(Real u_bot, Real v_bot, Real dh_dx, Real dh_dy) {
    // For sigma-following coordinates with sigma(-1) at bottom,
    // the bottom is always at sigma = -1, so omega(-1) = 0 by definition.
    // The physical no-flow BC is automatically satisfied.
    return 0.0;
}

Real VerticalVelocityDiagnosis::continuity_residual(const VecX &div_Hu, const VecX &H,
                                                    const VecX &omega,
                                                    const VecX &dHomega_dsigma) const {
    // Residual = div_h(Hu, Hv) + d(H*omega)/dsigma
    const int n = static_cast<int>(div_Hu.size());
    Real residual_sq = 0.0;

    for (int i = 0; i < n; ++i) {
        Real res = div_Hu(i) + dHomega_dsigma(i);
        residual_sq += res * res;
    }

    return std::sqrt(residual_sq / n);
}

// ============================================================================
// ColumnVerticalVelocity implementation
// ============================================================================

ColumnVerticalVelocity::ColumnVerticalVelocity(int order) : order_(order), n_levels_(order + 1) {
    // Generate LGL nodes for reference interval [-1, 1]
    // Then map to sigma [-1, 0]
    sigma_nodes_.resize(n_levels_);
    weights_.resize(n_levels_);

    // For LGL nodes, use the Gauss-Lobatto points
    // These include the endpoints
    if (n_levels_ == 1) {
        sigma_nodes_(0) = -0.5;
        weights_(0) = 1.0;
    } else if (n_levels_ == 2) {
        sigma_nodes_(0) = -1.0;
        sigma_nodes_(1) = 0.0;
        weights_(0) = 0.5;
        weights_(1) = 0.5;
    } else {
        // Use Legendre-Gauss-Lobatto nodes
        // These are roots of (1-x^2)*P'_n(x) = 0 where P_n is Legendre polynomial
        // Map from [-1, 1] to [-1, 0]: sigma = 0.5*(zeta - 1)

        // For simplicity, use pre-computed nodes for small orders
        // A production code would compute these numerically

        VecX zeta_nodes(n_levels_);
        VecX zeta_weights(n_levels_);

        if (n_levels_ == 3) {
            zeta_nodes << -1.0, 0.0, 1.0;
            zeta_weights << 1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0;
        } else if (n_levels_ == 4) {
            zeta_nodes << -1.0, -std::sqrt(1.0 / 5.0), std::sqrt(1.0 / 5.0), 1.0;
            zeta_weights << 1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0;
        } else if (n_levels_ == 5) {
            zeta_nodes << -1.0, -std::sqrt(3.0 / 7.0), 0.0, std::sqrt(3.0 / 7.0), 1.0;
            zeta_weights << 1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0;
        } else {
            // For higher orders, compute numerically or use a table
            // Here we just use uniform spacing as a fallback
            for (int k = 0; k < n_levels_; ++k) {
                zeta_nodes(k) = -1.0 + 2.0 * k / (n_levels_ - 1);
                zeta_weights(k) = 2.0 / (n_levels_ - 1);
            }
            zeta_weights(0) *= 0.5;
            zeta_weights(n_levels_ - 1) *= 0.5;
        }

        // Map to sigma space [-1, 0]
        for (int k = 0; k < n_levels_; ++k) {
            sigma_nodes_(k) = 0.5 * (zeta_nodes(k) - 1.0);
            weights_(k) = 0.5 * zeta_weights(k); // dsigma = 0.5 * dzeta
        }
    }

    // Build integration matrix
    // I(k, j) = cumulative integral weight for point j up to level k
    integration_matrix_.resize(n_levels_, n_levels_);
    integration_matrix_.setZero();

    for (int k = 0; k < n_levels_; ++k) {
        for (int j = 0; j <= k; ++j) {
            integration_matrix_(k, j) = weights_(j);
        }
    }
}

void ColumnVerticalVelocity::diagnose_column(const VecX &div_Hu_col, const VecX &H_col,
                                             Real omega_bottom, VecX &omega_col) const {
    if (div_Hu_col.size() != n_levels_) {
        throw std::invalid_argument("Column size mismatch");
    }

    omega_col.resize(n_levels_);
    omega_col(0) = omega_bottom;

    // Integrate from bottom to each level
    // omega(k) = omega(0) - (1/H) * integral_{-1}^{sigma_k} div_Hu dsigma

    Real integral = 0.0;
    for (int k = 1; k < n_levels_; ++k) {
        // Trapezoidal rule for this segment
        const Real dsigma = sigma_nodes_(k) - sigma_nodes_(k - 1);
        const Real div_avg = 0.5 * (div_Hu_col(k - 1) + div_Hu_col(k));
        integral += div_avg * dsigma;

        // Average H for this layer
        const Real H_avg = 0.5 * (H_col(k - 1) + H_col(k));

        if (H_avg > 1e-10) {
            omega_col(k) = omega_bottom - integral / H_avg;
        } else {
            omega_col(k) = omega_bottom;
        }
    }
}

} // namespace drifter
