#include "bathymetry/bezier_basis_2d.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

BezierBasis2D::BezierBasis2D() { compute_binomial_coefficients(); }

void BezierBasis2D::compute_binomial_coefficients() {
    // Initialize Pascal's triangle for binomial coefficients
    for (int n = 0; n <= 2 * DEGREE; ++n) {
        for (int k = 0; k <= n; ++k) {
            if (k == 0 || k == n) {
                binomial_[n][k] = 1.0;
            } else {
                binomial_[n][k] = binomial_[n - 1][k - 1] + binomial_[n - 1][k];
            }
        }
        // Zero out unused entries
        for (int k = n + 1; k <= 2 * DEGREE; ++k) {
            binomial_[n][k] = 0.0;
        }
    }
}

Real BezierBasis2D::binom(int n, int k) const {
    if (k < 0 || k > n || n < 0 || n > 2 * DEGREE)
        return 0.0;
    return binomial_[n][k];
}

Vec2 BezierBasis2D::control_point_position(int dof) const {
    int i, j;
    dof_ij(dof, i, j);
    return Vec2(static_cast<Real>(i) / DEGREE, static_cast<Real>(j) / DEGREE);
}

// =============================================================================
// 1D Bernstein basis evaluation
// =============================================================================

VecX BezierBasis2D::evaluate_bernstein_1d(int n, Real t) const {
    // Use de Casteljau-like recurrence for numerical stability
    // B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
    //
    // Recurrence: B_{i,n}(t) = (1-t)*B_{i,n-1}(t) + t*B_{i-1,n-1}(t)
    // with B_{0,0} = 1

    VecX B(n + 1);

    if (n == 0) {
        B(0) = 1.0;
        return B;
    }

    Real one_minus_t = 1.0 - t;

    // Start with degree 0: B_{0,0} = 1
    std::vector<Real> prev(1, 1.0);

    // Build up to degree n
    for (int deg = 1; deg <= n; ++deg) {
        std::vector<Real> curr(deg + 1);

        // B_{0,d} = (1-t) * B_{0,d-1}
        curr[0] = one_minus_t * prev[0];

        // B_{i,d} = t * B_{i-1,d-1} + (1-t) * B_{i,d-1}
        for (int i = 1; i < deg; ++i) {
            curr[i] = t * prev[i - 1] + one_minus_t * prev[i];
        }

        // B_{d,d} = t * B_{d-1,d-1}
        curr[deg] = t * prev[deg - 1];

        prev = std::move(curr);
    }

    for (int i = 0; i <= n; ++i) {
        B(i) = prev[i];
    }

    return B;
}

VecX BezierBasis2D::evaluate_bernstein_derivative_1d(
    int n, Real t, int k) const {
    // Use the relation: d/dt B_{i,n}(t) = n * [B_{i-1,n-1}(t) - B_{i,n-1}(t)]
    // Applied recursively for higher derivatives.
    //
    // For k-th derivative, we use the relation:
    // d^k B_{i,n}/dt^k = (n!/(n-k)!) * Delta^k c_i evaluated at degree n-k
    // where Delta is the forward difference operator.
    //
    // Concretely:
    // d^k B_{i,n}/dt^k = C(n,k) * k! * sum_{j=0}^{k} (-1)^{k-j} C(k,j)
    // B_{i+j-k,n-k}(t)
    //
    // But it's cleaner to use recursive application of first derivative.

    VecX dB(n + 1);

    if (k > n) {
        dB.setZero();
        return dB;
    }

    if (k == 0) {
        return evaluate_bernstein_1d(n, t);
    }

    // For first derivative: d/dt B_{i,n}(t) = n * [B_{i-1,n-1}(t) -
    // B_{i,n-1}(t)]
    if (k == 1) {
        VecX B_lower = evaluate_bernstein_1d(n - 1, t);
        for (int i = 0; i <= n; ++i) {
            Real b_im1 = (i > 0) ? B_lower(i - 1) : 0.0;
            Real b_i = (i < n) ? B_lower(i) : 0.0;
            dB(i) = n * (b_im1 - b_i);
        }
        return dB;
    }

    // For higher derivatives, apply recursively
    // d^k/dt^k B_{i,n} = d/dt [d^{k-1}/dt^{k-1} B_{i,n}]
    // But d^{k-1} B_{i,n} is a linear combination of
    // B_{i-k+1,n-k+1}...B_{i,n-k+1}
    //
    // More efficient: use the direct formula
    // d^k B_{i,n}/dt^k = n*(n-1)*...*(n-k+1) * sum_{j=0}^{k} (-1)^{k-j} C(k,j)
    // B_{i-k+j,n-k}(t)

    Real factor = 1.0;
    for (int m = 0; m < k; ++m) {
        factor *= (n - m);
    }

    VecX B_lower = evaluate_bernstein_1d(n - k, t);

    for (int i = 0; i <= n; ++i) {
        Real sum = 0.0;
        for (int j = 0; j <= k; ++j) {
            int idx = i - k + j;
            if (idx >= 0 && idx <= n - k) {
                Real sign = ((k - j) % 2 == 0) ? 1.0 : -1.0;
                sum += sign * binom(k, j) * B_lower(idx);
            }
        }
        dB(i) = factor * sum;
    }

    return dB;
}

// =============================================================================
// 2D basis evaluation
// =============================================================================

VecX BezierBasis2D::evaluate(Real u, Real v) const {
    VecX Bu = evaluate_bernstein_1d(DEGREE, u);
    VecX Bv = evaluate_bernstein_1d(DEGREE, v);

    VecX phi(NDOF);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            phi(dof_index(i, j)) = Bu(i) * Bv(j);
        }
    }
    return phi;
}

VecX BezierBasis2D::evaluate_du(Real u, Real v) const {
    VecX dBu = evaluate_bernstein_derivative_1d(DEGREE, u, 1);
    VecX Bv = evaluate_bernstein_1d(DEGREE, v);

    VecX dphi(NDOF);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            dphi(dof_index(i, j)) = dBu(i) * Bv(j);
        }
    }
    return dphi;
}

VecX BezierBasis2D::evaluate_dv(Real u, Real v) const {
    VecX Bu = evaluate_bernstein_1d(DEGREE, u);
    VecX dBv = evaluate_bernstein_derivative_1d(DEGREE, v, 1);

    VecX dphi(NDOF);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            dphi(dof_index(i, j)) = Bu(i) * dBv(j);
        }
    }
    return dphi;
}

MatX BezierBasis2D::evaluate_gradient(Real u, Real v) const {
    VecX Bu = evaluate_bernstein_1d(DEGREE, u);
    VecX Bv = evaluate_bernstein_1d(DEGREE, v);
    VecX dBu = evaluate_bernstein_derivative_1d(DEGREE, u, 1);
    VecX dBv = evaluate_bernstein_derivative_1d(DEGREE, v, 1);

    MatX grad(NDOF, 2);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            int dof = dof_index(i, j);
            grad(dof, 0) = dBu(i) * Bv(j); // d/du
            grad(dof, 1) = Bu(i) * dBv(j); // d/dv
        }
    }
    return grad;
}

// =============================================================================
// Second derivatives
// =============================================================================

VecX BezierBasis2D::evaluate_d2u(Real u, Real v) const {
    VecX d2Bu = evaluate_bernstein_derivative_1d(DEGREE, u, 2);
    VecX Bv = evaluate_bernstein_1d(DEGREE, v);

    VecX d2phi(NDOF);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            d2phi(dof_index(i, j)) = d2Bu(i) * Bv(j);
        }
    }
    return d2phi;
}

VecX BezierBasis2D::evaluate_d2v(Real u, Real v) const {
    VecX Bu = evaluate_bernstein_1d(DEGREE, u);
    VecX d2Bv = evaluate_bernstein_derivative_1d(DEGREE, v, 2);

    VecX d2phi(NDOF);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            d2phi(dof_index(i, j)) = Bu(i) * d2Bv(j);
        }
    }
    return d2phi;
}

VecX BezierBasis2D::evaluate_d2uv(Real u, Real v) const {
    VecX dBu = evaluate_bernstein_derivative_1d(DEGREE, u, 1);
    VecX dBv = evaluate_bernstein_derivative_1d(DEGREE, v, 1);

    VecX d2phi(NDOF);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            d2phi(dof_index(i, j)) = dBu(i) * dBv(j);
        }
    }
    return d2phi;
}

void BezierBasis2D::evaluate_second_derivatives(
    Real u, Real v, VecX &d2u, VecX &d2v, VecX &d2uv) const {
    VecX Bu = evaluate_bernstein_1d(DEGREE, u);
    VecX Bv = evaluate_bernstein_1d(DEGREE, v);
    VecX dBu = evaluate_bernstein_derivative_1d(DEGREE, u, 1);
    VecX dBv = evaluate_bernstein_derivative_1d(DEGREE, v, 1);
    VecX d2Bu = evaluate_bernstein_derivative_1d(DEGREE, u, 2);
    VecX d2Bv = evaluate_bernstein_derivative_1d(DEGREE, v, 2);

    d2u.resize(NDOF);
    d2v.resize(NDOF);
    d2uv.resize(NDOF);

    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            int dof = dof_index(i, j);
            d2u(dof) = d2Bu(i) * Bv(j);
            d2v(dof) = Bu(i) * d2Bv(j);
            d2uv(dof) = dBu(i) * dBv(j);
        }
    }
}

// =============================================================================
// Higher derivatives (for C^2 constraints)
// =============================================================================

VecX BezierBasis2D::evaluate_derivative(Real u, Real v, int nu, int nv) const {
    VecX dnBu = evaluate_bernstein_derivative_1d(DEGREE, u, nu);
    VecX dnBv = evaluate_bernstein_derivative_1d(DEGREE, v, nv);

    VecX dphi(NDOF);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            dphi(dof_index(i, j)) = dnBu(i) * dnBv(j);
        }
    }
    return dphi;
}

VecX BezierBasis2D::evaluate_d3uuu(Real u, Real v) const {
    return evaluate_derivative(u, v, 3, 0);
}

VecX BezierBasis2D::evaluate_d3uuv(Real u, Real v) const {
    return evaluate_derivative(u, v, 2, 1);
}

VecX BezierBasis2D::evaluate_d3uvv(Real u, Real v) const {
    return evaluate_derivative(u, v, 1, 2);
}

VecX BezierBasis2D::evaluate_d3vvv(Real u, Real v) const {
    return evaluate_derivative(u, v, 0, 3);
}

VecX BezierBasis2D::evaluate_d4uuvv(Real u, Real v) const {
    return evaluate_derivative(u, v, 2, 2);
}

// =============================================================================
// Scalar interpolation using de Casteljau algorithm
// =============================================================================

Real BezierBasis2D::evaluate_scalar(const VecX &coeffs, Real u, Real v) const {
    // 2D de Casteljau algorithm
    // First apply de Casteljau in u direction for each row
    // Then apply de Casteljau in v direction to the results

    const int n = DEGREE;

    // Temporary storage for de Casteljau in u direction
    std::vector<Real> row_values(N1D);

    // Process each row (fixed j) using de Casteljau in u
    for (int j = 0; j < N1D; ++j) {
        // Extract row coefficients
        std::vector<Real> row(N1D);
        for (int i = 0; i < N1D; ++i) {
            row[i] = coeffs(dof_index(i, j));
        }

        // de Casteljau in u direction
        for (int r = 1; r <= n; ++r) {
            for (int i = 0; i <= n - r; ++i) {
                row[i] = (1.0 - u) * row[i] + u * row[i + 1];
            }
        }
        row_values[j] = row[0];
    }

    // de Casteljau in v direction
    for (int r = 1; r <= n; ++r) {
        for (int j = 0; j <= n - r; ++j) {
            row_values[j] = (1.0 - v) * row_values[j] + v * row_values[j + 1];
        }
    }

    return row_values[0];
}

// =============================================================================
// Corner and edge access
// =============================================================================

int BezierBasis2D::corner_dof(int corner_id) const {
    // Corner mapping (in parameter space [0,1]^2):
    // 0: (0, 0) -> i=0, j=0
    // 1: (1, 0) -> i=5, j=0
    // 2: (0, 1) -> i=0, j=5
    // 3: (1, 1) -> i=5, j=5
    switch (corner_id) {
    case 0:
        return dof_index(0, 0);
    case 1:
        return dof_index(DEGREE, 0);
    case 2:
        return dof_index(0, DEGREE);
    case 3:
        return dof_index(DEGREE, DEGREE);
    default:
        throw std::invalid_argument("corner_id must be 0-3");
    }
}

int BezierBasis2D::dof_to_corner(int dof) const {
    // Inverse of corner_dof: given a DOF index, return corner ID or -1 if not a
    // corner Corner DOFs: 0 (corner 0), 5 (corner 2), 30 (corner 1), 35 (corner
    // 3)
    if (dof == dof_index(0, 0))
        return 0; // (0, 0)
    if (dof == dof_index(DEGREE, 0))
        return 1; // (1, 0)
    if (dof == dof_index(0, DEGREE))
        return 2; // (0, 1)
    if (dof == dof_index(DEGREE, DEGREE))
        return 3; // (1, 1)
    return -1;    // Not a corner DOF
}

Vec2 BezierBasis2D::corner_param(int corner_id) const {
    switch (corner_id) {
    case 0:
        return Vec2(0.0, 0.0);
    case 1:
        return Vec2(1.0, 0.0);
    case 2:
        return Vec2(0.0, 1.0);
    case 3:
        return Vec2(1.0, 1.0);
    default:
        throw std::invalid_argument("corner_id must be 0-3");
    }
}

std::vector<int> BezierBasis2D::edge_dofs(int edge_id) const {
    std::vector<int> dofs(N1D);

    switch (edge_id) {
    case 0: // u = 0 (left edge)
        for (int j = 0; j < N1D; ++j) {
            dofs[j] = dof_index(0, j);
        }
        break;
    case 1: // u = 1 (right edge)
        for (int j = 0; j < N1D; ++j) {
            dofs[j] = dof_index(DEGREE, j);
        }
        break;
    case 2: // v = 0 (bottom edge)
        for (int i = 0; i < N1D; ++i) {
            dofs[i] = dof_index(i, 0);
        }
        break;
    case 3: // v = 1 (top edge)
        for (int i = 0; i < N1D; ++i) {
            dofs[i] = dof_index(i, DEGREE);
        }
        break;
    default:
        throw std::invalid_argument("edge_id must be 0-3");
    }

    return dofs;
}

// =============================================================================
// Non-conforming interface support (Bezier subdivision)
// =============================================================================

MatX BezierBasis2D::compute_1d_extraction_matrix(Real t0, Real t1) const {
    const int n = DEGREE; // 5 for quintic
    MatX S = MatX::Zero(N1D, N1D);

    // For the common case of [0, 0.5] and [0.5, 1], use precomputed formulas
    // based on de Casteljau subdivision.
    //
    // Left half [0, 0.5] subdivision matrix:
    //   Row k gives coefficients for control point Q_k of the left sub-curve.
    //   Q_k = sum_{j=0}^{k} C(k,j) / 2^k * P_j
    //
    // Right half [0.5, 1] subdivision matrix:
    //   Row k gives coefficients for control point Q_k of the right sub-curve.
    //   Q_k = sum_{j=k}^{n} C(n-k,j-k) / 2^(n-k) * P_j

    if (std::abs(t0) < 1e-14 && std::abs(t1 - 0.5) < 1e-14) {
        // Left half subdivision matrix
        // Q_k = sum_{j=0}^{k} C(k,j) * (1/2)^k * P_j
        for (int k = 0; k <= n; ++k) {
            Real denom = std::pow(2.0, k);
            for (int j = 0; j <= k; ++j) {
                S(k, j) = binom(k, j) / denom;
            }
        }
    } else if (std::abs(t0 - 0.5) < 1e-14 && std::abs(t1 - 1.0) < 1e-14) {
        // Right half subdivision matrix
        // Q_k = sum_{j=k}^{n} C(n-k,j-k) * (1/2)^(n-k) * P_j
        for (int k = 0; k <= n; ++k) {
            Real denom = std::pow(2.0, n - k);
            for (int j = k; j <= n; ++j) {
                S(k, j) = binom(n - k, j - k) / denom;
            }
        }
    } else {
        // General case: for interval [t0, t1], use composition of subdivision
        // First subdivide at t1 to get left part [0, t1], then at t0/t1 to get
        // right part This can be computed as matrix product of two subdivision
        // matrices.
        //
        // For 2:1 refinement, we only need [0, 0.5] and [0.5, 1], so this
        // branch is not implemented. Throw an error for now.
        throw std::runtime_error(
            "BezierBasis2D::compute_1d_extraction_matrix: "
            "General intervals not yet supported. Only [0, "
            "0.5] and [0.5, 1] are implemented.");
    }

    return S;
}

} // namespace drifter
