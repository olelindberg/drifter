#include "bathymetry/thb_spline/bspline_basis_1d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace drifter {

BSplineBasis1D::BSplineBasis1D(int num_spans) : num_spans_(num_spans) {
    if (num_spans < 1) {
        throw std::invalid_argument("BSplineBasis1D: num_spans must be at least 1");
    }
    build_knot_vector();
}

void BSplineBasis1D::build_knot_vector() {
    // Open (clamped) uniform knot vector for cubic B-splines:
    // [0, 0, 0, 0, 1, 2, ..., n-1, n, n, n, n]
    // Total knots = num_spans + 2*(degree+1) = num_spans + 8

    const int num_knots = num_spans_ + 2 * (DEGREE + 1);
    knots_.resize(num_knots);

    // First degree+1 knots are 0 (clamped at start)
    for (int i = 0; i <= DEGREE; ++i) {
        knots_[i] = 0.0;
    }

    // Interior knots: 1, 2, ..., num_spans-1
    for (int i = 1; i < num_spans_; ++i) {
        knots_[DEGREE + i] = static_cast<Real>(i);
    }

    // Last degree+1 knots are num_spans (clamped at end)
    for (int i = 0; i <= DEGREE; ++i) {
        knots_[DEGREE + num_spans_ + i] = static_cast<Real>(num_spans_);
    }
}

int BSplineBasis1D::find_span(Real t) const {
    const int n = num_basis() - 1;  // Last basis function index

    // Clamp to domain
    if (t <= knots_[DEGREE]) {
        return DEGREE;
    }
    if (t >= knots_[n + 1]) {
        return n;
    }

    // Binary search for span
    int low = DEGREE;
    int high = n + 1;
    int mid = (low + high) / 2;

    while (t < knots_[mid] || t >= knots_[mid + 1]) {
        if (t < knots_[mid]) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }

    return mid;
}

Real BSplineBasis1D::cox_de_boor(int i, int p, Real t) const {
    // Base case: degree 0
    if (p == 0) {
        // Handle boundary case for rightmost span
        if (i == num_basis() - 1 && t == knots_[i + 1]) {
            return 1.0;
        }
        return (t >= knots_[i] && t < knots_[i + 1]) ? 1.0 : 0.0;
    }

    // Recursive case
    Real left = 0.0;
    Real right = 0.0;

    Real denom1 = knots_[i + p] - knots_[i];
    if (std::abs(denom1) > 1e-14) {
        left = (t - knots_[i]) / denom1 * cox_de_boor(i, p - 1, t);
    }

    Real denom2 = knots_[i + p + 1] - knots_[i + 1];
    if (std::abs(denom2) > 1e-14) {
        right = (knots_[i + p + 1] - t) / denom2 * cox_de_boor(i + 1, p - 1, t);
    }

    return left + right;
}

Real BSplineBasis1D::cox_de_boor_derivative(int i, int p, Real t, int deriv) const {
    if (deriv == 0) {
        return cox_de_boor(i, p, t);
    }

    if (p == 0) {
        return 0.0;  // Derivative of constant is 0
    }

    // Derivative formula: dN_{i,p}/dt = p * [N_{i,p-1}/(t_{i+p}-t_i) -
    // N_{i+1,p-1}/(t_{i+p+1}-t_{i+1})]
    Real left = 0.0;
    Real right = 0.0;

    Real denom1 = knots_[i + p] - knots_[i];
    if (std::abs(denom1) > 1e-14) {
        left = cox_de_boor_derivative(i, p - 1, t, deriv - 1) / denom1;
    }

    Real denom2 = knots_[i + p + 1] - knots_[i + 1];
    if (std::abs(denom2) > 1e-14) {
        right = cox_de_boor_derivative(i + 1, p - 1, t, deriv - 1) / denom2;
    }

    return static_cast<Real>(p) * (left - right);
}

VecX BSplineBasis1D::evaluate_nonzero(Real t) const {
    const int span = find_span(t);
    VecX N(DEGREE + 1);

    // Initialize
    N.setZero();
    N(0) = 1.0;

    // Temporary storage for left/right differences
    std::vector<Real> left(DEGREE + 1);
    std::vector<Real> right(DEGREE + 1);

    for (int j = 1; j <= DEGREE; ++j) {
        left[j] = t - knots_[span + 1 - j];
        right[j] = knots_[span + j] - t;

        Real saved = 0.0;
        for (int r = 0; r < j; ++r) {
            Real temp = N(r) / (right[r + 1] + left[j - r]);
            N(r) = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N(j) = saved;
    }

    return N;
}

MatX BSplineBasis1D::evaluate_nonzero_derivs(Real t, int num_derivs) const {
    const int span = find_span(t);
    const int n = std::min(num_derivs, DEGREE);

    MatX ders(n + 1, DEGREE + 1);
    ders.setZero();

    // Build basis function table using triangular algorithm
    MatX ndu(DEGREE + 1, DEGREE + 1);
    ndu.setZero();
    ndu(0, 0) = 1.0;

    std::vector<Real> left(DEGREE + 1);
    std::vector<Real> right(DEGREE + 1);

    for (int j = 1; j <= DEGREE; ++j) {
        left[j] = t - knots_[span + 1 - j];
        right[j] = knots_[span + j] - t;

        Real saved = 0.0;
        for (int r = 0; r < j; ++r) {
            // Lower triangle
            ndu(j, r) = right[r + 1] + left[j - r];
            Real temp = ndu(r, j - 1) / ndu(j, r);
            // Upper triangle
            ndu(r, j) = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu(j, j) = saved;
    }

    // Load the basis functions
    for (int j = 0; j <= DEGREE; ++j) {
        ders(0, j) = ndu(j, DEGREE);
    }

    // Compute derivatives
    MatX a(2, DEGREE + 1);

    for (int r = 0; r <= DEGREE; ++r) {
        int s1 = 0, s2 = 1;
        a(0, 0) = 1.0;

        for (int k = 1; k <= n; ++k) {
            Real d = 0.0;
            int rk = r - k;
            int pk = DEGREE - k;

            if (r >= k) {
                a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk);
                d = a(s2, 0) * ndu(rk, pk);
            }

            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (r - 1 <= pk) ? k - 1 : DEGREE - r;

            for (int j = j1; j <= j2; ++j) {
                a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j);
                d += a(s2, j) * ndu(rk + j, pk);
            }

            if (r <= pk) {
                a(s2, k) = -a(s1, k - 1) / ndu(pk + 1, r);
                d += a(s2, k) * ndu(r, pk);
            }

            ders(k, r) = d;
            std::swap(s1, s2);
        }
    }

    // Multiply by correct factors
    Real r = static_cast<Real>(DEGREE);
    for (int k = 1; k <= n; ++k) {
        for (int j = 0; j <= DEGREE; ++j) {
            ders(k, j) *= r;
        }
        r *= static_cast<Real>(DEGREE - k);
    }

    return ders;
}

Real BSplineBasis1D::evaluate(int i, Real t) const {
    if (i < 0 || i >= num_basis()) {
        return 0.0;
    }

    // Check if t is in support of basis function i
    auto [t_min, t_max] = support(i);
    if (t < t_min || t > t_max) {
        return 0.0;
    }

    return cox_de_boor(i, DEGREE, t);
}

Real BSplineBasis1D::evaluate_derivative(int i, Real t) const {
    if (i < 0 || i >= num_basis()) {
        return 0.0;
    }

    auto [t_min, t_max] = support(i);
    if (t < t_min || t > t_max) {
        return 0.0;
    }

    return cox_de_boor_derivative(i, DEGREE, t, 1);
}

Real BSplineBasis1D::evaluate_second_derivative(int i, Real t) const {
    if (i < 0 || i >= num_basis()) {
        return 0.0;
    }

    auto [t_min, t_max] = support(i);
    if (t < t_min || t > t_max) {
        return 0.0;
    }

    return cox_de_boor_derivative(i, DEGREE, t, 2);
}

VecX BSplineBasis1D::evaluate_all(Real t) const {
    VecX result(num_basis());
    result.setZero();

    const int span = find_span(t);
    VecX nonzero = evaluate_nonzero(t);

    // Place non-zero values at correct indices
    for (int i = 0; i <= DEGREE; ++i) {
        int idx = span - DEGREE + i;
        if (idx >= 0 && idx < num_basis()) {
            result(idx) = nonzero(i);
        }
    }

    return result;
}

VecX BSplineBasis1D::evaluate_all_derivatives(Real t) const {
    VecX result(num_basis());
    result.setZero();

    const int span = find_span(t);
    MatX ders = evaluate_nonzero_derivs(t, 1);

    // Place derivative values at correct indices
    for (int i = 0; i <= DEGREE; ++i) {
        int idx = span - DEGREE + i;
        if (idx >= 0 && idx < num_basis()) {
            result(idx) = ders(1, i);
        }
    }

    return result;
}

VecX BSplineBasis1D::evaluate_all_second_derivatives(Real t) const {
    VecX result(num_basis());
    result.setZero();

    const int span = find_span(t);
    MatX ders = evaluate_nonzero_derivs(t, 2);

    // Place second derivative values at correct indices
    for (int i = 0; i <= DEGREE; ++i) {
        int idx = span - DEGREE + i;
        if (idx >= 0 && idx < num_basis()) {
            result(idx) = ders(2, i);
        }
    }

    return result;
}

std::pair<Real, Real> BSplineBasis1D::support(int i) const {
    // Basis function i is non-zero on [knot[i], knot[i+degree+1]]
    return {knots_[i], knots_[i + DEGREE + 1]};
}

std::vector<Real> BSplineBasis1D::refinement_coefficients() {
    // Two-scale relation for cubic B-splines with dyadic refinement:
    // B^l_j(t) = Σ_{k=0}^{4} c_k × B^{l+1}_{2j+k-1}(t)
    // Coefficients: binomial(4, k) / 2^4 = [1/16, 4/16, 6/16, 4/16, 1/16]
    // Wait, for B-splines the refinement uses degree+1 = 4 coefficients? Let me recalculate.
    //
    // Actually for cubic B-splines (degree 3), the refinement coefficients are:
    // c = [1, 4, 6, 4, 1] / 8 (not 16)
    // This comes from the subdivision rule for B-splines.
    return {1.0 / 8.0, 4.0 / 8.0, 6.0 / 8.0, 4.0 / 8.0, 1.0 / 8.0};
}

}  // namespace drifter
