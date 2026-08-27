#include "bathymetry/hermite_basis_1d.hpp"
#include <stdexcept>

namespace drifter {

namespace {

/// Ascending-power coefficients of the 1D Hermite bases, indexed by continuity
/// order r then by node-major DOF index i = s*(r+1) + m.
/// See docs/hermite_smoothness_operator.md S3.3.

// r = 0 (linear; these are the degree-1 Bernstein polynomials)
//   H_00 = 1 - t,  H_10 = t
const std::vector<std::vector<Real>> kCoeffsR0 = {
    {1.0, -1.0},
    {0.0, 1.0},
};

// r = 1 (cubic, Bogner-Fox-Schmit)
//   H_00 = 2t^3 - 3t^2 + 1,  H_01 = t^3 - 2t^2 + t
//   H_10 = -2t^3 + 3t^2,     H_11 = t^3 - t^2
const std::vector<std::vector<Real>> kCoeffsR1 = {
    {1.0, 0.0, -3.0, 2.0},
    {0.0, 1.0, -2.0, 1.0},
    {0.0, 0.0, 3.0, -2.0},
    {0.0, 0.0, -1.0, 1.0},
};

// r = 2 (quintic)
//   H_00 = -6t^5 + 15t^4 - 10t^3 + 1
//   H_01 = -3t^5 +  8t^4 -  6t^3 + t
//   H_02 = -1/2 t^5 + 3/2 t^4 - 3/2 t^3 + 1/2 t^2
//   H_10 =  6t^5 - 15t^4 + 10t^3
//   H_11 = -3t^5 +  7t^4 -  4t^3
//   H_12 =  1/2 t^5 - t^4 + 1/2 t^3
const std::vector<std::vector<Real>> kCoeffsR2 = {
    {1.0, 0.0, 0.0, -10.0, 15.0, -6.0},
    {0.0, 1.0, 0.0, -6.0, 8.0, -3.0},
    {0.0, 0.0, 0.5, -1.5, 1.5, -0.5},
    {0.0, 0.0, 0.0, 10.0, -15.0, 6.0},
    {0.0, 0.0, 0.0, -4.0, 7.0, -3.0},
    {0.0, 0.0, 0.0, 0.5, -1.0, 0.5},
};

} // namespace

HermiteBasis1D::HermiteBasis1D(int r) : r_(r) {
    switch (r) {
    case 0:
        coeffs_ = kCoeffsR0;
        break;
    case 1:
        coeffs_ = kCoeffsR1;
        break;
    case 2:
        coeffs_ = kCoeffsR2;
        break;
    default:
        throw std::invalid_argument(
            "HermiteBasis1D: continuity order r must be 0, 1 or 2, got " + std::to_string(r));
    }
}

std::vector<Real> HermiteBasis1D::poly(int i, int deriv) const {
    if (i < 0 || i >= num_dofs()) {
        throw std::invalid_argument("HermiteBasis1D::poly: DOF index out of range: " +
                                    std::to_string(i));
    }
    if (deriv < 0) {
        throw std::invalid_argument("HermiteBasis1D::poly: derivative order must be >= 0");
    }

    std::vector<Real> c = coeffs_[static_cast<size_t>(i)];

    // Differentiate `deriv` times: d/dt sum a_k t^k = sum k a_k t^(k-1)
    for (int d = 0; d < deriv; ++d) {
        if (c.size() <= 1) {
            return {0.0};
        }
        std::vector<Real> dc(c.size() - 1);
        for (size_t k = 1; k < c.size(); ++k) {
            dc[k - 1] = static_cast<Real>(k) * c[k];
        }
        c = std::move(dc);
    }
    return c;
}

Real HermiteBasis1D::eval(int i, int deriv, Real t) const {
    const std::vector<Real> c = poly(i, deriv);

    // Horner
    Real result = 0.0;
    for (size_t k = c.size(); k-- > 0;) {
        result = result * t + c[k];
    }
    return result;
}

} // namespace drifter
