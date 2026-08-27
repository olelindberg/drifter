#include "bathymetry/hermite_energy_matrices.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

namespace {

/// Exact integral over [0, 1] of the product of two polynomials given in
/// ascending-power coefficient form:
///     integral_0^1 (sum a_k t^k)(sum b_l t^l) dt = sum_kl a_k b_l / (k+l+1)
Real integrate_product_01(const std::vector<Real> &a, const std::vector<Real> &b) {
    Real sum = 0.0;
    for (size_t k = 0; k < a.size(); ++k) {
        if (a[k] == 0.0) {
            continue;
        }
        for (size_t l = 0; l < b.size(); ++l) {
            sum += a[k] * b[l] / static_cast<Real>(k + l + 1);
        }
    }
    return sum;
}

} // namespace

HermiteEnergyMatrices1D::HermiteEnergyMatrices1D(int r) : basis_(r) {}

MatX HermiteEnergyMatrices1D::reference(int m, int n) const {
    const int nd = num_dofs();
    MatX K(nd, nd);

    // Cache the derivative polynomials so each is differentiated once
    std::vector<std::vector<Real>> dm(static_cast<size_t>(nd));
    std::vector<std::vector<Real>> dn(static_cast<size_t>(nd));
    for (int i = 0; i < nd; ++i) {
        dm[static_cast<size_t>(i)] = basis_.poly(i, m);
        dn[static_cast<size_t>(i)] = basis_.poly(i, n);
    }

    for (int i = 0; i < nd; ++i) {
        for (int j = 0; j < nd; ++j) {
            K(i, j) = integrate_product_01(dm[static_cast<size_t>(i)], dn[static_cast<size_t>(j)]);
        }
    }
    return K;
}

VecX HermiteEnergyMatrices1D::lambda_scaling(Real h) const {
    const int nd = num_dofs();
    VecX lam(nd);
    for (int i = 0; i < nd; ++i) {
        lam(i) = std::pow(h, basis_.deriv_of(i));
    }
    return lam;
}

MatX HermiteEnergyMatrices1D::physical(int m, int n, Real h) const {
    if (h <= 0.0) {
        throw std::invalid_argument("HermiteEnergyMatrices1D::physical: h must be positive");
    }

    const VecX lam = lambda_scaling(h);
    const Real jac = std::pow(h, 1 - m - n);

    // jac * Lambda(h) * K^(m,n) * Lambda(h)
    MatX K = reference(m, n);
    return jac * (lam.asDiagonal() * K * lam.asDiagonal());
}

} // namespace drifter
