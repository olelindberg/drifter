#pragma once

/// @file hermite_basis_1d.hpp
/// @brief 1D Hermite basis on the reference interval [0, 1]
///
/// The tensor-product Hermite elements of the Hermite bathymetry smoother are
/// built from these 1D bases. For continuity order r the polynomial degree is
/// p = 2r+1 and there are 2(r+1) basis functions, dual to the derivatives at the
/// two endpoints:
///
///     d^m' / dt^m' H_{s,m} |_{t=s'} = delta_{ss'} delta_{mm'}
///
/// with s, s' in {0, 1} (node) and m, m' in {0, ..., r} (derivative order).
///
/// Basis functions are stored as polynomial coefficients in ascending powers, so
/// differentiation and integration are exact - which is what lets the energy
/// matrices of hermite_energy_matrices.hpp avoid quadrature entirely.
///
/// See docs/hermite_smoothness_operator.md S3.3 for the closed forms.

#include "core/types.hpp"
#include <vector>

namespace drifter {

/// @brief 1D Hermite basis for continuity order r
///
/// DOFs are indexed node-major, matching docs/hermite_smoothness_operator.md S3.2:
///
///     i = s * (r+1) + m
///
/// so for r = 1 the order is (z_0, z_0', z_1, z_1').
class HermiteBasis1D {
public:
    /// @brief Construct the basis for continuity order r
    /// @param r Continuity order (0, 1 or 2)
    /// @throws std::invalid_argument if r is outside [0, 2]
    explicit HermiteBasis1D(int r);

    /// @brief Continuity order
    int r() const { return r_; }

    /// @brief Polynomial degree p = 2r+1
    int degree() const { return 2 * r_ + 1; }

    /// @brief Number of basis functions, 2(r+1) = p+1
    int num_dofs() const { return 2 * (r_ + 1); }

    /// @brief Node index (0 or 1) of DOF i
    int node_of(int i) const { return i / (r_ + 1); }

    /// @brief Derivative order (0..r) of DOF i
    int deriv_of(int i) const { return i % (r_ + 1); }

    /// @brief DOF index for node s and derivative order m
    int dof_index(int s, int m) const { return s * (r_ + 1) + m; }

    /// @brief Evaluate the `deriv`-th derivative of basis function i at t
    /// @param i Basis function index in [0, num_dofs)
    /// @param deriv Derivative order (>= 0; returns 0 beyond the degree)
    /// @param t Parameter, normally in [0, 1]
    Real eval(int i, int deriv, Real t) const;

    /// @brief Polynomial coefficients of the `deriv`-th derivative of function i
    /// @return Coefficients in ascending powers of t (exact, no quadrature)
    std::vector<Real> poly(int i, int deriv = 0) const;

private:
    int r_;
    /// coeffs_[i] = ascending-power polynomial coefficients of H_i, length p+1
    std::vector<std::vector<Real>> coeffs_;
};

} // namespace drifter
