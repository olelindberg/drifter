#pragma once

/// @file hermite_energy_matrices.hpp
/// @brief Exact 1D Hermite energy matrices K^(m,n)
///
/// The tensor-product structure of the Hermite element lets its energy matrix be
/// written in closed Kronecker form, with no quadrature at all. The building
/// blocks are the 1D matrices
///
///     [K^(m,n)]_ij = integral_0^1 H_i^(m)(t) H_j^(n)(t) dt
///
/// which are integrals of polynomials and therefore exact rationals. Immediately
/// K^(m,n) = (K^(n,m))^T, so K^(0,0) (a mass matrix), K^(1,1) and K^(2,2) are
/// symmetric while K^(2,0) is not.
///
/// The *physical* counterpart on an interval of length h, in physical-derivative
/// DOFs, folds in both the chain rule and the DOF scaling:
///
///     K_phys^(m,n)(h) = h^(1-m-n) Lambda(h) K^(m,n) Lambda(h)
///
/// with Lambda(h) = diag(h^mu_i) and mu_i the derivative order of the i-th DOF
/// (h^1 from the Jacobian, h^(-m-n) from the two chain rules, Lambda twice from
/// the physical DOFs).
///
/// See docs/hermite_smoothness_operator.md S5.1.

#include "bathymetry/hermite_basis_1d.hpp"
#include "core/types.hpp"

namespace drifter {

/// @brief Exact 1D Hermite energy matrices for a given continuity order
///
/// Computes K^(m,n) once at construction by exact polynomial integration; no
/// Gauss rule is involved, so there is no quadrature error and no risk of the
/// spurious kernel modes that under-integration would introduce.
class HermiteEnergyMatrices1D {
public:
    /// @brief Construct for continuity order r
    /// @param r Continuity order (0, 1 or 2)
    explicit HermiteEnergyMatrices1D(int r);

    /// @brief Continuity order
    int r() const { return basis_.r(); }

    /// @brief Number of 1D DOFs, 2(r+1)
    int num_dofs() const { return basis_.num_dofs(); }

    /// @brief Reference energy matrix K^(m,n) on [0, 1]
    ///
    /// Computed on demand and exactly; entries are integrals of polynomials.
    ///
    /// @param m, n Derivative orders (>= 0)
    /// @return (num_dofs x num_dofs) matrix
    MatX reference(int m, int n) const;

    /// @brief Physical energy matrix K_phys^(m,n)(h) in physical-derivative DOFs
    /// @param m, n Derivative orders (>= 0)
    /// @param h Interval length (> 0)
    MatX physical(int m, int n, Real h) const;

    /// @brief Diagonal DOF scaling Lambda(h) = diag(h^mu_i)
    /// @param h Interval length
    VecX lambda_scaling(Real h) const;

private:
    HermiteBasis1D basis_;
};

} // namespace drifter
