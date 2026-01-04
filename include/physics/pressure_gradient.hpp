#pragma once

// Pressure Gradient Computation in Sigma Coordinates
//
// The pressure gradient in sigma coordinates has two components:
//
// 1. Barotropic: -g * grad(eta)  (surface pressure gradient)
// 2. Baroclinic: -g/rho_0 * integral_sigma^0 grad_h(rho) * H dsigma'
//
// The challenge is that grad_h operates at constant z, but we compute
// at constant sigma. This leads to the "sigma-coordinate error" which
// can cause spurious currents over steep topography.
//
// This module provides several methods to compute the pressure gradient:
//
// 1. Standard method: Direct computation with sigma-to-z transform
// 2. Density Jacobian method (Shchepetkin & McWilliams, 2003)
// 3. Polynomial reconstruction method
//
// The Density Jacobian method is more accurate over steep slopes.

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/operators_3d.hpp"
#include "physics/equation_of_state.hpp"
#include <functional>

namespace drifter {

/// @brief Pressure gradient computation method
enum class PressureGradientMethod {
    Standard,            ///< Direct sigma-to-z transform
    DensityJacobian,     ///< Shchepetkin & McWilliams (2003)
    Polynomial           ///< High-order polynomial reconstruction
};

/// @brief Pressure gradient parameters
struct PressureGradientParams {
    PressureGradientMethod method = PressureGradientMethod::DensityJacobian;
    Real rho_0 = 1025.0;    ///< Reference density [kg/m³]
    Real g = 9.81;          ///< Gravitational acceleration [m/s²]
    bool use_split = true;  ///< Split barotropic/baroclinic
};

/// @brief Pressure gradient computation for a single element
class PressureGradientElement {
public:
    /// @brief Construct pressure gradient calculator
    PressureGradientElement(const HexahedronBasis& basis,
                             const GaussQuadrature3D& quad,
                             const PressureGradientParams& params = PressureGradientParams());

    /// @brief Set bathymetry
    void set_bathymetry(const VecX& h, const VecX& dh_dx, const VecX& dh_dy);

    /// @brief Set equation of state
    void set_eos(const EquationOfState& eos) { eos_ = &eos; }

    // =========================================================================
    // Barotropic pressure gradient
    // =========================================================================

    /// @brief Compute barotropic pressure gradient (-gH * grad(eta))
    /// @param eta Free surface elevation
    /// @param H Water depth
    /// @param[out] pg_x X-component of pressure gradient force
    /// @param[out] pg_y Y-component
    void barotropic_gradient(const VecX& eta, const VecX& H,
                              VecX& pg_x, VecX& pg_y) const;

    // =========================================================================
    // Baroclinic pressure gradient
    // =========================================================================

    /// @brief Compute baroclinic pressure gradient using standard method
    /// @param rho Density at DOFs
    /// @param eta Free surface elevation
    /// @param H Water depth
    /// @param sigma Sigma values at DOFs
    /// @param[out] pg_x X-component
    /// @param[out] pg_y Y-component
    void baroclinic_gradient_standard(const VecX& rho, const VecX& eta, const VecX& H,
                                        const VecX& sigma,
                                        VecX& pg_x, VecX& pg_y) const;

    /// @brief Compute baroclinic pressure gradient using Density Jacobian method
    /// @details More accurate over steep topography
    void baroclinic_gradient_density_jacobian(const VecX& rho, const VecX& z,
                                                 const VecX& H,
                                                 VecX& pg_x, VecX& pg_y) const;

    /// @brief Compute full pressure gradient (barotropic + baroclinic)
    void full_gradient(const VecX& eta, const VecX& rho, const VecX& H,
                        const VecX& sigma,
                        VecX& pg_x, VecX& pg_y) const;

    // =========================================================================
    // Hydrostatic pressure
    // =========================================================================

    /// @brief Compute hydrostatic pressure from density
    /// @details p(sigma) = g * integral_sigma^0 rho * H dsigma'
    void hydrostatic_pressure(const VecX& rho, const VecX& H,
                               VecX& pressure) const;

    /// @brief Compute pressure at specific depth
    Real pressure_at_depth(const VecX& rho, const VecX& H,
                            int i_horiz, Real sigma) const;

private:
    const HexahedronBasis& basis_;
    const GaussQuadrature3D& quad_;
    PressureGradientParams params_;
    const EquationOfState* eos_ = nullptr;

    VecX h_;
    VecX dh_dx_;
    VecX dh_dy_;

    int n_horiz_;
    int n_vert_;

    // Vertical integration weights
    VecX sigma_weights_;

    // Integration matrix for hydrostatic pressure
    // P(k) = integral from sigma(k) to 0 of rho * H dsigma
    MatX pressure_integration_matrix_;

    void build_integration_matrix();

    /// @brief Horizontal density gradient at constant z
    /// @details drho/dx|_z = drho/dx|_sigma - (dsigma/dx) * drho/dsigma
    void horizontal_rho_gradient_at_z(const VecX& rho, const VecX& sigma,
                                        const VecX& eta, const VecX& H,
                                        VecX& drho_dx, VecX& drho_dy) const;
};

/// @brief Global pressure gradient solver
class PressureGradientSolver {
public:
    /// @brief Construct solver
    PressureGradientSolver(int order,
                            const PressureGradientParams& params = PressureGradientParams());

    /// @brief Initialize with mesh
    void initialize(int num_elements,
                     const std::vector<VecX>& bathymetry,
                     const std::vector<VecX>& dh_dx,
                     const std::vector<VecX>& dh_dy);

    /// @brief Set equation of state
    void set_eos(const EquationOfState& eos);

    /// @brief Compute pressure gradient for all elements
    /// @param eta Free surface at each element
    /// @param rho Density at each element
    /// @param H Water depth at each element
    /// @param sigma Sigma values at each element
    /// @param[out] pg_x X-component at each element
    /// @param[out] pg_y Y-component at each element
    void compute(const std::vector<VecX>& eta,
                  const std::vector<VecX>& rho,
                  const std::vector<VecX>& H,
                  const std::vector<VecX>& sigma,
                  std::vector<VecX>& pg_x,
                  std::vector<VecX>& pg_y) const;

    /// @brief Compute pressure gradient from T, S (using EOS)
    void compute_from_tracers(const std::vector<VecX>& eta,
                                const std::vector<VecX>& T,
                                const std::vector<VecX>& S,
                                const std::vector<VecX>& H,
                                const std::vector<VecX>& sigma,
                                std::vector<VecX>& pg_x,
                                std::vector<VecX>& pg_y) const;

private:
    int order_;
    PressureGradientParams params_;

    HexahedronBasis basis_;
    GaussQuadrature3D quad_;

    std::vector<std::unique_ptr<PressureGradientElement>> elements_;
    const EquationOfState* eos_ = nullptr;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline PressureGradientElement::PressureGradientElement(
    const HexahedronBasis& basis, const GaussQuadrature3D& quad,
    const PressureGradientParams& params)
    : basis_(basis)
    , quad_(quad)
    , params_(params)
    , n_horiz_((basis.order() + 1) * (basis.order() + 1))
    , n_vert_(basis.order() + 1)
{
    // Set up vertical integration weights
    const VecX& lgl_weights = basis.lgl_weights_1d();
    sigma_weights_.resize(n_vert_);
    for (int k = 0; k < n_vert_; ++k) {
        sigma_weights_(k) = 0.5 * lgl_weights(k);
    }

    build_integration_matrix();
}

inline void PressureGradientElement::set_bathymetry(
    const VecX& h, const VecX& dh_dx, const VecX& dh_dy) {
    h_ = h;
    dh_dx_ = dh_dx;
    dh_dy_ = dh_dy;
}

inline void PressureGradientElement::build_integration_matrix() {
    // Build matrix for downward integration from surface
    // P(k) = sum_{j=k}^{n_vert-1} w_j * rho(j) * H(j)

    pressure_integration_matrix_.resize(n_vert_, n_vert_);
    pressure_integration_matrix_.setZero();

    for (int k = 0; k < n_vert_; ++k) {
        for (int j = k; j < n_vert_; ++j) {
            pressure_integration_matrix_(k, j) = sigma_weights_(j);
        }
    }
}

inline void PressureGradientElement::barotropic_gradient(
    const VecX& eta, const VecX& H, VecX& pg_x, VecX& pg_y) const {

    int ndof = static_cast<int>(eta.size());
    pg_x.resize(ndof);
    pg_y.resize(ndof);

    // Compute gradient of eta
    MatX grad_eta;
    DG3DElementOperator elem_op(basis_, quad_);
    elem_op.gradient_reference(eta, grad_eta, true);

    // -gH * grad(eta)
    for (int i = 0; i < ndof; ++i) {
        pg_x(i) = -params_.g * H(i) * grad_eta(i, 0);
        pg_y(i) = -params_.g * H(i) * grad_eta(i, 1);
    }
}

inline void PressureGradientElement::hydrostatic_pressure(
    const VecX& rho, const VecX& H, VecX& pressure) const {

    int n_total = n_horiz_ * n_vert_;
    pressure.resize(n_total);

    for (int i_h = 0; i_h < n_horiz_; ++i_h) {
        // Integrate from surface downward
        for (int k = 0; k < n_vert_; ++k) {
            Real p = 0.0;
            for (int j = k; j < n_vert_; ++j) {
                int idx = i_h * n_vert_ + j;
                p += sigma_weights_(j) * params_.g * rho(idx) * H(idx);
            }
            pressure(i_h * n_vert_ + k) = p;
        }
    }
}

inline void PressureGradientElement::baroclinic_gradient_standard(
    const VecX& rho, const VecX& eta, const VecX& H, const VecX& sigma,
    VecX& pg_x, VecX& pg_y) const {

    int n_total = n_horiz_ * n_vert_;
    pg_x.resize(n_total);
    pg_y.resize(n_total);

    // Compute hydrostatic pressure
    VecX pressure;
    hydrostatic_pressure(rho, H, pressure);

    // Compute horizontal gradient of pressure at constant sigma
    MatX grad_p;
    DG3DElementOperator elem_op(basis_, quad_);
    elem_op.gradient_reference(pressure, grad_p, true);

    // Correct for sigma-to-z transform
    // dp/dx|_z = dp/dx|_sigma + (dp/dsigma) * (dsigma/dx)
    // But we want -grad_z(p)/rho, which in sigma coords has extra terms

    VecX dp_dsigma = basis_.D_zeta_lgl() * pressure;

    for (int i = 0; i < n_total; ++i) {
        // dsigma/dx = (sigma * dH/dx - deta/dx) / H
        int i_h = i / n_vert_;
        Real deta_dx = 0.0;  // Need to compute from eta gradient
        Real deta_dy = 0.0;
        Real dH_dx = deta_dx + dh_dx_(i_h);
        Real dH_dy = deta_dy + dh_dy_(i_h);

        Real dsigma_dx = (sigma(i) * dH_dx - deta_dx) / H(i);
        Real dsigma_dy = (sigma(i) * dH_dy - deta_dy) / H(i);

        // Correct gradient
        Real dp_dx_z = grad_p(i, 0) + dp_dsigma(i) * dsigma_dx;
        Real dp_dy_z = grad_p(i, 1) + dp_dsigma(i) * dsigma_dy;

        // Force per unit mass: -1/rho * grad(p)
        pg_x(i) = -dp_dx_z / params_.rho_0;
        pg_y(i) = -dp_dy_z / params_.rho_0;
    }
}

inline void PressureGradientElement::baroclinic_gradient_density_jacobian(
    const VecX& rho, const VecX& z, const VecX& H, VecX& pg_x, VecX& pg_y) const {

    // Density Jacobian method (Shchepetkin & McWilliams, 2003)
    // Uses coordinate-invariant formulation to reduce sigma-coordinate errors

    int n_total = n_horiz_ * n_vert_;
    pg_x.resize(n_total);
    pg_y.resize(n_total);

    // Compute density gradient at constant sigma
    MatX grad_rho;
    DG3DElementOperator elem_op(basis_, quad_);
    elem_op.gradient_reference(rho, grad_rho, true);

    // Compute z gradient
    MatX grad_z;
    elem_op.gradient_reference(z, grad_z, true);

    // Vertical density derivative
    VecX drho_dsigma = basis_.D_zeta_lgl() * rho;

    // Density Jacobian: J = drho/dx * dz/dy - drho/dy * dz/dx
    // Plus vertical integration term

    for (int i_h = 0; i_h < n_horiz_; ++i_h) {
        // Integrate from bottom to each level
        Real integral_x = 0.0;
        Real integral_y = 0.0;

        for (int k = 0; k < n_vert_; ++k) {
            int idx = i_h * n_vert_ + k;

            // Jacobian contribution at this level
            Real J_x = grad_rho(idx, 0) * grad_z(idx, 2) - drho_dsigma(idx) * grad_z(idx, 0);
            Real J_y = grad_rho(idx, 1) * grad_z(idx, 2) - drho_dsigma(idx) * grad_z(idx, 1);

            integral_x += sigma_weights_(k) * J_x;
            integral_y += sigma_weights_(k) * J_y;

            // Pressure gradient at this level
            pg_x(idx) = -params_.g / params_.rho_0 * integral_x;
            pg_y(idx) = -params_.g / params_.rho_0 * integral_y;
        }
    }
}

inline void PressureGradientElement::full_gradient(
    const VecX& eta, const VecX& rho, const VecX& H, const VecX& sigma,
    VecX& pg_x, VecX& pg_y) const {

    // Barotropic component
    VecX pg_baro_x, pg_baro_y;
    barotropic_gradient(eta, H, pg_baro_x, pg_baro_y);

    // Baroclinic component
    VecX pg_clinic_x, pg_clinic_y;

    if (params_.method == PressureGradientMethod::DensityJacobian) {
        // Need z values for density Jacobian method
        VecX z(rho.size());
        for (int i = 0; i < static_cast<int>(rho.size()); ++i) {
            int i_h = i / n_vert_;
            z(i) = eta(i) + sigma(i) * H(i);
        }
        baroclinic_gradient_density_jacobian(rho, z, H, pg_clinic_x, pg_clinic_y);
    } else {
        baroclinic_gradient_standard(rho, eta, H, sigma, pg_clinic_x, pg_clinic_y);
    }

    // Combine
    pg_x = pg_baro_x + pg_clinic_x;
    pg_y = pg_baro_y + pg_clinic_y;
}

}  // namespace drifter
