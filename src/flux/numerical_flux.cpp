#include "flux/numerical_flux.hpp"
#include <cmath>

namespace drifter {

// =============================================================================
// ShallowWaterRoeFlux implementation
// =============================================================================

ShallowWaterRoeFlux::ShallowWaterRoeFlux(Real g, Real entropy_fix)
    : g_(g), entropy_fix_(entropy_fix)
{
}

VecX ShallowWaterRoeFlux::flux(const VecX& U_L, const VecX& U_R, const Vec3& n) const {
    // State: [h, hu, hv]
    Real h_L = U_L(0);
    Real hu_L = U_L(1);
    Real hv_L = U_L(2);

    Real h_R = U_R(0);
    Real hu_R = U_R(1);
    Real hv_R = U_R(2);

    // Avoid division by zero
    Real h_min = 1e-10;
    if (h_L < h_min) h_L = h_min;
    if (h_R < h_min) h_R = h_min;

    Real u_L = hu_L / h_L;
    Real v_L = hv_L / h_L;
    Real u_R = hu_R / h_R;
    Real v_R = hv_R / h_R;

    Real nx = n(0);
    Real ny = n(1);

    // Normal and tangent velocities
    Real un_L = u_L * nx + v_L * ny;
    Real ut_L = -u_L * ny + v_L * nx;
    Real un_R = u_R * nx + v_R * ny;
    Real ut_R = -u_R * ny + v_R * nx;

    // Roe averages
    Real h_roe, u_roe, v_roe;
    roe_average(h_L, u_L, v_L, h_R, u_R, v_R, h_roe, u_roe, v_roe);

    Real un_roe = u_roe * nx + v_roe * ny;
    Real c_roe = std::sqrt(g_ * h_roe);

    // Eigenvalues
    Real lambda1 = un_roe - c_roe;
    Real lambda2 = un_roe;
    Real lambda3 = un_roe + c_roe;

    // Entropy fix (Harten)
    Real delta = entropy_fix_ * c_roe;
    if (std::abs(lambda1) < delta) {
        lambda1 = (lambda1 * lambda1 + delta * delta) / (2.0 * delta);
        if (lambda1 < 0) lambda1 = -lambda1;
    }
    if (std::abs(lambda3) < delta) {
        lambda3 = (lambda3 * lambda3 + delta * delta) / (2.0 * delta);
        if (lambda3 < 0) lambda3 = -lambda3;
    }

    // Wave strengths
    Real dh = h_R - h_L;
    Real dun = un_R - un_L;
    Real dut = ut_R - ut_L;

    Real alpha1 = 0.5 * (dh - h_roe / c_roe * dun);
    Real alpha2 = h_roe * dut;
    Real alpha3 = 0.5 * (dh + h_roe / c_roe * dun);

    // Physical fluxes
    Real F1_L = h_L * un_L;
    Real F2_L = h_L * u_L * un_L + 0.5 * g_ * h_L * h_L * nx;
    Real F3_L = h_L * v_L * un_L + 0.5 * g_ * h_L * h_L * ny;

    Real F1_R = h_R * un_R;
    Real F2_R = h_R * u_R * un_R + 0.5 * g_ * h_R * h_R * nx;
    Real F3_R = h_R * v_R * un_R + 0.5 * g_ * h_R * h_R * ny;

    // Roe flux
    VecX result(3);

    // Wave contributions
    Real abs_lambda1 = std::abs(lambda1);
    Real abs_lambda2 = std::abs(lambda2);
    Real abs_lambda3 = std::abs(lambda3);

    // Right eigenvectors times wave strengths
    Real r1_1 = alpha1;
    Real r1_2 = alpha1 * (u_roe - c_roe * nx);
    Real r1_3 = alpha1 * (v_roe - c_roe * ny);

    Real r2_1 = 0.0;
    Real r2_2 = alpha2 * (-ny);
    Real r2_3 = alpha2 * (nx);

    Real r3_1 = alpha3;
    Real r3_2 = alpha3 * (u_roe + c_roe * nx);
    Real r3_3 = alpha3 * (v_roe + c_roe * ny);

    result(0) = 0.5 * (F1_L + F1_R) - 0.5 * (abs_lambda1 * r1_1 + abs_lambda2 * r2_1 + abs_lambda3 * r3_1);
    result(1) = 0.5 * (F2_L + F2_R) - 0.5 * (abs_lambda1 * r1_2 + abs_lambda2 * r2_2 + abs_lambda3 * r3_2);
    result(2) = 0.5 * (F3_L + F3_R) - 0.5 * (abs_lambda1 * r1_3 + abs_lambda2 * r2_3 + abs_lambda3 * r3_3);

    return result;
}

void ShallowWaterRoeFlux::roe_average(
    Real h_L, Real u_L, Real v_L,
    Real h_R, Real u_R, Real v_R,
    Real& h_roe, Real& u_roe, Real& v_roe) const {

    Real sqrt_h_L = std::sqrt(h_L);
    Real sqrt_h_R = std::sqrt(h_R);
    Real denom = sqrt_h_L + sqrt_h_R;

    h_roe = 0.5 * (h_L + h_R);
    u_roe = (sqrt_h_L * u_L + sqrt_h_R * u_R) / denom;
    v_roe = (sqrt_h_L * v_L + sqrt_h_R * v_R) / denom;
}

// =============================================================================
// NumericalFluxFactory implementation
// =============================================================================

std::unique_ptr<NumericalFlux> NumericalFluxFactory::shallow_water_lax_friedrichs(Real g) {
    auto physical_flux = [g](const VecX& U) -> Tensor3 {
        Real h = U(0);
        Real hu = U(1);
        Real hv = U(2);

        if (h < 1e-10) h = 1e-10;
        Real u = hu / h;
        Real v = hv / h;

        Tensor3 F;
        // F_x
        F[0].resize(3, 1);
        F[0](0, 0) = hu;
        F[0](1, 0) = hu * u + 0.5 * g * h * h;
        F[0](2, 0) = hu * v;

        // F_y
        F[1].resize(3, 1);
        F[1](0, 0) = hv;
        F[1](1, 0) = hv * u;
        F[1](2, 0) = hv * v + 0.5 * g * h * h;

        // F_z (zero for 2D)
        F[2].resize(3, 1);
        F[2].setZero();

        return F;
    };

    auto max_wave_speed = [g](const VecX& U_L, const VecX& U_R, const Vec3& n) -> Real {
        Real h_L = std::max(U_L(0), 1e-10);
        Real h_R = std::max(U_R(0), 1e-10);
        Real u_L = U_L(1) / h_L;
        Real v_L = U_L(2) / h_L;
        Real u_R = U_R(1) / h_R;
        Real v_R = U_R(2) / h_R;

        Real c_L = std::sqrt(g * h_L);
        Real c_R = std::sqrt(g * h_R);

        Real un_L = u_L * n(0) + v_L * n(1);
        Real un_R = u_R * n(0) + v_R * n(1);

        return std::max(std::abs(un_L) + c_L, std::abs(un_R) + c_R);
    };

    return std::make_unique<LaxFriedrichsFlux>(physical_flux, max_wave_speed, 3);
}

std::unique_ptr<NumericalFlux> NumericalFluxFactory::shallow_water_hllc(Real g) {
    return std::make_unique<ShallowWaterHLLCFlux>(g);
}

std::unique_ptr<NumericalFlux> NumericalFluxFactory::shallow_water_roe(Real g) {
    return std::make_unique<ShallowWaterRoeFlux>(g);
}

std::unique_ptr<NumericalFlux> NumericalFluxFactory::advection_upwind(const Vec3& velocity) {
    return std::make_unique<UpwindFlux>(velocity);
}

std::unique_ptr<NumericalFlux> NumericalFluxFactory::central(
    PhysicalFluxFunc phys_flux, int nvars) {
    return std::make_unique<CentralFlux>(std::move(phys_flux), nvars);
}

}  // namespace drifter
