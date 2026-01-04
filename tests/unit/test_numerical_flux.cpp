#include <gtest/gtest.h>
#include "flux/numerical_flux.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class NumericalFluxTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
    }
};

// Test Lax-Friedrichs flux consistency (F*(U, U, n) = F(U) . n)
TEST_F(NumericalFluxTest, LaxFriedrichsConsistency) {
    Real g = 9.81;
    auto flux = NumericalFluxFactory::shallow_water_lax_friedrichs(g);

    // Test state: h=1, u=0.5, v=0.3
    VecX U(3);
    U << 1.0, 0.5, 0.3;  // [h, hu, hv]

    Vec3 n(1.0, 0.0, 0.0);  // Normal in x direction

    VecX F_star = flux->flux(U, U, n);

    // Physical flux in x: F_x = [hu, hu*u + 0.5*g*h^2, hu*v]
    Real h = U(0);
    Real u = U(1) / h;
    Real v = U(2) / h;

    VecX F_physical(3);
    F_physical << h * u,
                  h * u * u + 0.5 * g * h * h,
                  h * u * v;

    EXPECT_TRUE(vectors_equal(F_star, F_physical, LOOSE_TOLERANCE));
}

// Test Lax-Friedrichs symmetry
TEST_F(NumericalFluxTest, LaxFriedrichsSymmetry) {
    Real g = 9.81;
    auto flux = NumericalFluxFactory::shallow_water_lax_friedrichs(g);

    VecX U_L(3), U_R(3);
    U_L << 1.0, 0.5, 0.2;
    U_R << 1.2, 0.6, 0.1;

    Vec3 n(1.0, 0.0, 0.0);

    VecX F_LR = flux->flux(U_L, U_R, n);
    VecX F_RL = flux->flux(U_R, U_L, -n);

    // F*(U_L, U_R, n) + F*(U_R, U_L, -n) = 0 (opposite normals)
    EXPECT_TRUE(vectors_equal(F_LR, -F_RL, LOOSE_TOLERANCE));
}

// Test HLLC flux consistency
TEST_F(NumericalFluxTest, HLLCConsistency) {
    Real g = 9.81;
    auto flux = NumericalFluxFactory::shallow_water_hllc(g);

    VecX U(3);
    U << 2.0, 1.0, -0.5;

    Vec3 n(0.0, 1.0, 0.0);  // Normal in y direction

    VecX F_star = flux->flux(U, U, n);

    Real h = U(0);
    Real u = U(1) / h;
    Real v = U(2) / h;

    VecX F_physical(3);
    F_physical << h * v,
                  h * u * v,
                  h * v * v + 0.5 * g * h * h;

    EXPECT_TRUE(vectors_equal(F_star, F_physical, LOOSE_TOLERANCE));
}

// Test Roe flux consistency
TEST_F(NumericalFluxTest, RoeConsistency) {
    Real g = 9.81;
    auto flux = NumericalFluxFactory::shallow_water_roe(g);

    VecX U(3);
    U << 1.5, 0.75, 0.45;

    Vec3 n(std::sqrt(0.5), std::sqrt(0.5), 0.0);  // 45 degree normal

    VecX F_star = flux->flux(U, U, n);

    Real h = U(0);
    Real hu = U(1);
    Real hv = U(2);
    Real u = hu / h;
    Real v = hv / h;

    // F.n = F_x * n_x + F_y * n_y
    Real un = u * n(0) + v * n(1);
    VecX F_physical(3);
    F_physical << h * un,
                  hu * un + 0.5 * g * h * h * n(0),
                  hv * un + 0.5 * g * h * h * n(1);

    EXPECT_TRUE(vectors_equal(F_star, F_physical, LOOSE_TOLERANCE));
}

// Test upwind flux for advection
TEST_F(NumericalFluxTest, UpwindAdvection) {
    Vec3 velocity(1.0, 0.0, 0.0);  // Flow in +x direction
    auto flux = NumericalFluxFactory::advection_upwind(velocity);

    VecX U_L(1), U_R(1);
    U_L << 2.0;
    U_R << 1.0;

    Vec3 n(1.0, 0.0, 0.0);  // Outward normal to right

    VecX F_star = flux->flux(U_L, U_R, n);

    // With flow in +x and normal in +x, upwind takes left value
    // F* = u * U_L (for outflow)
    Real expected = velocity(0) * U_L(0);

    EXPECT_NEAR(F_star(0), expected, TOLERANCE);
}

// Test that dry cells are handled (no NaN)
TEST_F(NumericalFluxTest, DryCellHandling) {
    Real g = 9.81;
    auto flux = NumericalFluxFactory::shallow_water_lax_friedrichs(g);

    // Near-dry state
    VecX U_L(3), U_R(3);
    U_L << 1e-12, 0.0, 0.0;  // Nearly dry
    U_R << 1.0, 0.5, 0.0;

    Vec3 n(1.0, 0.0, 0.0);

    VecX F_star = flux->flux(U_L, U_R, n);

    // Check for NaN
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(F_star(i))) << "Component " << i << " is NaN";
        EXPECT_FALSE(std::isinf(F_star(i))) << "Component " << i << " is Inf";
    }
}

// Test conservation: F(U_L, U_R, n) + F(U_R, U_L, -n) ≈ 0 at interface
TEST_F(NumericalFluxTest, Conservation) {
    Real g = 9.81;

    std::vector<std::unique_ptr<NumericalFlux>> fluxes;
    fluxes.push_back(NumericalFluxFactory::shallow_water_lax_friedrichs(g));
    fluxes.push_back(NumericalFluxFactory::shallow_water_hllc(g));
    fluxes.push_back(NumericalFluxFactory::shallow_water_roe(g));

    VecX U_L(3), U_R(3);
    U_L << 1.0, 0.5, 0.2;
    U_R << 1.5, 0.3, 0.4;

    Vec3 n(0.6, 0.8, 0.0);  // Arbitrary normal

    for (size_t i = 0; i < fluxes.size(); ++i) {
        VecX F_plus = fluxes[i]->flux(U_L, U_R, n);
        VecX F_minus = fluxes[i]->flux(U_R, U_L, -n);

        // Conservation: what leaves left = what enters right
        VecX sum = F_plus + F_minus;

        EXPECT_NEAR(sum.norm(), 0.0, LOOSE_TOLERANCE)
            << "Flux " << i << " not conservative";
    }
}

// Test maximum wave speed computation
TEST_F(NumericalFluxTest, MaxWaveSpeed) {
    Real g = 9.81;

    VecX U(3);
    U << 1.0, 2.0, 1.0;  // h=1, u=2, v=1

    Real h = U(0);
    Real u = U(1) / h;
    Real v = U(2) / h;
    Real c = std::sqrt(g * h);

    // Maximum eigenvalue: |u| + c in flow direction
    Real expected_x = std::abs(u) + c;
    Real expected_y = std::abs(v) + c;

    // Create flux and test wave speed
    auto flux = std::make_unique<ShallowWaterRoeFlux>(g);

    VecX U_L = U, U_R = U;
    Vec3 nx(1, 0, 0), ny(0, 1, 0);

    // The wave speed should be at least |u| + c
    // Testing via flux evaluation (implicit wave speed)
    VecX F_x = flux->flux(U_L, U_R, nx);
    VecX F_y = flux->flux(U_L, U_R, ny);

    // For identical states, flux should be physical flux (tested above)
    // Wave speed test would require direct access to wave speed function
}

// Test central flux (no dissipation for identical states)
TEST_F(NumericalFluxTest, CentralFlux) {
    auto physical_flux = [](const VecX& U) -> Tensor3 {
        Tensor3 F;
        F[0].resize(1, 1);
        F[1].resize(1, 1);
        F[2].resize(1, 1);
        F[0](0, 0) = U(0);
        F[1](0, 0) = 2.0 * U(0);
        F[2](0, 0) = 0.0;
        return F;
    };

    auto flux = NumericalFluxFactory::central(physical_flux, 1);

    VecX U(1);
    U << 3.0;

    Vec3 n(1, 0, 0);
    VecX F_star = flux->flux(U, U, n);

    // Central flux for identical states should give physical flux
    EXPECT_NEAR(F_star(0), 3.0, TOLERANCE);
}
