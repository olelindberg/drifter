#include <gtest/gtest.h>
#include "physics/mode_splitting.hpp"
#include "physics/primitive_equations.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class ModeSplittingTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
        order_ = 2;
        n1d_ = order_ + 1;
        n_horiz_ = n1d_ * n1d_;
        n_vert_ = n1d_;
        n_total_ = n_horiz_ * n_vert_;

        // Create basis and quadrature
        basis_ = std::make_unique<HexahedronBasis>(order_);
        quad_ = std::make_unique<GaussQuadrature3D>(order_, QuadratureType::GaussLegendre);

        // Create element
        element_ = std::make_unique<ModeSplittingElement>(*basis_, *quad_);

        // Set up flat bathymetry at 100m depth
        VecX h(n_horiz_), dh_dx(n_horiz_), dh_dy(n_horiz_);
        h.setConstant(100.0);
        dh_dx.setZero();
        dh_dy.setZero();
        element_->set_bathymetry(h, dh_dx, dh_dy);

        // Set Coriolis parameter (mid-latitudes)
        VecX f(n_horiz_);
        f.setConstant(1e-4);  // f = 10^-4 s^-1
        element_->set_coriolis(f);
    }

    int order_;
    int n1d_;
    int n_horiz_;
    int n_vert_;
    int n_total_;

    std::unique_ptr<HexahedronBasis> basis_;
    std::unique_ptr<GaussQuadrature3D> quad_;
    std::unique_ptr<ModeSplittingElement> element_;
};

// Test that depth averaging of a constant gives the same constant
TEST_F(ModeSplittingTest, DepthAverageConstant) {
    VecX u_3d(n_total_);
    u_3d.setConstant(1.5);

    VecX U_bar;
    element_->depth_average(u_3d, U_bar);

    ASSERT_EQ(U_bar.size(), n_horiz_);
    for (int i = 0; i < n_horiz_; ++i) {
        EXPECT_NEAR(U_bar(i), 1.5, TOLERANCE);
    }
}

// Test that depth averaging is linear
TEST_F(ModeSplittingTest, DepthAverageLinear) {
    VecX u1_3d(n_total_), u2_3d(n_total_);
    u1_3d.setRandom();
    u2_3d.setRandom();

    Real alpha = 2.5, beta = -1.3;
    VecX u_combined = alpha * u1_3d + beta * u2_3d;

    VecX U1_bar, U2_bar, U_combined_bar;
    element_->depth_average(u1_3d, U1_bar);
    element_->depth_average(u2_3d, U2_bar);
    element_->depth_average(u_combined, U_combined_bar);

    VecX expected = alpha * U1_bar + beta * U2_bar;

    for (int i = 0; i < n_horiz_; ++i) {
        EXPECT_NEAR(U_combined_bar(i), expected(i), LOOSE_TOLERANCE);
    }
}

// Test that depth-integrated transport is consistent
TEST_F(ModeSplittingTest, DepthIntegrate) {
    VecX u_3d(n_total_);
    u_3d.setConstant(0.5);  // 0.5 m/s

    VecX H(n_total_);
    H.setConstant(100.0);  // 100m depth

    VecX HU_bar;
    element_->depth_integrate(u_3d, H, HU_bar);

    // For constant u and H, HU_bar should be approximately u * H
    // (depends on quadrature weights)
    ASSERT_EQ(HU_bar.size(), n_horiz_);
    for (int i = 0; i < n_horiz_; ++i) {
        EXPECT_GT(HU_bar(i), 0.0);
    }
}

// Test that compute_deviation satisfies u' = u - U_bar
TEST_F(ModeSplittingTest, ComputeDeviation) {
    VecX u_3d(n_total_);
    u_3d.setRandom();

    VecX U_bar;
    element_->depth_average(u_3d, U_bar);

    VecX u_prime;
    element_->compute_deviation(u_3d, U_bar, u_prime);

    ASSERT_EQ(u_prime.size(), n_total_);

    // Check u' = u - U_bar for each column
    for (int i_h = 0; i_h < n_horiz_; ++i_h) {
        for (int k = 0; k < n_vert_; ++k) {
            int idx = i_h * n_vert_ + k;
            EXPECT_NEAR(u_prime(idx), u_3d(idx) - U_bar(i_h), TOLERANCE);
        }
    }
}

// Test that depth average of deviation is zero
TEST_F(ModeSplittingTest, DeviationAveragesZero) {
    VecX u_3d(n_total_);
    u_3d.setRandom();

    VecX U_bar;
    element_->depth_average(u_3d, U_bar);

    VecX u_prime;
    element_->compute_deviation(u_3d, U_bar, u_prime);

    VecX U_prime_bar;
    element_->depth_average(u_prime, U_prime_bar);

    // Depth average of deviation should be zero
    for (int i = 0; i < n_horiz_; ++i) {
        EXPECT_NEAR(U_prime_bar(i), 0.0, LOOSE_TOLERANCE);
    }
}

// Test barotropic RHS for quiescent ocean
TEST_F(ModeSplittingTest, BarotropicRHSQuiescent) {
    BarotropicState state;
    state.resize(n_horiz_);
    state.eta.setZero();
    state.HU_bar.setZero();
    state.HV_bar.setZero();
    state.H.setConstant(100.0);
    state.U_bar.setZero();
    state.V_bar.setZero();

    VecX forcing_x(n_horiz_), forcing_y(n_horiz_);
    forcing_x.setZero();
    forcing_y.setZero();

    BarotropicTendencies tendency;
    element_->barotropic_rhs(state, forcing_x, forcing_y, tendency);

    // For quiescent ocean with flat surface, all tendencies should be zero
    for (int i = 0; i < n_horiz_; ++i) {
        EXPECT_NEAR(tendency.deta_dt(i), 0.0, TOLERANCE);
        EXPECT_NEAR(tendency.dHU_bar_dt(i), 0.0, TOLERANCE);
        EXPECT_NEAR(tendency.dHV_bar_dt(i), 0.0, TOLERANCE);
    }
}

// Test 2D divergence computation
TEST_F(ModeSplittingTest, Divergence2D) {
    // For constant transport, divergence should be zero
    VecX HU_bar(n_horiz_), HV_bar(n_horiz_);
    HU_bar.setConstant(100.0);
    HV_bar.setConstant(50.0);

    VecX div = element_->compute_2d_divergence(HU_bar, HV_bar);

    ASSERT_EQ(div.size(), n_horiz_);
    for (int i = 0; i < n_horiz_; ++i) {
        EXPECT_NEAR(div(i), 0.0, TOLERANCE);
    }
}

// Test that barotropic step preserves mass for closed domain
TEST_F(ModeSplittingTest, BarotropicStepConservation) {
    BarotropicState state;
    state.resize(n_horiz_);
    state.eta.setConstant(0.1);  // Small positive elevation
    state.H.setConstant(100.1);
    state.HU_bar.setConstant(0.0);
    state.HV_bar.setConstant(0.0);
    state.U_bar.setZero();
    state.V_bar.setZero();

    VecX forcing_x(n_horiz_), forcing_y(n_horiz_);
    forcing_x.setZero();
    forcing_y.setZero();

    Real total_eta_before = state.eta.sum();

    // Do a barotropic step
    Real dt = 0.1;
    element_->barotropic_step_euler(dt, forcing_x, forcing_y, state);

    Real total_eta_after = state.eta.sum();

    // Total eta should be conserved (approximately, depends on BC handling)
    EXPECT_NEAR(total_eta_after, total_eta_before, LOOSE_TOLERANCE);
}

// =============================================================================
// Primitive Equations Tests
// =============================================================================

class PrimitiveEquationsTest : public DrifterTestBase {
protected:
    void SetUp() override {
        DrifterTestBase::SetUp();
        order_ = 2;
        n1d_ = order_ + 1;
        n_horiz_ = n1d_ * n1d_;
        n_vert_ = n1d_;
        n_total_ = n_horiz_ * n_vert_;

        // Create basis and quadrature
        basis_ = std::make_unique<HexahedronBasis>(order_);
        quad_ = std::make_unique<GaussQuadrature3D>(order_, QuadratureType::GaussLegendre);

        // Create element
        element_ = std::make_unique<PrimitiveEquationsElement>(*basis_, *quad_);

        // Set up flat bathymetry
        VecX h(n_total_), dh_dx(n_total_), dh_dy(n_total_);
        h.setConstant(100.0);
        dh_dx.setZero();
        dh_dy.setZero();
        element_->set_bathymetry(h, dh_dx, dh_dy);

        // Set Coriolis
        VecX y_positions(n_total_);
        y_positions.setZero();
        CoriolisParameter coriolis(1e-4, 0.0, 0.0);
        element_->set_coriolis(coriolis, y_positions);
    }

    int order_;
    int n1d_;
    int n_horiz_;
    int n_vert_;
    int n_total_;

    std::unique_ptr<HexahedronBasis> basis_;
    std::unique_ptr<GaussQuadrature3D> quad_;
    std::unique_ptr<PrimitiveEquationsElement> element_;
};

// Test state resize
TEST_F(PrimitiveEquationsTest, StateResize) {
    PrimitiveState state;
    state.resize(n_total_);

    EXPECT_EQ(state.Hu.size(), n_total_);
    EXPECT_EQ(state.Hv.size(), n_total_);
    EXPECT_EQ(state.eta.size(), n_total_);
    EXPECT_EQ(state.HT.size(), n_total_);
    EXPECT_EQ(state.HS.size(), n_total_);
    EXPECT_EQ(state.H.size(), n_total_);
    EXPECT_EQ(state.u.size(), n_total_);
    EXPECT_EQ(state.v.size(), n_total_);
    EXPECT_EQ(state.T.size(), n_total_);
    EXPECT_EQ(state.S.size(), n_total_);
    EXPECT_EQ(state.omega.size(), n_total_);
    EXPECT_EQ(state.rho.size(), n_total_);
}

// Test update derived for PrimitiveState
TEST_F(PrimitiveEquationsTest, UpdateDerived) {
    PrimitiveState state;
    state.resize(n_total_);

    // Set conserved variables
    state.eta.setConstant(0.5);
    state.Hu.setConstant(50.0);  // H*u = 100 * 0.5
    state.Hv.setConstant(25.0);  // H*v = 100 * 0.25
    state.HT.setConstant(1000.0);  // H*T
    state.HS.setConstant(3500.0);  // H*S

    VecX h(n_total_);
    h.setConstant(100.0);

    state.update_derived(h);

    // Check H = eta + h
    for (int i = 0; i < n_total_; ++i) {
        EXPECT_NEAR(state.H(i), 100.5, TOLERANCE);
    }

    // Check u = Hu / H
    for (int i = 0; i < n_total_; ++i) {
        EXPECT_NEAR(state.u(i), 50.0 / 100.5, LOOSE_TOLERANCE);
    }

    // Check v = Hv / H
    for (int i = 0; i < n_total_; ++i) {
        EXPECT_NEAR(state.v(i), 25.0 / 100.5, LOOSE_TOLERANCE);
    }
}

// Test tendency resize and set_zero
TEST_F(PrimitiveEquationsTest, TendencyResize) {
    PrimitiveTendencies tendency;
    tendency.resize(n_total_);

    EXPECT_EQ(tendency.dHu_dt.size(), n_total_);
    EXPECT_EQ(tendency.dHv_dt.size(), n_total_);
    EXPECT_EQ(tendency.deta_dt.size(), n_total_);
    EXPECT_EQ(tendency.dHT_dt.size(), n_total_);
    EXPECT_EQ(tendency.dHS_dt.size(), n_total_);

    tendency.set_zero();

    for (int i = 0; i < n_total_; ++i) {
        EXPECT_EQ(tendency.dHu_dt(i), 0.0);
        EXPECT_EQ(tendency.dHv_dt(i), 0.0);
        EXPECT_EQ(tendency.deta_dt(i), 0.0);
        EXPECT_EQ(tendency.dHT_dt(i), 0.0);
        EXPECT_EQ(tendency.dHS_dt(i), 0.0);
    }
}

// Test Coriolis terms for geostrophic balance
TEST_F(PrimitiveEquationsTest, CoriolisTerms) {
    PrimitiveState state;
    state.resize(n_total_);

    // Set up geostrophic flow
    state.H.setConstant(100.0);
    state.u.setConstant(0.0);
    state.v.setConstant(1.0);  // 1 m/s northward
    state.Hu = state.H.cwiseProduct(state.u);
    state.Hv = state.H.cwiseProduct(state.v);

    VecX cor_Hu, cor_Hv;
    element_->coriolis_terms(state, cor_Hu, cor_Hv);

    // Coriolis force on northward flow should be eastward (positive x)
    // cor_Hu = f * Hv
    for (int i = 0; i < n_total_; ++i) {
        EXPECT_GT(cor_Hu(i), 0.0);
    }
}

// Test compute_eta_rhs for quiescent ocean
TEST_F(PrimitiveEquationsTest, EtaRHSQuiescent) {
    PrimitiveState state;
    state.resize(n_total_);

    state.Hu.setZero();
    state.Hv.setZero();
    state.H.setConstant(100.0);

    VecX deta_dt;
    element_->compute_eta_rhs(state, deta_dt);

    // For zero transport, deta/dt should be zero
    // Note: deta_dt is returned on the 3D grid (same value at all vertical levels)
    ASSERT_EQ(deta_dt.size(), n_total_);
    for (int i = 0; i < n_total_; ++i) {
        EXPECT_NEAR(deta_dt(i), 0.0, TOLERANCE);
    }
}

// Test that tracer advection with zero velocity gives zero tendency
TEST_F(PrimitiveEquationsTest, TracerAdvectionZeroVelocity) {
    PrimitiveState state;
    state.resize(n_total_);

    state.u.setZero();
    state.v.setZero();
    state.omega.setZero();
    state.H.setConstant(100.0);
    state.Hu.setZero();
    state.Hv.setZero();

    VecX HT(n_total_);
    HT.setConstant(1000.0);

    VecX adv_HT;
    element_->tracer_advection(state, HT, adv_HT);

    // Zero velocity means zero advection
    ASSERT_EQ(adv_HT.size(), n_total_);
    for (int i = 0; i < n_total_; ++i) {
        EXPECT_NEAR(adv_HT(i), 0.0, TOLERANCE);
    }
}
