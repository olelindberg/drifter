#include "bathymetry/thb_spline/bspline_basis_1d.hpp"
#include "bathymetry/thb_spline/bspline_basis_2d.hpp"
#include "bathymetry/thb_spline/bspline_knot_vector.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace drifter {
namespace {

constexpr Real TOLERANCE = 1e-10;

// =============================================================================
// BSplineBasis1D Tests
// =============================================================================

class BSplineBasis1DTest : public ::testing::Test {
  protected:
    void SetUp() override { basis_ = std::make_unique<BSplineBasis1D>(4); }

    std::unique_ptr<BSplineBasis1D> basis_;
};

TEST_F(BSplineBasis1DTest, Construction) {
    // 4 spans means 4 + 3 = 7 basis functions for cubic
    EXPECT_EQ(basis_->num_spans(), 4);
    EXPECT_EQ(basis_->num_basis(), 7);
    EXPECT_EQ(basis_->domain_min(), 0.0);
    EXPECT_EQ(basis_->domain_max(), 4.0);
}

TEST_F(BSplineBasis1DTest, KnotVector) {
    const auto& knots = basis_->knots();
    // Open cubic: [0,0,0,0, 1,2,3, 4,4,4,4]
    EXPECT_EQ(knots.size(), 12u);  // 4 + 8 = 12 knots

    // First 4 knots are 0
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(knots[i], 0.0);
    }

    // Interior knots
    EXPECT_DOUBLE_EQ(knots[4], 1.0);
    EXPECT_DOUBLE_EQ(knots[5], 2.0);
    EXPECT_DOUBLE_EQ(knots[6], 3.0);

    // Last 4 knots are 4
    for (int i = 7; i < 11; ++i) {
        EXPECT_DOUBLE_EQ(knots[i], 4.0);
    }
}

TEST_F(BSplineBasis1DTest, PartitionOfUnity) {
    // Sum of all basis functions should equal 1 at any point
    const int num_samples = 100;
    for (int i = 0; i <= num_samples; ++i) {
        Real t = basis_->domain_min() +
                 (basis_->domain_max() - basis_->domain_min()) * i / num_samples;
        VecX vals = basis_->evaluate_all(t);
        Real sum = vals.sum();
        EXPECT_NEAR(sum, 1.0, TOLERANCE) << "at t = " << t;
    }
}

TEST_F(BSplineBasis1DTest, NonNegativity) {
    // B-spline basis functions are non-negative
    const int num_samples = 100;
    for (int i = 0; i <= num_samples; ++i) {
        Real t = basis_->domain_min() +
                 (basis_->domain_max() - basis_->domain_min()) * i / num_samples;
        VecX vals = basis_->evaluate_all(t);
        for (int j = 0; j < vals.size(); ++j) {
            EXPECT_GE(vals(j), -TOLERANCE) << "Basis " << j << " at t = " << t;
        }
    }
}

TEST_F(BSplineBasis1DTest, LocalSupport) {
    // Each basis function has support on at most degree+1 = 4 spans
    for (int i = 0; i < basis_->num_basis(); ++i) {
        auto [t_min, t_max] = basis_->support(i);
        Real support_width = t_max - t_min;
        // Support width should be at most 4 spans
        EXPECT_LE(support_width, 4.0 + TOLERANCE);
    }
}

TEST_F(BSplineBasis1DTest, BoundaryInterpolation) {
    // Open knot vectors interpolate at endpoints
    // At t=0, only the first basis function is non-zero (value = 1)
    VecX vals_start = basis_->evaluate_all(0.0);
    EXPECT_NEAR(vals_start(0), 1.0, TOLERANCE);
    for (int i = 1; i < basis_->num_basis(); ++i) {
        EXPECT_NEAR(vals_start(i), 0.0, TOLERANCE);
    }

    // At t=4, only the last basis function is non-zero (value = 1)
    VecX vals_end = basis_->evaluate_all(4.0);
    EXPECT_NEAR(vals_end(basis_->num_basis() - 1), 1.0, TOLERANCE);
    for (int i = 0; i < basis_->num_basis() - 1; ++i) {
        EXPECT_NEAR(vals_end(i), 0.0, TOLERANCE);
    }
}

TEST_F(BSplineBasis1DTest, DerivativePartitionOfUnity) {
    // Derivatives of partition of unity sum to 0
    const int num_samples = 50;
    for (int i = 1; i < num_samples; ++i) {  // Avoid boundary
        Real t = basis_->domain_min() +
                 (basis_->domain_max() - basis_->domain_min()) * i / num_samples;
        VecX derivs = basis_->evaluate_all_derivatives(t);
        Real sum = derivs.sum();
        EXPECT_NEAR(sum, 0.0, TOLERANCE) << "at t = " << t;
    }
}

TEST_F(BSplineBasis1DTest, RefinementCoefficients) {
    auto coeffs = BSplineBasis1D::refinement_coefficients();
    EXPECT_EQ(coeffs.size(), 5u);

    // Refinement mask for B-splines sums to 2 (standard subdivision property)
    // Mask: [1, 4, 6, 4, 1] / 8 = [1/8, 4/8, 6/8, 4/8, 1/8]
    Real sum = 0.0;
    for (Real c : coeffs) {
        sum += c;
    }
    EXPECT_NEAR(sum, 2.0, TOLERANCE);

    // Symmetric
    EXPECT_DOUBLE_EQ(coeffs[0], coeffs[4]);
    EXPECT_DOUBLE_EQ(coeffs[1], coeffs[3]);

    // Check specific values
    EXPECT_NEAR(coeffs[0], 1.0 / 8.0, TOLERANCE);
    EXPECT_NEAR(coeffs[1], 4.0 / 8.0, TOLERANCE);
    EXPECT_NEAR(coeffs[2], 6.0 / 8.0, TOLERANCE);
}

// =============================================================================
// BSplineKnotVector Tests
// =============================================================================

class BSplineKnotVectorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Domain [0, 10], 2 spans at level 0, max level 3
        knots_ = std::make_unique<BSplineKnotVector>(0.0, 10.0, 2, 3);
    }

    std::unique_ptr<BSplineKnotVector> knots_;
};

TEST_F(BSplineKnotVectorTest, Construction) {
    EXPECT_DOUBLE_EQ(knots_->domain_min(), 0.0);
    EXPECT_DOUBLE_EQ(knots_->domain_max(), 10.0);
    EXPECT_EQ(knots_->num_spans_level0(), 2);
    EXPECT_EQ(knots_->max_level(), 3);
}

TEST_F(BSplineKnotVectorTest, DyadicRefinement) {
    // Level l has 2^l * num_spans_level0 spans
    EXPECT_EQ(knots_->num_spans(0), 2);
    EXPECT_EQ(knots_->num_spans(1), 4);
    EXPECT_EQ(knots_->num_spans(2), 8);
    EXPECT_EQ(knots_->num_spans(3), 16);
}

TEST_F(BSplineKnotVectorTest, SpanSize) {
    // Span size halves with each level
    EXPECT_DOUBLE_EQ(knots_->span_size(0), 5.0);
    EXPECT_DOUBLE_EQ(knots_->span_size(1), 2.5);
    EXPECT_DOUBLE_EQ(knots_->span_size(2), 1.25);
    EXPECT_DOUBLE_EQ(knots_->span_size(3), 0.625);
}

TEST_F(BSplineKnotVectorTest, NumBasis) {
    // num_basis = num_spans + degree (cubic)
    EXPECT_EQ(knots_->num_basis(0), 5);
    EXPECT_EQ(knots_->num_basis(1), 7);
    EXPECT_EQ(knots_->num_basis(2), 11);
    EXPECT_EQ(knots_->num_basis(3), 19);
}

TEST_F(BSplineKnotVectorTest, ParameterMapping) {
    // Physical x = 5 should map to t = 1.0 at level 0 (middle of domain)
    Real t = knots_->physical_to_parameter(0, 5.0);
    EXPECT_NEAR(t, 1.0, TOLERANCE);

    // At level 1, same physical point maps to t = 2.0
    t = knots_->physical_to_parameter(1, 5.0);
    EXPECT_NEAR(t, 2.0, TOLERANCE);

    // Inverse mapping
    Real x = knots_->parameter_to_physical(0, 1.0);
    EXPECT_NEAR(x, 5.0, TOLERANCE);
}

TEST_F(BSplineKnotVectorTest, LevelForSpanSize) {
    // Find level matching span size
    EXPECT_EQ(knots_->level_for_span_size(5.0), 0);
    EXPECT_EQ(knots_->level_for_span_size(2.5), 1);
    EXPECT_EQ(knots_->level_for_span_size(1.25), 2);
    EXPECT_EQ(knots_->level_for_span_size(0.625), 3);

    // Intermediate values should round to nearest
    EXPECT_EQ(knots_->level_for_span_size(3.5), 1);  // Closer to 2.5 than 5.0
}

TEST_F(BSplineKnotVectorTest, FromOctreeSizes) {
    // Create from octree-like element sizes
    auto kv = BSplineKnotVector::from_octree_sizes(0.0, 100.0, 50.0, 6.25);

    EXPECT_DOUBLE_EQ(kv.domain_min(), 0.0);
    EXPECT_DOUBLE_EQ(kv.domain_max(), 100.0);

    // Level 0 should have span size close to max_element_size
    EXPECT_LE(kv.span_size(0), 50.0 + TOLERANCE);

    // Max level should have span size close to min_element_size
    EXPECT_LE(kv.span_size(kv.max_level()), 6.25 + TOLERANCE);
}

// =============================================================================
// BSplineBasis2D Tests
// =============================================================================

class BSplineBasis2DTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // 3 x 4 spans
        basis_ = std::make_unique<BSplineBasis2D>(3, 4);
    }

    std::unique_ptr<BSplineBasis2D> basis_;
};

TEST_F(BSplineBasis2DTest, Construction) {
    EXPECT_EQ(basis_->num_spans_u(), 3);
    EXPECT_EQ(basis_->num_spans_v(), 4);
    EXPECT_EQ(basis_->num_basis_u(), 6);  // 3 + 3
    EXPECT_EQ(basis_->num_basis_v(), 7);  // 4 + 3
    EXPECT_EQ(basis_->num_basis(), 42);   // 6 * 7
}

TEST_F(BSplineBasis2DTest, DOFIndexing) {
    // dof = i + num_basis_u * j
    EXPECT_EQ(basis_->dof_index(0, 0), 0);
    EXPECT_EQ(basis_->dof_index(1, 0), 1);
    EXPECT_EQ(basis_->dof_index(0, 1), 6);
    EXPECT_EQ(basis_->dof_index(5, 6), 41);

    // Inverse
    auto [i, j] = basis_->dof_to_ij(13);
    EXPECT_EQ(i, 1);
    EXPECT_EQ(j, 2);
}

TEST_F(BSplineBasis2DTest, PartitionOfUnity2D) {
    // Sum of all 2D basis functions should equal 1
    const int num_samples = 20;
    for (int j = 0; j <= num_samples; ++j) {
        for (int i = 0; i <= num_samples; ++i) {
            Real u = static_cast<Real>(i) / num_samples * basis_->num_spans_u();
            Real v = static_cast<Real>(j) / num_samples * basis_->num_spans_v();
            VecX vals = basis_->evaluate_all(u, v);
            Real sum = vals.sum();
            EXPECT_NEAR(sum, 1.0, TOLERANCE) << "at (u, v) = (" << u << ", " << v << ")";
        }
    }
}

TEST_F(BSplineBasis2DTest, TensorProduct) {
    // 2D basis should be tensor product of 1D bases
    Real u = 1.5;
    Real v = 2.0;

    VecX vals_2d = basis_->evaluate_all(u, v);
    VecX vals_u = basis_->basis_u().evaluate_all(u);
    VecX vals_v = basis_->basis_v().evaluate_all(v);

    for (int j = 0; j < basis_->num_basis_v(); ++j) {
        for (int i = 0; i < basis_->num_basis_u(); ++i) {
            Real expected = vals_u(i) * vals_v(j);
            Real actual = vals_2d(basis_->dof_index(i, j));
            EXPECT_NEAR(actual, expected, TOLERANCE)
                << "at (i, j) = (" << i << ", " << j << ")";
        }
    }
}

TEST_F(BSplineBasis2DTest, DerivativesTensorProduct) {
    Real u = 1.5;
    Real v = 2.0;

    for (int i = 0; i < basis_->num_basis_u(); ++i) {
        for (int j = 0; j < basis_->num_basis_v(); ++j) {
            // du derivative
            Real du_val = basis_->evaluate_du(i, j, u, v);
            Real expected_du = basis_->basis_u().evaluate_derivative(i, u) *
                               basis_->basis_v().evaluate(j, v);
            EXPECT_NEAR(du_val, expected_du, TOLERANCE);

            // dv derivative
            Real dv_val = basis_->evaluate_dv(i, j, u, v);
            Real expected_dv = basis_->basis_u().evaluate(i, u) *
                               basis_->basis_v().evaluate_derivative(j, v);
            EXPECT_NEAR(dv_val, expected_dv, TOLERANCE);
        }
    }
}

TEST_F(BSplineBasis2DTest, Support2D) {
    // Test support computation
    auto [u_min, u_max, v_min, v_max] = basis_->support(2, 3);

    // Check it's consistent with 1D supports
    auto [u_min_1d, u_max_1d] = basis_->basis_u().support(2);
    auto [v_min_1d, v_max_1d] = basis_->basis_v().support(3);

    EXPECT_DOUBLE_EQ(u_min, u_min_1d);
    EXPECT_DOUBLE_EQ(u_max, u_max_1d);
    EXPECT_DOUBLE_EQ(v_min, v_min_1d);
    EXPECT_DOUBLE_EQ(v_max, v_max_1d);
}

}  // namespace
}  // namespace drifter
