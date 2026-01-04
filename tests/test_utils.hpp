#pragma once

// Common test utilities for DRIFTER unit tests

#include <gtest/gtest.h>
#include "core/types.hpp"
#include <cmath>
#include <vector>

namespace drifter {
namespace testing {

/// @brief Floating point comparison tolerance
constexpr Real TOLERANCE = 1e-12;
constexpr Real LOOSE_TOLERANCE = 1e-8;

/// @brief Check if two Real values are approximately equal
inline bool approx_equal(Real a, Real b, Real tol = TOLERANCE) {
    return std::abs(a - b) < tol * (1.0 + std::max(std::abs(a), std::abs(b)));
}

/// @brief Check if two vectors are approximately equal
inline bool vectors_equal(const VecX& a, const VecX& b, Real tol = TOLERANCE) {
    if (a.size() != b.size()) return false;
    for (Index i = 0; i < a.size(); ++i) {
        if (!approx_equal(a(i), b(i), tol)) return false;
    }
    return true;
}

/// @brief Check if two matrices are approximately equal
inline bool matrices_equal(const MatX& a, const MatX& b, Real tol = TOLERANCE) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (Index i = 0; i < a.rows(); ++i) {
        for (Index j = 0; j < a.cols(); ++j) {
            if (!approx_equal(a(i, j), b(i, j), tol)) return false;
        }
    }
    return true;
}

/// @brief Generate LGL nodes for testing
inline VecX lgl_nodes(int n) {
    VecX nodes(n);
    if (n == 1) {
        nodes(0) = 0.0;
    } else if (n == 2) {
        nodes(0) = -1.0;
        nodes(1) = 1.0;
    } else if (n == 3) {
        nodes(0) = -1.0;
        nodes(1) = 0.0;
        nodes(2) = 1.0;
    } else if (n == 4) {
        nodes(0) = -1.0;
        nodes(1) = -std::sqrt(1.0/5.0);
        nodes(2) = std::sqrt(1.0/5.0);
        nodes(3) = 1.0;
    } else if (n == 5) {
        nodes(0) = -1.0;
        nodes(1) = -std::sqrt(3.0/7.0);
        nodes(2) = 0.0;
        nodes(3) = std::sqrt(3.0/7.0);
        nodes(4) = 1.0;
    } else {
        // Uniform for higher orders (not accurate but sufficient for basic tests)
        for (int i = 0; i < n; ++i) {
            nodes(i) = -1.0 + 2.0 * i / (n - 1);
        }
    }
    return nodes;
}

/// @brief Polynomial evaluation at a point
inline Real evaluate_polynomial(const VecX& coeffs, Real x) {
    Real result = 0.0;
    Real x_power = 1.0;
    for (Index i = 0; i < coeffs.size(); ++i) {
        result += coeffs(i) * x_power;
        x_power *= x;
    }
    return result;
}

/// @brief Polynomial derivative evaluation
inline Real evaluate_polynomial_derivative(const VecX& coeffs, Real x) {
    Real result = 0.0;
    Real x_power = 1.0;
    for (Index i = 1; i < coeffs.size(); ++i) {
        result += i * coeffs(i) * x_power;
        x_power *= x;
    }
    return result;
}

/// @brief Create a test function: f(x,y,z) = x^2 + 2*y^2 + 3*z^2
inline Real test_function_1(Real x, Real y, Real z) {
    return x*x + 2.0*y*y + 3.0*z*z;
}

/// @brief Gradient of test_function_1
inline Vec3 test_function_1_gradient(Real x, Real y, Real z) {
    return Vec3(2.0*x, 4.0*y, 6.0*z);
}

/// @brief Create a test function: f(x,y,z) = sin(pi*x)*cos(pi*y)*sin(pi*z)
inline Real test_function_2(Real x, Real y, Real z) {
    constexpr Real pi = 3.14159265358979323846;
    return std::sin(pi * x) * std::cos(pi * y) * std::sin(pi * z);
}

/// @brief Assert that integration of a constant is exact
#define ASSERT_INTEGRATION_CONSTANT(quad, value, expected) \
    ASSERT_NEAR((quad).integrate([](Real, Real, Real) { return (value); }), \
                (expected), TOLERANCE)

/// @brief Assert that differentiation of a polynomial is exact
#define ASSERT_DIFFERENTIATION_EXACT(D, u, du_expected) \
    ASSERT_TRUE(vectors_equal((D) * (u), (du_expected), LOOSE_TOLERANCE))

/// @brief Test fixture base class with common setup
class DrifterTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }

    void TearDown() override {
        // Common cleanup if needed
    }

    // Helper to create a simple hex element
    std::vector<Vec3> create_unit_cube_nodes(int order) {
        int n = order + 1;
        std::vector<Vec3> nodes;
        nodes.reserve(n * n * n);

        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    Real x = -1.0 + 2.0 * i / order;
                    Real y = -1.0 + 2.0 * j / order;
                    Real z = -1.0 + 2.0 * k / order;
                    nodes.push_back(Vec3(x, y, z));
                }
            }
        }

        return nodes;
    }
};

}  // namespace testing
}  // namespace drifter
