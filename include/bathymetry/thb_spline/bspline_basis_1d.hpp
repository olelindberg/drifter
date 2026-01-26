#pragma once

#include "core/types.hpp"
#include <vector>

namespace drifter {

/**
 * @brief 1D cubic B-spline basis evaluation
 *
 * Implements uniform cubic B-splines with open (clamped) knot vectors at domain
 * boundaries. Uses Cox-de Boor recursion for stable evaluation.
 *
 * Cubic B-splines (degree 3) provide C² continuity, making them suitable for
 * smooth surface fitting without explicit continuity constraints.
 *
 * Knot vector structure for n spans:
 *   [0, 0, 0, 0, 1, 2, ..., n-1, n, n, n, n]
 *   (4 repeated knots at each end for clamping, uniform interior)
 *
 * Number of basis functions = n + 3 (for cubic)
 */
class BSplineBasis1D {
  public:
    static constexpr int DEGREE = 3;  // Cubic

    /**
     * @brief Construct a 1D B-spline basis
     * @param num_spans Number of uniform spans (each span has one polynomial piece)
     *
     * Creates an open (clamped) uniform knot vector with num_spans interior spans.
     * Domain is [0, num_spans] in parameter space.
     */
    explicit BSplineBasis1D(int num_spans);

    /// Number of basis functions = num_spans + degree
    int num_basis() const { return num_spans_ + DEGREE; }

    /// Number of spans (polynomial pieces)
    int num_spans() const { return num_spans_; }

    /// Domain bounds in parameter space
    Real domain_min() const { return 0.0; }
    Real domain_max() const { return static_cast<Real>(num_spans_); }

    /// Access the knot vector
    const std::vector<Real>& knots() const { return knots_; }

    /**
     * @brief Find the span index containing parameter t
     * @param t Parameter value in [0, num_spans]
     * @return Span index i such that knot[i] <= t < knot[i+1]
     *
     * Returns degree for t <= 0, and num_spans + degree - 1 for t >= num_spans.
     */
    int find_span(Real t) const;

    /**
     * @brief Evaluate all non-zero basis functions at parameter t
     * @param t Parameter value in [0, num_spans]
     * @return Vector of 4 values (N_{span-3}, N_{span-2}, N_{span-1}, N_{span})
     *
     * Uses Cox-de Boor recursion for numerical stability.
     */
    VecX evaluate_nonzero(Real t) const;

    /**
     * @brief Evaluate all non-zero basis functions and their derivatives
     * @param t Parameter value in [0, num_spans]
     * @param num_derivs Number of derivatives to compute (0, 1, or 2)
     * @return Matrix where row k is the k-th derivative (k=0 is value)
     *
     * Returns (num_derivs+1) x 4 matrix.
     */
    MatX evaluate_nonzero_derivs(Real t, int num_derivs) const;

    /**
     * @brief Evaluate a single basis function at parameter t
     * @param i Basis function index (0 to num_basis()-1)
     * @param t Parameter value
     * @return Basis function value N_i(t)
     */
    Real evaluate(int i, Real t) const;

    /**
     * @brief Evaluate first derivative of basis function i at t
     * @param i Basis function index
     * @param t Parameter value
     * @return dN_i/dt
     */
    Real evaluate_derivative(int i, Real t) const;

    /**
     * @brief Evaluate second derivative of basis function i at t
     * @param i Basis function index
     * @param t Parameter value
     * @return d²N_i/dt²
     */
    Real evaluate_second_derivative(int i, Real t) const;

    /**
     * @brief Evaluate all basis functions at parameter t
     * @param t Parameter value
     * @return Vector of all num_basis() values
     */
    VecX evaluate_all(Real t) const;

    /**
     * @brief Evaluate all basis function derivatives at parameter t
     * @param t Parameter value
     * @return Vector of all num_basis() first derivatives
     */
    VecX evaluate_all_derivatives(Real t) const;

    /**
     * @brief Evaluate all basis function second derivatives at parameter t
     * @param t Parameter value
     * @return Vector of all num_basis() second derivatives
     */
    VecX evaluate_all_second_derivatives(Real t) const;

    /**
     * @brief Get the support interval of basis function i
     * @param i Basis function index
     * @return Pair (t_min, t_max) where basis function is non-zero
     */
    std::pair<Real, Real> support(int i) const;

    /**
     * @brief Two-scale refinement coefficients for subdivision
     *
     * For cubic B-splines with dyadic refinement:
     *   B^l_j(t) = Σ_{k=0}^{4} c_k × B^{l+1}_{2j+k-1}(t)
     *
     * @return Vector of 5 refinement coefficients [1/8, 4/8, 6/8, 4/8, 1/8]
     */
    static std::vector<Real> refinement_coefficients();

  private:
    int num_spans_;
    std::vector<Real> knots_;

    /// Build the open (clamped) uniform knot vector
    void build_knot_vector();

    /// Cox-de Boor recursion for basis evaluation
    Real cox_de_boor(int i, int p, Real t) const;

    /// Derivative of Cox-de Boor (recursive formula)
    Real cox_de_boor_derivative(int i, int p, Real t, int deriv) const;
};

}  // namespace drifter
