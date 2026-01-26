#pragma once

#include "bathymetry/thb_spline/bspline_basis_1d.hpp"
#include "bathymetry/thb_spline/bspline_knot_vector.hpp"
#include "core/types.hpp"
#include <memory>
#include <utility>

namespace drifter {

/**
 * @brief 2D tensor-product B-spline basis evaluation
 *
 * Combines two 1D B-spline bases (u and v directions) via tensor product.
 * Supports evaluation at a single level within a hierarchical structure.
 *
 * DOF indexing: dof = i + num_basis_u * j
 * where i is the u-direction index and j is the v-direction index.
 */
class BSplineBasis2D {
  public:
    static constexpr int DEGREE = 3;  // Cubic

    /**
     * @brief Construct a 2D basis from hierarchical knot vectors
     * @param knots_u Knot vector for u-direction
     * @param knots_v Knot vector for v-direction
     * @param level Refinement level for evaluation
     */
    BSplineBasis2D(const BSplineKnotVector& knots_u, const BSplineKnotVector& knots_v,
                   int level);

    /**
     * @brief Construct a uniform 2D basis at a single level
     * @param num_spans_u Number of spans in u-direction
     * @param num_spans_v Number of spans in v-direction
     */
    BSplineBasis2D(int num_spans_u, int num_spans_v);

    /// Number of basis functions in u-direction
    int num_basis_u() const { return basis_u_.num_basis(); }

    /// Number of basis functions in v-direction
    int num_basis_v() const { return basis_v_.num_basis(); }

    /// Total number of basis functions
    int num_basis() const { return num_basis_u() * num_basis_v(); }

    /// Number of spans in u-direction
    int num_spans_u() const { return basis_u_.num_spans(); }

    /// Number of spans in v-direction
    int num_spans_v() const { return basis_v_.num_spans(); }

    /// Current refinement level
    int level() const { return level_; }

    /// Access underlying 1D bases
    const BSplineBasis1D& basis_u() const { return basis_u_; }
    const BSplineBasis1D& basis_v() const { return basis_v_; }

    /**
     * @brief Convert 2D index to global DOF index
     * @param i u-direction index
     * @param j v-direction index
     * @return Global DOF index
     */
    int dof_index(int i, int j) const { return i + num_basis_u() * j; }

    /**
     * @brief Convert global DOF index to 2D indices
     * @param dof Global DOF index
     * @return Pair (i, j) of 2D indices
     */
    std::pair<int, int> dof_to_ij(int dof) const {
        int i = dof % num_basis_u();
        int j = dof / num_basis_u();
        return {i, j};
    }

    /**
     * @brief Find spans containing parameter point (u, v)
     * @param u Parameter in u-direction
     * @param v Parameter in v-direction
     * @return Pair (span_u, span_v)
     */
    std::pair<int, int> find_spans(Real u, Real v) const;

    /**
     * @brief Evaluate non-zero basis functions at (u, v)
     * @param u Parameter in u-direction (in [0, num_spans_u])
     * @param v Parameter in v-direction (in [0, num_spans_v])
     * @return 4x4 matrix of non-zero values: result(iu, jv) = N_iu(u) * N_jv(v)
     *
     * Indices in result correspond to:
     *   iu in [span_u - 3, span_u]
     *   jv in [span_v - 3, span_v]
     */
    MatX evaluate_nonzero(Real u, Real v) const;

    /**
     * @brief Evaluate non-zero basis functions and derivatives at (u, v)
     * @param u Parameter in u-direction
     * @param v Parameter in v-direction
     * @param num_derivs_u Number of u-derivatives (0, 1, or 2)
     * @param num_derivs_v Number of v-derivatives (0, 1, or 2)
     * @return 3D structure: [du][dv] -> 4x4 matrix of basis values
     */
    std::vector<std::vector<MatX>> evaluate_nonzero_derivs(Real u, Real v,
                                                           int num_derivs_u,
                                                           int num_derivs_v) const;

    /**
     * @brief Evaluate a single basis function at (u, v)
     * @param i u-direction index
     * @param j v-direction index
     * @param u Parameter in u-direction
     * @param v Parameter in v-direction
     * @return Basis function value N_{i,j}(u, v) = N_i(u) * N_j(v)
     */
    Real evaluate(int i, int j, Real u, Real v) const;

    /**
     * @brief Evaluate all basis functions at (u, v)
     * @param u Parameter in u-direction
     * @param v Parameter in v-direction
     * @return Vector of all num_basis() values in DOF order
     */
    VecX evaluate_all(Real u, Real v) const;

    /**
     * @brief Evaluate du derivative of basis function (i, j) at (u, v)
     * @return dN_{i,j}/du = dN_i/du * N_j
     */
    Real evaluate_du(int i, int j, Real u, Real v) const;

    /**
     * @brief Evaluate dv derivative of basis function (i, j) at (u, v)
     * @return dN_{i,j}/dv = N_i * dN_j/dv
     */
    Real evaluate_dv(int i, int j, Real u, Real v) const;

    /**
     * @brief Evaluate d²/dudv mixed derivative
     * @return d²N_{i,j}/dudv = dN_i/du * dN_j/dv
     */
    Real evaluate_dudv(int i, int j, Real u, Real v) const;

    /**
     * @brief Evaluate d²/du² second derivative
     * @return d²N_{i,j}/du² = d²N_i/du² * N_j
     */
    Real evaluate_d2u(int i, int j, Real u, Real v) const;

    /**
     * @brief Evaluate d²/dv² second derivative
     * @return d²N_{i,j}/dv² = N_i * d²N_j/dv²
     */
    Real evaluate_d2v(int i, int j, Real u, Real v) const;

    /**
     * @brief Evaluate all basis function u-derivatives at (u, v)
     * @return Vector of all num_basis() du values
     */
    VecX evaluate_all_du(Real u, Real v) const;

    /**
     * @brief Evaluate all basis function v-derivatives at (u, v)
     * @return Vector of all num_basis() dv values
     */
    VecX evaluate_all_dv(Real u, Real v) const;

    /**
     * @brief Get support interval of basis function (i, j) in parameter space
     * @return Tuple (u_min, u_max, v_min, v_max)
     */
    std::tuple<Real, Real, Real, Real> support(int i, int j) const;

    /**
     * @brief Check if basis function (i, j) is non-zero at (u, v)
     */
    bool is_nonzero_at(int i, int j, Real u, Real v) const;

  private:
    BSplineBasis1D basis_u_;
    BSplineBasis1D basis_v_;
    int level_ = 0;
};

}  // namespace drifter
