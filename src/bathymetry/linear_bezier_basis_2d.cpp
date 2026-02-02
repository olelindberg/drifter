#include "bathymetry/linear_bezier_basis_2d.hpp"
#include <cassert>
#include <stdexcept>

namespace drifter {

Vec2 LinearBezierBasis2D::control_point_position(int dof) const {
  assert(dof >= 0 && dof < NDOF);
  int i, j;
  dof_ij(dof, i, j);
  // Linear: control points at corners (0 or 1)
  return Vec2(static_cast<Real>(i), static_cast<Real>(j));
}

VecX LinearBezierBasis2D::evaluate(Real u, Real v) const {
  // Linear Bernstein basis: B_0(t) = 1-t, B_1(t) = t
  VecX Bu = evaluate_bernstein_1d(DEGREE, u);
  VecX Bv = evaluate_bernstein_1d(DEGREE, v);

  VecX result(NDOF);
  for (int j = 0; j < N1D; ++j) {
    for (int i = 0; i < N1D; ++i) {
      result(dof_index(i, j)) = Bu(i) * Bv(j);
    }
  }
  return result;
}

VecX LinearBezierBasis2D::evaluate_bernstein_1d(int n, Real t) const {
  VecX B(n + 1);
  if (n == 0) {
    B(0) = 1.0;
  } else if (n == 1) {
    // Linear: B_0 = 1-t, B_1 = t
    B(0) = 1.0 - t;
    B(1) = t;
  } else {
    // General case using de Casteljau-style recurrence
    Real s = 1.0 - t;
    B(0) = 1.0;
    for (int i = 1; i <= n; ++i) {
      B(i) = B(i - 1) * t / static_cast<Real>(i) * static_cast<Real>(n - i + 1);
    }
    // Actually use binomial formula: B_i,n(t) = C(n,i) * t^i * (1-t)^(n-i)
    Real s_power = 1.0;
    Real t_power = 1.0;
    for (int i = 0; i <= n; ++i) {
      if (i > 0) {
        t_power *= t;
        s_power = 1.0;
        for (int k = 0; k < n - i; ++k) {
          s_power *= s;
        }
      } else {
        t_power = 1.0;
        s_power = 1.0;
        for (int k = 0; k < n; ++k) {
          s_power *= s;
        }
      }
      // Binomial coefficient C(n,i)
      Real binom = 1.0;
      for (int k = 0; k < i; ++k) {
        binom *= static_cast<Real>(n - k) / static_cast<Real>(k + 1);
      }
      B(i) = binom * t_power * s_power;
    }
  }
  return B;
}

VecX LinearBezierBasis2D::evaluate_du(Real u, Real v) const {
  // d/du of B_i(u) * B_j(v) = B'_i(u) * B_j(v)
  // For linear: B'_0(u) = -1, B'_1(u) = 1
  VecX dBu = evaluate_bernstein_derivative_1d(DEGREE, u, 1);
  VecX Bv = evaluate_bernstein_1d(DEGREE, v);

  VecX result(NDOF);
  for (int j = 0; j < N1D; ++j) {
    for (int i = 0; i < N1D; ++i) {
      result(dof_index(i, j)) = dBu(i) * Bv(j);
    }
  }
  return result;
}

VecX LinearBezierBasis2D::evaluate_dv(Real u, Real v) const {
  // d/dv of B_i(u) * B_j(v) = B_i(u) * B'_j(v)
  VecX Bu = evaluate_bernstein_1d(DEGREE, u);
  VecX dBv = evaluate_bernstein_derivative_1d(DEGREE, v, 1);

  VecX result(NDOF);
  for (int j = 0; j < N1D; ++j) {
    for (int i = 0; i < N1D; ++i) {
      result(dof_index(i, j)) = Bu(i) * dBv(j);
    }
  }
  return result;
}

MatX LinearBezierBasis2D::evaluate_gradient(Real u, Real v) const {
  MatX grad(NDOF, 2);
  grad.col(0) = evaluate_du(u, v);
  grad.col(1) = evaluate_dv(u, v);
  return grad;
}

VecX LinearBezierBasis2D::evaluate_bernstein_derivative_1d(int n, Real t,
                                                           int k) const {
  if (k == 0) {
    return evaluate_bernstein_1d(n, t);
  }

  VecX dB(n + 1);

  if (n == 0) {
    // Derivative of constant is zero
    dB(0) = 0.0;
  } else if (n == 1 && k == 1) {
    // First derivative of linear Bernstein
    // B_0(t) = 1-t  =>  B'_0 = -1
    // B_1(t) = t    =>  B'_1 = 1
    dB(0) = -1.0;
    dB(1) = 1.0;
  } else if (k >= n + 1) {
    // k-th derivative of degree n polynomial is zero if k > n
    dB.setZero();
  } else {
    // General formula: d^k/dt^k B_i,n(t) = n!/(n-k)! * sum over j of
    // combinations This is getting complex for higher k, but we only need k=1
    // here

    // Use the recurrence: B'_i,n(t) = n * (B_{i-1,n-1}(t) - B_{i,n-1}(t))
    // with B_{-1,m} = B_{m+1,m} = 0
    if (k == 1) {
      VecX B_lower = evaluate_bernstein_1d(n - 1, t);
      for (int i = 0; i <= n; ++i) {
        Real left = (i > 0) ? B_lower(i - 1) : 0.0;
        Real right = (i < n) ? B_lower(i) : 0.0;
        dB(i) = static_cast<Real>(n) * (left - right);
      }
    } else {
      // Higher derivatives: apply recurrence recursively
      VecX dB_lower = evaluate_bernstein_derivative_1d(n - 1, t, k - 1);
      for (int i = 0; i <= n; ++i) {
        Real left = (i > 0) ? dB_lower(i - 1) : 0.0;
        Real right = (i < n) ? dB_lower(i) : 0.0;
        dB(i) = static_cast<Real>(n) * (left - right);
      }
    }
  }

  return dB;
}

Real LinearBezierBasis2D::evaluate_scalar(const VecX &coeffs, Real u,
                                          Real v) const {
  assert(coeffs.size() == NDOF);
  // Bilinear interpolation using de Casteljau
  // First interpolate in u direction for each row
  Real c0 = (1.0 - u) * coeffs(0) + u * coeffs(2); // j=0 row
  Real c1 = (1.0 - u) * coeffs(1) + u * coeffs(3); // j=1 row
  // Then interpolate in v direction
  return (1.0 - v) * c0 + v * c1;
}

int LinearBezierBasis2D::corner_dof(int corner_id) const {
  // Corner mapping:
  // corner 0: (u=0, v=0) -> dof 0
  // corner 1: (u=1, v=0) -> dof 2
  // corner 2: (u=0, v=1) -> dof 1
  // corner 3: (u=1, v=1) -> dof 3
  static constexpr int corner_to_dof[4] = {0, 2, 1, 3};
  assert(corner_id >= 0 && corner_id < 4);
  return corner_to_dof[corner_id];
}

int LinearBezierBasis2D::dof_to_corner(int dof) const {
  // Inverse mapping
  // dof 0 -> corner 0
  // dof 1 -> corner 2
  // dof 2 -> corner 1
  // dof 3 -> corner 3
  static constexpr int dof_to_corner_map[4] = {0, 2, 1, 3};
  if (dof < 0 || dof >= NDOF) {
    return -1;
  }
  return dof_to_corner_map[dof];
}

std::vector<int> LinearBezierBasis2D::edge_dofs(int edge_id) const {
  // For linear elements, all edge DOFs are corners (2 per edge)
  // Edge 0 (u=0, left): j varies, i=0 -> DOFs 0, 1
  // Edge 1 (u=1, right): j varies, i=1 -> DOFs 2, 3
  // Edge 2 (v=0, bottom): i varies, j=0 -> DOFs 0, 2
  // Edge 3 (v=1, top): i varies, j=1 -> DOFs 1, 3
  switch (edge_id) {
  case 0:
    return {0, 1}; // left edge
  case 1:
    return {2, 3}; // right edge
  case 2:
    return {0, 2}; // bottom edge
  case 3:
    return {1, 3}; // top edge
  default:
    throw std::invalid_argument("Invalid edge_id: " + std::to_string(edge_id));
  }
}

Vec2 LinearBezierBasis2D::corner_param(int corner_id) const {
  // Corner 0: (0, 0), Corner 1: (1, 0), Corner 2: (0, 1), Corner 3: (1, 1)
  switch (corner_id) {
  case 0:
    return Vec2(0.0, 0.0);
  case 1:
    return Vec2(1.0, 0.0);
  case 2:
    return Vec2(0.0, 1.0);
  case 3:
    return Vec2(1.0, 1.0);
  default:
    throw std::invalid_argument("Invalid corner_id: " +
                                std::to_string(corner_id));
  }
}

MatX LinearBezierBasis2D::compute_1d_extraction_matrix(Real t0, Real t1) const {
  // For linear Bezier subdivision from [0,1] to [t0, t1]:
  // New control points are evaluated at t0 and t1 on the original curve.
  //
  // Original: c0, c1 define curve c(t) = (1-t)*c0 + t*c1
  // New curve on [t0, t1]: need c'0 = c(t0), c'1 = c(t1)
  //
  // c'0 = (1-t0)*c0 + t0*c1
  // c'1 = (1-t1)*c0 + t1*c1
  //
  // S = [ (1-t0)  t0 ]
  //     [ (1-t1)  t1 ]

  MatX S(N1D, N1D);
  S(0, 0) = 1.0 - t0;
  S(0, 1) = t0;
  S(1, 0) = 1.0 - t1;
  S(1, 1) = t1;
  return S;
}

} // namespace drifter
