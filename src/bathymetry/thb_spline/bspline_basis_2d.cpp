#include "bathymetry/thb_spline/bspline_basis_2d.hpp"

#include <algorithm>
#include <cmath>

namespace drifter {

BSplineBasis2D::BSplineBasis2D(const BSplineKnotVector& knots_u,
                               const BSplineKnotVector& knots_v, int level)
    : basis_u_(knots_u.num_spans(level)),
      basis_v_(knots_v.num_spans(level)),
      level_(level) {}

BSplineBasis2D::BSplineBasis2D(int num_spans_u, int num_spans_v)
    : basis_u_(num_spans_u), basis_v_(num_spans_v), level_(0) {}

std::pair<int, int> BSplineBasis2D::find_spans(Real u, Real v) const {
    return {basis_u_.find_span(u), basis_v_.find_span(v)};
}

MatX BSplineBasis2D::evaluate_nonzero(Real u, Real v) const {
    VecX Nu = basis_u_.evaluate_nonzero(u);
    VecX Nv = basis_v_.evaluate_nonzero(v);

    // Outer product: result(i, j) = Nu(i) * Nv(j)
    MatX result(DEGREE + 1, DEGREE + 1);
    for (int j = 0; j <= DEGREE; ++j) {
        for (int i = 0; i <= DEGREE; ++i) {
            result(i, j) = Nu(i) * Nv(j);
        }
    }
    return result;
}

std::vector<std::vector<MatX>> BSplineBasis2D::evaluate_nonzero_derivs(
    Real u, Real v, int num_derivs_u, int num_derivs_v) const {
    MatX ders_u = basis_u_.evaluate_nonzero_derivs(u, num_derivs_u);
    MatX ders_v = basis_v_.evaluate_nonzero_derivs(v, num_derivs_v);

    std::vector<std::vector<MatX>> result(num_derivs_u + 1);
    for (int du = 0; du <= num_derivs_u; ++du) {
        result[du].resize(num_derivs_v + 1);
        for (int dv = 0; dv <= num_derivs_v; ++dv) {
            result[du][dv].resize(DEGREE + 1, DEGREE + 1);
            for (int j = 0; j <= DEGREE; ++j) {
                for (int i = 0; i <= DEGREE; ++i) {
                    result[du][dv](i, j) = ders_u(du, i) * ders_v(dv, j);
                }
            }
        }
    }
    return result;
}

Real BSplineBasis2D::evaluate(int i, int j, Real u, Real v) const {
    return basis_u_.evaluate(i, u) * basis_v_.evaluate(j, v);
}

VecX BSplineBasis2D::evaluate_all(Real u, Real v) const {
    VecX Nu = basis_u_.evaluate_all(u);
    VecX Nv = basis_v_.evaluate_all(v);

    VecX result(num_basis());
    for (int j = 0; j < num_basis_v(); ++j) {
        for (int i = 0; i < num_basis_u(); ++i) {
            result(dof_index(i, j)) = Nu(i) * Nv(j);
        }
    }
    return result;
}

Real BSplineBasis2D::evaluate_du(int i, int j, Real u, Real v) const {
    return basis_u_.evaluate_derivative(i, u) * basis_v_.evaluate(j, v);
}

Real BSplineBasis2D::evaluate_dv(int i, int j, Real u, Real v) const {
    return basis_u_.evaluate(i, u) * basis_v_.evaluate_derivative(j, v);
}

Real BSplineBasis2D::evaluate_dudv(int i, int j, Real u, Real v) const {
    return basis_u_.evaluate_derivative(i, u) * basis_v_.evaluate_derivative(j, v);
}

Real BSplineBasis2D::evaluate_d2u(int i, int j, Real u, Real v) const {
    return basis_u_.evaluate_second_derivative(i, u) * basis_v_.evaluate(j, v);
}

Real BSplineBasis2D::evaluate_d2v(int i, int j, Real u, Real v) const {
    return basis_u_.evaluate(i, u) * basis_v_.evaluate_second_derivative(j, v);
}

VecX BSplineBasis2D::evaluate_all_du(Real u, Real v) const {
    VecX dNu = basis_u_.evaluate_all_derivatives(u);
    VecX Nv = basis_v_.evaluate_all(v);

    VecX result(num_basis());
    for (int j = 0; j < num_basis_v(); ++j) {
        for (int i = 0; i < num_basis_u(); ++i) {
            result(dof_index(i, j)) = dNu(i) * Nv(j);
        }
    }
    return result;
}

VecX BSplineBasis2D::evaluate_all_dv(Real u, Real v) const {
    VecX Nu = basis_u_.evaluate_all(u);
    VecX dNv = basis_v_.evaluate_all_derivatives(v);

    VecX result(num_basis());
    for (int j = 0; j < num_basis_v(); ++j) {
        for (int i = 0; i < num_basis_u(); ++i) {
            result(dof_index(i, j)) = Nu(i) * dNv(j);
        }
    }
    return result;
}

std::tuple<Real, Real, Real, Real> BSplineBasis2D::support(int i, int j) const {
    auto [u_min, u_max] = basis_u_.support(i);
    auto [v_min, v_max] = basis_v_.support(j);
    return {u_min, u_max, v_min, v_max};
}

bool BSplineBasis2D::is_nonzero_at(int i, int j, Real u, Real v) const {
    auto [u_min, u_max, v_min, v_max] = support(i, j);
    return (u >= u_min && u <= u_max && v >= v_min && v <= v_max);
}

}  // namespace drifter
