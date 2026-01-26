#include "bathymetry/thb_spline/thb_hierarchy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace drifter {

THBHierarchy::THBHierarchy(const BSplineKnotVector& knots_u,
                           const BSplineKnotVector& knots_v)
    : knots_u_(knots_u), knots_v_(knots_v) {
    if (knots_u.max_level() != knots_v.max_level()) {
        throw std::invalid_argument(
            "THBHierarchy: knot vectors must have same max_level");
    }
    max_level_ = knots_u.max_level();
    initialize();
}

THBHierarchy::THBHierarchy(Real domain_min_u, Real domain_max_u, Real domain_min_v,
                           Real domain_max_v, int num_spans_level0_u,
                           int num_spans_level0_v, int max_level)
    : knots_u_(domain_min_u, domain_max_u, num_spans_level0_u, max_level),
      knots_v_(domain_min_v, domain_max_v, num_spans_level0_v, max_level),
      max_level_(max_level) {
    initialize();
}

void THBHierarchy::initialize() {
    // Build 2D bases for each level
    bases_.reserve(max_level_ + 1);
    for (int l = 0; l <= max_level_; ++l) {
        bases_.emplace_back(knots_u_, knots_v_, l);
    }

    // Compute cumulative DOF offsets
    level_offsets_.resize(max_level_ + 2);
    level_offsets_[0] = 0;
    for (int l = 0; l <= max_level_; ++l) {
        level_offsets_[l + 1] = level_offsets_[l] + num_basis(l);
    }
}

std::tuple<int, int, int> THBHierarchy::dof_to_level_ij(Index dof) const {
    // Find level by binary search in level_offsets
    int level = 0;
    for (int l = 0; l <= max_level_; ++l) {
        if (dof < level_offsets_[l + 1]) {
            level = l;
            break;
        }
    }

    Index local_dof = dof - level_offsets_[level];
    int nbu = num_basis_u(level);
    int i = static_cast<int>(local_dof % nbu);
    int j = static_cast<int>(local_dof / nbu);

    return {level, i, j};
}

Eigen::VectorBlock<VecX> THBHierarchy::level_coefficients(int level) {
    Index start = level_offsets_[level];
    Index size = num_basis(level);
    return coefficients_.segment(start, size);
}

Eigen::VectorBlock<const VecX> THBHierarchy::level_coefficients(int level) const {
    Index start = level_offsets_[level];
    Index size = num_basis(level);
    return coefficients_.segment(start, size);
}

void THBHierarchy::allocate_coefficients() {
    coefficients_.setZero(total_dofs());
}

std::pair<Real, Real> THBHierarchy::physical_to_parameter(int level, Real x,
                                                          Real y) const {
    Real u = knots_u_.physical_to_parameter(level, x);
    Real v = knots_v_.physical_to_parameter(level, y);
    return {u, v};
}

std::pair<Real, Real> THBHierarchy::parameter_to_physical(int level, Real u,
                                                          Real v) const {
    Real x = knots_u_.parameter_to_physical(level, u);
    Real y = knots_v_.parameter_to_physical(level, v);
    return {x, y};
}

int THBHierarchy::level_for_element_size(Real element_size_u,
                                         Real element_size_v) const {
    // Use minimum dimension to ensure adequate resolution in all directions
    Real min_size = std::min(element_size_u, element_size_v);

    // Use average span size at level 0
    Real avg_span_level0 =
        0.5 * (knots_u_.span_size(0) + knots_v_.span_size(0));

    // Find the finest level where span_size(l) >= min_size
    // This ensures the THB basis functions can resolve features at the element scale
    // span_size(l) = avg_span_level0 / 2^l
    // We want: avg_span_level0 / 2^l >= min_size
    // 2^l <= avg_span_level0 / min_size
    // l <= log2(avg_span_level0 / min_size)
    // l = floor(log2(avg_span_level0 / min_size))
    Real l_exact = std::log2(avg_span_level0 / min_size);
    int l = static_cast<int>(std::floor(l_exact));
    return std::clamp(l, 0, max_level_);
}

}  // namespace drifter
