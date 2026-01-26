#include "bathymetry/thb_spline/bspline_knot_vector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace drifter {

BSplineKnotVector::BSplineKnotVector(Real domain_min, Real domain_max,
                                     int num_spans_level0, int max_level)
    : domain_min_(domain_min),
      domain_max_(domain_max),
      num_spans_level0_(num_spans_level0),
      max_level_(max_level) {
    if (domain_max <= domain_min) {
        throw std::invalid_argument("BSplineKnotVector: domain_max must be > domain_min");
    }
    if (num_spans_level0 < 1) {
        throw std::invalid_argument("BSplineKnotVector: num_spans_level0 must be >= 1");
    }
    if (max_level < 0) {
        throw std::invalid_argument("BSplineKnotVector: max_level must be >= 0");
    }

    // Build knot vectors for all levels
    knots_per_level_.resize(max_level_ + 1);
    for (int l = 0; l <= max_level_; ++l) {
        build_knot_vector(l);
    }
}

void BSplineKnotVector::build_knot_vector(int level) {
    const int n = num_spans(level);
    const Real h = span_size(level);

    // Open (clamped) uniform knot vector:
    // [domain_min repeated degree+1 times, uniform interior, domain_max repeated
    // degree+1 times] Total knots = n + 2*(degree+1) = n + 8

    const int num_knots = n + 2 * (DEGREE + 1);
    knots_per_level_[level].resize(num_knots);

    auto& knots = knots_per_level_[level];

    // First degree+1 knots are domain_min (clamped at start)
    for (int i = 0; i <= DEGREE; ++i) {
        knots[i] = domain_min_;
    }

    // Interior knots: uniform spacing
    for (int i = 1; i < n; ++i) {
        knots[DEGREE + i] = domain_min_ + i * h;
    }

    // Last degree+1 knots are domain_max (clamped at end)
    for (int i = 0; i <= DEGREE; ++i) {
        knots[DEGREE + n + i] = domain_max_;
    }
}

int BSplineKnotVector::find_span(int level, Real x) const {
    // Convert physical to parameter space for this level
    Real t = physical_to_parameter(level, x);

    const int n = num_basis(level) - 1;  // Last basis function index
    const auto& knots = knots_per_level_[level];

    // Clamp to valid span range
    if (t <= 0.0) {
        return DEGREE;
    }
    if (t >= static_cast<Real>(num_spans(level))) {
        return n;
    }

    // Binary search for span
    int low = DEGREE;
    int high = n + 1;
    int mid = (low + high) / 2;

    // Convert t to knot space (knots are in physical coordinates for this impl)
    Real x_clamped = std::clamp(x, domain_min_, domain_max_);

    while (x_clamped < knots[mid] || x_clamped >= knots[mid + 1]) {
        if (x_clamped < knots[mid]) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }

    return mid;
}

Real BSplineKnotVector::physical_to_parameter(int level, Real x) const {
    // Map [domain_min, domain_max] -> [0, num_spans(level)]
    const int n = num_spans(level);
    return (x - domain_min_) / (domain_max_ - domain_min_) * n;
}

Real BSplineKnotVector::parameter_to_physical(int level, Real t) const {
    // Map [0, num_spans(level)] -> [domain_min, domain_max]
    const int n = num_spans(level);
    return domain_min_ + t / n * (domain_max_ - domain_min_);
}

std::pair<Real, Real> BSplineKnotVector::support_physical(int level, int i) const {
    const auto& knots = knots_per_level_[level];
    // Basis function i is non-zero on [knot[i], knot[i+degree+1]]
    return {knots[i], knots[i + DEGREE + 1]};
}

int BSplineKnotVector::level_for_span_size(Real target_span_size) const {
    // Find level l such that span_size(l) is closest to target
    // span_size(l) = (domain_max - domain_min) / (num_spans_level0 * 2^l)

    const Real domain_size = domain_max_ - domain_min_;
    const Real base_span = domain_size / num_spans_level0_;

    // Solve: base_span / 2^l = target
    // l = log2(base_span / target)
    Real l_exact = std::log2(base_span / target_span_size);

    // Clamp to valid level range
    int l = static_cast<int>(std::round(l_exact));
    return std::clamp(l, 0, max_level_);
}

BSplineKnotVector BSplineKnotVector::from_octree_sizes(Real domain_min, Real domain_max,
                                                       Real max_element_size,
                                                       Real min_element_size) {
    const Real domain_size = domain_max - domain_min;

    // Level 0 span size should match max element size
    // num_spans_level0 = ceil(domain_size / max_element_size)
    int num_spans_level0 = static_cast<int>(std::ceil(domain_size / max_element_size));

    // Ensure at least 1 span
    num_spans_level0 = std::max(1, num_spans_level0);

    // Actual span size at level 0
    Real actual_span_level0 = domain_size / num_spans_level0;

    // Max level: span_size(max_level) <= min_element_size
    // span_size(l) = actual_span_level0 / 2^l
    // l = ceil(log2(actual_span_level0 / min_element_size))
    Real l_exact = std::log2(actual_span_level0 / min_element_size);
    int max_level = static_cast<int>(std::ceil(l_exact));
    max_level = std::max(0, max_level);

    return BSplineKnotVector(domain_min, domain_max, num_spans_level0, max_level);
}

}  // namespace drifter
