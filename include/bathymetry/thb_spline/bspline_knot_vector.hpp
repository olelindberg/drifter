#pragma once

#include "core/types.hpp"
#include <vector>

namespace drifter {

/**
 * @brief Hierarchical dyadic knot vector for THB-splines
 *
 * Manages uniform knot sequences at multiple resolution levels with dyadic
 * refinement. Each level l has twice the spans of level l-1.
 *
 * Level structure:
 *   Level 0: num_spans_level0 spans (coarsest)
 *   Level l: 2^l × num_spans_level0 spans
 *
 * Open (clamped) knot vectors at boundaries ensure interpolation at endpoints.
 */
class BSplineKnotVector {
  public:
    static constexpr int DEGREE = 3;  // Cubic

    /**
     * @brief Construct a hierarchical knot vector
     * @param domain_min Physical domain minimum
     * @param domain_max Physical domain maximum
     * @param num_spans_level0 Number of spans at coarsest level
     * @param max_level Maximum refinement level (inclusive)
     */
    BSplineKnotVector(Real domain_min, Real domain_max, int num_spans_level0,
                      int max_level);

    /// Default constructor for delayed initialization
    BSplineKnotVector() = default;

    /// Physical domain bounds
    Real domain_min() const { return domain_min_; }
    Real domain_max() const { return domain_max_; }

    /// Number of spans at coarsest level
    int num_spans_level0() const { return num_spans_level0_; }

    /// Maximum refinement level
    int max_level() const { return max_level_; }

    /// Number of spans at level l
    int num_spans(int level) const { return num_spans_level0_ * (1 << level); }

    /// Number of basis functions at level l
    int num_basis(int level) const { return num_spans(level) + DEGREE; }

    /// Span size in physical coordinates at level l
    Real span_size(int level) const {
        return (domain_max_ - domain_min_) / num_spans(level);
    }

    /// Access knot vector at level l
    const std::vector<Real>& knots(int level) const { return knots_per_level_[level]; }

    /**
     * @brief Find the span index containing physical coordinate x at level l
     * @param level Refinement level
     * @param x Physical coordinate
     * @return Span index i such that knot[i] <= x < knot[i+1]
     */
    int find_span(int level, Real x) const;

    /**
     * @brief Map physical coordinate to parameter space [0, num_spans(level)]
     * @param x Physical coordinate
     * @return Parameter value t in [0, num_spans(level)]
     */
    Real physical_to_parameter(int level, Real x) const;

    /**
     * @brief Map parameter value to physical coordinate
     * @param level Refinement level
     * @param t Parameter value in [0, num_spans(level)]
     * @return Physical coordinate x
     */
    Real parameter_to_physical(int level, Real t) const;

    /**
     * @brief Get the support interval of basis function i at level l
     * @param level Refinement level
     * @param i Basis function index
     * @return Pair (x_min, x_max) in physical coordinates
     */
    std::pair<Real, Real> support_physical(int level, int i) const;

    /**
     * @brief Determine which level best matches a given span size
     * @param target_span_size Desired span size in physical coordinates
     * @return Level l such that span_size(l) is closest to target
     */
    int level_for_span_size(Real target_span_size) const;

    /**
     * @brief Auto-configure from octree element sizes
     * @param domain_min Physical domain minimum
     * @param domain_max Physical domain maximum
     * @param max_element_size Size of largest octree element
     * @param min_element_size Size of smallest octree element
     * @return Configured knot vector matching octree hierarchy
     */
    static BSplineKnotVector from_octree_sizes(Real domain_min, Real domain_max,
                                               Real max_element_size,
                                               Real min_element_size);

  private:
    Real domain_min_ = 0.0;
    Real domain_max_ = 1.0;
    int num_spans_level0_ = 1;
    int max_level_ = 0;

    /// Knot vectors for each level
    std::vector<std::vector<Real>> knots_per_level_;

    /// Build knot vector for specified level
    void build_knot_vector(int level);
};

}  // namespace drifter
