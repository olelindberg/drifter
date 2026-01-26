#pragma once

#include "bathymetry/thb_spline/bspline_basis_2d.hpp"
#include "bathymetry/thb_spline/bspline_knot_vector.hpp"
#include "core/types.hpp"
#include <memory>
#include <vector>

namespace drifter {

/**
 * @brief Hierarchical B-spline structure for THB-splines
 *
 * Manages multi-level B-spline basis functions and control points.
 * Each level l has 2^l times the resolution of level 0.
 *
 * The hierarchy stores:
 * - Knot vectors for u and v directions (same structure, may differ in spans)
 * - Control points for all levels (only active ones used in evaluation)
 * - Global DOF indexing that maps (level, i, j) to flat storage
 */
class THBHierarchy {
  public:
    /**
     * @brief Construct hierarchy from knot vectors
     * @param knots_u Knot vector for u-direction
     * @param knots_v Knot vector for v-direction
     *
     * Both knot vectors should have the same max_level.
     */
    THBHierarchy(const BSplineKnotVector& knots_u, const BSplineKnotVector& knots_v);

    /**
     * @brief Construct uniform hierarchy
     * @param domain_min_u U-direction domain minimum
     * @param domain_max_u U-direction domain maximum
     * @param domain_min_v V-direction domain minimum
     * @param domain_max_v V-direction domain maximum
     * @param num_spans_level0_u Level 0 spans in u
     * @param num_spans_level0_v Level 0 spans in v
     * @param max_level Maximum refinement level
     */
    THBHierarchy(Real domain_min_u, Real domain_max_u, Real domain_min_v,
                 Real domain_max_v, int num_spans_level0_u, int num_spans_level0_v,
                 int max_level);

    /// Maximum refinement level
    int max_level() const { return max_level_; }

    /// Domain bounds
    Real domain_min_u() const { return knots_u_.domain_min(); }
    Real domain_max_u() const { return knots_u_.domain_max(); }
    Real domain_min_v() const { return knots_v_.domain_min(); }
    Real domain_max_v() const { return knots_v_.domain_max(); }

    /// Access knot vectors
    const BSplineKnotVector& knots_u() const { return knots_u_; }
    const BSplineKnotVector& knots_v() const { return knots_v_; }

    /// Number of basis functions at level l in each direction
    int num_basis_u(int level) const { return knots_u_.num_basis(level); }
    int num_basis_v(int level) const { return knots_v_.num_basis(level); }

    /// Number of basis functions at level l (total)
    int num_basis(int level) const { return num_basis_u(level) * num_basis_v(level); }

    /// Total DOFs across all levels
    Index total_dofs() const { return level_offsets_.back(); }

    /// DOF offset for start of level l
    Index level_offset(int level) const { return level_offsets_[level]; }

    /**
     * @brief Map (level, i, j) to global DOF index
     * @param level Refinement level
     * @param i U-direction basis index
     * @param j V-direction basis index
     * @return Global DOF index
     */
    Index global_dof(int level, int i, int j) const {
        return level_offsets_[level] + i + num_basis_u(level) * j;
    }

    /**
     * @brief Map global DOF to (level, i, j)
     * @param dof Global DOF index
     * @return Tuple (level, i, j)
     */
    std::tuple<int, int, int> dof_to_level_ij(Index dof) const;

    /**
     * @brief Access 2D basis at level l
     * @param level Refinement level
     */
    const BSplineBasis2D& basis(int level) const { return bases_[level]; }

    /// Access control point coefficients (all levels flattened)
    const VecX& coefficients() const { return coefficients_; }
    VecX& coefficients() { return coefficients_; }

    /// Set coefficient for (level, i, j)
    void set_coefficient(int level, int i, int j, Real value) {
        coefficients_(global_dof(level, i, j)) = value;
    }

    /// Get coefficient for (level, i, j)
    Real get_coefficient(int level, int i, int j) const {
        return coefficients_(global_dof(level, i, j));
    }

    /**
     * @brief Get coefficients for a single level
     * @param level Refinement level
     * @return View of coefficients for this level
     */
    Eigen::VectorBlock<VecX> level_coefficients(int level);
    Eigen::VectorBlock<const VecX> level_coefficients(int level) const;

    /**
     * @brief Resize coefficients to total_dofs() and zero-initialize
     */
    void allocate_coefficients();

    /**
     * @brief Map physical coordinates to parameter space at given level
     * @param level Refinement level
     * @param x Physical x-coordinate
     * @param y Physical y-coordinate
     * @return Pair (u, v) in parameter space [0, num_spans_u/v(level)]
     */
    std::pair<Real, Real> physical_to_parameter(int level, Real x, Real y) const;

    /**
     * @brief Map parameter coordinates to physical space
     * @param level Refinement level
     * @param u Parameter in [0, num_spans_u(level)]
     * @param v Parameter in [0, num_spans_v(level)]
     * @return Pair (x, y) in physical space
     */
    std::pair<Real, Real> parameter_to_physical(int level, Real u, Real v) const;

    /**
     * @brief Determine which level matches a given element size
     * @param element_size_u Element size in u-direction
     * @param element_size_v Element size in v-direction
     * @return Level where span size ≈ element size
     */
    int level_for_element_size(Real element_size_u, Real element_size_v) const;

  private:
    BSplineKnotVector knots_u_;
    BSplineKnotVector knots_v_;
    int max_level_;

    /// 2D basis objects for each level
    std::vector<BSplineBasis2D> bases_;

    /// Cumulative DOF offsets: level_offsets_[l] = sum of DOFs for levels 0..l-1
    std::vector<Index> level_offsets_;

    /// Control point coefficients (flat storage for all levels)
    VecX coefficients_;

    /// Build bases and compute offsets
    void initialize();
};

}  // namespace drifter
