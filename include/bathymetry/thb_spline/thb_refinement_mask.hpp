#pragma once

#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/thb_spline/thb_hierarchy.hpp"
#include "core/types.hpp"
#include <set>
#include <vector>

namespace drifter {

/**
 * @brief Tracks active basis functions for THB-splines
 *
 * Determines which basis functions at each level are "active" based on the
 * adaptive refinement pattern. A basis function is active at level l if:
 *   1. Its support overlaps with at least one region refined to exactly level l
 *   2. Its support is NOT fully contained in regions refined to level l+1 or finer
 *
 * This implements the "selection" step of the THB-spline algorithm, where
 * only active functions contribute to the surface representation.
 */
class THBRefinementMask {
  public:
    /**
     * @brief Construct from quadtree and hierarchy
     * @param quadtree Adaptive 2D mesh defining refinement regions
     * @param hierarchy THB hierarchy with knot vectors
     *
     * Automatically determines active functions from quadtree element sizes.
     */
    THBRefinementMask(const QuadtreeAdapter& quadtree, const THBHierarchy& hierarchy);

    /**
     * @brief Construct uniform mask (all functions active at specified level)
     * @param hierarchy THB hierarchy
     * @param active_level Level at which all functions are active
     */
    THBRefinementMask(const THBHierarchy& hierarchy, int active_level);

    /**
     * @brief Check if basis function (level, i, j) is active
     * @return true if function contributes to THB surface
     */
    bool is_active(int level, int i, int j) const;

    /**
     * @brief Check if global DOF is active
     * @return true if function contributes to THB surface
     */
    bool is_active_dof(Index dof) const;

    /// Get all active functions as (level, i, j) tuples
    const std::vector<std::tuple<int, int, int>>& active_functions() const {
        return active_functions_;
    }

    /// Number of active functions (active DOFs)
    Index num_active() const { return static_cast<Index>(active_functions_.size()); }

    /**
     * @brief Get active function indices for a given level
     * @param level Refinement level
     * @return Vector of (i, j) pairs for active functions at this level
     */
    std::vector<std::pair<int, int>> active_at_level(int level) const;

    /**
     * @brief Get the refinement level assigned to each quadtree element
     * @return Vector indexed by element index
     */
    const std::vector<int>& element_levels() const { return element_levels_; }

    /**
     * @brief Get elements that are refined to a specific level
     * @param level Target level
     * @return Vector of element indices
     */
    std::vector<Index> elements_at_level(int level) const;

    /// Access underlying hierarchy
    const THBHierarchy& hierarchy() const { return hierarchy_; }

    /**
     * @brief Map active function index to (level, i, j)
     * @param active_idx Index in active_functions_ vector
     * @return Tuple (level, i, j)
     */
    std::tuple<int, int, int> active_to_level_ij(Index active_idx) const {
        return active_functions_[active_idx];
    }

    /**
     * @brief Map (level, i, j) to active function index
     * @return Index in active_functions_, or -1 if not active
     */
    Index level_ij_to_active(int level, int i, int j) const;

    /**
     * @brief Get mapping from active DOFs to global DOFs
     * @return Vector where result[active_idx] = global_dof
     */
    std::vector<Index> active_to_global_map() const;

    /**
     * @brief Check if support of (level, i, j) overlaps a refined region
     * @param level Basis function level
     * @param i U-direction index
     * @param j V-direction index
     * @param target_level Target refinement level
     * @return true if support intersects any region at target_level or finer
     */
    bool support_overlaps_level(int level, int i, int j, int target_level) const;

    /**
     * @brief Get the finest refinement level at a given physical point
     * @param x Physical x-coordinate
     * @param y Physical y-coordinate
     * @return Finest level l such that (x,y) is in Ω^l, or 0 if not in any refined region
     *
     * This determines which level's basis functions should be evaluated at this point.
     */
    int level_at_point(Real x, Real y) const;

  private:
    const THBHierarchy& hierarchy_;

    /// Element-to-level mapping
    std::vector<int> element_levels_;

    /// Set of active (level, i, j) tuples for fast lookup
    std::set<std::tuple<int, int, int>> active_set_;

    /// Ordered list of active functions
    std::vector<std::tuple<int, int, int>> active_functions_;

    /// Regions refined to each level: refined_regions_[l] = list of (xmin, xmax, ymin,
    /// ymax)
    std::vector<std::vector<std::tuple<Real, Real, Real, Real>>> refined_regions_;

    /// Build active set from quadtree
    void build_from_quadtree(const QuadtreeAdapter& quadtree);

    /// Check if a region is fully covered by finer-level regions
    bool is_fully_refined(Real xmin, Real xmax, Real ymin, Real ymax,
                          int current_level) const;

    /// Get support bounds of basis function in physical coordinates
    std::tuple<Real, Real, Real, Real> get_support_bounds(int level, int i,
                                                          int j) const;
};

}  // namespace drifter
