#include "bathymetry/thb_spline/thb_refinement_mask.hpp"

#include <algorithm>
#include <cmath>

namespace drifter {

THBRefinementMask::THBRefinementMask(const QuadtreeAdapter& quadtree,
                                     const THBHierarchy& hierarchy)
    : hierarchy_(hierarchy) {
    build_from_quadtree(quadtree);
}

THBRefinementMask::THBRefinementMask(const THBHierarchy& hierarchy, int active_level)
    : hierarchy_(hierarchy) {
    // All functions at active_level are active, none at other levels
    element_levels_.clear();
    refined_regions_.resize(hierarchy.max_level() + 1);

    // Add entire domain as refined to active_level
    refined_regions_[active_level].push_back(
        {hierarchy.domain_min_u(), hierarchy.domain_max_u(), hierarchy.domain_min_v(),
         hierarchy.domain_max_v()});

    // Mark all functions at active_level as active
    for (int j = 0; j < hierarchy.num_basis_v(active_level); ++j) {
        for (int i = 0; i < hierarchy.num_basis_u(active_level); ++i) {
            active_set_.insert({active_level, i, j});
            active_functions_.push_back({active_level, i, j});
        }
    }
}

void THBRefinementMask::build_from_quadtree(const QuadtreeAdapter& quadtree) {
    const int max_level = hierarchy_.max_level();
    refined_regions_.resize(max_level + 1);
    element_levels_.resize(quadtree.num_elements());

    // Step 1: Assign each element to a THB level based on its size
    for (Index e = 0; e < quadtree.num_elements(); ++e) {
        Vec2 size = quadtree.element_size(e);
        int level = hierarchy_.level_for_element_size(size.x(), size.y());
        element_levels_[e] = level;

        // Record this element's bounds as refined to this level
        const QuadBounds& bounds = quadtree.element_bounds(e);
        refined_regions_[level].push_back(
            {bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax});
    }

    // Step 2: For each level, determine which basis functions are active
    //
    // With single-level evaluation (each point uses only functions at its level),
    // a function at level l is active if its support overlaps at least one region
    // at level l. We don't need to check for refinement by finer levels because
    // we only evaluate functions at the point's level.
    //
    // Note: This is a simplified approach compared to standard THB-splines which
    // use truncation for multi-level evaluation. The single-level approach:
    // - Maintains partition of unity (B-splines at each level sum to 1)
    // - May have discontinuities at level boundaries (C^0 instead of C^2)
    // - Is simpler to implement and debug

    for (int level = 0; level <= max_level; ++level) {
        const int nbu = hierarchy_.num_basis_u(level);
        const int nbv = hierarchy_.num_basis_v(level);

        for (int j = 0; j < nbv; ++j) {
            for (int i = 0; i < nbu; ++i) {
                auto [xmin, xmax, ymin, ymax] = get_support_bounds(level, i, j);

                // Check if support overlaps any region at this level
                bool overlaps_this_level = false;
                for (const auto& [rxmin, rxmax, rymin, rymax] :
                     refined_regions_[level]) {
                    // Check for overlap
                    if (xmax > rxmin && xmin < rxmax && ymax > rymin && ymin < rymax) {
                        overlaps_this_level = true;
                        break;
                    }
                }

                if (overlaps_this_level) {
                    active_set_.insert({level, i, j});
                    active_functions_.push_back({level, i, j});
                }
            }
        }
    }

    // Sort active functions by (level, j, i) for consistent ordering
    std::sort(active_functions_.begin(), active_functions_.end());
}

bool THBRefinementMask::is_fully_refined(Real xmin, Real xmax, Real ymin, Real ymax,
                                         int current_level) const {
    // A region is "fully refined" if it's entirely covered by regions at finer levels
    // This is a conservative check - we check if all corners are in finer regions

    const Real eps = 1e-10;
    const int max_level = hierarchy_.max_level();

    if (current_level >= max_level) {
        return false;  // Can't be refined beyond max level
    }

    // Sample points within the support region
    const int num_samples = 4;  // Check corners
    std::vector<std::pair<Real, Real>> sample_points = {
        {xmin + eps, ymin + eps},
        {xmax - eps, ymin + eps},
        {xmin + eps, ymax - eps},
        {xmax - eps, ymax - eps}};

    for (const auto& [px, py] : sample_points) {
        bool point_in_finer_region = false;

        // Check if this point is in any region at a finer level
        for (int finer_level = current_level + 1; finer_level <= max_level;
             ++finer_level) {
            for (const auto& [rxmin, rxmax, rymin, rymax] :
                 refined_regions_[finer_level]) {
                if (px >= rxmin && px <= rxmax && py >= rymin && py <= rymax) {
                    point_in_finer_region = true;
                    break;
                }
            }
            if (point_in_finer_region)
                break;
        }

        if (!point_in_finer_region) {
            return false;  // At least one corner is not in a finer region
        }
    }

    return true;  // All corners are covered by finer regions
}

std::tuple<Real, Real, Real, Real> THBRefinementMask::get_support_bounds(int level,
                                                                         int i,
                                                                         int j) const {
    auto [u_min, u_max] = hierarchy_.knots_u().support_physical(level, i);
    auto [v_min, v_max] = hierarchy_.knots_v().support_physical(level, j);
    return {u_min, u_max, v_min, v_max};
}

bool THBRefinementMask::is_active(int level, int i, int j) const {
    return active_set_.count({level, i, j}) > 0;
}

bool THBRefinementMask::is_active_dof(Index dof) const {
    auto [level, i, j] = hierarchy_.dof_to_level_ij(dof);
    return is_active(level, i, j);
}

std::vector<std::pair<int, int>> THBRefinementMask::active_at_level(int level) const {
    std::vector<std::pair<int, int>> result;
    for (const auto& [l, i, j] : active_functions_) {
        if (l == level) {
            result.push_back({i, j});
        }
    }
    return result;
}

std::vector<Index> THBRefinementMask::elements_at_level(int level) const {
    std::vector<Index> result;
    for (Index e = 0; e < static_cast<Index>(element_levels_.size()); ++e) {
        if (element_levels_[e] == level) {
            result.push_back(e);
        }
    }
    return result;
}

Index THBRefinementMask::level_ij_to_active(int level, int i, int j) const {
    auto it = std::find(active_functions_.begin(), active_functions_.end(),
                        std::make_tuple(level, i, j));
    if (it == active_functions_.end()) {
        return -1;
    }
    return static_cast<Index>(std::distance(active_functions_.begin(), it));
}

std::vector<Index> THBRefinementMask::active_to_global_map() const {
    std::vector<Index> result;
    result.reserve(active_functions_.size());
    for (const auto& [level, i, j] : active_functions_) {
        result.push_back(hierarchy_.global_dof(level, i, j));
    }
    return result;
}

bool THBRefinementMask::support_overlaps_level(int level, int i, int j,
                                               int target_level) const {
    if (target_level > hierarchy_.max_level()) {
        return false;
    }

    auto [xmin, xmax, ymin, ymax] = get_support_bounds(level, i, j);

    // Check overlap with any region at target_level or finer
    for (int l = target_level; l <= hierarchy_.max_level(); ++l) {
        for (const auto& [rxmin, rxmax, rymin, rymax] : refined_regions_[l]) {
            if (xmax > rxmin && xmin < rxmax && ymax > rymin && ymin < rymax) {
                return true;
            }
        }
    }
    return false;
}

int THBRefinementMask::level_at_point(Real x, Real y) const {
    // Find the finest level l such that point (x, y) is in Ω^l
    // Search from finest to coarsest
    const int max_level = hierarchy_.max_level();

    for (int level = max_level; level >= 0; --level) {
        for (const auto& [rxmin, rxmax, rymin, rymax] : refined_regions_[level]) {
            if (x >= rxmin && x <= rxmax && y >= rymin && y <= rymax) {
                return level;
            }
        }
    }

    // Point not in any explicitly defined region - return level 0 (coarsest)
    return 0;
}

}  // namespace drifter
