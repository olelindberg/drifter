#pragma once

#include "bathymetry/thb_spline/thb_hierarchy.hpp"
#include "bathymetry/thb_spline/thb_refinement_mask.hpp"
#include "core/types.hpp"
#include <map>
#include <vector>

namespace drifter {

/**
 * @brief Computes truncation coefficients for THB-splines
 *
 * Implements the truncation mechanism that ensures partition of unity when
 * some basis functions are replaced by finer ones.
 *
 * For cubic B-splines with dyadic refinement:
 *   Parent B^l_j = Σ_{k=0}^4 c_k × B^{l+1}_{2j+k-1}
 * where c = [1/8, 4/8, 6/8, 4/8, 1/8]
 *
 * When a parent is NOT active (replaced by children), its children need to
 * be "truncated" - their evaluation must subtract the inactive parent's
 * contribution to maintain partition of unity.
 *
 * Truncated child: T^{l+1}_i = B^{l+1}_i - Σ_j (c_{i-2j+1} × B^l_j)
 * where the sum is over inactive parents j at level l.
 */
class THBTruncation {
  public:
    /**
     * @brief Construct truncation data from refinement mask
     * @param hierarchy THB hierarchy with knot vectors
     * @param mask Active function information
     */
    THBTruncation(const THBHierarchy& hierarchy, const THBRefinementMask& mask);

    /**
     * @brief Evaluate truncated basis function at (u, v)
     * @param level Basis function level
     * @param i U-direction index
     * @param j V-direction index
     * @param u Parameter in u-direction
     * @param v Parameter in v-direction
     * @return Truncated basis value
     *
     * If the function has no truncation (level 0 or no inactive parents),
     * returns the standard B-spline value.
     */
    Real evaluate_truncated(int level, int i, int j, Real u, Real v) const;

    /**
     * @brief Evaluate all active truncated basis functions at (u, v)
     * @param u Parameter in u-direction
     * @param v Parameter in v-direction
     * @return Vector of values for each active function (in active function order)
     */
    VecX evaluate_all_active(Real u, Real v) const;

    /**
     * @brief Check if basis function needs truncation
     * @param level Basis function level
     * @param i U-direction index
     * @param j V-direction index
     * @return true if truncation coefficients exist for this function
     */
    bool needs_truncation(int level, int i, int j) const;

    /**
     * @brief Get truncation coefficients for a child function
     * @param level Child level (must be > 0)
     * @param i U-direction index at child level
     * @param j V-direction index at child level
     * @return Vector of (parent_level, parent_i, parent_j, coefficient)
     *
     * Returns empty vector if no truncation needed.
     */
    std::vector<std::tuple<int, int, int, Real>> get_truncation_coeffs(int level, int i,
                                                                       int j) const;

    /// Access underlying hierarchy
    const THBHierarchy& hierarchy() const { return hierarchy_; }

    /// Access refinement mask
    const THBRefinementMask& mask() const { return mask_; }

    /**
     * @brief 1D refinement coefficients for cubic B-splines
     * @return [1/8, 4/8, 6/8, 4/8, 1/8]
     */
    static std::vector<Real> refinement_coeffs_1d();

    /**
     * @brief 2D refinement coefficient (tensor product of 1D)
     * @param ki Offset in u-direction (0-4)
     * @param kj Offset in v-direction (0-4)
     * @return c_ki × c_kj
     */
    static Real refinement_coeff_2d(int ki, int kj);

  private:
    const THBHierarchy& hierarchy_;
    const THBRefinementMask& mask_;

    /// Truncation data: (level, i, j) -> list of (parent_level, parent_i, parent_j,
    /// coeff)
    std::map<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int, Real>>>
        truncation_data_;

    /// Precompute truncation coefficients for all active functions
    void build_truncation_data();

    /**
     * @brief Find which parents at level l-1 affect child (l, i, j) in u-direction
     * @param level Child level
     * @param i Child u-index
     * @return Vector of (parent_i, coefficient) pairs
     */
    std::vector<std::pair<int, Real>> parents_1d_u(int level, int i) const;

    /**
     * @brief Find which parents at level l-1 affect child (l, i, j) in v-direction
     * @param level Child level
     * @param j Child v-index
     * @return Vector of (parent_j, coefficient) pairs
     */
    std::vector<std::pair<int, Real>> parents_1d_v(int level, int j) const;
};

}  // namespace drifter
