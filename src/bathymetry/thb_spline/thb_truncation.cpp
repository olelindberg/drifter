#include "bathymetry/thb_spline/thb_truncation.hpp"

#include <algorithm>

namespace drifter {

// Static 1D refinement coefficients for cubic B-splines
// These come from the two-scale relation: B^l_j = Σ_k c_k × B^{l+1}_{2j+k-1}
static const std::vector<Real> REFINE_COEFFS = {1.0 / 8.0, 4.0 / 8.0, 6.0 / 8.0,
                                                4.0 / 8.0, 1.0 / 8.0};

THBTruncation::THBTruncation(const THBHierarchy& hierarchy,
                             const THBRefinementMask& mask)
    : hierarchy_(hierarchy), mask_(mask) {
    build_truncation_data();
}

std::vector<Real> THBTruncation::refinement_coeffs_1d() {
    return REFINE_COEFFS;
}

Real THBTruncation::refinement_coeff_2d(int ki, int kj) {
    if (ki < 0 || ki > 4 || kj < 0 || kj > 4) {
        return 0.0;
    }
    return REFINE_COEFFS[ki] * REFINE_COEFFS[kj];
}

std::vector<std::pair<int, Real>> THBTruncation::parents_1d_u(int level, int i) const {
    // Child function i at level l is affected by parents at level l-1
    // Two-scale relation: B^{l-1}_p = Σ_{k=0}^4 c_k × B^l_{2p+k-1}
    // Solving for which parents affect child i:
    //   2p + k - 1 = i  =>  p = (i - k + 1) / 2
    // For k in [0, 4], p must be integer in [0, num_basis_u(level-1))

    std::vector<std::pair<int, Real>> result;
    if (level <= 0)
        return result;

    const int num_parents = hierarchy_.num_basis_u(level - 1);

    for (int k = 0; k <= 4; ++k) {
        // 2p + k - 1 = i  =>  p = (i + 1 - k) / 2
        int numerator = i + 1 - k;
        if (numerator < 0 || numerator % 2 != 0)
            continue;

        int p = numerator / 2;
        if (p >= 0 && p < num_parents) {
            result.push_back({p, REFINE_COEFFS[k]});
        }
    }

    return result;
}

std::vector<std::pair<int, Real>> THBTruncation::parents_1d_v(int level, int j) const {
    // Same logic as parents_1d_u but for v-direction
    std::vector<std::pair<int, Real>> result;
    if (level <= 0)
        return result;

    const int num_parents = hierarchy_.num_basis_v(level - 1);

    for (int k = 0; k <= 4; ++k) {
        int numerator = j + 1 - k;
        if (numerator < 0 || numerator % 2 != 0)
            continue;

        int p = numerator / 2;
        if (p >= 0 && p < num_parents) {
            result.push_back({p, REFINE_COEFFS[k]});
        }
    }

    return result;
}

void THBTruncation::build_truncation_data() {
    // For each active function at level > 0, find which inactive parents
    // contribute to it and need to be truncated

    for (const auto& [level, i, j] : mask_.active_functions()) {
        if (level == 0)
            continue;

        // Get parent contributions in each direction
        auto parents_u = parents_1d_u(level, i);
        auto parents_v = parents_1d_v(level, j);

        // For each 2D parent (tensor product), check if it's inactive
        std::vector<std::tuple<int, int, int, Real>> truncation_entries;

        for (const auto& [pi, ci] : parents_u) {
            for (const auto& [pj, cj] : parents_v) {
                // Check if this parent is NOT active (needs truncation)
                if (!mask_.is_active(level - 1, pi, pj)) {
                    Real coeff = ci * cj;
                    truncation_entries.push_back({level - 1, pi, pj, coeff});
                }
            }
        }

        if (!truncation_entries.empty()) {
            truncation_data_[{level, i, j}] = std::move(truncation_entries);
        }
    }
}

bool THBTruncation::needs_truncation(int level, int i, int j) const {
    return truncation_data_.count({level, i, j}) > 0;
}

std::vector<std::tuple<int, int, int, Real>>
THBTruncation::get_truncation_coeffs(int level, int i, int j) const {
    auto it = truncation_data_.find({level, i, j});
    if (it != truncation_data_.end()) {
        return it->second;
    }
    return {};
}

Real THBTruncation::evaluate_truncated(int level, int i, int j, Real u, Real v) const {
    // Start with standard B-spline value
    const auto& basis = hierarchy_.basis(level);
    Real value = basis.evaluate(i, j, u, v);

    // Subtract inactive parent contributions
    auto it = truncation_data_.find({level, i, j});
    if (it != truncation_data_.end()) {
        for (const auto& [parent_level, pi, pj, coeff] : it->second) {
            const auto& parent_basis = hierarchy_.basis(parent_level);

            // Convert parameter coordinates from child level to parent level
            // For dyadic refinement: child level l has 2^(l-parent_level) times more spans
            // So u_parent = u / 2^(level - parent_level)
            int level_diff = level - parent_level;
            Real scale = 1.0 / (1 << level_diff);  // 1 / 2^level_diff
            Real u_parent = u * scale;
            Real v_parent = v * scale;

            value -= coeff * parent_basis.evaluate(pi, pj, u_parent, v_parent);
        }
    }

    return value;
}

VecX THBTruncation::evaluate_all_active(Real u, Real v) const {
    const auto& active_funcs = mask_.active_functions();
    VecX result(active_funcs.size());

    for (Index idx = 0; idx < static_cast<Index>(active_funcs.size()); ++idx) {
        const auto& [level, i, j] = active_funcs[idx];
        result(idx) = evaluate_truncated(level, i, j, u, v);
    }

    return result;
}

}  // namespace drifter
