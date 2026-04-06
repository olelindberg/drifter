#include "bathymetry/static_condensation.hpp"
#include <Eigen/Dense>
#include <stdexcept>

namespace drifter {

MatX StaticCondensationManager::condense(const MatX &K_elem, const VecX &f_elem) {
    if (K_elem.rows() != NUM_TOTAL || K_elem.cols() != NUM_TOTAL) {
        throw std::invalid_argument(
            "StaticCondensationManager::condense: K_elem must be 16x16");
    }

    // Extract submatrices
    MatX K_SS(NUM_SKELETON, NUM_SKELETON);
    MatX K_SI(NUM_SKELETON, NUM_INTERIOR);
    MatX K_IS(NUM_INTERIOR, NUM_SKELETON);
    MatX K_II(NUM_INTERIOR, NUM_INTERIOR);

    for (int i = 0; i < NUM_SKELETON; ++i) {
        for (int j = 0; j < NUM_SKELETON; ++j) {
            K_SS(i, j) = K_elem(SKELETON_INDICES[i], SKELETON_INDICES[j]);
        }
        for (int j = 0; j < NUM_INTERIOR; ++j) {
            K_SI(i, j) = K_elem(SKELETON_INDICES[i], INTERIOR_INDICES[j]);
        }
    }
    for (int i = 0; i < NUM_INTERIOR; ++i) {
        for (int j = 0; j < NUM_SKELETON; ++j) {
            K_IS(i, j) = K_elem(INTERIOR_INDICES[i], SKELETON_INDICES[j]);
        }
        for (int j = 0; j < NUM_INTERIOR; ++j) {
            K_II(i, j) = K_elem(INTERIOR_INDICES[i], INTERIOR_INDICES[j]);
        }
    }

    // Solve K_II * X = K_IS using LDLT (symmetric positive definite)
    Eigen::LDLT<MatX> K_II_ldlt(K_II);
    if (K_II_ldlt.info() != Eigen::Success) {
        throw std::runtime_error(
            "StaticCondensationManager::condense: K_II factorization failed");
    }

    // Store K_II inverse for recovery with non-zero f_I
    cache_.K_II_inv = K_II_ldlt.solve(MatX::Identity(NUM_INTERIOR, NUM_INTERIOR));

    // Recovery operator: K_II^{-1} * K_IS (4x12)
    cache_.recovery = K_II_ldlt.solve(K_IS);

    // Schur complement: S = K_SS - K_SI * K_II^{-1} * K_IS
    cache_.schur = K_SS - K_SI * cache_.recovery;

    // Symmetrize (numerical precision)
    cache_.schur = 0.5 * (cache_.schur + cache_.schur.transpose());

    // Handle RHS condensation if provided
    if (f_elem.size() == NUM_TOTAL) {
        VecX f_I(NUM_INTERIOR);
        for (int i = 0; i < NUM_INTERIOR; ++i) {
            f_I(i) = f_elem(INTERIOR_INDICES[i]);
        }
        // K_SI * K_II^{-1} * f_I
        cache_.rhs_contrib = K_SI * K_II_ldlt.solve(f_I);
    } else {
        cache_.rhs_contrib = VecX::Zero(NUM_SKELETON);
    }

    cache_.is_valid = true;
    return cache_.schur;
}

VecX StaticCondensationManager::get_rhs_contribution() const {
    if (!cache_.is_valid) {
        throw std::runtime_error(
            "StaticCondensationManager::get_rhs_contribution: cache not valid");
    }
    return cache_.rhs_contrib;
}

VecX StaticCondensationManager::extract_skeleton(const VecX &x_full) {
    if (x_full.size() != NUM_TOTAL) {
        throw std::invalid_argument(
            "StaticCondensationManager::extract_skeleton: x_full must be 16x1");
    }
    VecX x_skeleton(NUM_SKELETON);
    for (int i = 0; i < NUM_SKELETON; ++i) {
        x_skeleton(i) = x_full(SKELETON_INDICES[i]);
    }
    return x_skeleton;
}

VecX StaticCondensationManager::extract_interior(const VecX &x_full) {
    if (x_full.size() != NUM_TOTAL) {
        throw std::invalid_argument(
            "StaticCondensationManager::extract_interior: x_full must be 16x1");
    }
    VecX x_interior(NUM_INTERIOR);
    for (int i = 0; i < NUM_INTERIOR; ++i) {
        x_interior(i) = x_full(INTERIOR_INDICES[i]);
    }
    return x_interior;
}

VecX StaticCondensationManager::recover_interior(const VecX &x_skeleton,
                                                  const VecX &f_interior) const {
    if (!cache_.is_valid) {
        throw std::runtime_error(
            "StaticCondensationManager::recover_interior: cache not valid");
    }
    if (x_skeleton.size() != NUM_SKELETON) {
        throw std::invalid_argument(
            "StaticCondensationManager::recover_interior: x_skeleton must be 12x1");
    }

    // x_I = K_II^{-1} * (f_I - K_IS * x_S)
    //     = K_II^{-1} * f_I - K_II^{-1} * K_IS * x_S
    //     = K_II^{-1} * f_I - recovery * x_S
    VecX x_interior = -cache_.recovery * x_skeleton;

    // Add contribution from interior RHS if provided
    if (f_interior.size() == NUM_INTERIOR) {
        x_interior += cache_.K_II_inv * f_interior;
    }

    return x_interior;
}

VecX StaticCondensationManager::assemble_full(const VecX &x_skeleton,
                                               const VecX &x_interior) {
    if (x_skeleton.size() != NUM_SKELETON) {
        throw std::invalid_argument(
            "StaticCondensationManager::assemble_full: x_skeleton must be 12x1");
    }
    if (x_interior.size() != NUM_INTERIOR) {
        throw std::invalid_argument(
            "StaticCondensationManager::assemble_full: x_interior must be 4x1");
    }

    VecX x_full(NUM_TOTAL);
    for (int i = 0; i < NUM_SKELETON; ++i) {
        x_full(SKELETON_INDICES[i]) = x_skeleton(i);
    }
    for (int i = 0; i < NUM_INTERIOR; ++i) {
        x_full(INTERIOR_INDICES[i]) = x_interior(i);
    }
    return x_full;
}

} // namespace drifter
