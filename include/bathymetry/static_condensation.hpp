#pragma once

/// @file static_condensation.hpp
/// @brief Static condensation for interior DOFs in CG Bezier elements
///
/// Implements Schur complement reduction to eliminate interior (bubble) DOFs
/// at element level, reducing global system size by ~25% for cubic Bezier.

#include "core/types.hpp"
#include <array>
#include <vector>

namespace drifter {

/// @brief Cache entry for static condensation of a single element
struct CondensationCache {
    /// Schur complement on skeleton DOFs: S = K_SS - K_SI * K_II^{-1} * K_IS
    MatX schur;

    /// Recovery operator for back-substitution: K_II^{-1} * K_IS
    /// Interior DOF values: x_I = -recovery * x_S (when f_I = 0)
    MatX recovery;

    /// K_II inverse for recovery with non-zero f_I
    MatX K_II_inv;

    /// RHS contribution from interior condensation: K_SI * K_II^{-1} * f_I
    VecX rhs_contrib;

    /// Flag indicating cache is valid
    bool is_valid = false;
};

/// @brief Static condensation manager for cubic Bezier elements (16 DOFs)
///
/// Performs element-level Schur complement reduction to eliminate interior
/// (bubble) DOFs, keeping only skeleton DOFs (vertices + edges) in the global
/// system. This reduces DOF count by 4/16 = 25%.
///
/// For element stiffness K partitioned as:
///   K = [K_SS  K_SI]
///       [K_IS  K_II]
///
/// The condensed system is:
///   S * x_S = f_S - K_SI * K_II^{-1} * f_I
///
/// where S = K_SS - K_SI * K_II^{-1} * K_IS is the Schur complement.
///
/// After solving for x_S, interior DOFs are recovered:
///   x_I = K_II^{-1} * (f_I - K_IS * x_S)
class StaticCondensationManager {
public:
    /// Number of skeleton DOFs per cubic Bezier element (4 corners + 8 edge)
    static constexpr int NUM_SKELETON = 12;

    /// Number of interior DOFs per cubic Bezier element (2x2 center)
    static constexpr int NUM_INTERIOR = 4;

    /// Total DOFs per element
    static constexpr int NUM_TOTAL = 16;

    /// Skeleton local DOF indices (corners + edges)
    static constexpr std::array<int, NUM_SKELETON> SKELETON_INDICES = {
        0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15};

    /// Interior local DOF indices (2x2 center: i,j in {1,2})
    static constexpr std::array<int, NUM_INTERIOR> INTERIOR_INDICES = {5, 6, 9, 10};

    /// Default constructor
    StaticCondensationManager() = default;

    /// @brief Condense a full 16x16 element matrix to 12x12 skeleton Schur complement
    /// @param K_elem Full element stiffness matrix (16x16)
    /// @param f_elem Element RHS vector (16x1), can be empty if no RHS
    /// @return Condensed 12x12 Schur complement matrix
    ///
    /// Also caches the recovery operator for later back-substitution.
    MatX condense(const MatX &K_elem, const VecX &f_elem = VecX());

    /// @brief Get condensed RHS contribution from interior DOFs
    /// @return K_SI * K_II^{-1} * f_I (12x1)
    ///
    /// Call after condense() if f_elem was non-empty.
    /// Full condensed RHS: f_condensed = f_S - get_rhs_contribution()
    VecX get_rhs_contribution() const;

    /// @brief Get skeleton DOF values from full element DOF vector
    /// @param x_full Full element DOF vector (16x1)
    /// @return Skeleton DOF values (12x1)
    static VecX extract_skeleton(const VecX &x_full);

    /// @brief Get interior DOF values from full element DOF vector
    /// @param x_full Full element DOF vector (16x1)
    /// @return Interior DOF values (4x1)
    static VecX extract_interior(const VecX &x_full);

    /// @brief Recover interior DOF values from skeleton solution
    /// @param x_skeleton Skeleton DOF solution (12x1)
    /// @param f_interior Interior RHS values (4x1), zero if not provided
    /// @return Interior DOF values (4x1)
    ///
    /// Computes: x_I = K_II^{-1} * (f_I - K_IS * x_S)
    VecX recover_interior(const VecX &x_skeleton,
                          const VecX &f_interior = VecX()) const;

    /// @brief Assemble full element DOF vector from skeleton and interior parts
    /// @param x_skeleton Skeleton DOF values (12x1)
    /// @param x_interior Interior DOF values (4x1)
    /// @return Full element DOF vector (16x1)
    static VecX assemble_full(const VecX &x_skeleton, const VecX &x_interior);

    /// @brief Get the recovery operator for manual computation
    /// @return K_II^{-1} * K_IS (4x12)
    const MatX &recovery_operator() const { return cache_.recovery; }

    /// @brief Check if condensation cache is valid
    bool is_valid() const { return cache_.is_valid; }

    /// @brief Clear the condensation cache
    void clear() { cache_.is_valid = false; }

private:
    CondensationCache cache_;
};

} // namespace drifter
