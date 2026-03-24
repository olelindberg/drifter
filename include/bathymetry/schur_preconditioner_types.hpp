#pragma once

/// @file schur_preconditioner_types.hpp
/// @brief Preconditioner type enum for Schur complement CG solver

namespace drifter {

/// @brief Preconditioner type for Schur complement CG
///
/// The Schur complement system S * lambda = rhs arises from eliminating
/// primal variables in the KKT system. S = C * A^{-1} * C^T is SPD.
///
/// Available preconditioners:
/// - None: Unpreconditioned CG (monitors ||r||_2)
/// - Diagonal: M_S = diag(S), cheap but weak
/// - PhysicsBased: M_S = C * K^{-1} * C^T where K = smoothness Hessian
/// - MultigridVCycle: M_S^{-1} v = C * A_mg^{-1} * C^T * v (variable, requires FCG)
/// - GaussSeidel: Symmetric Gauss-Seidel on assembled S
/// - SchwarzColored: Colored multiplicative Schwarz on assembled S
/// - DiagonalApproxCG: M_S = C * diag(Q)^{-1} * C^T, inner CG solve (variable, requires FCG)
/// - BlockDiagApproxCG: M_S = C * blockdiag(Q)^{-1} * C^T, edge-based blocks (variable, requires FCG)
enum class SchurPreconditionerType {
    None,            ///< Unpreconditioned CG (current default behavior)
    Diagonal,        ///< Diagonal scaling M_S = diag(S)
    PhysicsBased,    ///< Physics-based M_S = C * K^{-1} * C^T (recommended)
    MultigridVCycle, ///< MG V-cycle as approximate A^{-1} (variable, requires FCG)
    GaussSeidel,     ///< Symmetric Gauss-Seidel on assembled S
    SchwarzColored,  ///< Colored multiplicative Schwarz on assembled S
    DiagonalApproxCG, ///< M_S = C * diag(Q)^{-1} * C^T, inner CG solve (variable)
    BlockDiagApproxCG ///< M_S = C * blockdiag(Q)^{-1} * C^T, edge blocks (variable)
};

} // namespace drifter
