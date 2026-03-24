#pragma once

/// @file block_diag_approx_cg_schur_preconditioner.hpp
/// @brief Block-diagonal approximation Schur preconditioner with inner CG solve

#include "bathymetry/cg_cubic_bezier_dof_manager.hpp"
#include "bathymetry/schur_preconditioner.hpp"
#include "core/types.hpp"
#include <Eigen/LU>
#include <set>
#include <vector>

namespace drifter {

/// @brief Element block data for block-diagonal Q approximation
struct ElementBlockData {
    Index element_id;                   ///< Element index owning this block
    std::vector<Index> free_dofs;       ///< Owned free DOF indices
    Eigen::PartialPivLU<MatX> block_lu; ///< LU factorization of Q[dofs, dofs]
};

/// @brief Block-diagonal approximation Schur complement preconditioner
///
/// Assembles M_S = C * blockdiag(Q)^{-1} * C^T where blocks are element-based.
/// Each block contains DOFs owned by a single element (non-overlapping partition).
/// DOF ownership: first element (by index) to reference a DOF owns it.
/// This captures more coupling than the scalar diagonal in DiagonalApproxCG.
///
/// Assembly process (during construction):
///   1. Determine DOF ownership (first element to reference owns the DOF)
///   2. For each element, collect its owned free DOFs
///   3. Extract Q[owned_dofs, owned_dofs] for each element
///   4. LU factorize each block
///   5. Build sparse D = blockdiag(blocks)^{-1}
///   6. M_S = C * D * C^T via sparse matrix multiplication
///
/// Application: Solve M_S * z = r using inner diagonal-preconditioned CG.
///
/// Setup cost: O(n_elements * block_size^3 + nnz(C)^2)
/// Apply cost: O(inner_iterations * nnz(M_S))
///
/// This is a variable preconditioner (inner CG is iterative) and requires
/// Flexible CG (FCG) for the outer solve.
class BlockDiagApproxCGSchurPreconditioner : public ISchurPreconditioner {
public:
    /// @brief Setup block-diagonal approximation preconditioner
    /// @param Q System matrix on free DOFs (n_free x n_free)
    /// @param C Constraint matrix on free DOFs (n_c x n_free)
    /// @param dof_manager DOF manager for edge enumeration and DOF mappings
    /// @param inner_tolerance CG tolerance for M_S^{-1} solve (default: 1e-6)
    /// @param inner_max_iterations Max inner CG iterations (default: 100)
    BlockDiagApproxCGSchurPreconditioner(const SpMat& Q, const SpMat& C,
                                         const CGCubicBezierDofManager& dof_manager,
                                         Real inner_tolerance = 1e-6,
                                         int inner_max_iterations = 100);

    /// @brief Apply preconditioner: z = M_S^{-1} * r via inner CG
    VecX apply(const VecX& r) const override;

    /// @brief This is a variable preconditioner (inner CG is iterative)
    bool is_variable() const override { return true; }

    /// @brief Number of constraints
    Index num_constraints() const override { return n_c_; }

    /// @brief Get assembled M_S matrix for inspection (testing)
    const SpMat& assembled_matrix() const { return M_S_; }

    /// @brief Get number of element blocks (for diagnostics)
    Index num_element_blocks() const { return static_cast<Index>(element_blocks_.size()); }

private:
    SpMat M_S_;                                  ///< Assembled C * blockdiag(Q)^{-1} * C^T
    VecX diag_M_S_inv_;                          ///< 1/diag(M_S) for inner CG preconditioning
    std::vector<ElementBlockData> element_blocks_; ///< LU factorizations for each element
    Index n_c_;                                  ///< Number of constraints
    Real inner_tol_;                             ///< Inner CG tolerance
    int inner_max_iter_;                         ///< Max inner CG iterations

    /// @brief Build element blocks with owned DOFs and LU factorizations
    void build_element_blocks(const SpMat& Q, const CGCubicBezierDofManager& dof_manager);

    /// @brief Assemble block-diagonal inverse matrix D
    SpMat build_block_diagonal_inverse(Index n_free) const;
};

} // namespace drifter
