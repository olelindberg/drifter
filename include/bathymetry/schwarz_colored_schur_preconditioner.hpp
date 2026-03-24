#pragma once

/// @file schwarz_colored_schur_preconditioner.hpp
/// @brief Colored multiplicative Schwarz Schur complement preconditioner

#include "bathymetry/cg_cubic_bezier_dof_manager.hpp"
#include "bathymetry/schur_preconditioner.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <functional>
#include <vector>

namespace drifter {

/// @brief Colored multiplicative Schwarz preconditioner in constraint space
///
/// Operates directly on the assembled Schur complement S = C * Q^{-1} * C^T
/// using edge-based block decomposition:
/// - Each edge (internal or boundary) defines a block of constraints
/// - Blocks are colored so same-colored blocks share no constraints
/// - Apply: for each color, solve local block problems (can parallelize)
///
/// This is stronger than point-wise Gauss-Seidel because it captures
/// coupling between constraints along the same edge.
///
/// Setup cost: O(n_c) Schur matvecs + O(n_c^2) storage + block factorizations
/// Apply cost: O(num_colors * nnz(S) + sum(block_size^2)) per iteration
class SchwarzColoredSchurPreconditioner : public ISchurPreconditioner {
public:
    /// @brief Build edge-based block structure from constraint information
    /// @param schur_matvec Function computing S * v
    /// @param num_constraints Number of constraints
    /// @param edge_constraints Edge derivative constraints (for block detection)
    /// @param curvature_constraints Boundary curvature constraints
    /// @param gradient_constraints Boundary gradient constraints
    /// @param num_iterations Number of Schwarz sweeps per apply (default: 3)
    SchwarzColoredSchurPreconditioner(
        std::function<VecX(const VecX &)> schur_matvec,
        Index num_constraints,
        const std::vector<CubicEdgeDerivativeConstraint> &edge_constraints,
        const std::vector<BoundaryCurvatureConstraint> &curvature_constraints,
        const std::vector<BoundaryGradientConstraint> &gradient_constraints,
        int num_iterations = 3);

    /// @brief Apply preconditioner: z = M_S^{-1} * r via colored Schwarz
    VecX apply(const VecX &r) const override;

    /// @brief Number of constraints
    Index num_constraints() const override { return num_constraints_; }

    /// @brief Get number of edge blocks
    int num_blocks() const {
        return static_cast<int>(block_constraint_indices_.size());
    }

    /// @brief Get number of colors
    int num_colors() const { return num_colors_; }

    /// @brief Get number of Schwarz iterations
    int num_iterations() const { return num_iterations_; }

    /// @brief Get assembled Schur complement matrix for inspection (testing)
    const SpMat &schur_matrix() const { return S_; }

private:
    Index num_constraints_;
    int num_iterations_;
    SpMat S_; ///< Assembled Schur complement matrix

    /// Constraint indices for each block
    std::vector<std::vector<Index>> block_constraint_indices_;

    /// LU factorizations of block matrices
    std::vector<Eigen::PartialPivLU<MatX>> block_lu_;

    /// Block indices grouped by color
    std::vector<std::vector<Index>> blocks_by_color_;

    /// Number of colors
    int num_colors_;

    /// Build edge-based blocks from constraint structure
    void build_edge_blocks(
        const std::vector<CubicEdgeDerivativeConstraint> &edge_constraints,
        const std::vector<BoundaryCurvatureConstraint> &curvature_constraints,
        const std::vector<BoundaryGradientConstraint> &gradient_constraints);

    /// Color blocks using greedy graph coloring
    void color_blocks();

    /// Extract and factorize block matrices from S
    void factorize_blocks();
};

} // namespace drifter
