#pragma once

/// @file schwarz_block_builder.hpp
/// @brief Builder for Schwarz element block data structures
///
/// Creates element blocks and graph coloring for Schwarz methods
/// independently of the multigrid preconditioner. This allows using
/// Schwarz-based Schur preconditioners without requiring multigrid.

#include "bathymetry/cg_bezier_dof_manager_base.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <map>
#include <set>
#include <vector>

namespace drifter {

/// @brief Data structure holding precomputed element blocks for Schwarz methods
struct SchwarzBlockData {
    /// Free DOF indices for each element block
    std::vector<std::vector<Index>> element_free_dofs;

    /// Element block LU factorizations
    std::vector<Eigen::PartialPivLU<MatX>> element_block_lu;

    /// Element indices grouped by color (graph coloring)
    /// elements_by_color[c] contains indices of elements with color c
    std::vector<std::vector<Index>> elements_by_color;

    /// Number of colors used in coloring
    int num_colors = 0;
};

/// @brief Builder for Schwarz element block data
///
/// Extracts element-to-DOF mappings, computes block LU factorizations,
/// and performs graph coloring for Schwarz smoothers/preconditioners.
///
/// This allows using Schwarz methods without requiring the full multigrid
/// hierarchy to be built. The element block data is the same as what
/// BezierMultigridPreconditioner builds for its smoothers.
class SchwarzBlockBuilder {
public:
    /// @brief Build element blocks from system matrix and DOF manager
    /// @param Q System matrix (num_free_dofs x num_free_dofs)
    /// @param dof_manager DOF manager for element DOFs
    /// @param compute_coloring Whether to compute graph coloring (for ColoredSchwarz)
    /// @return SchwarzBlockData with element blocks and optionally coloring
    static SchwarzBlockData build(const SpMat &Q,
                                  const CGBezierDofManagerBase &dof_manager,
                                  bool compute_coloring = true);

private:
    /// @brief Build element block LU factorizations
    static void build_element_blocks(SchwarzBlockData &data, const SpMat &Q,
                                     const CGBezierDofManagerBase &dof_manager);

    /// @brief Build element coloring using greedy graph coloring
    static void build_element_coloring(SchwarzBlockData &data);
};

} // namespace drifter
