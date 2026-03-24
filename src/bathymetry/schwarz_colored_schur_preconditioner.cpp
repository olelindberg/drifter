#include "bathymetry/schwarz_colored_schur_preconditioner.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

namespace drifter {

SchwarzColoredSchurPreconditioner::SchwarzColoredSchurPreconditioner(
    std::function<VecX(const VecX &)> schur_matvec,
    Index num_constraints,
    const std::vector<CubicEdgeDerivativeConstraint> &edge_constraints,
    const std::vector<BoundaryCurvatureConstraint> &curvature_constraints,
    const std::vector<BoundaryGradientConstraint> &gradient_constraints,
    int num_iterations)
    : num_constraints_(num_constraints), num_iterations_(num_iterations), num_colors_(0) {
    // 1. Assemble S column by column (same as GaussSeidel)
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_constraints * num_constraints);

    VecX e = VecX::Zero(num_constraints);
    for (Index i = 0; i < num_constraints; ++i) {
        e(i) = 1.0;
        VecX Se = schur_matvec(e);

        for (Index j = 0; j < num_constraints; ++j) {
            if (std::abs(Se(j)) > 1e-14) {
                triplets.emplace_back(j, i, Se(j));
            }
        }
        e(i) = 0.0;
    }

    S_.resize(num_constraints, num_constraints);
    S_.setFromTriplets(triplets.begin(), triplets.end());
    S_.makeCompressed();

    // 2. Build edge-based block structure
    build_edge_blocks(edge_constraints, curvature_constraints, gradient_constraints);

    // 3. Color blocks using greedy graph coloring
    color_blocks();

    // 4. Extract and factorize block matrices
    factorize_blocks();
}

void SchwarzColoredSchurPreconditioner::build_edge_blocks(
    const std::vector<CubicEdgeDerivativeConstraint> &edge_constraints,
    const std::vector<BoundaryCurvatureConstraint> &curvature_constraints,
    const std::vector<BoundaryGradientConstraint> &gradient_constraints) {
    // Edge key: (elem1, elem2, edge1, edge2) for internal edges
    //           (elem, -1, edge, type) for boundary edges
    using EdgeKey = std::tuple<Index, Index, int, int>;
    std::map<EdgeKey, std::vector<Index>> edge_to_constraints;

    Index constraint_idx = 0;

    // Process edge derivative constraints (rows 0 to num_edge - 1)
    for (const auto &ec : edge_constraints) {
        // Normalize edge key: smaller element first
        Index e1 = std::min(ec.elem1, ec.elem2);
        Index e2 = std::max(ec.elem1, ec.elem2);
        int ed1 = (ec.elem1 <= ec.elem2) ? ec.edge1 : ec.edge2;
        int ed2 = (ec.elem1 <= ec.elem2) ? ec.edge2 : ec.edge1;

        EdgeKey key = std::make_tuple(e1, e2, ed1, ed2);
        edge_to_constraints[key].push_back(constraint_idx);
        ++constraint_idx;
    }

    // Process boundary curvature constraints
    // (rows num_edge to num_edge + num_curvature - 1)
    for (const auto &bc : curvature_constraints) {
        // Boundary edge key: (elem, -1, edge, 0) for curvature
        EdgeKey key = std::make_tuple(bc.elem, Index(-1), bc.edge, 0);
        edge_to_constraints[key].push_back(constraint_idx);
        ++constraint_idx;
    }

    // Process boundary gradient constraints
    // (rows num_edge + num_curvature to total - 1)
    for (const auto &gc : gradient_constraints) {
        // Boundary edge key: (elem, -1, edge, 1) for gradient
        EdgeKey key = std::make_tuple(gc.elem, Index(-1), gc.edge, 1);
        edge_to_constraints[key].push_back(constraint_idx);
        ++constraint_idx;
    }

    // Convert to vector form
    block_constraint_indices_.clear();
    block_constraint_indices_.reserve(edge_to_constraints.size());
    for (const auto &[key, indices] : edge_to_constraints) {
        block_constraint_indices_.push_back(indices);
    }
}

void SchwarzColoredSchurPreconditioner::color_blocks() {
    size_t num_blocks = block_constraint_indices_.size();
    if (num_blocks == 0) {
        num_colors_ = 0;
        return;
    }

    // Build constraint -> block mapping (vector for O(1) lookup)
    std::vector<Index> constraint_to_block(num_constraints_, -1);
    for (size_t b = 0; b < num_blocks; ++b) {
        for (Index c : block_constraint_indices_[b]) {
            constraint_to_block[c] = static_cast<Index>(b);
        }
    }

    // Build block adjacency via SINGLE pass over S columns - O(nnz(S))
    // Two blocks are adjacent if S has non-zero coupling between them
    std::vector<std::set<Index>> block_neighbors(num_blocks);

    for (int col = 0; col < S_.outerSize(); ++col) {
        Index col_block = constraint_to_block[col];
        if (col_block < 0) {
            continue;
        }

        for (SpMat::InnerIterator it(S_, col); it; ++it) {
            if (std::abs(it.value()) < 1e-14) {
                continue;
            }
            Index row_block = constraint_to_block[it.row()];
            if (row_block >= 0 && row_block != col_block) {
                block_neighbors[col_block].insert(row_block);
                block_neighbors[row_block].insert(col_block); // Symmetric
            }
        }
    }

    // Greedy graph coloring
    std::vector<int> block_color(num_blocks, -1);
    int max_color = -1;

    for (size_t b = 0; b < num_blocks; ++b) {
        std::set<int> neighbor_colors;
        for (Index neighbor : block_neighbors[b]) {
            if (block_color[neighbor] >= 0) {
                neighbor_colors.insert(block_color[neighbor]);
            }
        }

        // Find lowest available color
        int color = 0;
        while (neighbor_colors.count(color)) {
            ++color;
        }

        block_color[b] = color;
        max_color = std::max(max_color, color);
    }

    // Group blocks by color
    num_colors_ = max_color + 1;
    blocks_by_color_.clear();
    blocks_by_color_.resize(num_colors_);

    for (size_t b = 0; b < num_blocks; ++b) {
        blocks_by_color_[block_color[b]].push_back(static_cast<Index>(b));
    }
}

void SchwarzColoredSchurPreconditioner::factorize_blocks() {
    size_t num_blocks = block_constraint_indices_.size();
    block_lu_.resize(num_blocks);

    for (size_t b = 0; b < num_blocks; ++b) {
        const auto &indices = block_constraint_indices_[b];
        int block_size = static_cast<int>(indices.size());

        if (block_size == 0) {
            continue;
        }

        // Extract block matrix from S
        MatX S_block(block_size, block_size);
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                S_block(i, j) = S_.coeff(indices[i], indices[j]);
            }
        }

        // Add small diagonal regularization for numerical stability
        Real trace = S_block.trace();
        Real reg = std::max(1e-12, 1e-8 * std::abs(trace) / block_size);
        for (int i = 0; i < block_size; ++i) {
            S_block(i, i) += reg;
        }

        // Compute LU factorization
        block_lu_[b] = S_block.partialPivLu();
    }
}

VecX SchwarzColoredSchurPreconditioner::apply(const VecX &r) const {
    VecX z = VecX::Zero(num_constraints_);

    for (int iter = 0; iter < num_iterations_; ++iter) {
        // Compute S*z ONCE per iteration, then update incrementally
        VecX Sz = S_ * z;

        // Process each color sequentially (Gauss-Seidel between colors)
        for (int color = 0; color < num_colors_; ++color) {
            // Same-colored blocks don't couple, can be processed in parallel
            for (Index b : blocks_by_color_[color]) {
                const auto &indices = block_constraint_indices_[b];
                int block_size = static_cast<int>(indices.size());

                if (block_size == 0) {
                    continue;
                }

                // Gather local residual: r_local = r - Sz for block rows
                VecX r_local(block_size);
                for (int i = 0; i < block_size; ++i) {
                    r_local(i) = r(indices[i]) - Sz(indices[i]);
                }

                // Solve local system: S_block * dz = r_local
                VecX dz_local = block_lu_[b].solve(r_local);

                // Scatter to solution and incrementally update Sz
                for (int i = 0; i < block_size; ++i) {
                    Real dz_i = dz_local(i);
                    z(indices[i]) += dz_i;

                    // Update Sz incrementally: Sz += S[:, indices[i]] * dz_i
                    for (SpMat::InnerIterator it(S_, indices[i]); it; ++it) {
                        Sz(it.row()) += it.value() * dz_i;
                    }
                }
            }
        }
    }

    return z;
}

} // namespace drifter
