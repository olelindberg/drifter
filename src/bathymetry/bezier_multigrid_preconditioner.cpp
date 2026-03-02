#include "bathymetry/bezier_multigrid_preconditioner.hpp"
#include "core/scoped_timer.hpp"
#include "mesh/morton.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace drifter {

namespace {

/// @brief Compute 2D parent Morton code for quadtree
/// @details Since quadtree uses Morton3D::encode(x, y, 0), parent is encode(x/2, y/2, 0)
uint64_t parent_2d(uint64_t code) {
    uint32_t x, y, z;
    Morton3D::decode(code, x, y, z);
    return Morton3D::encode(x / 2, y / 2, 0);
}

/// @brief Get child index within parent (0-3) for 2D quadtree
/// @return Child index: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
int child_index_2d(uint64_t code) {
    uint32_t x, y, z;
    Morton3D::decode(code, x, y, z);
    return (x & 1) + 2 * (y & 1);
}

} // namespace

BezierMultigridPreconditioner::BezierMultigridPreconditioner(
    const MultigridConfig &config)
    : config_(config) {
    // Precompute Bezier subdivision matrices
    CubicBezierBasis2D basis;
    S_left_ = basis.compute_1d_extraction_matrix(0.0, 0.5);
    S_right_ = basis.compute_1d_extraction_matrix(0.5, 1.0);
}

MatX BezierMultigridPreconditioner::kronecker_product(const MatX &A,
                                                      const MatX &B) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int p = static_cast<int>(B.rows());
    const int q = static_cast<int>(B.cols());

    MatX result(m * p, n * q);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result.block(i * p, j * q, p, q) = A(i, j) * B;
        }
    }
    return result;
}

void BezierMultigridPreconditioner::setup(
    const SpMat &Q_fine, const QuadtreeAdapter &mesh,
    const CGCubicBezierDofManager &dof_manager) {
    OptionalScopedTimer t_total(profile_ ? &profile_->setup_total_ms : nullptr);

    levels_.clear();

    // Level 0 is the finest level (input)
    {
        OptionalScopedTimer t(profile_ ? &profile_->setup_finest_init_ms
                                       : nullptr);
        MultigridLevel finest;
        finest.Q = Q_fine;
        finest.num_dofs = Q_fine.rows();
        finest.diag_inv = VecX(finest.num_dofs);
        for (Index i = 0; i < finest.num_dofs; ++i) {
            Real diag = Q_fine.coeff(i, i);
            finest.diag_inv(i) = (std::abs(diag) > 1e-14) ? 1.0 / diag : 0.0;
        }
        levels_.push_back(std::move(finest));
    }

    // Build element blocks for Schwarz smoother on finest level
    if (config_.smoother_type == SmootherType::MultiplicativeSchwarz ||
        config_.smoother_type == SmootherType::AdditiveSchwarz ||
        config_.smoother_type == SmootherType::ColoredMultiplicativeSchwarz) {
        OptionalScopedTimer t(profile_ ? &profile_->setup_element_blocks_ms
                                       : nullptr);
        build_element_blocks(0, dof_manager);

        // Build coloring for colored Schwarz
        if (config_.smoother_type == SmootherType::ColoredMultiplicativeSchwarz) {
            build_element_coloring(0);
        }
    }

    // Build coarse levels
    int target_levels =
        std::min(config_.num_levels, static_cast<int>(mesh.num_elements()));

    // We need enough elements to coarsen
    // Each level reduces elements by ~4x, so we need at least 4^(levels-1) elements
    Index min_elements_for_levels = 1;
    for (int l = 1; l < target_levels; ++l) {
        min_elements_for_levels *= 4;
    }

    if (mesh.num_elements() < min_elements_for_levels) {
        target_levels = 1;
        Index n = mesh.num_elements();
        while (n >= 4) {
            target_levels++;
            n /= 4;
        }
    }

    if (config_.verbose) {
        std::cerr << "[Multigrid] Building " << target_levels << " levels\n";
        std::cerr << "[Multigrid] Level 0: " << levels_[0].num_dofs << " DOFs\n";
    }

    // Build each coarse level
    for (int l = 1; l < target_levels; ++l) {
        size_t levels_before = levels_.size();
        build_coarse_level(l - 1, mesh, dof_manager);

        // Check if a level was actually added
        if (levels_.size() == levels_before) {
            // Coarsening stopped (couldn't reduce DOFs significantly)
            break;
        }

        if (config_.verbose) {
            std::cerr << "[Multigrid] Level " << levels_.size() - 1 << ": "
                      << levels_.back().num_dofs << " DOFs\n";
        }

        // Stop if we've reached the coarsest size threshold
        if (levels_.back().num_dofs <= config_.coarsest_direct_size) {
            break;
        }
    }

    // Setup direct solver for coarsest level
    {
        OptionalScopedTimer t(profile_ ? &profile_->setup_coarse_lu_ms : nullptr);
        int coarsest = static_cast<int>(levels_.size()) - 1;
        levels_[coarsest].solver = std::make_unique<Eigen::SparseLU<SpMat>>();
        levels_[coarsest].solver->compute(levels_[coarsest].Q);

        if (levels_[coarsest].solver->info() != Eigen::Success) {
            throw std::runtime_error(
                "BezierMultigridPreconditioner: Coarsest level LU factorization "
                "failed");
        }
    }
}

SpMat BezierMultigridPreconditioner::build_prolongation(
    const QuadtreeAdapter &mesh,
    const CGCubicBezierDofManager &fine_dof_manager, Index &coarse_num_dofs) {

    // Group fine elements by parent Morton code
    std::map<uint64_t, std::vector<Index>> parent_to_fine;
    for (Index e = 0; e < mesh.num_elements(); ++e) {
        uint64_t morton = mesh.elements()[e]->morton;
        uint64_t parent_morton = parent_2d(morton);
        parent_to_fine[parent_morton].push_back(e);
    }

    // For uniform meshes, all parents should have 4 children
    // For adaptive meshes, some parents may have fewer children (not fully refined)
    // We handle both cases by building the prolongation element-by-element

    // Build coarse DOF numbering using position-based deduplication
    // Same strategy as CGCubicBezierDofManager
    static constexpr Real POSITION_SCALE = 1e8;
    auto quantize = [](Real x, Real y) -> std::pair<int64_t, int64_t> {
        return {static_cast<int64_t>(std::round(x * POSITION_SCALE)),
                static_cast<int64_t>(std::round(y * POSITION_SCALE))};
    };

    // Map from coarse element (parent Morton) to coarse DOF indices
    std::map<uint64_t, std::vector<Index>> coarse_elem_to_dofs;
    std::map<std::pair<int64_t, int64_t>, Index> coarse_position_to_dof;
    coarse_num_dofs = 0;

    // 2D Bezier subdivision matrices for each child quadrant
    // Child 0: [0,0.5]×[0,0.5], Child 1: [0.5,1]×[0,0.5]
    // Child 2: [0,0.5]×[0.5,1], Child 3: [0.5,1]×[0.5,1]
    std::array<MatX, 4> P_local;
    P_local[0] = kronecker_product(S_left_, S_left_);   // (0,0)
    P_local[1] = kronecker_product(S_right_, S_left_);  // (1,0)
    P_local[2] = kronecker_product(S_left_, S_right_);  // (0,1)
    P_local[3] = kronecker_product(S_right_, S_right_); // (1,1)

    // Build coarse DOF numbering
    // For each coarse element (parent), compute the 16 DOF positions
    for (const auto &[parent_morton, children] : parent_to_fine) {
        // Use the first child to compute parent bounds
        if (children.empty())
            continue;

        // Compute parent element bounds from children
        Real xmin = std::numeric_limits<Real>::max();
        Real xmax = std::numeric_limits<Real>::lowest();
        Real ymin = std::numeric_limits<Real>::max();
        Real ymax = std::numeric_limits<Real>::lowest();

        for (Index child_elem : children) {
            const auto &bounds = mesh.element_bounds(child_elem);
            xmin = std::min(xmin, bounds.xmin);
            xmax = std::max(xmax, bounds.xmax);
            ymin = std::min(ymin, bounds.ymin);
            ymax = std::max(ymax, bounds.ymax);
        }

        Real dx = xmax - xmin;
        Real dy = ymax - ymin;

        // 16 DOFs in 4×4 grid (cubic Bezier)
        std::vector<Index> coarse_dofs(16);
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                int local_dof = i + 4 * j;
                Real u = i / 3.0;
                Real v = j / 3.0;
                Real x = xmin + u * dx;
                Real y = ymin + v * dy;

                auto key = quantize(x, y);
                auto it = coarse_position_to_dof.find(key);
                if (it != coarse_position_to_dof.end()) {
                    coarse_dofs[local_dof] = it->second;
                } else {
                    coarse_position_to_dof[key] = coarse_num_dofs;
                    coarse_dofs[local_dof] = coarse_num_dofs;
                    coarse_num_dofs++;
                }
            }
        }
        coarse_elem_to_dofs[parent_morton] = std::move(coarse_dofs);
    }

    // Build prolongation matrix
    Index fine_num_dofs = fine_dof_manager.num_free_dofs();
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(mesh.num_elements() * 16 * 16);

    for (Index e = 0; e < mesh.num_elements(); ++e) {
        uint64_t morton = mesh.elements()[e]->morton;
        uint64_t parent_morton = parent_2d(morton);
        int child_idx = child_index_2d(morton);

        const auto &fine_dofs = fine_dof_manager.element_dofs(e);
        const auto &coarse_dofs = coarse_elem_to_dofs[parent_morton];
        const MatX &P_elem = P_local[child_idx];

        // P_elem: maps coarse DOFs (16) -> fine DOFs (16) for this element
        // P_elem(fine_local, coarse_local) = weight
        for (int fine_local = 0; fine_local < 16; ++fine_local) {
            Index fine_global = fine_dofs[fine_local];
            Index fine_free = fine_dof_manager.global_to_free(fine_global);

            // Skip constrained DOFs
            if (fine_free < 0)
                continue;

            for (int coarse_local = 0; coarse_local < 16; ++coarse_local) {
                Real weight = P_elem(fine_local, coarse_local);
                if (std::abs(weight) < 1e-14)
                    continue;

                Index coarse_free = coarse_dofs[coarse_local];
                triplets.emplace_back(fine_free, coarse_free, weight);
            }
        }
    }

    SpMat P(fine_num_dofs, coarse_num_dofs);
    P.setFromTriplets(triplets.begin(), triplets.end());

    return P;
}

void BezierMultigridPreconditioner::build_coarse_level(
    int fine_level, const QuadtreeAdapter &mesh,
    const CGCubicBezierDofManager &fine_dof_manager) {

    // Build prolongation operator
    Index coarse_num_dofs = 0;
    SpMat P;
    {
        OptionalScopedTimer t(profile_ ? &profile_->setup_prolongation_ms
                                       : nullptr);
        P = build_prolongation(mesh, fine_dof_manager, coarse_num_dofs);
    }

    // If coarsening didn't reduce DOFs significantly, stop
    if (coarse_num_dofs >= levels_[fine_level].num_dofs * 0.9) {
        // Can't coarsen further meaningfully
        return;
    }

    MultigridLevel coarse;
    coarse.num_dofs = coarse_num_dofs;

    // Restriction = transpose of prolongation (scaled by mass if needed, but
    // simple transpose works for Galerkin)
    coarse.R = P.transpose();

    // Store prolongation in fine level (for use in V-cycle)
    levels_[fine_level].P = P;

    // Galerkin projection: Q_coarse = R * Q_fine * P
    {
        OptionalScopedTimer t(profile_ ? &profile_->setup_galerkin_ms : nullptr);
        SpMat temp = levels_[fine_level].Q * P;
        coarse.Q = coarse.R * temp;

        // Symmetrize (numerical cleanup)
        coarse.Q = 0.5 * (coarse.Q + SpMat(coarse.Q.transpose()));
    }

    // Compute inverse diagonal for Jacobi smoother
    coarse.diag_inv = VecX(coarse_num_dofs);
    for (Index i = 0; i < coarse_num_dofs; ++i) {
        Real diag = coarse.Q.coeff(i, i);
        coarse.diag_inv(i) = (std::abs(diag) > 1e-14) ? 1.0 / diag : 0.0;
    }

    levels_.push_back(std::move(coarse));
}

void BezierMultigridPreconditioner::build_element_blocks(
    int level, const CGCubicBezierDofManager &dof_manager) {

    auto &L = levels_[level];
    Index num_elements = dof_manager.mesh().num_elements();

    L.element_free_dofs.resize(num_elements);
    L.element_block_lu.resize(num_elements);

    for (Index e = 0; e < num_elements; ++e) {
        const auto &global_dofs = dof_manager.element_dofs(e);

        // Map to free DOF indices, skipping constrained DOFs
        std::vector<Index> free_dofs;
        free_dofs.reserve(16);
        for (int local = 0; local < 16; ++local) {
            Index global = global_dofs[local];
            Index free = dof_manager.global_to_free(global);
            if (free >= 0) {
                free_dofs.push_back(free);
            }
        }

        L.element_free_dofs[e] = free_dofs;

        // Extract element block from Q
        int block_size = static_cast<int>(free_dofs.size());
        if (block_size == 0) {
            // All DOFs constrained, store empty factorization
            L.element_block_lu[e] = Eigen::PartialPivLU<MatX>();
            continue;
        }

        MatX Q_block(block_size, block_size);
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                Q_block(i, j) = L.Q.coeff(free_dofs[i], free_dofs[j]);
            }
        }

        // LU factorize for local solves
        L.element_block_lu[e] = Q_block.partialPivLu();
    }
}

VecX BezierMultigridPreconditioner::apply(const VecX &r) const {
    if (levels_.empty()) {
        throw std::runtime_error(
            "BezierMultigridPreconditioner::apply: setup() not called");
    }

    OptionalScopedTimer t(profile_ ? &profile_->apply_total_ms : nullptr);
    if (profile_) {
        profile_->apply_calls++;
    }

    VecX x = VecX::Zero(r.size());
    v_cycle(0, x, r);
    return x;
}

void BezierMultigridPreconditioner::v_cycle(int level, VecX &x,
                                            const VecX &b) const {
    if (profile_) {
        profile_->vcycle_calls++;
    }

    const auto &L = levels_[level];

    // Coarsest level: direct solve
    if (level == static_cast<int>(levels_.size()) - 1) {
        OptionalScopedTimer t(profile_ ? &profile_->vcycle_coarse_solve_ms
                                       : nullptr);
        if (profile_) {
            profile_->coarse_solves++;
        }
        x = L.solver->solve(b);
        return;
    }

    // Pre-smoothing (use Schwarz on finest level if configured)
    {
        OptionalScopedTimer t(profile_ ? &profile_->vcycle_pre_smooth_ms
                                       : nullptr);
        if (config_.smoother_type == SmootherType::MultiplicativeSchwarz &&
            level == 0) {
            smooth_schwarz(level, x, b, config_.pre_smoothing);
        } else if (config_.smoother_type == SmootherType::AdditiveSchwarz &&
                   level == 0) {
            smooth_schwarz_additive(level, x, b, config_.pre_smoothing);
        } else if (config_.smoother_type ==
                       SmootherType::ColoredMultiplicativeSchwarz &&
                   level == 0) {
            smooth_schwarz_colored(level, x, b, config_.pre_smoothing);
        } else {
            smooth_jacobi(level, x, b, config_.pre_smoothing);
        }
    }

    // Compute residual: r = b - Q*x
    VecX r;
    {
        OptionalScopedTimer t(profile_ ? &profile_->vcycle_residual_ms : nullptr);
        if (profile_) {
            profile_->matvec_products++;
        }
        r = b - L.Q * x;
    }

    // Restrict to coarser level: r_c = R * r
    const auto &Lc = levels_[level + 1];
    VecX r_coarse;
    {
        OptionalScopedTimer t(profile_ ? &profile_->vcycle_restrict_ms : nullptr);
        r_coarse = Lc.R * r;
    }

    // Solve on coarse level (recursively)
    VecX e_coarse = VecX::Zero(Lc.num_dofs);
    v_cycle(level + 1, e_coarse, r_coarse);

    // Prolongate correction: e = P * e_c
    VecX e;
    {
        OptionalScopedTimer t(profile_ ? &profile_->vcycle_prolong_ms : nullptr);
        e = L.P * e_coarse;
    }

    // Update solution: x = x + e
    x += e;

    // Post-smoothing (use Schwarz on finest level if configured)
    {
        OptionalScopedTimer t(profile_ ? &profile_->vcycle_post_smooth_ms
                                       : nullptr);
        if (config_.smoother_type == SmootherType::MultiplicativeSchwarz &&
            level == 0) {
            smooth_schwarz(level, x, b, config_.post_smoothing);
        } else if (config_.smoother_type == SmootherType::AdditiveSchwarz &&
                   level == 0) {
            smooth_schwarz_additive(level, x, b, config_.post_smoothing);
        } else if (config_.smoother_type ==
                       SmootherType::ColoredMultiplicativeSchwarz &&
                   level == 0) {
            smooth_schwarz_colored(level, x, b, config_.post_smoothing);
        } else {
            smooth_jacobi(level, x, b, config_.post_smoothing);
        }
    }
}

void BezierMultigridPreconditioner::smooth_jacobi(int level, VecX &x,
                                                  const VecX &b,
                                                  int iters) const {
    OptionalScopedTimer t(profile_ ? &profile_->jacobi_total_ms : nullptr);

    const auto &L = levels_[level];
    Real omega = config_.jacobi_omega;

    for (int iter = 0; iter < iters; ++iter) {
        if (profile_) {
            profile_->jacobi_iterations++;
            profile_->matvec_products++;
        }
        VecX r = b - L.Q * x;
        x += omega * (L.diag_inv.asDiagonal() * r);
    }
}

void BezierMultigridPreconditioner::smooth_schwarz(int level, VecX &x,
                                                   const VecX &b,
                                                   int iters) const {
    const auto &L = levels_[level];

    // If no element blocks, fall back to Jacobi
    if (L.element_free_dofs.empty()) {
        smooth_jacobi(level, x, b, iters);
        return;
    }

    size_t num_elements = L.element_free_dofs.size();

    for (int iter = 0; iter < iters; ++iter) {
        if (profile_) {
            profile_->schwarz_iterations++;
        }

        // Compute full residual once (sparse matmul is efficient)
        VecX Qx;
        {
            OptionalScopedTimer t(profile_ ? &profile_->schwarz_matvec_ms
                                           : nullptr);
            if (profile_) {
                profile_->matvec_products++;
            }
            Qx = L.Q * x;
        }

        // Forward sweep through elements
        for (size_t e = 0; e < num_elements; ++e) {
            const auto &free_dofs = L.element_free_dofs[e];
            int block_size = static_cast<int>(free_dofs.size());
            if (block_size == 0)
                continue;

            // Gather local residual
            VecX r_local(block_size);
            {
                OptionalScopedTimer t(profile_ ? &profile_->schwarz_gather_ms
                                               : nullptr);
                for (int i = 0; i < block_size; ++i) {
                    r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
                }
            }

            // Solve local system: Q_block * dx = r_local
            VecX dx_local;
            {
                OptionalScopedTimer t(profile_ ? &profile_->schwarz_local_solve_ms
                                               : nullptr);
                if (profile_) {
                    profile_->schwarz_element_solves++;
                }
                dx_local = L.element_block_lu[e].solve(r_local);
            }

            // Update solution and Qx immediately (Gauss-Seidel style)
            {
                OptionalScopedTimer t(
                    profile_ ? &profile_->schwarz_scatter_update_ms : nullptr);
                for (int i = 0; i < block_size; ++i) {
                    Index dof = free_dofs[i];
                    x(dof) += dx_local(i);
                    // Update Qx for subsequent elements (column iteration)
                    for (SpMat::InnerIterator it(L.Q, dof); it; ++it) {
                        Qx(it.index()) += it.value() * dx_local(i);
                    }
                }
            }
        }
    }
}

void BezierMultigridPreconditioner::smooth_schwarz_additive(int level, VecX &x,
                                                             const VecX &b,
                                                             int iters) const {
    const auto &L = levels_[level];

    // If no element blocks, fall back to Jacobi
    if (L.element_free_dofs.empty()) {
        smooth_jacobi(level, x, b, iters);
        return;
    }

    size_t num_elements = L.element_free_dofs.size();
    Real omega = config_.jacobi_omega; // Use same damping as Jacobi

    for (int iter = 0; iter < iters; ++iter) {
        if (profile_) {
            profile_->schwarz_iterations++;
        }

        // Compute full residual once
        VecX Qx;
        {
            OptionalScopedTimer t(profile_ ? &profile_->schwarz_matvec_ms
                                           : nullptr);
            if (profile_) {
                profile_->matvec_products++;
            }
            Qx = L.Q * x;
        }

        // Accumulate all corrections (additive: no immediate updates)
        VecX dx_total = VecX::Zero(x.size());

        // Process all elements (can be parallelized)
        for (size_t e = 0; e < num_elements; ++e) {
            const auto &free_dofs = L.element_free_dofs[e];
            int block_size = static_cast<int>(free_dofs.size());
            if (block_size == 0)
                continue;

            // Gather local residual
            VecX r_local(block_size);
            {
                OptionalScopedTimer t(profile_ ? &profile_->schwarz_gather_ms
                                               : nullptr);
                for (int i = 0; i < block_size; ++i) {
                    r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
                }
            }

            // Solve local system: Q_block * dx = r_local
            VecX dx_local;
            {
                OptionalScopedTimer t(profile_ ? &profile_->schwarz_local_solve_ms
                                               : nullptr);
                if (profile_) {
                    profile_->schwarz_element_solves++;
                }
                dx_local = L.element_block_lu[e].solve(r_local);
            }

            // Accumulate corrections (no Qx update - that's the key difference!)
            {
                OptionalScopedTimer t(
                    profile_ ? &profile_->schwarz_scatter_update_ms : nullptr);
                for (int i = 0; i < block_size; ++i) {
                    dx_total(free_dofs[i]) += dx_local(i);
                }
            }
        }

        // Apply all corrections at once with damping
        x += omega * dx_total;
    }
}

void BezierMultigridPreconditioner::build_element_coloring(int level) {
    auto &L = levels_[level];

    // Clear existing coloring
    L.elements_by_color.clear();
    L.num_colors = 0;

    size_t num_elements = L.element_free_dofs.size();
    if (num_elements == 0)
        return;

    // Build DOF -> elements adjacency map
    std::map<Index, std::vector<Index>> dof_to_elements;
    for (size_t e = 0; e < num_elements; ++e) {
        for (Index dof : L.element_free_dofs[e]) {
            dof_to_elements[dof].push_back(static_cast<Index>(e));
        }
    }

    // Greedy graph coloring based on DOF adjacency
    std::vector<int> element_color(num_elements, -1);
    int max_color = -1;

    for (size_t e = 0; e < num_elements; ++e) {
        // Find colors used by adjacent elements (elements sharing DOFs)
        std::set<int> neighbor_colors;
        for (Index dof : L.element_free_dofs[e]) {
            for (Index neighbor : dof_to_elements[dof]) {
                if (neighbor != static_cast<Index>(e) &&
                    element_color[neighbor] >= 0) {
                    neighbor_colors.insert(element_color[neighbor]);
                }
            }
        }

        // Find lowest available color
        int color = 0;
        while (neighbor_colors.count(color)) {
            color++;
        }

        element_color[e] = color;
        max_color = std::max(max_color, color);
    }

    // Build color groups
    L.num_colors = max_color + 1;
    L.elements_by_color.resize(L.num_colors);

    for (size_t e = 0; e < num_elements; ++e) {
        L.elements_by_color[element_color[e]].push_back(static_cast<Index>(e));
    }

    if (config_.verbose) {
        std::cerr << "[Multigrid] Graph coloring: " << L.num_colors
                  << " colors for " << num_elements << " elements (";
        for (int c = 0; c < L.num_colors; ++c) {
            std::cerr << L.elements_by_color[c].size();
            if (c < L.num_colors - 1)
                std::cerr << "/";
        }
        std::cerr << ")\n";
    }
}

void BezierMultigridPreconditioner::smooth_schwarz_colored(int level, VecX &x,
                                                            const VecX &b,
                                                            int iters) const {
    const auto &L = levels_[level];

    // If no element blocks or coloring, fall back to regular multiplicative
    if (L.element_free_dofs.empty() || L.num_colors == 0) {
        smooth_jacobi(level, x, b, iters);
        return;
    }

    for (int iter = 0; iter < iters; ++iter) {
        if (profile_) {
            profile_->schwarz_iterations++;
        }

        // Process each color sequentially (Gauss-Seidel between colors)
        for (int color = 0; color < L.num_colors; ++color) {
            const auto &elements = L.elements_by_color[color];
            if (elements.empty())
                continue;

            // Compute Qx once per color (updated after previous color)
            VecX Qx;
            {
                OptionalScopedTimer t(profile_ ? &profile_->schwarz_matvec_ms
                                               : nullptr);
                if (profile_) {
                    profile_->matvec_products++;
                }
                Qx = L.Q * x;
            }

            // Accumulate corrections for all elements of this color
            // (They don't share DOFs, so no conflicts)
            VecX dx_color = VecX::Zero(x.size());

            for (Index e : elements) {
                const auto &free_dofs = L.element_free_dofs[e];
                int block_size = static_cast<int>(free_dofs.size());
                if (block_size == 0)
                    continue;

                // Gather local residual
                VecX r_local(block_size);
                {
                    OptionalScopedTimer t(profile_ ? &profile_->schwarz_gather_ms
                                                   : nullptr);
                    for (int i = 0; i < block_size; ++i) {
                        r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
                    }
                }

                // Solve local system
                VecX dx_local;
                {
                    OptionalScopedTimer t(
                        profile_ ? &profile_->schwarz_local_solve_ms : nullptr);
                    if (profile_) {
                        profile_->schwarz_element_solves++;
                    }
                    dx_local = L.element_block_lu[e].solve(r_local);
                }

                // Accumulate (no Qx update needed - same-colored elements
                // don't share DOFs)
                {
                    OptionalScopedTimer t(
                        profile_ ? &profile_->schwarz_scatter_update_ms
                                 : nullptr);
                    for (int i = 0; i < block_size; ++i) {
                        dx_color(free_dofs[i]) += dx_local(i);
                    }
                }
            }

            // Apply all corrections from this color (Gauss-Seidel step)
            x += dx_color;
        }
    }
}

} // namespace drifter
