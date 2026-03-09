#include "bathymetry/schwarz_method.hpp"
#include <stdexcept>

namespace drifter {

// =============================================================================
// SchwarzMethodBase
// =============================================================================

SchwarzMethodBase::SchwarzMethodBase(
    const SpMat &Q,
    const std::vector<std::vector<Index>> &element_free_dofs,
    const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu)
    : Q_(Q), element_free_dofs_(element_free_dofs),
      element_block_lu_(element_block_lu) {
    if (element_free_dofs.size() != element_block_lu.size()) {
        throw std::invalid_argument(
            "SchwarzMethodBase: element_free_dofs and element_block_lu must "
            "have same size");
    }
}

// =============================================================================
// MultiplicativeSchwarzMethod
// =============================================================================

void MultiplicativeSchwarzMethod::apply(VecX &x, const VecX &b,
                                         int iters) const {
    size_t num_elements = element_free_dofs_.size();

    for (int iter = 0; iter < iters; ++iter) {
        // Compute full residual once per iteration
        VecX Qx = Q_ * x;

        // Forward sweep through elements
        for (size_t e = 0; e < num_elements; ++e) {
            const auto &free_dofs = element_free_dofs_[e];
            int block_size = static_cast<int>(free_dofs.size());
            if (block_size == 0)
                continue;

            // Gather local residual
            VecX r_local(block_size);
            for (int i = 0; i < block_size; ++i) {
                r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
            }

            // Solve local system: Q_block * dx = r_local
            VecX dx_local = element_block_lu_[e].solve(r_local);

            // Update solution and Qx immediately (Gauss-Seidel style)
            for (int i = 0; i < block_size; ++i) {
                Index dof = free_dofs[i];
                x(dof) += dx_local(i);
                // Update Qx for subsequent elements (column iteration)
                for (SpMat::InnerIterator it(Q_, dof); it; ++it) {
                    Qx(it.index()) += it.value() * dx_local(i);
                }
            }
        }
    }
}

// =============================================================================
// AdditiveSchwarzMethod
// =============================================================================

AdditiveSchwarzMethod::AdditiveSchwarzMethod(
    const SpMat &Q,
    const std::vector<std::vector<Index>> &element_free_dofs,
    const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu,
    Real omega)
    : SchwarzMethodBase(Q, element_free_dofs, element_block_lu),
      omega_(omega) {}

void AdditiveSchwarzMethod::apply(VecX &x, const VecX &b, int iters) const {
    size_t num_elements = element_free_dofs_.size();

    for (int iter = 0; iter < iters; ++iter) {
        // Compute full residual once
        VecX Qx = Q_ * x;

        // Accumulate all corrections (additive: no immediate updates)
        VecX dx_total = VecX::Zero(x.size());

        // Process all elements (can be parallelized)
        for (size_t e = 0; e < num_elements; ++e) {
            const auto &free_dofs = element_free_dofs_[e];
            int block_size = static_cast<int>(free_dofs.size());
            if (block_size == 0)
                continue;

            // Gather local residual
            VecX r_local(block_size);
            for (int i = 0; i < block_size; ++i) {
                r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
            }

            // Solve local system
            VecX dx_local = element_block_lu_[e].solve(r_local);

            // Accumulate corrections
            for (int i = 0; i < block_size; ++i) {
                dx_total(free_dofs[i]) += dx_local(i);
            }
        }

        // Apply all corrections at once with damping
        x += omega_ * dx_total;
    }
}

// =============================================================================
// ColoredSchwarzMethod
// =============================================================================

ColoredSchwarzMethod::ColoredSchwarzMethod(
    const SpMat &Q,
    const std::vector<std::vector<Index>> &element_free_dofs,
    const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu,
    const std::vector<std::vector<Index>> &elements_by_color)
    : SchwarzMethodBase(Q, element_free_dofs, element_block_lu),
      elements_by_color_(elements_by_color) {}

void ColoredSchwarzMethod::apply(VecX &x, const VecX &b, int iters) const {
    for (int iter = 0; iter < iters; ++iter) {
        // Process each color sequentially (Gauss-Seidel between colors)
        for (size_t color = 0; color < elements_by_color_.size(); ++color) {
            const auto &elements = elements_by_color_[color];
            if (elements.empty())
                continue;

            // Compute Qx once per color (updated after previous color)
            VecX Qx = Q_ * x;

            // Accumulate corrections for all elements of this color
            // (They don't share DOFs, so no conflicts)
            VecX dx_color = VecX::Zero(x.size());

            for (Index e : elements) {
                const auto &free_dofs = element_free_dofs_[e];
                int block_size = static_cast<int>(free_dofs.size());
                if (block_size == 0)
                    continue;

                // Gather local residual
                VecX r_local(block_size);
                for (int i = 0; i < block_size; ++i) {
                    r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
                }

                // Solve local system
                VecX dx_local = element_block_lu_[e].solve(r_local);

                // Accumulate (same-colored elements don't share DOFs)
                for (int i = 0; i < block_size; ++i) {
                    dx_color(free_dofs[i]) += dx_local(i);
                }
            }

            // Apply all corrections from this color (Gauss-Seidel step)
            x += dx_color;
        }
    }
}

} // namespace drifter
