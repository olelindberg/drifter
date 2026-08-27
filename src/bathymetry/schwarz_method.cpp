#include "bathymetry/schwarz_method.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Set to true to see parallelization diagnostics
static constexpr bool SCHWARZ_DEBUG = false;

namespace drifter {

// =============================================================================
// SchwarzMethodBase
// =============================================================================

SchwarzMethodBase::SchwarzMethodBase(const SpMat &Q, const std::vector<std::vector<Index>> &element_free_dofs, const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu) : Q_(Q), element_free_dofs_(element_free_dofs), element_block_lu_(element_block_lu) {
  if (element_free_dofs.size() != element_block_lu.size()) {
    throw std::invalid_argument("SchwarzMethodBase: element_free_dofs and element_block_lu must "
                                "have same size");
  }
}

// =============================================================================
// MultiplicativeSchwarzMethod
// =============================================================================

void MultiplicativeSchwarzMethod::apply(VecX &x, const VecX &b, int iters) const {
  size_t num_elements = element_free_dofs_.size();

  for (int iter = 0; iter < iters; ++iter) {
        // Compute full residual once per iteration
    VecX Qx = Q_ * x;

        // Forward sweep through elements
    for (size_t e = 0; e < num_elements; ++e) {
      const auto &free_dofs = element_free_dofs_[e];
      int block_size        = static_cast<int>(free_dofs.size());
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
        Index dof  = free_dofs[i];
        x(dof)    += dx_local(i);
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

AdditiveSchwarzMethod::AdditiveSchwarzMethod(const SpMat &Q, const std::vector<std::vector<Index>> &element_free_dofs, const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu, Real omega) : SchwarzMethodBase(Q, element_free_dofs, element_block_lu), omega_(omega) {}

void AdditiveSchwarzMethod::apply(VecX &x, const VecX &b, int iters) const {
  size_t num_elements = element_free_dofs_.size();

  // Find max block size to pre-allocate buffers
  int max_block_size = 0;
  for (const auto &free_dofs : element_free_dofs_) {
    max_block_size = std::max(max_block_size, static_cast<int>(free_dofs.size()));
  }

  // Minimum elements to benefit from parallelization
  constexpr size_t MIN_PARALLEL_ELEMENTS = 16;

  for (int iter = 0; iter < iters; ++iter) {
    // Compute full residual once
    VecX Qx = Q_ * x;

#ifdef _OPENMP
    if (num_elements >= MIN_PARALLEL_ELEMENTS) {
      // Per-thread correction accumulation (elements may share DOFs)
      int num_threads = 1;
#pragma omp parallel
      {
#pragma omp single
        num_threads = omp_get_num_threads();
      }
      std::vector<VecX> thread_dx(num_threads, VecX::Zero(x.size()));

#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        // Pre-allocate thread-local buffers
        VecX r_local(max_block_size);
        VecX dx_local(max_block_size);

#pragma omp for schedule(static)
        for (size_t e = 0; e < num_elements; ++e) {
          const auto &free_dofs = element_free_dofs_[e];
          int block_size        = static_cast<int>(free_dofs.size());
          if (block_size == 0)
            continue;

          // Gather local residual
          for (int i = 0; i < block_size; ++i) {
            r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
          }

          // Solve local system
          dx_local.head(block_size) =
              element_block_lu_[e].solve(r_local.head(block_size));

          // Accumulate to thread-local vector
          for (int i = 0; i < block_size; ++i) {
            thread_dx[tid](free_dofs[i]) += dx_local(i);
          }
        }
      }

      // Reduce thread-local corrections and apply with damping
      VecX dx_total = VecX::Zero(x.size());
      for (const auto &dx : thread_dx) {
        dx_total += dx;
      }
      x += omega_ * dx_total;
    } else
#endif
    {
      // Sequential fallback for small element counts
      VecX dx_total = VecX::Zero(x.size());
      VecX r_local(max_block_size);
      VecX dx_local(max_block_size);

      for (size_t e = 0; e < num_elements; ++e) {
        const auto &free_dofs = element_free_dofs_[e];
        int block_size        = static_cast<int>(free_dofs.size());
        if (block_size == 0)
          continue;

        // Gather local residual
        for (int i = 0; i < block_size; ++i) {
          r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
        }

        // Solve local system
        dx_local.head(block_size) =
            element_block_lu_[e].solve(r_local.head(block_size));

        // Accumulate corrections
        for (int i = 0; i < block_size; ++i) {
          dx_total(free_dofs[i]) += dx_local(i);
        }
      }
      x += omega_ * dx_total;
    }
  }
}

// =============================================================================
// ColoredSchwarzMethod
// =============================================================================

ColoredSchwarzMethod::ColoredSchwarzMethod(const SpMat &Q, const std::vector<std::vector<Index>> &element_free_dofs, const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu, const std::vector<std::vector<Index>> &elements_by_color) : SchwarzMethodBase(Q, element_free_dofs, element_block_lu), elements_by_color_(elements_by_color) {}

void ColoredSchwarzMethod::apply(VecX &x, const VecX &b, int iters) const {
  // Find max block size to pre-allocate buffers
  int max_block_size = 0;
  for (const auto &free_dofs : element_free_dofs_) {
    max_block_size = std::max(max_block_size, static_cast<int>(free_dofs.size()));
  }

  // Minimum elements per color to benefit from parallelization
  constexpr size_t MIN_PARALLEL_ELEMENTS = 16;

  for (int iter = 0; iter < iters; ++iter) {
    // Process each color sequentially (Gauss-Seidel between colors)
    for (size_t color = 0; color < elements_by_color_.size(); ++color) {
      const auto &elements = elements_by_color_[color];
      if (elements.empty())
        continue;

      // Compute Qx once per color (updated after previous color)
      VecX Qx = Q_ * x;

      // Only parallelize if enough elements to overcome thread overhead
#ifdef _OPENMP
      if (elements.size() >= MIN_PARALLEL_ELEMENTS) {
        if (SCHWARZ_DEBUG && iter == 0 && color == 0) {
          std::cout << "[ColoredSchwarz] Parallel: " << elements.size()
                    << " elements, " << omp_get_max_threads() << " threads\n";
        }
#pragma omp parallel
        {
          // Pre-allocate thread-local buffers (avoid heap alloc in inner loop)
          VecX r_local(max_block_size);
          VecX dx_local(max_block_size);

#pragma omp for schedule(static)
          for (size_t idx = 0; idx < elements.size(); ++idx) {
            Index e               = elements[idx];
            const auto &free_dofs = element_free_dofs_[e];
            int block_size        = static_cast<int>(free_dofs.size());
            if (block_size == 0)
              continue;

            // Gather local residual
            for (int i = 0; i < block_size; ++i) {
              r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
            }

            // Solve local system
            dx_local.head(block_size) =
                element_block_lu_[e].solve(r_local.head(block_size));

            // Direct write (safe - same-color elements don't share DOFs)
            for (int i = 0; i < block_size; ++i) {
              x(free_dofs[i]) += dx_local(i);
            }
          }
        }
      } else
#endif
      {
        // Sequential fallback for small element counts
        if (SCHWARZ_DEBUG && iter == 0 && color == 0) {
          std::cout << "[ColoredSchwarz] Sequential: " << elements.size()
                    << " elements (< " << MIN_PARALLEL_ELEMENTS << ")\n";
        }
        VecX r_local(max_block_size);
        VecX dx_local(max_block_size);

        for (size_t idx = 0; idx < elements.size(); ++idx) {
          Index e               = elements[idx];
          const auto &free_dofs = element_free_dofs_[e];
          int block_size        = static_cast<int>(free_dofs.size());
          if (block_size == 0)
            continue;

          // Gather local residual
          for (int i = 0; i < block_size; ++i) {
            r_local(i) = b(free_dofs[i]) - Qx(free_dofs[i]);
          }

          // Solve local system
          dx_local.head(block_size) =
              element_block_lu_[e].solve(r_local.head(block_size));

          // Direct write
          for (int i = 0; i < block_size; ++i) {
            x(free_dofs[i]) += dx_local(i);
          }
        }
      }
    }
  }
}

} // namespace drifter
