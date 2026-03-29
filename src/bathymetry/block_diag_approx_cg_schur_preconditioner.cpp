#include "bathymetry/block_diag_approx_cg_schur_preconditioner.hpp"
#include <cmath>
#include <iostream>

namespace drifter {

BlockDiagApproxCGSchurPreconditioner::BlockDiagApproxCGSchurPreconditioner(const SpMat &Q, const SpMat &C, const CGCubicBezierDofManager &dof_manager, Real inner_tolerance, int inner_max_iterations, Real drop_tolerance) : n_c_(C.rows()), inner_tol_(inner_tolerance), inner_max_iter_(inner_max_iterations), drop_tolerance_(drop_tolerance) {
    // Step 1: Build element blocks with LU factorizations (non-overlapping DOF ownership)
  build_element_blocks(Q, dof_manager);

    // Step 2: Build sparse block-diagonal inverse matrix D
  SpMat D = build_block_diagonal_inverse(Q.rows());

    // Step 3: M_S = C * D * C^T (sparse matrix multiplication)
  C_T_     = C.transpose();  // Cache for future matrix-free mode
  SpMat CD = C * D;
  M_S_     = CD * C_T_;

    // Step 4: Extract diagonal of M_S for inner CG preconditioning
  VecX diag_M_S = M_S_.diagonal();
  diag_M_S_inv_.resize(n_c_);
  for (Index i = 0; i < n_c_; ++i) {
    Real d = diag_M_S(i);
    if (std::abs(d) > 1e-14) {
      diag_M_S_inv_(i) = 1.0 / d;
    } else {
      diag_M_S_inv_(i) = 1.0 / 1e-14;
    }
  }
}

void BlockDiagApproxCGSchurPreconditioner::build_element_blocks(const SpMat &Q, const CGCubicBezierDofManager &dof_manager) {
  const auto &mesh = dof_manager.mesh();
  Index n_elem     = mesh.num_elements();
  Index n_global   = dof_manager.num_global_dofs();

    // Step 1: Determine DOF ownership - first element to reference a DOF owns it
    // This creates a non-overlapping partition of DOFs across elements
  std::vector<Index> dof_owner(n_global, -1);

  for (Index e = 0; e < n_elem; ++e) {
    const auto &elem_dofs = dof_manager.element_dofs(e);
    for (Index global_dof : elem_dofs) {
      if (dof_owner[global_dof] < 0) {
        dof_owner[global_dof] = e;
      }
    }
  }

    // Step 2: Build blocks - each element gets only its owned free DOFs
  for (Index e = 0; e < n_elem; ++e) {
    const auto &elem_dofs = dof_manager.element_dofs(e);

        // Collect DOFs owned by this element that are also free (not constrained)
    std::vector<Index> owned_free_dofs;
    owned_free_dofs.reserve(elem_dofs.size());

    for (Index global_dof : elem_dofs) {
      if (dof_owner[global_dof] == e) {
        Index free_dof = dof_manager.global_to_free(global_dof);
        if (free_dof >= 0) {
          owned_free_dofs.push_back(free_dof);
        }
      }
    }

        // Skip if no owned free DOFs
    if (owned_free_dofs.empty()) {
      continue;
    }

        // Extract Q block for owned DOFs
    int block_size = static_cast<int>(owned_free_dofs.size());
    MatX Q_block(block_size, block_size);
    for (int i = 0; i < block_size; ++i) {
      for (int j = 0; j < block_size; ++j) {
        Q_block(i, j) = Q.coeff(owned_free_dofs[i], owned_free_dofs[j]);
      }
    }

        // LU factorize and store
    ElementBlockData block;
    block.element_id = e;
    block.free_dofs  = std::move(owned_free_dofs);
    block.block_lu   = Q_block.partialPivLu();
    element_blocks_.push_back(std::move(block));
  }
}

SpMat BlockDiagApproxCGSchurPreconditioner::build_block_diagonal_inverse(Index n_free) const {
  std::vector<Eigen::Triplet<Real>> triplets;

    // Estimate nnz: each block contributes block_size^2 entries
  size_t estimated_nnz = 0;
  for (const auto &block : element_blocks_) {
    estimated_nnz += block.free_dofs.size() * block.free_dofs.size();
  }
  triplets.reserve(estimated_nnz);

  for (const auto &block : element_blocks_) {
    int block_size = static_cast<int>(block.free_dofs.size());

        // Compute block inverse via LU solve: inv = LU.solve(I)
    MatX block_inv = block.block_lu.solve(MatX::Identity(block_size, block_size));

        // Add entries to sparse matrix (drop small values for sparsity)
    for (int i = 0; i < block_size; ++i) {
      for (int j = 0; j < block_size; ++j) {
        Real val = block_inv(i, j);
        if (std::abs(val) > drop_tolerance_) {
          triplets.emplace_back(block.free_dofs[i], block.free_dofs[j], val);
        }
      }
    }
  }

  SpMat D_inv(n_free, n_free);
  D_inv.setFromTriplets(triplets.begin(), triplets.end());
  return D_inv;
}

VecX BlockDiagApproxCGSchurPreconditioner::apply(const VecX &r) const {
    // Solve M_S * z = r using diagonal-preconditioned CG
    // (Same as DiagonalApproxCG - inner CG on assembled M_S)
  VecX z = VecX::Zero(n_c_);

    // Handle zero RHS
  Real r_norm = r.norm();
  if (r_norm < 1e-14) {
    return z;
  }

    // Adaptive inner tolerance: allow looser tolerance early, but ensure accuracy late
    // On first call, record initial outer residual norm
  if (initial_outer_norm_ < 0.0) {
    initial_outer_norm_ = r_norm;
  }
    // Scale tolerance: looser early (up to 100x inner_tol), strict late (inner_tol)
    // This ensures convergence isn't stalled by poor preconditioner quality
  Real adaptive_tol = inner_tol_ * std::max(1.0, 100.0 * std::sqrt(r_norm / initial_outer_norm_));

  VecX residual  = r; // r - M_S * z, but z=0 initially
  VecX precond_r = diag_M_S_inv_.cwiseProduct(residual);
  VecX p         = precond_r;
  Real rz        = residual.dot(precond_r);

  int iterations = 0;
  for (int iter = 0; iter < inner_max_iter_; ++iter) {
    iterations = iter + 1;
    VecX Ap    = M_S_ * p;
    Real pAp   = p.dot(Ap);

        // Check for breakdown
    if (std::abs(pAp) < 1e-14) {
      break;
    }

    Real alpha = rz / pAp;

    z        += alpha * p;
    residual -= alpha * Ap;

        // Check convergence (using adaptive tolerance)
    if (residual.norm() < adaptive_tol * r_norm) {
      break;
    }

    VecX precond_r_new = diag_M_S_inv_.cwiseProduct(residual);
    Real rz_new        = residual.dot(precond_r_new);

        // Check for breakdown
    if (std::abs(rz) < 1e-14) {
      break;
    }

    Real beta = rz_new / rz;

    p  = precond_r_new + beta * p;
    rz = rz_new;
  }

    // Output final iteration summary
  Real relative_residual = residual.norm() / r_norm;
  std::cout << "[BlockDiagApproxCG] iter=" << iterations << ", relative_residual=" << relative_residual << "\n";

  return z;
}

} // namespace drifter
