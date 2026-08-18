#include "bathymetry/block_diag_approx_cg_schur_preconditioner.hpp"
#include <Eigen/LU>
#include <cmath>
#include <iostream>

namespace drifter {

BlockDiagApproxCGSchurPreconditioner::BlockDiagApproxCGSchurPreconditioner(const SpMat &Q, const SpMat &C, const CGCubicBezierDofManager &dof_manager, Real inner_tolerance, int inner_max_iterations, Real /*drop_tolerance*/) : n_c_(C.rows()), n_free_(C.cols()), inner_tol_(inner_tolerance), inner_max_iter_(inner_max_iterations) {
  // Step 1: Build element blocks with explicit inverses (non-overlapping DOF ownership)
  build_element_blocks(Q, dof_manager);

  // Step 2: Store C and C^T for matrix-free M_S*p computation
  C_   = C;
  C_T_ = C.transpose();

  // Step 3: Compute diagonal of M_S for inner CG preconditioning (matrix-free)
  VecX diag_M_S = compute_M_S_diagonal();
  diag_M_S_inv_.resize(n_c_);
  for (Index i = 0; i < n_c_; ++i) {
    Real d = diag_M_S(i);
    if (std::abs(d) > 1e-14) {
      diag_M_S_inv_(i) = 1.0 / d;
    } else {
      diag_M_S_inv_(i) = 1.0 / 1e-14;
    }
  }

  // Allocate workspace vectors
  z_.resize(n_c_);
  precond_r_new_.resize(n_c_);
  Ap_.resize(n_c_);
  residual_.resize(n_c_);
  precond_r_.resize(n_c_);
  p_.resize(n_c_);
  temp1_.resize(n_free_);
  temp2_.resize(n_free_);
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

    // Compute explicit inverse via LU and store
    ElementBlockData block;
    block.element_id = e;
    block.free_dofs  = std::move(owned_free_dofs);
    Eigen::PartialPivLU<MatX> lu(Q_block);
    block.block_inv = lu.solve(MatX::Identity(block_size, block_size));
    block.local_vec.resize(block_size); // Pre-allocate work buffer
    element_blocks_.push_back(std::move(block));
  }
}

void BlockDiagApproxCGSchurPreconditioner::apply_block_diagonal(const VecX &v, VecX &result) const {
  result.setZero();

  for (const auto &block : element_blocks_) {
    const int block_size = static_cast<int>(block.free_dofs.size());

    // Gather into pre-allocated local buffer
    for (int i = 0; i < block_size; ++i) {
      block.local_vec(i) = v(block.free_dofs[i]);
    }

    // Dense matvec (no allocation): result_local = block_inv * local_vec
    // Use local_vec as both input and output via .eval()
    block.local_vec = block.block_inv * block.local_vec.eval();

    // Scatter to result (blocks are non-overlapping, so direct assignment)
    for (int i = 0; i < block_size; ++i) {
      result(block.free_dofs[i]) = block.local_vec(i);
    }
  }
}

void BlockDiagApproxCGSchurPreconditioner::apply_M_S(const VecX &p, VecX &result) const {
  // Matrix-free: M_S * p = C * D * C^T * p
  // where D = blockdiag(Q)^{-1}
  temp1_.noalias() = C_T_ * p;           // C^T * p: n_free vector
  apply_block_diagonal(temp1_, temp2_);  // D * (C^T * p): n_free vector
  result.noalias() = C_ * temp2_;        // C * D * C^T * p: n_c vector
}

VecX BlockDiagApproxCGSchurPreconditioner::compute_M_S_diagonal() const {
  // Compute diag(M_S) = diag(C * D * C^T) without forming full matrix
  // For each constraint i: diag(M_S)[i] = C[i,:] * D * C[i,:]^T
  VecX diag(n_c_);
  VecX c_row(n_free_);
  VecX d_row(n_free_);

  // Use row-major iteration for efficiency
  for (Index i = 0; i < n_c_; ++i) {
    // Extract row i of C into a dense vector
    c_row.setZero();
    for (SpMat::InnerIterator it(C_, i); it; ++it) {
      c_row(it.col()) = it.value();
    }

    // Apply D to the row: d_row = D * c_row
    apply_block_diagonal(c_row, d_row);

    // Diagonal entry is dot product: c_row^T * d_row
    diag(i) = c_row.dot(d_row);
  }

  return diag;
}

SpMat BlockDiagApproxCGSchurPreconditioner::assembled_matrix() const {
  // Assemble M_S = C * D * C^T on demand for testing
  // Build D as sparse matrix from block inverses
  std::vector<Eigen::Triplet<Real>> triplets;
  size_t estimated_nnz = 0;
  for (const auto &block : element_blocks_) {
    estimated_nnz += block.free_dofs.size() * block.free_dofs.size();
  }
  triplets.reserve(estimated_nnz);

  for (const auto &block : element_blocks_) {
    int block_size = static_cast<int>(block.free_dofs.size());

    for (int i = 0; i < block_size; ++i) {
      for (int j = 0; j < block_size; ++j) {
        Real val = block.block_inv(i, j);
        if (std::abs(val) > 1e-14) {
          triplets.emplace_back(block.free_dofs[i], block.free_dofs[j], val);
        }
      }
    }
  }

  SpMat D(n_free_, n_free_);
  D.setFromTriplets(triplets.begin(), triplets.end());

  // M_S = C * D * C^T
  SpMat CD = C_ * D;
  return CD * C_T_;
}

VecX BlockDiagApproxCGSchurPreconditioner::apply(const VecX &r) const {
  // Solve M_S * z = r using diagonal-preconditioned CG
  // Using allocation-free matrix-free M_S * p computation

  z_.fill(0.0);

  // Handle zero RHS
  Real r_norm = r.norm();
  if (r_norm < 1e-14) {
    return z_;
  }

  // Adaptive inner tolerance: allow looser tolerance early, but ensure accuracy late
  // On first call, record initial outer residual norm
  if (initial_outer_norm_ < 0.0) {
    initial_outer_norm_ = r_norm;
  }
  // Scale tolerance: looser early (up to 100x inner_tol), strict late (inner_tol)
  // This ensures convergence isn't stalled by poor preconditioner quality
  Real adaptive_tol = inner_tol_ * std::max(1.0, 100.0 * std::sqrt(r_norm / initial_outer_norm_));

  residual_  = r; // r - M_S * z, but z=0 initially
  precond_r_ = diag_M_S_inv_.cwiseProduct(residual_);
  p_         = precond_r_;
  Real rz    = residual_.dot(precond_r_);

  int iterations = 0;
  for (int iter = 0; iter < inner_max_iter_; ++iter) {
    iterations = iter + 1;
    apply_M_S(p_, Ap_); // Matrix-free: Ap = M_S * p
    Real pAp = p_.dot(Ap_);

    // Check for breakdown
    if (std::abs(pAp) < 1e-14) {
      break;
    }

    Real alpha = rz / pAp;

    z_        += alpha * p_;
    residual_ -= alpha * Ap_;

    // Check convergence (using adaptive tolerance)
    if (residual_.norm() < adaptive_tol * r_norm) {
      break;
    }

    precond_r_new_ = diag_M_S_inv_.cwiseProduct(residual_);
    Real rz_new    = residual_.dot(precond_r_new_);

    // Check for breakdown
    if (std::abs(rz) < 1e-14) {
      break;
    }

    Real beta = rz_new / rz;

    p_ = precond_r_new_ + beta * p_;
    rz = rz_new;
  }

  // Output final iteration summary
  Real relative_residual = residual_.norm() / r_norm;
  std::cout << "[BlockDiagApproxCG] iter=" << iterations << ", relative_residual=" << relative_residual << "\n";

  return z_;
}

} // namespace drifter
