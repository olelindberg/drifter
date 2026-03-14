#include "bathymetry/physics_based_schur_preconditioner.hpp"
#include <Eigen/SparseLU>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace drifter {

PhysicsBasedSchurPreconditioner::PhysicsBasedSchurPreconditioner(
    const SpMat& K_reduced,
    const SpMat& C,
    Real regularization)
    : num_constraints_(C.rows())
{
    // Factor K for repeated solves
    Eigen::SparseLU<SpMat> K_solver;
    K_solver.compute(K_reduced);
    if (K_solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "PhysicsBasedSchurPreconditioner: K LU factorization failed");
    }

    // Assemble M_S = C * K^{-1} * C^T column by column
    // For each column i: solve K * v = C^T * e_i, then M_S[:, i] = C * v
    Index n_c = C.rows();

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(n_c * n_c); // Dense upper bound

    VecX e = VecX::Zero(n_c);

    for (Index i = 0; i < n_c; ++i) {
        e(i) = 1.0;

        // Compute C^T * e_i (column i of C^T = row i of C)
        VecX Ct_ei = C.transpose() * e;

        // Solve K * v = C^T * e_i
        VecX v = K_solver.solve(Ct_ei);

        // Compute M_S[:, i] = C * v
        VecX M_S_col = C * v;

        // Store non-zero entries
        for (Index j = 0; j < n_c; ++j) {
            if (std::abs(M_S_col(j)) > 1e-14) {
                triplets.emplace_back(j, i, M_S_col(j));
            }
        }

        e(i) = 0.0;
    }

    M_S_.resize(n_c, n_c);
    M_S_.setFromTriplets(triplets.begin(), triplets.end());

    // Add diagonal regularization to handle rank-deficient M_S
    // (when C has redundant constraints, e.g., overlapping boundary and edge constraints)
    // Use both:
    //   1. Relative regularization: regularization * trace(M_S) / n_c
    //   2. Absolute minimum: regularization * max_diag to ensure we handle near-zero traces
    Real reg_applied = 0.0;
    if (regularization > 0.0 && n_c > 0) {
        Real trace_MS = 0.0;
        Real max_diag = 0.0;
        for (Index i = 0; i < n_c; ++i) {
            Real diag_i = M_S_.coeff(i, i);
            trace_MS += diag_i;
            max_diag = std::max(max_diag, std::abs(diag_i));
        }
        // Use the larger of relative and absolute regularization
        Real rel_reg = (trace_MS > 0.0) ? (regularization * trace_MS / n_c) : 0.0;
        Real abs_reg = regularization * std::max(max_diag, 1.0);
        reg_applied = std::max(rel_reg, abs_reg);
        for (Index i = 0; i < n_c; ++i) {
            M_S_.coeffRef(i, i) += reg_applied;
        }
    }

    // Factor M_S with Cholesky (M_S is SPD since K is SPD and C has full row rank)
    cholesky_.compute(M_S_);
    if (cholesky_.info() != Eigen::Success) {
        // Compute diagnostics
        Real min_diag = std::numeric_limits<Real>::max();
        Real max_diag = 0.0;
        for (Index i = 0; i < n_c; ++i) {
            Real d = M_S_.coeff(i, i);
            min_diag = std::min(min_diag, d);
            max_diag = std::max(max_diag, d);
        }
        std::string msg = "PhysicsBasedSchurPreconditioner: M_S Cholesky factorization failed. "
                          "n_c=" + std::to_string(n_c) +
                          ", reg_applied=" + std::to_string(reg_applied) +
                          ", min_diag=" + std::to_string(min_diag) +
                          ", max_diag=" + std::to_string(max_diag);
        throw std::runtime_error(msg);
    }

    is_valid_ = true;
}

VecX PhysicsBasedSchurPreconditioner::apply(const VecX& r) const {
    return cholesky_.solve(r);
}

} // namespace drifter
