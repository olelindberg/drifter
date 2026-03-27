#include "bathymetry/diagonal_approx_cg_schur_preconditioner.hpp"
#include <cmath>
#include <iostream>

namespace drifter {

DiagonalApproxCGSchurPreconditioner::DiagonalApproxCGSchurPreconditioner(
    const SpMat& Q,
    const SpMat& C,
    Real inner_tolerance,
    int inner_max_iterations)
    : n_c_(C.rows())
    , inner_tol_(inner_tolerance)
    , inner_max_iter_(inner_max_iterations)
{
    // Extract diagonal of Q
    VecX diag_Q = Q.diagonal();

    // Build sparse diagonal matrix D = diag(1/diag(Q))
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(diag_Q.size());
    for (Index i = 0; i < diag_Q.size(); ++i) {
        Real d = diag_Q(i);
        if (std::abs(d) > 1e-14) {
            triplets.emplace_back(i, i, 1.0 / d);
        }
    }
    SpMat D(Q.rows(), Q.cols());
    D.setFromTriplets(triplets.begin(), triplets.end());

    // M_S = C * D * C^T (sparse matrix multiplication)
    SpMat CD = C * D;
    M_S_ = CD * C.transpose();

    // Extract diagonal of M_S for inner CG preconditioning
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

VecX DiagonalApproxCGSchurPreconditioner::apply(const VecX& r) const {
    // Solve M_S * z = r using diagonal-preconditioned CG
    VecX z = VecX::Zero(n_c_);

    // Handle zero RHS
    Real r_norm = r.norm();
    if (r_norm < 1e-14) {
        return z;
    }

    VecX residual = r; // r - M_S * z, but z=0 initially
    VecX precond_r = diag_M_S_inv_.cwiseProduct(residual);
    VecX p = precond_r;
    Real rz = residual.dot(precond_r);

    int iterations = 0;
    for (int iter = 0; iter < inner_max_iter_; ++iter) {
        iterations = iter + 1;
        VecX Ap = M_S_ * p;
        Real pAp = p.dot(Ap);

        // Check for breakdown
        if (std::abs(pAp) < 1e-14) {
            break;
        }

        Real alpha = rz / pAp;

        z += alpha * p;
        residual -= alpha * Ap;

        // Check convergence
        if (residual.norm() < inner_tol_ * r_norm) {
            break;
        }

        VecX precond_r_new = diag_M_S_inv_.cwiseProduct(residual);
        Real rz_new = residual.dot(precond_r_new);

        // Check for breakdown
        if (std::abs(rz) < 1e-14) {
            break;
        }

        Real beta = rz_new / rz;

        p = precond_r_new + beta * p;
        rz = rz_new;
    }

    // Output final iteration summary
    Real relative_residual = residual.norm() / r_norm;
    std::cout << "[DiagApproxCG    ] iter=" << iterations
              << ", relative_residual=" << relative_residual << "\n";

    return z;
}

} // namespace drifter
