#include "bathymetry/flexible_cg.hpp"
#include <cmath>
#include <iostream>

namespace drifter {

FlexibleCG::FlexibleCG(
    std::function<VecX(const VecX&)> matvec,
    const ISchurPreconditioner& precond,
    Real tol,
    int max_iter)
    : matvec_(std::move(matvec))
    , precond_(precond)
    , tol_(tol)
    , max_iter_(max_iter)
{
}

FCGResult FlexibleCG::solve(VecX& x, const VecX& b) {
    FCGResult result;

    // r_0 = b - S * x_0
    VecX r = b - matvec_(x);

    // z_0 = M^{-1} * r_0
    VecX z = precond_.apply(r);

    // Initial preconditioned residual norm: ||r_0||_{M^{-1}} = sqrt(r^T z)
    Real rz_old = r.dot(z);

    // Handle zero RHS
    if (rz_old < 1e-30) {
        result.iterations = 0;
        result.initial_residual = 0.0;
        result.final_residual = 0.0;
        result.relative_residual = 0.0;
        result.converged = true;
        return result;
    }

    Real rz_init = rz_old;
    result.initial_residual = std::sqrt(rz_init);
    result.residual_history.push_back(result.initial_residual);

    // p_0 = z_0
    VecX p = z;

    Real tol_sq = tol_ * tol_;

    for (int iter = 0; iter < max_iter_; ++iter) {
        // q_k = S * p_k
        VecX q = matvec_(p);

        // alpha_k = (z_k, r_k) / (p_k, q_k)
        Real pq = p.dot(q);
        if (std::abs(pq) < 1e-30) {
            // Breakdown: p is S-conjugate to itself (shouldn't happen for SPD)
            result.iterations = iter;
            result.converged = false;
            break;
        }
        Real alpha = rz_old / pq;

        // x_{k+1} = x_k + alpha_k * p_k
        x += alpha * p;

        // r_{k+1} = r_k - alpha_k * q_k
        r -= alpha * q;

        // z_{k+1} = M^{-1} * r_{k+1}
        z = precond_.apply(r);

        // (z_{k+1}, r_{k+1})
        Real rz_new = r.dot(z);

        // Safeguard: detect indefinite preconditioner (r^T z < 0)
        // This can happen with variable preconditioners like multigrid if the
        // approximation becomes poor. Use absolute value to prevent NaN.
        if (rz_new < 0) {
            if (verbose_) {
                std::cerr << "[FCG] Warning: r^T z = " << rz_new << " < 0 at iter "
                          << (iter + 1) << " (indefinite preconditioner)\n";
            }
            rz_new = std::abs(rz_new);
        }

        // Preconditioned residual norm: ||r_{k+1}||_{M^{-1}} = sqrt(r^T z)
        Real precond_res_norm = std::sqrt(rz_new);
        Real unprecond_res_norm = r.norm();
        result.residual_history.push_back(precond_res_norm);

        if (verbose_) {
            std::cout << "[SchurFCG         ] iter=" << (iter + 1)
                      << ", relative_residual="
                      << (precond_res_norm / result.initial_residual) << "\n";
        }

        // Check convergence: ||r||_{M^{-1}} / ||r_0||_{M^{-1}} < tol
        if (rz_new / rz_init < tol_sq) {
            result.iterations = iter + 1;
            result.final_residual = precond_res_norm;
            result.relative_residual = precond_res_norm / result.initial_residual;
            result.converged = true;
            std::cout << "[SchurFCG         ] iter=" << result.iterations
                      << ", relative_residual=" << result.relative_residual << "\n";
            return result;
        }

        // beta_k = (z_{k+1}, r_{k+1}) / (z_k, r_k)
        Real beta = rz_new / rz_old;

        // p_{k+1} = z_{k+1} + beta_k * p_k
        p = z + beta * p;

        rz_old = rz_new;
    }

    // Max iterations reached
    result.iterations = max_iter_;
    result.final_residual = std::sqrt(rz_old);
    result.relative_residual = result.final_residual / result.initial_residual;
    result.converged = false;
    std::cout << "[SchurFCG         ] iter=" << result.iterations
              << ", relative_residual=" << result.relative_residual << "\n";

    return result;
}

} // namespace drifter
