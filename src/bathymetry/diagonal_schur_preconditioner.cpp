#include "bathymetry/diagonal_schur_preconditioner.hpp"
#include <cmath>

namespace drifter {

DiagonalSchurPreconditioner::DiagonalSchurPreconditioner(
    std::function<VecX(const VecX&)> schur_matvec,
    Index num_constraints)
{
    diag_inv_.resize(num_constraints);
    VecX e = VecX::Zero(num_constraints);

    // Extract diagonal: d_i = e_i^T * S * e_i
    for (Index i = 0; i < num_constraints; ++i) {
        e(i) = 1.0;
        VecX Se = schur_matvec(e);
        Real diag_val = Se(i);

        // Safeguard against near-zero diagonal entries
        // Use large value (small inverse) rather than zero to avoid NaN
        constexpr Real min_diag = 1e-14;
        if (std::abs(diag_val) > min_diag) {
            diag_inv_(i) = 1.0 / diag_val;
        } else {
            diag_inv_(i) = 1.0 / min_diag;
        }

        e(i) = 0.0;
    }
}

VecX DiagonalSchurPreconditioner::apply(const VecX& r) const {
    return diag_inv_.cwiseProduct(r);
}

VecX DiagonalSchurPreconditioner::diagonal() const {
    VecX diag(diag_inv_.size());
    for (Index i = 0; i < diag_inv_.size(); ++i) {
        diag(i) = 1.0 / diag_inv_(i);
    }
    return diag;
}

} // namespace drifter
