#include "bathymetry/jacobi_method.hpp"
#include <stdexcept>

namespace drifter {

JacobiMethod::JacobiMethod(const SpMat &Q, Real omega) : Q_(Q), omega_(omega) {
    if (Q.rows() != Q.cols()) {
        throw std::invalid_argument(
            "JacobiMethod: matrix must be square");
    }

    // Pre-compute inverse diagonal
    Index n = Q.rows();
    diag_inv_.resize(n);
    for (Index i = 0; i < n; ++i) {
        Real diag_val = Q.coeff(i, i);
        if (std::abs(diag_val) < 1e-30) {
            throw std::invalid_argument(
                "JacobiMethod: zero diagonal element at index " +
                std::to_string(i));
        }
        diag_inv_(i) = 1.0 / diag_val;
    }
}

void JacobiMethod::apply(VecX &x, const VecX &b, int iters) const {
    for (int iter = 0; iter < iters; ++iter) {
        VecX r = b - Q_ * x;
        x += omega_ * (diag_inv_.asDiagonal() * r);
    }
}

} // namespace drifter
