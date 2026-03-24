#include "bathymetry/gauss_seidel_schur_preconditioner.hpp"
#include <cmath>
#include <vector>

namespace drifter {

GaussSeidelSchurPreconditioner::GaussSeidelSchurPreconditioner(
    std::function<VecX(const VecX&)> schur_matvec,
    Index num_constraints,
    int num_iterations)
    : num_constraints_(num_constraints)
    , num_iterations_(num_iterations)
{
    // Assemble S column by column: S[:, i] = S * e_i
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_constraints * num_constraints);

    VecX e = VecX::Zero(num_constraints);
    diag_.resize(num_constraints);

    for (Index i = 0; i < num_constraints; ++i) {
        e(i) = 1.0;
        VecX Se = schur_matvec(e);

        // Store column i
        for (Index j = 0; j < num_constraints; ++j) {
            if (std::abs(Se(j)) > 1e-14) {
                triplets.emplace_back(j, i, Se(j));
            }
        }
        diag_(i) = Se(i);

        e(i) = 0.0;
    }

    S_.resize(num_constraints, num_constraints);
    S_.setFromTriplets(triplets.begin(), triplets.end());
    S_.makeCompressed();
}

VecX GaussSeidelSchurPreconditioner::apply(const VecX& r) const {
    // Symmetric Gauss-Seidel: forward sweep then backward sweep
    VecX x = VecX::Zero(num_constraints_);

    for (int iter = 0; iter < num_iterations_; ++iter) {
        // Forward sweep
        for (Index i = 0; i < num_constraints_; ++i) {
            Real sigma = r(i);
            for (SpMat::InnerIterator it(S_, i); it; ++it) {
                if (it.row() != i) {
                    sigma -= it.value() * x(it.row());
                }
            }
            x(i) = sigma / diag_(i);
        }

        // Backward sweep
        for (Index i = num_constraints_ - 1; i >= 0; --i) {
            Real sigma = r(i);
            for (SpMat::InnerIterator it(S_, i); it; ++it) {
                if (it.row() != i) {
                    sigma -= it.value() * x(it.row());
                }
            }
            x(i) = sigma / diag_(i);
        }
    }

    return x;
}

} // namespace drifter
