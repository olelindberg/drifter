#include "bathymetry/schwarz_schur_preconditioner.hpp"

namespace drifter {

SchwarzSchurPreconditioner::SchwarzSchurPreconditioner(
    std::unique_ptr<IIterativeMethod> smoother, const SpMat &Q, const SpMat &C,
    int num_iterations)
    : smoother_(std::move(smoother)), Q_(Q), C_(C), Ct_(C.transpose()),
      num_iterations_(num_iterations) {}

VecX SchwarzSchurPreconditioner::apply(const VecX &r) const {
    // M_S^{-1} * r ≈ C * M_Q^{-1} * C^T * r
    //
    // Where M_Q^{-1} is approximated by applying Schwarz iterations to solve
    // Q * x = b (starting from x = 0).

    // 1. Map from constraint space to primal space: b = C^T * r
    VecX b = Ct_ * r;

    // 2. Approximate Q^{-1} * b using Schwarz iterations
    // Starting from x = 0, apply num_iterations_ of the Schwarz smoother
    VecX x = VecX::Zero(b.size());
    smoother_->apply(x, b, num_iterations_);

    // 3. Map back to constraint space: result = C * x
    return C_ * x;
}

} // namespace drifter
