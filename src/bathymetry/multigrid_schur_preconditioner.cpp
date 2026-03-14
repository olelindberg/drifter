#include "bathymetry/multigrid_schur_preconditioner.hpp"

namespace drifter {

MultigridSchurPreconditioner::MultigridSchurPreconditioner(
    const BezierMultigridPreconditioner& mg_precond,
    const SpMat& Q,
    const SpMat& C,
    int num_vcycles)
    : mg_precond_(mg_precond)
    , Q_(Q)
    , C_(C)
    , Ct_(C.transpose())
    , num_vcycles_(num_vcycles)
{
}

VecX MultigridSchurPreconditioner::apply(const VecX& r) const {
    // M_S^{-1} * r ≈ C * Q^{-1} * C^T * r
    //
    // We approximate Q^{-1} * b using iterative refinement with V-cycles:
    //   x_0 = 0
    //   for i = 1..num_vcycles:
    //     residual = b - Q * x_{i-1}
    //     correction = mg.apply(residual)
    //     x_i = x_{i-1} + correction
    //
    // This is more accurate than a single V-cycle because each V-cycle
    // smooths the error, and the residual correction accumulates.

    // 1. Compute C^T * r (maps from constraint space to primal space)
    VecX b = Ct_ * r;

    // 2. Iterative refinement with MG V-cycles
    VecX x = VecX::Zero(b.size());
    for (int i = 0; i < num_vcycles_; ++i) {
        VecX residual = b - Q_ * x;
        VecX correction = mg_precond_.apply(residual);
        x += correction;
    }

    // 3. Compute C * result (maps back to constraint space)
    return C_ * x;
}

} // namespace drifter
