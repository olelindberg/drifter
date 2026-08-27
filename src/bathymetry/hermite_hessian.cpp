#include "bathymetry/hermite_hessian.hpp"

namespace drifter {

HermiteHessian::HermiteHessian(int r) : r_(r), n1d_(2 * (r + 1)), energy_(r) {
    unit_hessian_ = scaled_hessian(1.0, 1.0);
}

MatX HermiteHessian::kron(const MatX &Y, const MatX &X) const {
    MatX result(num_dofs(), num_dofs());
    for (int jr = 0; jr < n1d_; ++jr) {
        for (int jc = 0; jc < n1d_; ++jc) {
            result.block(jr * n1d_, jc * n1d_, n1d_, n1d_) = Y(jr, jc) * X;
        }
    }
    return result;
}

MatX HermiteHessian::scaled_hessian(Real dx, Real dy) const {
    const MatX K00x = energy_.physical(0, 0, dx);
    const MatX K00y = energy_.physical(0, 0, dy);
    const MatX K11x = energy_.physical(1, 1, dx);
    const MatX K11y = energy_.physical(1, 1, dy);

    MatX H;

    if (r_ == 0) {
        // Membrane: integral [ z_x^2 + z_y^2 ]
        H = kron(K00y, K11x) + kron(K11y, K00x);
    } else {
        // Thin plate: integral [ (z_xx + z_yy)^2 + 2 z_xy^2 ]
        const MatX K22x = energy_.physical(2, 2, dx);
        const MatX K22y = energy_.physical(2, 2, dy);
        const MatX K20x = energy_.physical(2, 0, dx);
        const MatX K02y = energy_.physical(0, 2, dy);

        // The cross term 2*int z_xx z_yy enters symmetrised, so the pair carries
        // coefficient 1 rather than 2: 2 x'Ax = x'(A + A')x for any x.
        const MatX cross = kron(K02y, K20x);

        H = kron(K00y, K22x)          // int z_xx w_xx
            + kron(K22y, K00x)        // int z_yy w_yy
            + cross + cross.transpose() // symmetrised 2 int z_xx w_yy
            + 2.0 * kron(K11y, K11x); // 2 int z_xy w_xy
    }

    // Remove round-off asymmetry, as the Bezier hessians do
    return 0.5 * (H + H.transpose());
}

} // namespace drifter
