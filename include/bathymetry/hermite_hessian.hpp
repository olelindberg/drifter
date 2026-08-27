#pragma once

/// @file hermite_hessian.hpp
/// @brief Smoothness energy Hessian for the Hermite element
///
/// Two energies, selected by continuity order:
///
///   r = 0, **membrane** (Dirichlet):   integral [ z_x^2 + z_y^2 ]
///   r >= 1, **thin plate**:            integral [ (z_xx + z_yy)^2 + 2 z_xy^2 ]
///
/// The membrane energy is used at r = 0 because the thin-plate energy degenerates
/// on Q_1: a bilinear function has z_xx = z_yy = 0, leaving only the twist term,
/// whose null space *grows with the mesh*. This mirrors the existing choice of
/// DirichletHessian for the linear Bezier smoother.
///
/// The r >= 1 form is the one implemented for the Bezier smoother, which is
/// (Laplacian)^2 plus an extra twist penalty rather than the standard bending
/// energy. Its null space is {1, x, y, x^2 - y^2} (dimension 4) rather than
/// {1, x, y} - inherited unchanged, since it is a property of the energy and not
/// of the basis. See docs/hermite_smoothness_operator.md S2.2.
///
/// Unlike CubicThinPlateHessian and DirichletHessian, this class uses **no
/// quadrature**. Both energies are sums of tensor-product terms and the Hermite
/// basis is a tensor product, so the element matrix factorises into exact 1D
/// energy matrices (hermite_energy_matrices.hpp) combined by Kronecker products.
/// That removes quadrature error entirely, and with it the risk of spurious
/// zero-energy modes from under-integration.

#include "bathymetry/hermite_energy_matrices.hpp"
#include "bathymetry/hessian_base.hpp"
#include "core/types.hpp"

namespace drifter {

/// @brief Hermite smoothness Hessian of continuity order r
///
/// scaled_hessian(dx, dy) already carries the physical-derivative DOF scaling, so
/// it is self-contained: CGSmootherBase::assemble_hessian_global() needs no
/// changes to assemble it.
class HermiteHessian : public HessianBase {
public:
    /// @brief Construct for continuity order r
    /// @param r Continuity order (0, 1 or 2)
    /// @throws std::invalid_argument if r is outside [0, 2]
    explicit HermiteHessian(int r);

    /// @brief Continuity order
    int r() const { return r_; }

    /// @brief Number of element DOFs, (2r+2)^2
    int num_dofs() const override { return n1d_ * n1d_; }

    /// @brief Element Hessian for a physical element of size dx by dy
    ///
    /// For r >= 1 (thin plate):
    ///
    ///   H_e = K00(h_y) kron K22(h_x) + K22(h_y) kron K00(h_x)
    ///       + [ K02(h_y) kron K20(h_x) + transpose ]
    ///       + 2 K11(h_y) kron K11(h_x)
    ///
    /// term by term: the z_xx, z_yy, symmetrised 2 z_xx z_yy, and 2 z_xy^2
    /// contributions. For r = 0 (membrane):
    ///
    ///   H_e = K00(h_y) kron K11(h_x) + K11(h_y) kron K00(h_x)
    ///
    /// The K matrices are the *physical* ones, so the h_y/h_x^3 and h_x/h_y^3
    /// anisotropic factors emerge rather than being asserted.
    MatX scaled_hessian(Real dx, Real dy) const override;

    /// @brief Element Hessian on the unit element
    const MatX &element_hessian() const override { return unit_hessian_; }

private:
    /// @brief Kronecker product with the u factor varying fastest: Y kron X
    MatX kron(const MatX &Y, const MatX &X) const;

    int r_;
    int n1d_; ///< 2(r+1)
    HermiteEnergyMatrices1D energy_;
    MatX unit_hessian_;
};

} // namespace drifter
