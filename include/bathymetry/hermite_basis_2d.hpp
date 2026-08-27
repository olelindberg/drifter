#pragma once

/// @file hermite_basis_2d.hpp
/// @brief 2D tensor-product Hermite basis for the Hermite bathymetry smoother
///
/// For continuity order r the element is the tensor-product Hermite rectangle of
/// degree p = 2r+1, with corner degrees of freedom given by the tensor-product
/// derivative set
///
///     { d^(a+b) z / dx^a dy^b : a, b in {0,...,r} }
///
/// i.e. (r+1)^2 DOFs per corner and 4(r+1)^2 = (p+1)^2 per element - exactly the
/// dimension of Q_p, so the element is unisolvent.
///
/// | r | p | element                     | DOFs/corner | DOFs/element |
/// |---|---|-----------------------------|-------------|--------------|
/// | 0 | 1 | bilinear                    | 1           | 4            |
/// | 1 | 3 | bicubic (Bogner-Fox-Schmit) | 4           | 16           |
/// | 2 | 5 | biquintic                   | 9           | 36           |
///
/// Identifying these DOFs between neighbouring elements makes C^r continuity
/// *structural* along every conforming edge - no collocation constraints, no KKT
/// system. See docs/hermite_bathymetry_system.md S3.
///
/// This class evaluates the **parametric** basis Nhat_I(u,v) = H_i(u) H_j(v). The
/// physical-derivative scaling h_x^a h_y^b is supplied separately by
/// dof_scaling(), because it depends on the element size; CGSmootherBase applies
/// it through its element_dof_scaling() hook.

#include "bathymetry/basis_2d_base.hpp"
#include "bathymetry/hermite_basis_1d.hpp"
#include "core/types.hpp"
#include <utility>
#include <vector>

namespace drifter {

/// @brief 2D tensor-product Hermite basis of continuity order r
///
/// Local DOF ordering follows docs/hermite_smoothness_operator.md S3.2 - the
/// *cubic* Bezier convention, with the u index varying fastest:
///
///     i = s_x*(r+1) + a,   j = s_y*(r+1) + b,   I = i + (p+1)*j
///
/// so for r = 1 the 1D DOFs run node-major as (z_0, z_0', z_1, z_1'). Note this
/// is the transpose of LinearBezierBasis2D's convention (j + 2i), which matters
/// only when comparing raw coefficient vectors against that class.
class HermiteBasis2D : public Basis2DBase {
public:
    /// @brief Construct the basis for continuity order r
    /// @param r Continuity order (0, 1 or 2)
    /// @throws std::invalid_argument if r is outside [0, 2]
    explicit HermiteBasis2D(int r);

    // =========================================================================
    // Basis2DBase interface
    // =========================================================================

    int degree() const override { return basis_1d_.degree(); }
    int num_nodes_1d() const override { return basis_1d_.num_dofs(); }
    int num_dofs() const override { return n1d_ * n1d_; }

    /// @brief Nodal position of a DOF in [0,1]^2
    /// @note Unlike a Bezier basis this is not injective: all (r+1)^2 DOFs of a
    ///       corner share one position. The DOF manager disambiguates them by
    ///       the derivative multi-index.
    Vec2 control_point_position(int dof) const override;

    VecX evaluate(Real u, Real v) const override;
    VecX evaluate_du(Real u, Real v) const override;
    VecX evaluate_dv(Real u, Real v) const override;
    MatX evaluate_gradient(Real u, Real v) const override;
    Real evaluate_scalar(const VecX &coeffs, Real u, Real v) const override;

    /// @brief Value DOF (a = b = 0) at a corner
    /// @param corner_id 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)
    int corner_dof(int corner_id) const override;

    /// @brief Corner a DOF belongs to
    /// @note Unlike a Bezier basis, *every* Hermite DOF sits at a corner, so this
    ///       never returns -1 for a valid index.
    int dof_to_corner(int dof) const override;

    /// @brief All DOFs at the two nodes on an edge
    /// @param edge_id 0: u=0 (left), 1: u=1 (right), 2: v=0 (bottom), 3: v=1 (top)
    /// @return 2(r+1)^2 DOF indices. This is every DOF *located* on the edge, which
    ///         is what boundary identification needs; it is a superset of the DOFs
    ///         that determine the value trace.
    std::vector<int> edge_dofs(int edge_id) const override;

    Vec2 corner_param(int corner_id) const override;

    // =========================================================================
    // Hermite-specific
    // =========================================================================

    /// @brief Continuity order
    int r() const { return r_; }

    /// @brief Derivative multi-index (a, b) of a DOF
    std::pair<int, int> deriv_order(int dof) const;

    /// @brief Total derivative order a + b of a DOF
    ///
    /// This is the exponent that governs the DOF's physical units and hence its
    /// row scaling in Q. See docs/hermite_bathymetry_system.md S11.
    int total_deriv_order(int dof) const;

    /// @brief Local DOF index from node/derivative indices
    /// @param sx, sy Node indices in {0, 1}
    /// @param a, b Derivative orders in {0, ..., r}
    int dof_index(int sx, int a, int sy, int b) const;

    /// @brief Diagonal DOF scaling Lambda_e = diag(h_x^a h_y^b)
    ///
    /// Converts between the parametric basis evaluated here and the basis dual to
    /// *physical* derivative DOFs. See docs/hermite_bathymetry_system.md S4.
    VecX dof_scaling(Real dx, Real dy) const;

    /// @brief 1D Bernstein <- Hermite change of basis M(h)
    ///
    /// Maps 1D Hermite DOFs (z_0, z_0', ..., z_1, z_1', ...) to the p+1 Bernstein
    /// control values of the same polynomial. Block-diagonal by node, which is
    /// docs/hermite_bathymetry_system.md S3 restated in Bernstein language.
    /// Invertible for every h > 0.
    MatX bernstein_change_of_basis_1d(Real h) const;

    /// @brief 2D Bernstein <- Hermite change of basis M_e = M(h_y) kron M(h_x)
    ///
    /// c_e = M_e q_e converts Hermite element DOFs to Bernstein control values in
    /// the i + (p+1)*j ordering (the CubicBezierBasis2D convention).
    MatX bernstein_change_of_basis(Real dx, Real dy) const;

    /// @brief Hanging-node midpoint matrix G_r^phys(h_t)
    ///
    /// Row i (slave derivative order) against column j (master 1D DOF):
    ///
    ///     [G_r^phys]_ij = h_t^(m_j - i) * H_j^(i)(1/2)
    ///
    /// Evaluates a coarse edge's Hermite interpolant and its first r derivatives at
    /// the edge midpoint, in physical-derivative DOFs. This is the constraint that
    /// makes C^r exact across a 2:1 T-junction; it can equivalently be derived by
    /// de Casteljau subdivision (see docs/hermite_bathymetry_system.md S8).
    ///
    /// @param h_t Physical length of the *coarse* edge
    /// @return (r+1) x (p+1) matrix
    MatX midpoint_matrix(Real h_t) const;

    /// @brief The underlying 1D basis
    const HermiteBasis1D &basis_1d() const { return basis_1d_; }

private:
    /// @brief 1D u-index of a 2D DOF
    int index_u(int dof) const { return dof % n1d_; }
    /// @brief 1D v-index of a 2D DOF
    int index_v(int dof) const { return dof / n1d_; }

    void check_dof(int dof) const;

    int r_;
    int n1d_; ///< p + 1 = 2(r+1)
    HermiteBasis1D basis_1d_;
};

} // namespace drifter
