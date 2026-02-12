// Mortar element implementation for non-conforming interfaces

#include "dg/mortar.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

// =============================================================================
// MortarSpace implementation
// =============================================================================

MortarSpace::MortarSpace(
    const FaceConnection &conn, const HexahedronBasis &basis_coarse,
    const HexahedronBasis &basis_fine, int face_id)
    : conn_type_(conn.type), face_id_(face_id),
      mortar_order_(std::max(basis_coarse.order(), basis_fine.order())),
      num_mortar_dofs_((mortar_order_ + 1) * (mortar_order_ + 1)),
      num_fine_(conn.num_fine_faces()),
      mortar_quad_(2 * mortar_order_ + 1, QuadratureType::GaussLegendre) {

    // Compute sub-face bounds for each fine element
    compute_subface_bounds();

    // Build projection and lift operators
    build_projection_operators(basis_coarse, basis_fine);
}

void MortarSpace::compute_subface_bounds() {
    subface_bounds_.resize(num_fine_);

    // Determine sub-face regions based on connection type
    // All coordinates are in the coarse face reference space [-1,1]^2

    switch (conn_type_) {
    case FaceConnectionType::SameLevel:
        // Single fine element covers entire face
        subface_bounds_[0] = {Vec2(-1.0, -1.0), Vec2(1.0, 1.0)};
        break;

    case FaceConnectionType::Fine2x1:
        // Split in first tangent direction (t1)
        subface_bounds_[0] = {Vec2(-1.0, -1.0), Vec2(0.0, 1.0)};
        subface_bounds_[1] = {Vec2(0.0, -1.0), Vec2(1.0, 1.0)};
        break;

    case FaceConnectionType::Fine1x2:
        // Split in second tangent direction (t2)
        subface_bounds_[0] = {Vec2(-1.0, -1.0), Vec2(1.0, 0.0)};
        subface_bounds_[1] = {Vec2(-1.0, 0.0), Vec2(1.0, 1.0)};
        break;

    case FaceConnectionType::Fine2x2:
        // 2x2 split
        subface_bounds_[0] = {Vec2(-1.0, -1.0), Vec2(0.0, 0.0)};
        subface_bounds_[1] = {Vec2(0.0, -1.0), Vec2(1.0, 0.0)};
        subface_bounds_[2] = {Vec2(-1.0, 0.0), Vec2(0.0, 1.0)};
        subface_bounds_[3] = {Vec2(0.0, 0.0), Vec2(1.0, 1.0)};
        break;

    case FaceConnectionType::Fine3_2plus1:
        // 2 in t1, 1 in t2 for bottom, full width for top
        subface_bounds_[0] = {Vec2(-1.0, -1.0), Vec2(0.0, 0.0)};
        subface_bounds_[1] = {Vec2(0.0, -1.0), Vec2(1.0, 0.0)};
        subface_bounds_[2] = {Vec2(-1.0, 0.0), Vec2(1.0, 1.0)};
        break;

    case FaceConnectionType::Fine3_1plus2:
        // 1 in t1 for left, 2 in t2 for right
        subface_bounds_[0] = {Vec2(-1.0, -1.0), Vec2(0.0, 0.0)};
        subface_bounds_[1] = {Vec2(-1.0, 0.0), Vec2(0.0, 1.0)};
        subface_bounds_[2] = {Vec2(0.0, -1.0), Vec2(1.0, 1.0)};
        break;

    case FaceConnectionType::Boundary:
    default:
        // No fine elements
        break;
    }
}

void MortarSpace::build_projection_operators(
    const HexahedronBasis &basis_coarse, const HexahedronBasis &basis_fine) {

    int nq = mortar_quad_.size();
    int np = mortar_order_ + 1;

    // Build 2D mortar basis functions on quadrature points
    // Using tensor product of 1D Lagrange polynomials on LGL nodes
    LagrangeBasis1D mortar_1d = LagrangeBasis1D::create_lgl(mortar_order_);

    // Mortar basis values at quadrature points: Phi(nq x num_mortar_dofs)
    MatX Phi_mortar(nq, num_mortar_dofs_);
    for (int q = 0; q < nq; ++q) {
        Vec2 xi_q = mortar_quad_.nodes()[q];
        VecX phi_t1 = mortar_1d.evaluate(xi_q(0));
        VecX phi_t2 = mortar_1d.evaluate(xi_q(1));

        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int dof = i + np * j;
                Phi_mortar(q, dof) = phi_t1(i) * phi_t2(j);
            }
        }
    }

    // Build mortar mass matrix: M_ij = ∫ φ_i φ_j dA
    mass_ = MatX::Zero(num_mortar_dofs_, num_mortar_dofs_);
    for (int q = 0; q < nq; ++q) {
        Real w = mortar_quad_.weights()(q);
        for (int i = 0; i < num_mortar_dofs_; ++i) {
            for (int j = 0; j < num_mortar_dofs_; ++j) {
                mass_(i, j) += w * Phi_mortar(q, i) * Phi_mortar(q, j);
            }
        }
    }

    // Invert mass matrix
    mass_inv_ = mass_.inverse();

    // =========================================================================
    // Coarse projection operator
    // =========================================================================

    // P_coarse: projects coarse face DOFs to mortar DOFs via L2 projection
    // M_mortar * P_coarse = B_coarse
    // where B_coarse_ij = ∫ φ_mortar_i * ψ_coarse_j dA

    int nf_coarse = basis_coarse.num_face_dofs_velocity();
    MatX B_coarse = MatX::Zero(num_mortar_dofs_, nf_coarse);

    // Evaluate coarse face basis at mortar quadrature points
    const LagrangeBasis1D &coarse_1d = basis_coarse.lgl_basis_1d();
    int np_coarse = basis_coarse.order() + 1;

    for (int q = 0; q < nq; ++q) {
        Vec2 xi_q = mortar_quad_.nodes()[q];
        Real w = mortar_quad_.weights()(q);

        VecX psi_t1 = coarse_1d.evaluate(xi_q(0));
        VecX psi_t2 = coarse_1d.evaluate(xi_q(1));

        for (int m = 0; m < num_mortar_dofs_; ++m) {
            for (int jc = 0; jc < np_coarse; ++jc) {
                for (int ic = 0; ic < np_coarse; ++ic) {
                    int dof_coarse = ic + np_coarse * jc;
                    B_coarse(m, dof_coarse) +=
                        w * Phi_mortar(q, m) * psi_t1(ic) * psi_t2(jc);
                }
            }
        }
    }

    P_coarse_ = mass_inv_ * B_coarse;

    // Lift operator for coarse: L = B^T (transpose of projection coupling)
    L_coarse_ = B_coarse.transpose();

    // =========================================================================
    // Fine projection operators
    // =========================================================================

    int nf_fine = basis_fine.num_face_dofs_velocity();
    int np_fine = basis_fine.order() + 1;
    const LagrangeBasis1D &fine_1d = basis_fine.lgl_basis_1d();

    P_fine_.resize(num_fine_);
    L_fine_.resize(num_fine_);

    for (int f = 0; f < num_fine_; ++f) {
        MatX B_fine = MatX::Zero(num_mortar_dofs_, nf_fine);

        // Get sub-face bounds for this fine element
        Vec2 sf_min = subface_bounds_[f].first;
        Vec2 sf_max = subface_bounds_[f].second;

        // Size of sub-face in mortar reference coords
        Real sf_area = (sf_max(0) - sf_min(0)) * (sf_max(1) - sf_min(1));
        Real scale = sf_area / 4.0; // Jacobian from [-1,1]^2 to sub-face

        // Create quadrature on sub-face (map mortar quad to sub-face)
        for (int q = 0; q < nq; ++q) {
            Vec2 xi_mortar = mortar_quad_.nodes()[q];
            Real w_mortar = mortar_quad_.weights()(q);

            // Check if this quad point is in the sub-face region
            if (xi_mortar(0) >= sf_min(0) && xi_mortar(0) <= sf_max(0) &&
                xi_mortar(1) >= sf_min(1) && xi_mortar(1) <= sf_max(1)) {

                // Map mortar coords to fine face reference coords [-1,1]^2
                Vec2 xi_fine;
                xi_fine(0) =
                    2.0 * (xi_mortar(0) - sf_min(0)) / (sf_max(0) - sf_min(0)) -
                    1.0;
                xi_fine(1) =
                    2.0 * (xi_mortar(1) - sf_min(1)) / (sf_max(1) - sf_min(1)) -
                    1.0;

                VecX psi_t1 = fine_1d.evaluate(xi_fine(0));
                VecX psi_t2 = fine_1d.evaluate(xi_fine(1));

                for (int m = 0; m < num_mortar_dofs_; ++m) {
                    for (int jf = 0; jf < np_fine; ++jf) {
                        for (int if_ = 0; if_ < np_fine; ++if_) {
                            int dof_fine = if_ + np_fine * jf;
                            B_fine(m, dof_fine) += w_mortar * Phi_mortar(q, m) *
                                                   psi_t1(if_) * psi_t2(jf);
                        }
                    }
                }
            }
        }

        P_fine_[f] = mass_inv_ * B_fine;
        L_fine_[f] = B_fine.transpose();
    }
}

void MortarSpace::compute_mortar_flux(
    const NumericalFluxFunc &flux_func, const VecX &U_coarse_face,
    const std::vector<VecX> &U_fine_faces, const Vec3 &normal,
    VecX &rhs_coarse_face, std::vector<VecX> &rhs_fine_faces) const {

    // Project coarse solution to mortar space
    VecX U_coarse_mortar = P_coarse_ * U_coarse_face;

    // Project fine solutions to mortar space
    VecX U_fine_mortar = VecX::Zero(num_mortar_dofs_);
    for (int f = 0; f < num_fine_; ++f) {
        U_fine_mortar += P_fine_[f] * U_fine_faces[f];
    }

    // Compute numerical flux at mortar quadrature points
    VecX F_star_mortar = VecX::Zero(num_mortar_dofs_);

    // For each mortar DOF (or quadrature point), evaluate flux
    // Here we use a simplified approach: evaluate at mortar nodes
    const LagrangeBasis1D mortar_1d =
        LagrangeBasis1D::create_lgl(mortar_order_);
    int np = mortar_order_ + 1;

    for (int j = 0; j < np; ++j) {
        for (int i = 0; i < np; ++i) {
            int dof = i + np * j;

            // Interpolate solutions at this mortar node
            // For nodal basis, the DOF value is the solution at that node
            VecX U_L(1), U_R(1); // Simplified: scalar case
            U_L(0) = U_coarse_mortar(dof);
            U_R(0) = U_fine_mortar(dof);

            // Compute numerical flux
            VecX F = flux_func(U_L, U_R, normal);
            F_star_mortar(dof) = F(0);
        }
    }

    // Lift flux back to element DOFs
    rhs_coarse_face = L_coarse_ * (mass_inv_ * F_star_mortar);

    rhs_fine_faces.resize(num_fine_);
    for (int f = 0; f < num_fine_; ++f) {
        // Fine elements see the opposite normal direction
        rhs_fine_faces[f] = -L_fine_[f] * (mass_inv_ * F_star_mortar);
    }
}

// =============================================================================
// MortarInterfaceManager implementation
// =============================================================================

MortarInterfaceManager::MortarInterfaceManager(int order) : order_(order) {
    basis_ = std::make_unique<HexahedronBasis>(order);
}

void MortarInterfaceManager::register_interface(const FaceConnection &conn) {
    if (conn.needs_mortar()) {
        connections_.push_back(conn);
    }
}

void MortarInterfaceManager::build_operators() {
    for (const auto &conn : connections_) {
        auto key = std::make_pair(conn.coarse_elem, conn.coarse_face_id);
        mortars_[key] = std::make_unique<MortarSpace>(
            conn, *basis_, *basis_, conn.coarse_face_id);
    }
}

const MortarSpace *
MortarInterfaceManager::get_mortar(Index coarse_elem, int face_id) const {
    auto it = mortars_.find({coarse_elem, face_id});
    if (it != mortars_.end()) {
        return it->second.get();
    }
    return nullptr;
}

bool MortarInterfaceManager::has_mortar(Index coarse_elem, int face_id) const {
    return mortars_.find({coarse_elem, face_id}) != mortars_.end();
}

// =============================================================================
// ConformingInterface implementation
// =============================================================================

ConformingInterface::ConformingInterface(
    const HexahedronBasis &basis, int face_id_left, int face_id_right)
    : needs_orientation_flip_(false) {

    // Get face interpolation matrices
    interp_left_ = basis.interp_to_face_lgl(face_id_left);
    interp_right_ = basis.interp_to_face_lgl(face_id_right);

    // Get face quadrature
    FaceQuadrature fquad(
        face_id_left, basis.order(), QuadratureType::GaussLobatto);
    num_quad_ = fquad.size();
    quad_weights_ = fquad.quad_2d().weights();

    // Build lift matrices (transpose of interpolation scaled by inverse mass)
    // For collocated nodes (LGL), this simplifies
    lift_left_ = interp_left_.transpose();
    lift_right_ = interp_right_.transpose();

    // Check if we need to flip orientation
    // Face orientations should be consistent across interface
    // For now, assume consistent orientation
}

void ConformingInterface::compute_flux(
    const NumericalFluxFunc &flux_func, const VecX &U_left_face,
    const VecX &U_right_face, const Vec3 &normal, VecX &rhs_left_face,
    VecX &rhs_right_face) const {

    rhs_left_face = VecX::Zero(U_left_face.size());
    rhs_right_face = VecX::Zero(U_right_face.size());

    // For collocated points (LGL), we can directly evaluate flux at nodes
    for (int q = 0; q < num_quad_; ++q) {
        // Get left and right states at this quadrature point
        // For collocated, the face DOF IS the value at the node
        VecX U_L(1), U_R(1); // Simplified: scalar case
        U_L(0) = U_left_face(q);
        U_R(0) = U_right_face(q);

        // Compute numerical flux
        VecX F = flux_func(U_L, U_R, normal);

        // Accumulate to RHS
        Real w = quad_weights_(q);
        rhs_left_face(q) += w * F(0);
        rhs_right_face(q) -= w * F(0); // Opposite sign for right element
    }
}

// =============================================================================
// Numerical flux implementations
// =============================================================================

namespace flux {

VecX lax_friedrichs(
    const VecX &U_L, const VecX &U_R, const Vec3 &n, Real max_speed,
    const std::function<VecX(const VecX &, const Vec3 &)> &physical_flux) {
    VecX F_L = physical_flux(U_L, n);
    VecX F_R = physical_flux(U_R, n);

    // F* = 0.5 * (F_L + F_R) - 0.5 * λ_max * (U_R - U_L)
    return 0.5 * (F_L + F_R) - 0.5 * max_speed * (U_R - U_L);
}

VecX central(
    const VecX &U_L, const VecX &U_R, const Vec3 &n,
    const std::function<VecX(const VecX &, const Vec3 &)> &physical_flux) {
    VecX F_L = physical_flux(U_L, n);
    VecX F_R = physical_flux(U_R, n);

    // F* = 0.5 * (F_L + F_R)
    return 0.5 * (F_L + F_R);
}

VecX upwind(
    const VecX &U_L, const VecX &U_R, const Vec3 &n, const Vec3 &velocity) {
    Real v_n = velocity.dot(n);

    // F* = v · n * U_upwind
    if (v_n >= 0) {
        return v_n * U_L;
    } else {
        return v_n * U_R;
    }
}

} // namespace flux

// =============================================================================
// Conservation verification
// =============================================================================

Real verify_mortar_conservation(
    const MortarSpace &mortar, const VecX &rhs_coarse,
    const std::vector<VecX> &rhs_fine, const VecX &quad_weights_coarse,
    const std::vector<VecX> &quad_weights_fine) {

    // Compute integral of flux on coarse side
    Real integral_coarse = 0.0;
    for (int i = 0; i < rhs_coarse.size(); ++i) {
        integral_coarse += quad_weights_coarse(i) * rhs_coarse(i);
    }

    // Compute integral of flux on fine side (summed)
    Real integral_fine = 0.0;
    for (int f = 0; f < static_cast<int>(rhs_fine.size()); ++f) {
        for (int i = 0; i < rhs_fine[f].size(); ++i) {
            integral_fine += quad_weights_fine[f](i) * rhs_fine[f](i);
        }
    }

    // Conservation requires these to sum to zero
    Real total = integral_coarse + integral_fine;

    // Compute normalization for relative error
    Real norm = std::abs(integral_coarse) + std::abs(integral_fine);
    if (norm < 1e-15)
        return 0.0;

    return std::abs(total) / norm;
}

} // namespace drifter
