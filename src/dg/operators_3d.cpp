// 3D DG integration operators implementation

#include "dg/operators_3d.hpp"
#include <omp.h>

namespace drifter {

// =============================================================================
// DG3DElementOperator implementation
// =============================================================================

DG3DElementOperator::DG3DElementOperator(
    const HexahedronBasis &basis, const GaussQuadrature3D &quad)
    : basis_(basis), quad_(quad), face_quads_{
                                      {FaceQuadrature(0, basis.order()),
                                       FaceQuadrature(1, basis.order()),
                                       FaceQuadrature(2, basis.order()),
                                       FaceQuadrature(3, basis.order()),
                                       FaceQuadrature(4, basis.order()),
                                       FaceQuadrature(5, basis.order())}} {

    int nq = quad.size();
    int ndof_lgl = basis.num_dofs_velocity();
    int ndof_gl = basis.num_dofs_tracer();

    // Precompute basis values at quadrature points
    phi_at_quad_lgl_.resize(nq, ndof_lgl);
    phi_at_quad_gl_.resize(nq, ndof_gl);

    for (int d = 0; d < 3; ++d) {
        dphi_at_quad_lgl_[d].resize(nq, ndof_lgl);
        dphi_at_quad_gl_[d].resize(nq, ndof_gl);
    }

    for (int q = 0; q < nq; ++q) {
        VecX phi_lgl = basis.evaluate_lgl(quad.nodes()[q]);
        VecX phi_gl = basis.evaluate_gl(quad.nodes()[q]);
        MatX grad_lgl = basis.evaluate_gradient_lgl(quad.nodes()[q]);
        MatX grad_gl = basis.evaluate_gradient_gl(quad.nodes()[q]);

        phi_at_quad_lgl_.row(q) = phi_lgl.transpose();
        phi_at_quad_gl_.row(q) = phi_gl.transpose();

        for (int d = 0; d < 3; ++d) {
            dphi_at_quad_lgl_[d].row(q) = grad_lgl.col(d).transpose();
            dphi_at_quad_gl_[d].row(q) = grad_gl.col(d).transpose();
        }
    }
}

void DG3DElementOperator::mass(const VecX &u, VecX &Mu, bool use_lgl) const {
    const MatX &M = use_lgl ? basis_.mass_lgl() : basis_.mass_gl();
    Mu = M * u;
}

void DG3DElementOperator::mass_inv(
    const VecX &u, VecX &Minv_u, bool use_lgl) const {
    const MatX &Minv = use_lgl ? basis_.mass_inv_lgl() : basis_.mass_inv_gl();
    Minv_u = Minv * u;
}

void DG3DElementOperator::gradient_reference(
    const VecX &u, MatX &grad_u, bool use_lgl) const {

    int ndof = use_lgl ? basis_.num_dofs_velocity() : basis_.num_dofs_tracer();
    const MatX &D_xi = use_lgl ? basis_.D_xi_lgl() : basis_.D_xi_gl();
    const MatX &D_eta = use_lgl ? basis_.D_eta_lgl() : basis_.D_eta_gl();
    const MatX &D_zeta = use_lgl ? basis_.D_zeta_lgl() : basis_.D_zeta_gl();

    grad_u.resize(ndof, 3);
    grad_u.col(0) = D_xi * u;
    grad_u.col(1) = D_eta * u;
    grad_u.col(2) = D_zeta * u;
}

void DG3DElementOperator::divergence_reference(
    const std::array<VecX, 3> &flux, VecX &div_flux, bool use_lgl) const {

    const MatX &D_xi = use_lgl ? basis_.D_xi_lgl() : basis_.D_xi_gl();
    const MatX &D_eta = use_lgl ? basis_.D_eta_lgl() : basis_.D_eta_gl();
    const MatX &D_zeta = use_lgl ? basis_.D_zeta_lgl() : basis_.D_zeta_gl();

    div_flux = D_xi * flux[0] + D_eta * flux[1] + D_zeta * flux[2];
}

void DG3DElementOperator::volume_integral(
    const VecX &U, const PhysicalFluxFunc &physical_flux,
    const std::vector<Mat3> &jacobian, const VecX &det_J, VecX &rhs,
    bool use_lgl) const {

    int ndof = use_lgl ? basis_.num_dofs_velocity() : basis_.num_dofs_tracer();
    int nq = quad_.size();

    rhs = VecX::Zero(ndof);

    const MatX &phi = use_lgl ? phi_at_quad_lgl_ : phi_at_quad_gl_;
    const std::array<MatX, 3> &dphi =
        use_lgl ? dphi_at_quad_lgl_ : dphi_at_quad_gl_;

    for (int q = 0; q < nq; ++q) {
        // Interpolate solution to quadrature point
        VecX U_q =
            phi.row(q)
                .transpose(); // This is just basis values, need to multiply
        Real u_val = phi.row(q) * U; // Scalar case

        VecX U_at_q(1);
        U_at_q(0) = u_val;

        // Compute physical flux at this point
        Tensor3 F = physical_flux(U_at_q);

        // Transform to reference coordinates using Jacobian inverse
        Mat3 Jinv = jacobian[q].inverse();
        Real w = quad_.weights()(q) * det_J(q);

        // Contribution: w * (dphi/dxi_ref . Jinv . F)
        // For each DOF i: rhs_i += w * (grad_phi_i)^T . Jinv^T . F
        for (int i = 0; i < ndof; ++i) {
            Vec3 grad_phi_i(dphi[0](q, i), dphi[1](q, i), dphi[2](q, i));
            Vec3 grad_phi_phys = Jinv.transpose() * grad_phi_i;

            // Contract with flux: grad_phi . F (sum over spatial dimensions)
            Real contrib = 0.0;
            for (int d = 0; d < 3; ++d) {
                contrib += grad_phi_phys(d) * F[d](0, 0); // Scalar flux
            }
            rhs(i) -= w * contrib; // Negative for RHS (integration by parts)
        }
    }
}

void DG3DElementOperator::face_integral(
    int face_id, const VecX &U_interior, const VecX &U_exterior,
    const Vec3 &normal, const NumericalFluxFunc &numerical_flux,
    const VecX &face_jacobian, VecX &rhs, bool use_lgl) const {

    const FaceQuadrature &fquad = face_quads_[face_id];
    const MatX &interp = use_lgl ? basis_.interp_to_face_lgl(face_id)
                                 : basis_.interp_to_face_gl(face_id);

    int ndof = use_lgl ? basis_.num_dofs_velocity() : basis_.num_dofs_tracer();
    int nfq = fquad.size();

    rhs = VecX::Zero(ndof);

    for (int q = 0; q < nfq; ++q) {
        // Get interior and exterior states at this face quadrature point
        VecX U_int_q(1), U_ext_q(1);
        U_int_q(0) = interp.row(q) * U_interior;
        U_ext_q(0) = interp.row(q) * U_exterior;

        // Compute numerical flux
        VecX F_star = numerical_flux(U_int_q, U_ext_q, normal);

        // Quadrature weight times surface Jacobian
        Real w = fquad.weights()(q) * face_jacobian(q);

        // Accumulate to RHS: rhs_i += w * phi_i * F_star
        for (int i = 0; i < ndof; ++i) {
            rhs(i) += w * interp(q, i) * F_star(0);
        }
    }
}

// =============================================================================
// DG3DIntegration implementation
// =============================================================================

DG3DIntegration::DG3DIntegration(int order, bool use_mortar)
    : order_(order), use_mortar_(use_mortar), basis_(order),
      quad_(order, QuadratureType::GaussLegendre), elem_op_(basis_, quad_) {

    if (use_mortar_) {
        mortar_manager_ = std::make_unique<MortarInterfaceManager>(order);
    }
}

void DG3DIntegration::gradient(
    const std::vector<VecX> &U,
    const std::vector<std::vector<FaceConnection>> &face_connections,
    const std::vector<BoundaryCondition> &bc, std::vector<VecX> &grad_U_x,
    std::vector<VecX> &grad_U_y, std::vector<VecX> &grad_U_z) const {

    size_t num_elems = U.size();
    int ndof = basis_.num_dofs_velocity();

    grad_U_x.resize(num_elems);
    grad_U_y.resize(num_elems);
    grad_U_z.resize(num_elems);

// Volume contribution (element-local, parallelizable)
#pragma omp parallel for
    for (size_t e = 0; e < num_elems; ++e) {
        MatX grad_ref;
        elem_op_.gradient_reference(U[e], grad_ref, true);

        grad_U_x[e] = grad_ref.col(0);
        grad_U_y[e] = grad_ref.col(1);
        grad_U_z[e] = grad_ref.col(2);

        // Apply inverse mass matrix
        elem_op_.mass_inv(grad_U_x[e], grad_U_x[e], true);
        elem_op_.mass_inv(grad_U_y[e], grad_U_y[e], true);
        elem_op_.mass_inv(grad_U_z[e], grad_U_z[e], true);
    }

    // Surface contribution (requires neighbor data, handle sequentially for
    // now)
    // TODO: Implement face flux contributions for gradient
}

void DG3DIntegration::divergence(
    const std::vector<VecX> &F_x, const std::vector<VecX> &F_y,
    const std::vector<VecX> &F_z,
    const std::vector<std::vector<FaceConnection>> &face_connections,
    const NumericalFluxFunc &numerical_flux,
    const std::vector<BoundaryCondition> &bc, std::vector<VecX> &div_F) const {

    size_t num_elems = F_x.size();

    div_F.resize(num_elems);

// Volume contribution
#pragma omp parallel for
    for (size_t e = 0; e < num_elems; ++e) {
        std::array<VecX, 3> flux = {F_x[e], F_y[e], F_z[e]};
        elem_op_.divergence_reference(flux, div_F[e], true);

        // Apply inverse mass matrix
        elem_op_.mass_inv(div_F[e], div_F[e], true);
    }

    // Surface contribution
    // TODO: Implement face flux contributions for divergence
}

void DG3DIntegration::compute_rhs(
    const std::vector<VecX> &U, const PhysicalFluxFunc &physical_flux,
    const NumericalFluxFunc &numerical_flux,
    const std::function<VecX(const VecX &, const Vec3 &)> &source_term,
    const std::vector<std::vector<FaceConnection>> &face_connections,
    const std::vector<BoundaryCondition> &bc, std::vector<VecX> &rhs) const {

    size_t num_elems = U.size();
    int ndof = basis_.num_dofs_velocity();

    rhs.resize(num_elems);

    // Initialize RHS
    for (size_t e = 0; e < num_elems; ++e) {
        rhs[e] = VecX::Zero(ndof);
    }

// Volume contribution (parallelizable over elements)
#pragma omp parallel for
    for (size_t e = 0; e < num_elems; ++e) {
        // For now, use identity Jacobian (reference element = physical element)
        std::vector<Mat3> jacobian(quad_.size(), Mat3::Identity());
        VecX det_J = VecX::Ones(quad_.size());

        VecX vol_contrib;
        elem_op_.volume_integral(
            U[e], physical_flux, jacobian, det_J, vol_contrib, true);
        rhs[e] += vol_contrib;
    }

    // Face contributions
    for (size_t e = 0; e < num_elems; ++e) {
        for (int f = 0; f < 6; ++f) {
            if (f < static_cast<int>(face_connections[e].size())) {
                const FaceConnection &conn = face_connections[e][f];

                if (conn.is_boundary()) {
                    // Boundary face
                    // TODO: Apply boundary conditions
                } else if (conn.is_conforming()) {
                    // Conforming interface
                    // TODO: Compute interface flux with neighbor
                } else if (use_mortar_) {
                    // Non-conforming interface with mortar
                    compute_interface_flux_nonconforming(
                        conn, U, numerical_flux, rhs);
                }
            }
        }
    }

// Apply inverse mass matrix
#pragma omp parallel for
    for (size_t e = 0; e < num_elems; ++e) {
        elem_op_.mass_inv(rhs[e], rhs[e], true);
    }
}

void DG3DIntegration::laplacian(
    const std::vector<VecX> &U, Real diffusivity,
    const std::vector<std::vector<FaceConnection>> &face_connections,
    const std::vector<BoundaryCondition> &bc, std::vector<VecX> &lap_U) const {

    // Compute gradient first
    std::vector<VecX> grad_x, grad_y, grad_z;
    gradient(U, face_connections, bc, grad_x, grad_y, grad_z);

    // Scale by diffusivity
    for (size_t e = 0; e < U.size(); ++e) {
        grad_x[e] *= diffusivity;
        grad_y[e] *= diffusivity;
        grad_z[e] *= diffusivity;
    }

    // Compute divergence of gradient
    auto central_flux = [](const VecX &U_L, const VecX &U_R, const Vec3 &n) {
        return 0.5 * (U_L + U_R);
    };

    divergence(
        grad_x, grad_y, grad_z, face_connections, central_flux, bc, lap_U);
}

void DG3DIntegration::register_nonconforming_interface(
    const FaceConnection &conn) {
    if (use_mortar_ && mortar_manager_) {
        mortar_manager_->register_interface(conn);
    }
}

void DG3DIntegration::build_mortar_operators() {
    if (use_mortar_ && mortar_manager_) {
        mortar_manager_->build_operators();
    }
}

void DG3DIntegration::compute_interface_flux_conforming(
    int elem_left, int face_left, int elem_right, int face_right,
    const VecX &U_left, const VecX &U_right, const Vec3 &normal,
    const NumericalFluxFunc &numerical_flux, VecX &rhs_left,
    VecX &rhs_right) const {

    ConformingInterface iface(basis_, face_left, face_right);

    // Extract face DOFs
    const MatX &interp_left = basis_.interp_to_face_lgl(face_left);
    const MatX &interp_right = basis_.interp_to_face_lgl(face_right);

    VecX U_left_face = interp_left * U_left;
    VecX U_right_face = interp_right * U_right;

    VecX rhs_left_face, rhs_right_face;
    iface.compute_flux(
        numerical_flux, U_left_face, U_right_face, normal, rhs_left_face,
        rhs_right_face);

    // Lift back to volume DOFs
    rhs_left = interp_left.transpose() * rhs_left_face;
    rhs_right = interp_right.transpose() * rhs_right_face;
}

void DG3DIntegration::compute_interface_flux_nonconforming(
    const FaceConnection &conn, const std::vector<VecX> &U,
    const NumericalFluxFunc &numerical_flux, std::vector<VecX> &rhs) const {

    if (!mortar_manager_)
        return;

    const MortarSpace *mortar =
        mortar_manager_->get_mortar(conn.coarse_elem, conn.coarse_face_id);

    if (!mortar)
        return;

    // Extract coarse face DOFs
    const MatX &interp_coarse = basis_.interp_to_face_lgl(conn.coarse_face_id);
    VecX U_coarse_face = interp_coarse * U[conn.coarse_elem];

    // Extract fine face DOFs
    std::vector<VecX> U_fine_faces(conn.fine_elems.size());
    for (size_t i = 0; i < conn.fine_elems.size(); ++i) {
        Index fine_elem = conn.fine_elems[i];
        int fine_face_id = conn.fine_face_ids[i];
        const MatX &interp_fine = basis_.interp_to_face_lgl(fine_face_id);
        U_fine_faces[i] = interp_fine * U[fine_elem];
    }

    // Compute normal (from coarse to fine)
    Vec3 normal = Vec3::Zero();
    int normal_axis = get_face_normal_axis(conn.coarse_face_id);
    normal(normal_axis) = is_positive_face(conn.coarse_face_id) ? 1.0 : -1.0;

    // Compute mortar flux
    VecX rhs_coarse_face;
    std::vector<VecX> rhs_fine_faces;
    mortar->compute_mortar_flux(
        numerical_flux, U_coarse_face, U_fine_faces, normal, rhs_coarse_face,
        rhs_fine_faces);

    // Lift back to volume DOFs
    rhs[conn.coarse_elem] += interp_coarse.transpose() * rhs_coarse_face;

    for (size_t i = 0; i < conn.fine_elems.size(); ++i) {
        Index fine_elem = conn.fine_elems[i];
        int fine_face_id = conn.fine_face_ids[i];
        const MatX &interp_fine = basis_.interp_to_face_lgl(fine_face_id);
        rhs[fine_elem] += interp_fine.transpose() * rhs_fine_faces[i];
    }
}

void DG3DIntegration::compute_boundary_flux(
    int elem, int face_id, const VecX &U_interior, const Vec3 &normal,
    const BoundaryCondition &bc, Real time, const Vec3 &face_center,
    const NumericalFluxFunc &numerical_flux, VecX &rhs) const {

    // Extract face DOFs
    const MatX &interp = basis_.interp_to_face_lgl(face_id);
    VecX U_face = interp * U_interior;

    // Compute exterior state based on BC type
    VecX U_exterior = U_face; // Default: mirror state

    switch (bc.type) {
    case BCType::Dirichlet:
        if (bc.value_func) {
            U_exterior = bc.value_func(face_center, time);
        }
        break;

    case BCType::Neumann:
        // For Neumann, use interior state (flux is specified separately)
        U_exterior = U_face;
        break;

    case BCType::Outflow:
        // Extrapolate interior state
        U_exterior = U_face;
        break;

    case BCType::NoSlip:
        // For velocity, negate to enforce u = 0
        U_exterior = -U_face;
        break;

    case BCType::FreeSlip:
        // Mirror normal component, keep tangential
        // For scalar, just use interior
        U_exterior = U_face;
        break;

    default:
        U_exterior = U_face;
        break;
    }

    // Compute numerical flux at face quadrature points
    FaceQuadrature fquad(face_id, order_, QuadratureType::GaussLobatto);
    int nfq = fquad.size();

    VecX rhs_face = VecX::Zero(U_face.size());
    for (int q = 0; q < nfq; ++q) {
        VecX U_int_q(1), U_ext_q(1);
        U_int_q(0) = U_face(q);
        U_ext_q(0) = U_exterior(q);

        VecX F_star = numerical_flux(U_int_q, U_ext_q, normal);

        Real w = fquad.weights()(q);
        rhs_face(q) += w * F_star(0);
    }

    // Lift to volume
    rhs += interp.transpose() * rhs_face;
}

// =============================================================================
// GeometricFactors implementation
// =============================================================================

GeometricFactors GeometricFactors::compute(
    const HexahedronBasis &basis, const GaussQuadrature3D &quad,
    const std::vector<Vec3> &physical_nodes) {

    GeometricFactors gf;
    int nq = quad.size();

    gf.jacobian.resize(nq);
    gf.jacobian_inv.resize(nq);
    gf.det_J.resize(nq);

    // Evaluate basis gradients at quadrature points and compute Jacobian
    for (int q = 0; q < nq; ++q) {
        MatX grad_phi = basis.evaluate_gradient_lgl(quad.nodes()[q]);

        // Jacobian: J_ij = dx_i / dxi_j = sum_k x_i^k * dphi_k/dxi_j
        Mat3 J = Mat3::Zero();
        for (int k = 0; k < static_cast<int>(physical_nodes.size()); ++k) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    J(i, j) += physical_nodes[k](i) * grad_phi(k, j);
                }
            }
        }

        gf.jacobian[q] = J;
        gf.det_J(q) = J.determinant();
        gf.jacobian_inv[q] = J.inverse();
    }

    // Compute face Jacobians
    for (int f = 0; f < 6; ++f) {
        FaceQuadrature fquad(f, basis.order());
        int nfq = fquad.size();
        gf.face_det_J[f].resize(nfq);

        for (int q = 0; q < nfq; ++q) {
            // Simplified: use average Jacobian at face
            // TODO: Proper face metric computation
            gf.face_det_J[f](q) = 1.0;
        }
    }

    return gf;
}

// =============================================================================
// SigmaCoordinateOperators implementation
// =============================================================================

SigmaCoordinateOperators::SigmaCoordinateOperators(
    const HexahedronBasis &basis, const GaussQuadrature3D &quad)
    : basis_(basis), quad_(quad) {}

void SigmaCoordinateOperators::horizontal_gradient(
    const VecX &U, const VecX &eta, const VecX &h, const VecX &deta_dx,
    const VecX &deta_dy, const VecX &dh_dx, const VecX &dh_dy, VecX &dU_dx,
    VecX &dU_dy) const {

    int ndof = basis_.num_dofs_velocity();

    // Compute reference gradients
    VecX dU_dxi = basis_.D_xi_lgl() * U;
    VecX dU_deta = basis_.D_eta_lgl() * U;
    VecX dU_dsigma = basis_.D_zeta_lgl() * U;

    // Apply sigma-coordinate transformation
    // du/dx|_z = du/dx|_sigma - (dsigma/dx) * du/dsigma
    // dsigma/dx = -sigma/H * (deta/dx + dh/dx) - 1/H * dh/dx  (for sigma in
    // [-1,0])

    dU_dx = dU_dxi;
    dU_dy = dU_deta;

    // Add metric correction terms
    for (int i = 0; i < ndof; ++i) {
        dU_dx(i) -= dsigma_dx_(i) * dU_dsigma(i);
        dU_dy(i) -= dsigma_dy_(i) * dU_dsigma(i);
    }
}

void SigmaCoordinateOperators::vertical_gradient(
    const VecX &U, const VecX &eta, const VecX &h, VecX &dU_dz) const {

    // dU/dz = (1/H) * dU/dsigma
    VecX dU_dsigma = basis_.D_zeta_lgl() * U;

    dU_dz.resize(dU_dsigma.size());
    for (int i = 0; i < dU_dsigma.size(); ++i) {
        dU_dz(i) = H_inv_(i) * dU_dsigma(i);
    }
}

void SigmaCoordinateOperators::sigma_divergence(
    const VecX &Hu, const VecX &Hv, const VecX &omega, const VecX &H,
    VecX &div) const {

    // Divergence in sigma coordinates:
    // div = d(Hu)/dx + d(Hv)/dy + d(omega)/dsigma

    VecX dHu_dx = basis_.D_xi_lgl() * Hu;
    VecX dHv_dy = basis_.D_eta_lgl() * Hv;
    VecX domega_dsigma = basis_.D_zeta_lgl() * omega;

    div = dHu_dx + dHv_dy + domega_dsigma;
}

void SigmaCoordinateOperators::update_sigma_metrics(
    const VecX &eta, const VecX &h, const VecX &deta_dx, const VecX &deta_dy,
    const VecX &dh_dx, const VecX &dh_dy) {

    // Call the time-dependent version with zero time derivative
    VecX zero_dt = VecX::Zero(eta.size());
    update_sigma_metrics_with_time(
        eta, h, deta_dx, deta_dy, dh_dx, dh_dy, zero_dt);
}

void SigmaCoordinateOperators::update_sigma_metrics_with_time(
    const VecX &eta, const VecX &h, const VecX &deta_dx, const VecX &deta_dy,
    const VecX &dh_dx, const VecX &dh_dy, const VecX &deta_dt) {

    int ndof = static_cast<int>(eta.size());
    H_.resize(ndof);
    H_inv_.resize(ndof);
    dsigma_dx_.resize(ndof);
    dsigma_dy_.resize(ndof);
    dsigma_dt_.resize(ndof);
    sigma_.resize(ndof);

    // Get sigma values at DOF points
    const auto &nodes = basis_.lgl_nodes();

    for (int i = 0; i < ndof; ++i) {
        H_(i) = eta(i) + h(i);
        H_inv_(i) = (H_(i) > 1e-10) ? 1.0 / H_(i) : 0.0;

        // Sigma value at this node (zeta coordinate in [-1,1] maps to sigma in
        // [-1,0])
        sigma_(i) = 0.5 * (nodes[i](2) - 1.0); // Map [-1,1] -> [-1,0]

        // dH/dx = deta/dx + dh/dx
        Real dH_dx = deta_dx(i) + dh_dx(i);
        Real dH_dy = deta_dy(i) + dh_dy(i);

        // dsigma/dx|_z = (sigma * dH/dx - deta/dx) / H
        dsigma_dx_(i) = (sigma_(i) * dH_dx - deta_dx(i)) * H_inv_(i);
        dsigma_dy_(i) = (sigma_(i) * dH_dy - deta_dy(i)) * H_inv_(i);

        // dsigma/dt|_z = -(1 + sigma) * deta/dt / H
        // For fixed bathymetry (dh/dt = 0): dH/dt = deta/dt
        dsigma_dt_(i) = -(1.0 + sigma_(i)) * deta_dt(i) * H_inv_(i);
    }
}

void SigmaCoordinateOperators::ale_correction(
    const VecX &U, const VecX &w_mesh, VecX &correction) const {

    // Compute dU/dz in sigma coordinates
    // dU/dz = H_inv * dU/dsigma
    VecX dU_dsigma = basis_.D_zeta_lgl() * U;

    int ndof = static_cast<int>(U.size());
    correction.resize(ndof);

    // ALE correction: -w_mesh * dU/dz
    for (int i = 0; i < ndof; ++i) {
        Real dU_dz = H_inv_(i) * dU_dsigma(i);
        correction(i) = -w_mesh(i) * dU_dz;
    }
}

void SigmaCoordinateOperators::material_derivative(
    const VecX &U, const VecX &dU_dt, const VecX &u, const VecX &v,
    const VecX &omega, const VecX &w_mesh, VecX &material_deriv) const {

    int ndof = static_cast<int>(U.size());

    // Compute gradients in sigma coordinates
    VecX dU_dxi = basis_.D_xi_lgl() * U;
    VecX dU_deta = basis_.D_eta_lgl() * U;
    VecX dU_dsigma = basis_.D_zeta_lgl() * U;

    material_deriv.resize(ndof);

    for (int i = 0; i < ndof; ++i) {
        // Horizontal gradients with sigma correction
        Real dU_dx = dU_dxi(i) - dsigma_dx_(i) * dU_dsigma(i);
        Real dU_dy = dU_deta(i) - dsigma_dy_(i) * dU_dsigma(i);

        // Vertical gradient in sigma space
        // The effective sigma velocity in moving frame is:
        // omega_eff = omega - w_mesh / H  (sigma velocity minus mesh motion
        // contribution)
        Real omega_eff = omega(i) - w_mesh(i) * H_inv_(i);

        // Full material derivative:
        // DU/Dt = dU/dt + u*dU/dx + v*dU/dy + omega_eff * dU/dsigma
        material_deriv(i) =
            dU_dt(i) + u(i) * dU_dx + v(i) * dU_dy + omega_eff * dU_dsigma(i);
    }
}

} // namespace drifter
