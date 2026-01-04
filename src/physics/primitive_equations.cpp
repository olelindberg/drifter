#include "physics/primitive_equations.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>

namespace drifter {

// =============================================================================
// PrimitiveState implementation
// =============================================================================

void PrimitiveState::update_derived(const VecX& h) {
    int n = static_cast<int>(Hu.size());

    H.resize(n);
    u.resize(n);
    v.resize(n);
    T.resize(n);
    S.resize(n);

    for (int i = 0; i < n; ++i) {
        H(i) = eta(i) + h(i);
        Real H_inv = (H(i) > 1e-10) ? 1.0 / H(i) : 0.0;
        u(i) = Hu(i) * H_inv;
        v(i) = Hv(i) * H_inv;
        T(i) = HT(i) * H_inv;
        S(i) = HS(i) * H_inv;
    }
}

void PrimitiveState::resize(int n) {
    Hu.resize(n); Hu.setZero();
    Hv.resize(n); Hv.setZero();
    eta.resize(n); eta.setZero();
    HT.resize(n); HT.setZero();
    HS.resize(n); HS.setZero();
    H.resize(n);
    u.resize(n);
    v.resize(n);
    T.resize(n);
    S.resize(n);
    omega.resize(n); omega.setZero();
    rho.resize(n); rho.setZero();
}

// =============================================================================
// PrimitiveTendencies implementation
// =============================================================================

void PrimitiveTendencies::resize(int n) {
    dHu_dt.resize(n);
    dHv_dt.resize(n);
    deta_dt.resize(n);
    dHT_dt.resize(n);
    dHS_dt.resize(n);
}

void PrimitiveTendencies::set_zero() {
    dHu_dt.setZero();
    dHv_dt.setZero();
    deta_dt.setZero();
    dHT_dt.setZero();
    dHS_dt.setZero();
}

// =============================================================================
// PrimitiveEquationsElement implementation
// =============================================================================

PrimitiveEquationsElement::PrimitiveEquationsElement(
    const HexahedronBasis& basis,
    const GaussQuadrature3D& quad,
    const OceanConstants& constants)
    : basis_(basis)
    , quad_(quad)
    , constants_(constants)
    , elem_op_(basis, quad)
    , sigma_op_(basis, quad)
    , n_horiz_((basis.order() + 1) * (basis.order() + 1))
    , n_vert_(basis.order() + 1)
{
    build_2d_operators();
}

void PrimitiveEquationsElement::build_2d_operators() {
    // Build 2D differentiation matrices for horizontal operations
    // These operate on the 2D horizontal slice of DOFs (for eta equation)

    int order = basis_.order();
    int n1d = order + 1;

    // Get 1D differentiation matrix
    const MatX& D_1d = basis_.D_xi_lgl().topLeftCorner(n1d, n1d);

    // 2D operators via tensor product
    D_x_2d_.resize(n_horiz_, n_horiz_);
    D_y_2d_.resize(n_horiz_, n_horiz_);
    D_x_2d_.setZero();
    D_y_2d_.setZero();

    // D_x = D_1d ⊗ I
    // D_y = I ⊗ D_1d
    for (int j = 0; j < n1d; ++j) {
        for (int i = 0; i < n1d; ++i) {
            int row = i + j * n1d;
            for (int ii = 0; ii < n1d; ++ii) {
                int col_x = ii + j * n1d;  // x-derivative: vary i
                int col_y = i + ii * n1d;  // y-derivative: vary j
                D_x_2d_(row, col_x) = D_1d(i, ii);
                D_y_2d_(row, col_y) = D_1d(j, ii);
            }
        }
    }

    // Set up vertical integration weights (LGL weights scaled for sigma [-1, 0])
    const VecX& lgl_weights = basis_.lgl_basis_1d().weights;
    sigma_weights_.resize(n_vert_);
    for (int k = 0; k < n_vert_; ++k) {
        sigma_weights_(k) = 0.5 * lgl_weights(k);  // Scale for [-1, 0]
    }
}

void PrimitiveEquationsElement::set_bathymetry(
    const VecX& h, const VecX& dh_dx, const VecX& dh_dy) {
    h_ = h;
    dh_dx_ = dh_dx;
    dh_dy_ = dh_dy;
}

void PrimitiveEquationsElement::set_coriolis(
    const CoriolisParameter& coriolis, const VecX& y_positions) {
    coriolis.compute(y_positions, f_);
}

void PrimitiveEquationsElement::compute_rhs(
    const PrimitiveState& state, PrimitiveTendencies& tendency) const {

    int ndof = basis_.num_dofs_velocity();

    tendency.resize(ndof);
    tendency.set_zero();

    // Momentum RHS
    compute_momentum_rhs(state, tendency.dHu_dt, tendency.dHv_dt);

    // Tracer RHS
    compute_tracer_rhs(state, tendency.dHT_dt, tendency.dHS_dt);

    // Free surface RHS
    compute_eta_rhs(state, tendency.deta_dt);
}

void PrimitiveEquationsElement::compute_momentum_rhs(
    const PrimitiveState& state, VecX& dHu_dt, VecX& dHv_dt) const {

    int ndof = basis_.num_dofs_velocity();
    dHu_dt = VecX::Zero(ndof);
    dHv_dt = VecX::Zero(ndof);

    // Advection
    VecX adv_Hu, adv_Hv;
    momentum_advection(state, adv_Hu, adv_Hv);
    dHu_dt -= adv_Hu;
    dHv_dt -= adv_Hv;

    // Coriolis
    VecX cor_Hu, cor_Hv;
    coriolis_terms(state, cor_Hu, cor_Hv);
    dHu_dt += cor_Hu;
    dHv_dt += cor_Hv;

    // Barotropic pressure gradient
    VecX pg_u, pg_v;
    barotropic_pressure_gradient(state, pg_u, pg_v);
    dHu_dt += pg_u;
    dHv_dt += pg_v;

    // Horizontal diffusion
    if (diffusivity_.nu_h > 0.0) {
        VecX diff_u_x, diff_u_y, diff_v_x, diff_v_y;
        horizontal_diffusion(state.u, diffusivity_.nu_h, diff_u_x, diff_u_y);
        horizontal_diffusion(state.v, diffusivity_.nu_h, diff_v_x, diff_v_y);

        for (int i = 0; i < ndof; ++i) {
            dHu_dt(i) += state.H(i) * (diff_u_x(i) + diff_u_y(i));
            dHv_dt(i) += state.H(i) * (diff_v_x(i) + diff_v_y(i));
        }
    }

    // Vertical diffusion
    if (diffusivity_.nu_v > 0.0) {
        VecX diff_u_z, diff_v_z;
        vertical_diffusion(state.u, diffusivity_.nu_v, state.H, diff_u_z);
        vertical_diffusion(state.v, diffusivity_.nu_v, state.H, diff_v_z);
        dHu_dt += diff_u_z;
        dHv_dt += diff_v_z;
    }
}

void PrimitiveEquationsElement::compute_tracer_rhs(
    const PrimitiveState& state, VecX& dHT_dt, VecX& dHS_dt) const {

    int ndof = basis_.num_dofs_tracer();
    dHT_dt = VecX::Zero(ndof);
    dHS_dt = VecX::Zero(ndof);

    // Advection
    VecX adv_HT, adv_HS;
    tracer_advection(state, state.HT, adv_HT);
    tracer_advection(state, state.HS, adv_HS);
    dHT_dt -= adv_HT;
    dHS_dt -= adv_HS;

    // Vertical diffusion for tracers
    if (diffusivity_.kappa_v_T > 0.0) {
        VecX diff_T_z;
        vertical_diffusion(state.T, diffusivity_.kappa_v_T, state.H, diff_T_z);
        dHT_dt += diff_T_z;
    }

    if (diffusivity_.kappa_v_S > 0.0) {
        VecX diff_S_z;
        vertical_diffusion(state.S, diffusivity_.kappa_v_S, state.H, diff_S_z);
        dHS_dt += diff_S_z;
    }
}

void PrimitiveEquationsElement::compute_eta_rhs(
    const PrimitiveState& state, VecX& deta_dt) const {

    // Free surface evolves according to:
    // deta/dt = -d/dx(integral Hu dsigma) - d/dy(integral Hv dsigma)
    //         = -div_h(HU_bar, HV_bar)
    //
    // where HU_bar = integral_{-1}^{0} Hu dsigma (depth-integrated transport)
    //
    // Note: eta is physically 2D (constant in vertical), but we store it on
    // the 3D grid for consistency. The 2D result is replicated to all vertical levels.

    // Step 1: Compute depth-integrated transports at each horizontal DOF
    VecX HU_bar(n_horiz_), HV_bar(n_horiz_);

    for (int i_h = 0; i_h < n_horiz_; ++i_h) {
        Real sum_Hu = 0.0;
        Real sum_Hv = 0.0;
        for (int k = 0; k < n_vert_; ++k) {
            int idx = i_h * n_vert_ + k;
            sum_Hu += sigma_weights_(k) * state.Hu(idx);
            sum_Hv += sigma_weights_(k) * state.Hv(idx);
        }
        HU_bar(i_h) = sum_Hu;
        HV_bar(i_h) = sum_Hv;
    }

    // Step 2: Compute horizontal divergence using 2D derivative matrices
    // deta/dt = -d(HU_bar)/dx - d(HV_bar)/dy
    VecX dHU_dx = D_x_2d_ * HU_bar;
    VecX dHV_dy = D_y_2d_ * HV_bar;

    VecX deta_dt_2d = -(dHU_dx + dHV_dy);

    // Step 3: Replicate 2D result to all vertical levels
    // deta_dt has size n_horiz_ * n_vert_, with vertical as fastest index
    int ndof = n_horiz_ * n_vert_;
    deta_dt.resize(ndof);
    for (int i_h = 0; i_h < n_horiz_; ++i_h) {
        for (int k = 0; k < n_vert_; ++k) {
            int idx = i_h * n_vert_ + k;
            deta_dt(idx) = deta_dt_2d(i_h);
        }
    }
}

void PrimitiveEquationsElement::momentum_advection(
    const PrimitiveState& state, VecX& adv_Hu, VecX& adv_Hv) const {

    // Advection: d(Hu*u)/dx + d(Hu*v)/dy + d(Hu*omega)/dsigma
    // Using divergence form for conservation

    int ndof = basis_.num_dofs_velocity();

    // Compute fluxes
    VecX F_Hu_x(ndof), F_Hu_y(ndof), F_Hu_z(ndof);
    VecX F_Hv_x(ndof), F_Hv_y(ndof), F_Hv_z(ndof);

    for (int i = 0; i < ndof; ++i) {
        // Hu fluxes
        F_Hu_x(i) = state.Hu(i) * state.u(i);
        F_Hu_y(i) = state.Hu(i) * state.v(i);
        F_Hu_z(i) = state.Hu(i) * state.omega(i);

        // Hv fluxes
        F_Hv_x(i) = state.Hv(i) * state.u(i);
        F_Hv_y(i) = state.Hv(i) * state.v(i);
        F_Hv_z(i) = state.Hv(i) * state.omega(i);
    }

    // Compute divergence in reference coordinates
    std::array<VecX, 3> flux_Hu = {F_Hu_x, F_Hu_y, F_Hu_z};
    std::array<VecX, 3> flux_Hv = {F_Hv_x, F_Hv_y, F_Hv_z};

    elem_op_.divergence_reference(flux_Hu, adv_Hu, true);
    elem_op_.divergence_reference(flux_Hv, adv_Hv, true);
}

void PrimitiveEquationsElement::coriolis_terms(
    const PrimitiveState& state, VecX& cor_Hu, VecX& cor_Hv) const {

    int ndof = basis_.num_dofs_velocity();

    cor_Hu.resize(ndof);
    cor_Hv.resize(ndof);

    // f * (Hv, -Hu)
    for (int i = 0; i < ndof; ++i) {
        cor_Hu(i) = f_(i) * state.Hv(i);
        cor_Hv(i) = -f_(i) * state.Hu(i);
    }
}

void PrimitiveEquationsElement::barotropic_pressure_gradient(
    const PrimitiveState& state, VecX& pg_u, VecX& pg_v) const {

    // -gH * grad(eta)
    MatX grad_eta;
    elem_op_.gradient_reference(state.eta, grad_eta, true);

    int ndof = basis_.num_dofs_velocity();
    pg_u.resize(ndof);
    pg_v.resize(ndof);

    for (int i = 0; i < ndof; ++i) {
        pg_u(i) = -constants_.g * state.H(i) * grad_eta(i, 0);
        pg_v(i) = -constants_.g * state.H(i) * grad_eta(i, 1);
    }
}

void PrimitiveEquationsElement::horizontal_diffusion(
    const VecX& field, Real nu, VecX& diff_x, VecX& diff_y) const {

    // d^2(field)/dx^2 and d^2(field)/dy^2
    MatX grad_field;
    elem_op_.gradient_reference(field, grad_field, true);

    VecX dfield_dx = grad_field.col(0);
    VecX dfield_dy = grad_field.col(1);

    MatX grad_dx, grad_dy;
    elem_op_.gradient_reference(dfield_dx, grad_dx, true);
    elem_op_.gradient_reference(dfield_dy, grad_dy, true);

    diff_x = nu * grad_dx.col(0);  // nu * d^2/dx^2
    diff_y = nu * grad_dy.col(1);  // nu * d^2/dy^2
}

void PrimitiveEquationsElement::vertical_diffusion(
    const VecX& field, Real kappa, const VecX& H, VecX& diff) const {

    // d/dsigma(kappa * dfield/dsigma) / H
    // = kappa * d^2(field)/dsigma^2 / H

    VecX dfield_dsigma = basis_.D_zeta_lgl() * field;
    VecX d2field_dsigma2 = basis_.D_zeta_lgl() * dfield_dsigma;

    int ndof = static_cast<int>(field.size());
    diff.resize(ndof);

    for (int i = 0; i < ndof; ++i) {
        Real H_inv = (H(i) > 1e-10) ? 1.0 / H(i) : 0.0;
        // Actually: d/dsigma(kappa/H * dfield/dsigma)
        // For constant kappa: kappa * d^2/dsigma^2 / H
        diff(i) = kappa * d2field_dsigma2(i) * H_inv;
    }
}

void PrimitiveEquationsElement::tracer_advection(
    const PrimitiveState& state, const VecX& HT, VecX& adv_HT) const {

    // div(u * HT)
    int ndof = basis_.num_dofs_velocity();

    VecX F_x(ndof), F_y(ndof), F_z(ndof);

    for (int i = 0; i < ndof; ++i) {
        F_x(i) = state.u(i) * HT(i);
        F_y(i) = state.v(i) * HT(i);
        F_z(i) = state.omega(i) * HT(i);
    }

    std::array<VecX, 3> flux = {F_x, F_y, F_z};
    elem_op_.divergence_reference(flux, adv_HT, true);
}

// =============================================================================
// PrimitiveEquationsSolver implementation
// =============================================================================

PrimitiveEquationsSolver::PrimitiveEquationsSolver(int order, const OceanConstants& constants)
    : order_(order)
    , constants_(constants)
    , basis_(order)
    , quad_(order, QuadratureType::GaussLegendre)
    , dg_(order, true)
{
}

void PrimitiveEquationsSolver::initialize(
    int num_elements,
    const std::vector<VecX>& bathymetry,
    const std::vector<VecX>& dh_dx,
    const std::vector<VecX>& dh_dy,
    const CoriolisParameter& coriolis,
    const std::vector<VecX>& y_positions) {

    elements_.resize(num_elements);

    for (int e = 0; e < num_elements; ++e) {
        elements_[e] = std::make_unique<PrimitiveEquationsElement>(
            basis_, quad_, constants_);

        elements_[e]->set_bathymetry(bathymetry[e], dh_dx[e], dh_dy[e]);
        elements_[e]->set_coriolis(coriolis, y_positions[e]);
    }
}

void PrimitiveEquationsSolver::set_boundary_conditions(
    const std::vector<BoundaryCondition>& bc) {
    boundary_conditions_ = bc;
}

void PrimitiveEquationsSolver::set_face_connections(
    const std::vector<std::vector<FaceConnection>>& conn) {
    face_connections_ = conn;
}

void PrimitiveEquationsSolver::compute_rhs(
    const std::vector<PrimitiveState>& states,
    std::vector<PrimitiveTendencies>& tendencies) const {

    size_t num_elements = states.size();
    tendencies.resize(num_elements);

    // Element-local contributions (parallelizable)
    #pragma omp parallel for
    for (size_t e = 0; e < num_elements; ++e) {
        elements_[e]->compute_rhs(states[e], tendencies[e]);
    }

    // TODO: Apply interface fluxes
    // TODO: Apply boundary conditions
}

void PrimitiveEquationsSolver::apply_interface_fluxes(
    const std::vector<PrimitiveState>& states,
    std::vector<PrimitiveTendencies>& tendencies) const {

    // DG interface flux computation using numerical fluxes
    // Uses mortar method for non-conforming interfaces, direct averaging for conforming

    // Define numerical flux function for primitive equations (Lax-Friedrichs)
    auto numerical_flux = [this](const VecX& U_L, const VecX& U_R, const Vec3& n) -> VecX {
        // State vector U = [Hu, Hv, eta, HT, HS]
        // Compute Lax-Friedrichs flux: F* = 0.5*(F_L + F_R) - 0.5*alpha*(U_R - U_L)

        // Get maximum wave speed for penalty term
        Real H_L = U_L.size() > 2 ? U_L(2) + 10.0 : 10.0;  // eta + h (assuming h ~ 10m)
        Real H_R = U_R.size() > 2 ? U_R(2) + 10.0 : 10.0;
        Real c_max = std::sqrt(constants_.g * std::max(H_L, H_R));

        // Physical flux in normal direction
        auto compute_flux = [&](const VecX& U) -> VecX {
            VecX F = VecX::Zero(U.size());
            if (U.size() < 3) return F;

            Real H = U(2) + 10.0;  // eta + h
            Real u = H > 1e-10 ? U(0) / H : 0.0;  // Hu/H
            Real v = H > 1e-10 ? U(1) / H : 0.0;  // Hv/H
            Real u_n = u * n(0) + v * n(1);

            // Mass flux
            F(2) = U(0) * n(0) + U(1) * n(1);  // Hu*nx + Hv*ny

            // Momentum flux with pressure
            F(0) = U(0) * u_n + 0.5 * constants_.g * H * H * n(0);
            F(1) = U(1) * u_n + 0.5 * constants_.g * H * H * n(1);

            // Tracer flux (if present)
            if (U.size() > 3) F(3) = U(3) * u_n;  // HT * u_n
            if (U.size() > 4) F(4) = U(4) * u_n;  // HS * u_n

            return F;
        };

        VecX F_L = compute_flux(U_L);
        VecX F_R = compute_flux(U_R);

        return 0.5 * (F_L + F_R) - 0.5 * c_max * (U_R - U_L);
    };

    for (size_t e = 0; e < states.size(); ++e) {
        if (e >= face_connections_.size()) continue;

        for (int f = 0; f < 6; ++f) {
            if (f >= static_cast<int>(face_connections_[e].size())) continue;

            const FaceConnection& conn = face_connections_[e][f];

            if (conn.is_boundary()) {
                // Handle in apply_boundary_conditions
                continue;
            }

            // Get face normal
            FaceQuadrature face_quad(f, order_ + 1);
            Vec3 normal = face_quad.normal();

            if (conn.is_conforming()) {
                // 1:1 same-level interface - use direct averaging
                if (conn.fine_elems.empty()) continue;

                Index neighbor_elem = conn.fine_elems[0];
                if (neighbor_elem >= states.size()) continue;

                // Extract face states (simplified - using mean values)
                const PrimitiveState& state_L = states[e];
                const PrimitiveState& state_N = states[neighbor_elem];

                // Build state vectors at interface
                VecX U_L(5), U_R(5);
                U_L << state_L.Hu.mean(), state_L.Hv.mean(), state_L.eta.mean(),
                       state_L.HT.mean(), state_L.HS.mean();
                U_R << state_N.Hu.mean(), state_N.Hv.mean(), state_N.eta.mean(),
                       state_N.HT.mean(), state_N.HS.mean();

                // Compute numerical flux
                VecX F_star = numerical_flux(U_L, U_R, normal);

                // The flux contribution would be added to tendencies via lift operator
                // Full implementation needs face-to-volume lifting

            } else {
                // Non-conforming interface - use mortar method
                const MortarSpace* mortar = dg_.mortar_manager() ?
                    dg_.mortar_manager()->get_mortar(e, f) : nullptr;

                if (!mortar) continue;

                // Extract face states for coarse element
                const PrimitiveState& state_coarse = states[e];
                VecX U_coarse_face(5);
                U_coarse_face << state_coarse.Hu.mean(), state_coarse.Hv.mean(),
                                 state_coarse.eta.mean(), state_coarse.HT.mean(),
                                 state_coarse.HS.mean();

                // Gather face states from fine elements
                std::vector<VecX> U_fine_faces;
                for (size_t fi = 0; fi < conn.fine_elems.size(); ++fi) {
                    Index fine_elem = conn.fine_elems[fi];
                    if (fine_elem >= states.size()) continue;

                    const PrimitiveState& state_fine = states[fine_elem];
                    VecX U_fine(5);
                    U_fine << state_fine.Hu.mean(), state_fine.Hv.mean(),
                              state_fine.eta.mean(), state_fine.HT.mean(),
                              state_fine.HS.mean();
                    U_fine_faces.push_back(U_fine);
                }

                // Compute mortar flux
                VecX rhs_coarse_face;
                std::vector<VecX> rhs_fine_faces;
                mortar->compute_mortar_flux(numerical_flux, U_coarse_face,
                                            U_fine_faces, normal,
                                            rhs_coarse_face, rhs_fine_faces);

                // Add flux contributions to tendencies
                // (Full implementation would lift face RHS to volume DOFs)
            }
        }
    }
}

void PrimitiveEquationsSolver::apply_boundary_conditions(
    const std::vector<PrimitiveState>& states,
    std::vector<PrimitiveTendencies>& tendencies,
    Real time) const {

    // Apply boundary fluxes based on BC type
    for (size_t e = 0; e < states.size(); ++e) {
        if (e >= face_connections_.size()) continue;

        for (int f = 0; f < 6; ++f) {
            if (f >= static_cast<int>(face_connections_[e].size())) continue;

            const FaceConnection& conn = face_connections_[e][f];

            if (!conn.is_boundary()) continue;

            // Find matching boundary condition
            const BoundaryCondition* bc = nullptr;
            for (const auto& b : boundary_conditions_) {
                if (b.boundary_id == conn.boundary_id) {
                    bc = &b;
                    break;
                }
            }
            if (!bc) continue;

            // Get face normal (outward pointing)
            FaceQuadrature face_quad(f, order_ + 1);
            Vec3 normal = face_quad.normal();
            const VecX& weights = face_quad.weights();
            int n_face_pts = face_quad.size();

            // Extract interior state at face quadrature points
            // For now, use a simple approach: apply penalty flux
            const PrimitiveState& state = states[e];

            switch (bc->type) {
                case BCType::NoSlip:
                case BCType::FreeSlip: {
                    // Wall boundary: no normal flow
                    // Ghost state has reflected normal velocity
                    // Penalty flux = (U_interior - U_ghost) * wave_speed
                    Real wave_speed = std::sqrt(constants_.g * state.H.mean());

                    // Apply penalty to momentum equations
                    for (int q = 0; q < n_face_pts; ++q) {
                        Real w = weights(q);

                        // Project velocity onto normal and tangent
                        Real u_n = state.u.mean() * normal(0) + state.v.mean() * normal(1);

                        // For wall BC: ghost has u_n_ghost = -u_n (no normal flow)
                        // Penalty contribution
                        Real penalty_coeff = wave_speed * w;

                        // Add to tendencies (weak enforcement)
                        // This is a simplified version - full implementation would
                        // interpolate to face points and lift back to element DOFs
                    }
                    break;
                }

                case BCType::Outflow: {
                    // Radiation/open boundary condition
                    // Allow waves to exit without reflection
                    // Ghost state extrapolated from interior

                    // For outflow: simply extrapolate interior to ghost
                    // Flux computed with same state on both sides -> zero penalty
                    // This allows smooth outflow
                    break;
                }

                case BCType::Inflow:
                case BCType::Dirichlet: {
                    // Specified inflow condition
                    // Ghost state from boundary condition function
                    if (bc->value_func) {
                        const auto& vol_nodes = face_quad.volume_nodes();
                        Real wave_speed = std::sqrt(constants_.g * state.H.mean());

                        for (int q = 0; q < n_face_pts; ++q) {
                            const Vec3& pt = vol_nodes[q];
                            VecX bc_val = bc->value_func(pt, time);

                            // Apply penalty towards BC value
                            // (Full implementation would do proper lifting)
                        }
                    }
                    break;
                }

                case BCType::Periodic: {
                    // Periodic BCs should be handled as interior interfaces
                    // with the periodic neighbor element
                    break;
                }

                case BCType::Neumann: {
                    // Specified flux - add directly to RHS
                    // (No ghost state needed, flux is prescribed)
                    break;
                }
            }
        }
    }
}

// =============================================================================
// WindStress implementation
// =============================================================================

void WindStress::compute(const VecX& x, const VecX& y, Real time,
                          VecX& tau_x, VecX& tau_y) const {
    int n = static_cast<int>(x.size());
    tau_x.resize(n);
    tau_y.resize(n);

    if (stress_func) {
        for (int i = 0; i < n; ++i) {
            Vec2 tau = stress_func(x(i), y(i), time);
            tau_x(i) = tau(0);
            tau_y(i) = tau(1);
        }
    } else {
        tau_x.setZero();
        tau_y.setZero();
    }
}

// =============================================================================
// BottomFriction implementation
// =============================================================================

void BottomFriction::compute(const VecX& u_bot, const VecX& v_bot, const VecX& H,
                              VecX& tau_x, VecX& tau_y) const {
    int n = static_cast<int>(u_bot.size());
    tau_x.resize(n);
    tau_y.resize(n);

    for (int i = 0; i < n; ++i) {
        Real speed = std::sqrt(u_bot(i) * u_bot(i) + v_bot(i) * v_bot(i));

        switch (type_) {
            case Type::Linear:
                tau_x(i) = -Cd_ * u_bot(i);
                tau_y(i) = -Cd_ * v_bot(i);
                break;

            case Type::Quadratic:
                tau_x(i) = -Cd_ * speed * u_bot(i);
                tau_y(i) = -Cd_ * speed * v_bot(i);
                break;

            case Type::ManningN:
                // tau = g * n^2 * |u| * u / h^(1/3)
                if (H(i) > 1e-6) {
                    Real coeff = Cd_ * Cd_ * 9.81 / std::pow(H(i), 1.0/3.0);
                    tau_x(i) = -coeff * speed * u_bot(i);
                    tau_y(i) = -coeff * speed * v_bot(i);
                } else {
                    tau_x(i) = 0.0;
                    tau_y(i) = 0.0;
                }
                break;
        }
    }
}

}  // namespace drifter
