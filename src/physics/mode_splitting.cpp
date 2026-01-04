#include "physics/mode_splitting.hpp"
#include <omp.h>
#include <cmath>
#include <algorithm>

namespace drifter {

// =============================================================================
// BarotropicState implementation
// =============================================================================

void BarotropicState::resize(int n) {
    eta.resize(n); eta.setZero();
    HU_bar.resize(n); HU_bar.setZero();
    HV_bar.resize(n); HV_bar.setZero();
    H.resize(n);
    U_bar.resize(n);
    V_bar.resize(n);
}

void BarotropicState::update_derived(const VecX& h) {
    int n = static_cast<int>(eta.size());
    H.resize(n);
    U_bar.resize(n);
    V_bar.resize(n);

    for (int i = 0; i < n; ++i) {
        H(i) = eta(i) + h(i);
        Real H_inv = (H(i) > 1e-10) ? 1.0 / H(i) : 0.0;
        U_bar(i) = HU_bar(i) * H_inv;
        V_bar(i) = HV_bar(i) * H_inv;
    }
}

// =============================================================================
// BaroclinicState implementation
// =============================================================================

void BaroclinicState::resize(int n) {
    u_prime.resize(n); u_prime.setZero();
    v_prime.resize(n); v_prime.setZero();
    HT.resize(n); HT.setZero();
    HS.resize(n); HS.setZero();
    omega.resize(n); omega.setZero();
}

// =============================================================================
// BarotropicTendencies implementation
// =============================================================================

void BarotropicTendencies::resize(int n) {
    deta_dt.resize(n);
    dHU_bar_dt.resize(n);
    dHV_bar_dt.resize(n);
}

void BarotropicTendencies::set_zero() {
    deta_dt.setZero();
    dHU_bar_dt.setZero();
    dHV_bar_dt.setZero();
}

// =============================================================================
// ModeSplittingElement implementation
// =============================================================================

ModeSplittingElement::ModeSplittingElement(
    const HexahedronBasis& basis,
    const GaussQuadrature3D& quad,
    const OceanConstants& constants)
    : basis_(basis)
    , quad_(quad)
    , constants_(constants)
    , n_horiz_((basis.order() + 1) * (basis.order() + 1))
    , n_vert_(basis.order() + 1)
{
    // Set up vertical integration weights
    const VecX& lgl_weights = basis.lgl_basis_1d().weights;
    sigma_weights_.resize(n_vert_);
    for (int k = 0; k < n_vert_; ++k) {
        sigma_weights_(k) = 0.5 * lgl_weights(k);  // Scale for [-1, 0]
    }

    build_2d_operators();
}

void ModeSplittingElement::build_2d_operators() {
    // Build 2D differentiation matrices for horizontal operations
    // These operate on the 2D horizontal slice of DOFs

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
}

void ModeSplittingElement::set_bathymetry(
    const VecX& h, const VecX& dh_dx, const VecX& dh_dy) {
    h_ = h;
    dh_dx_ = dh_dx;
    dh_dy_ = dh_dy;
}

void ModeSplittingElement::set_coriolis(const VecX& f) {
    f_ = f;
}

VecX ModeSplittingElement::compute_2d_divergence(const VecX& HU_bar, const VecX& HV_bar) const {
    // Compute divergence: div = d(HU_bar)/dx + d(HV_bar)/dy
    VecX dHU_dx = D_x_2d_ * HU_bar;
    VecX dHV_dy = D_y_2d_ * HV_bar;
    return dHU_dx + dHV_dy;
}

void ModeSplittingElement::depth_average(const VecX& u_3d, VecX& U_bar) const {
    U_bar.resize(n_horiz_);

    for (int i_h = 0; i_h < n_horiz_; ++i_h) {
        Real sum = 0.0;
        Real weight_sum = 0.0;
        for (int k = 0; k < n_vert_; ++k) {
            int idx = i_h * n_vert_ + k;
            sum += sigma_weights_(k) * u_3d(idx);
            weight_sum += sigma_weights_(k);
        }
        U_bar(i_h) = sum / weight_sum;
    }
}

void ModeSplittingElement::depth_integrate(const VecX& u_3d, const VecX& H, VecX& HU_bar) const {
    HU_bar.resize(n_horiz_);

    for (int i_h = 0; i_h < n_horiz_; ++i_h) {
        Real sum = 0.0;
        for (int k = 0; k < n_vert_; ++k) {
            int idx = i_h * n_vert_ + k;
            sum += sigma_weights_(k) * H(idx) * u_3d(idx);
        }
        HU_bar(i_h) = sum;
    }
}

void ModeSplittingElement::compute_deviation(const VecX& u_3d, const VecX& U_bar,
                                               VecX& u_prime) const {
    int n_total = n_horiz_ * n_vert_;
    u_prime.resize(n_total);

    for (int i_h = 0; i_h < n_horiz_; ++i_h) {
        for (int k = 0; k < n_vert_; ++k) {
            int idx = i_h * n_vert_ + k;
            u_prime(idx) = u_3d(idx) - U_bar(i_h);
        }
    }
}

void ModeSplittingElement::barotropic_eta_rhs(const BarotropicState& state,
                                                VecX& deta_dt) const {
    // deta/dt = -d(HU_bar)/dx - d(HV_bar)/dy

    VecX dHU_dx = D_x_2d_ * state.HU_bar;
    VecX dHV_dy = D_y_2d_ * state.HV_bar;

    deta_dt = -(dHU_dx + dHV_dy);
}

void ModeSplittingElement::barotropic_momentum_rhs(
    const BarotropicState& state,
    const VecX& forcing_x,
    const VecX& forcing_y,
    VecX& dHU_bar_dt,
    VecX& dHV_bar_dt) const {

    int n = n_horiz_;
    dHU_bar_dt.resize(n);
    dHV_bar_dt.resize(n);

    // Pressure gradient: -gH * grad(eta)
    VecX deta_dx = D_x_2d_ * state.eta;
    VecX deta_dy = D_y_2d_ * state.eta;

    for (int i = 0; i < n; ++i) {
        // Barotropic pressure gradient
        Real pg_x = -constants_.g * state.H(i) * deta_dx(i);
        Real pg_y = -constants_.g * state.H(i) * deta_dy(i);

        // Coriolis
        Real cor_x = f_(i) * state.HV_bar(i);
        Real cor_y = -f_(i) * state.HU_bar(i);

        // Total tendency
        dHU_bar_dt(i) = pg_x + cor_x + forcing_x(i);
        dHV_bar_dt(i) = pg_y + cor_y + forcing_y(i);
    }
}

void ModeSplittingElement::barotropic_rhs(
    const BarotropicState& state,
    const VecX& forcing_x,
    const VecX& forcing_y,
    BarotropicTendencies& tendency) const {

    tendency.resize(n_horiz_);

    barotropic_eta_rhs(state, tendency.deta_dt);
    barotropic_momentum_rhs(state, forcing_x, forcing_y,
                             tendency.dHU_bar_dt, tendency.dHV_bar_dt);
}

void ModeSplittingElement::baroclinic_pressure_forcing(
    const VecX& rho, const VecX& H, VecX& forcing_x, VecX& forcing_y) const {

    forcing_x.resize(n_horiz_);
    forcing_y.resize(n_horiz_);
    forcing_x.setZero();
    forcing_y.setZero();

    // Baroclinic pressure gradient: integral of (g/rho_0) * grad_h(rho') * H dσ
    // This is a depth-integrated quantity

    for (int i_h = 0; i_h < n_horiz_; ++i_h) {
        Real fx = 0.0, fy = 0.0;

        for (int k = 0; k < n_vert_; ++k) {
            int idx = i_h * n_vert_ + k;

            // Horizontal density gradient (simplified: use neighbor differences)
            // Full implementation would use proper DG gradient

            // For now, assume forcing is pre-computed and passed in
        }

        forcing_x(i_h) = fx;
        forcing_y(i_h) = fy;
    }
}

void ModeSplittingElement::bottom_stress_forcing(
    const VecX& u_bot, const VecX& v_bot, Real Cd,
    VecX& tau_x, VecX& tau_y) const {

    tau_x.resize(n_horiz_);
    tau_y.resize(n_horiz_);

    for (int i = 0; i < n_horiz_; ++i) {
        Real speed = std::sqrt(u_bot(i) * u_bot(i) + v_bot(i) * v_bot(i));
        tau_x(i) = -Cd * speed * u_bot(i);
        tau_y(i) = -Cd * speed * v_bot(i);
    }
}

void ModeSplittingElement::wind_stress_forcing(
    const VecX& tau_wind_x, const VecX& tau_wind_y, const VecX& H,
    VecX& forcing_x, VecX& forcing_y) const {

    forcing_x.resize(n_horiz_);
    forcing_y.resize(n_horiz_);

    // Wind stress is applied at surface: tau / (rho_0 * H)
    for (int i = 0; i < n_horiz_; ++i) {
        Real H_inv = (H(i) > 1e-10) ? 1.0 / H(i) : 0.0;
        forcing_x(i) = tau_wind_x(i) * H_inv / constants_.rho_0;
        forcing_y(i) = tau_wind_y(i) * H_inv / constants_.rho_0;
    }
}

void ModeSplittingElement::barotropic_step_euler(
    Real dt, const VecX& forcing_x, const VecX& forcing_y,
    BarotropicState& state) const {

    BarotropicTendencies tendency;
    barotropic_rhs(state, forcing_x, forcing_y, tendency);

    state.eta += dt * tendency.deta_dt;
    state.HU_bar += dt * tendency.dHU_bar_dt;
    state.HV_bar += dt * tendency.dHV_bar_dt;

    state.update_derived(h_);
}

void ModeSplittingElement::barotropic_step_predictor_corrector(
    Real dt, const VecX& forcing_x, const VecX& forcing_y,
    BarotropicState& state) const {

    // Predictor (forward Euler)
    BarotropicTendencies tend_n;
    barotropic_rhs(state, forcing_x, forcing_y, tend_n);

    BarotropicState state_star;
    state_star.resize(n_horiz_);
    state_star.eta = state.eta + dt * tend_n.deta_dt;
    state_star.HU_bar = state.HU_bar + dt * tend_n.dHU_bar_dt;
    state_star.HV_bar = state.HV_bar + dt * tend_n.dHV_bar_dt;
    state_star.update_derived(h_);

    // Corrector (trapezoidal)
    BarotropicTendencies tend_star;
    barotropic_rhs(state_star, forcing_x, forcing_y, tend_star);

    state.eta += 0.5 * dt * (tend_n.deta_dt + tend_star.deta_dt);
    state.HU_bar += 0.5 * dt * (tend_n.dHU_bar_dt + tend_star.dHU_bar_dt);
    state.HV_bar += 0.5 * dt * (tend_n.dHV_bar_dt + tend_star.dHV_bar_dt);

    state.update_derived(h_);
}

void ModeSplittingElement::subcycle(
    Real dt_3d, int n_subcycles,
    const VecX& forcing_x, const VecX& forcing_y,
    BarotropicState& state, VecX& eta_avg) const {

    Real dt_baro = dt_3d / n_subcycles;
    eta_avg = VecX::Zero(n_horiz_);

    for (int sub = 0; sub < n_subcycles; ++sub) {
        barotropic_step_predictor_corrector(dt_baro, forcing_x, forcing_y, state);
        eta_avg += state.eta;
    }

    eta_avg /= static_cast<Real>(n_subcycles);
}

// =============================================================================
// ModeSplittingSolver implementation
// =============================================================================

ModeSplittingSolver::ModeSplittingSolver(
    int order, const OceanConstants& constants, const ModeSplittingParams& params)
    : order_(order)
    , constants_(constants)
    , params_(params)
    , basis_(order)
    , quad_(order, QuadratureType::GaussLegendre)
{
}

void ModeSplittingSolver::initialize(
    int num_elements,
    const std::vector<VecX>& bathymetry,
    const std::vector<VecX>& dh_dx,
    const std::vector<VecX>& dh_dy,
    const std::vector<VecX>& coriolis) {

    elements_.resize(num_elements);

    for (int e = 0; e < num_elements; ++e) {
        elements_[e] = std::make_unique<ModeSplittingElement>(
            basis_, quad_, constants_);
        elements_[e]->set_bathymetry(bathymetry[e], dh_dx[e], dh_dy[e]);
        elements_[e]->set_coriolis(coriolis[e]);
    }

    baroclinic_forcing_x_.resize(num_elements);
    baroclinic_forcing_y_.resize(num_elements);
}

void ModeSplittingSolver::decompose(
    const std::vector<PrimitiveState>& full_state,
    std::vector<BarotropicState>& baro_state,
    std::vector<BaroclinicState>& clinic_state) const {

    size_t num_elems = full_state.size();
    baro_state.resize(num_elems);
    clinic_state.resize(num_elems);

    #pragma omp parallel for
    for (size_t e = 0; e < num_elems; ++e) {
        int n_horiz = (order_ + 1) * (order_ + 1);
        int n_total = full_state[e].u.size();

        baro_state[e].resize(n_horiz);
        clinic_state[e].resize(n_total);

        // Extract barotropic (depth-averaged)
        elements_[e]->depth_average(full_state[e].u, baro_state[e].U_bar);
        elements_[e]->depth_average(full_state[e].v, baro_state[e].V_bar);
        baro_state[e].eta = full_state[e].eta.head(n_horiz);

        // Compute depth-integrated transport
        elements_[e]->depth_integrate(full_state[e].u, full_state[e].H, baro_state[e].HU_bar);
        elements_[e]->depth_integrate(full_state[e].v, full_state[e].H, baro_state[e].HV_bar);

        baro_state[e].H = full_state[e].H.head(n_horiz);

        // Compute baroclinic deviation
        elements_[e]->compute_deviation(full_state[e].u, baro_state[e].U_bar,
                                          clinic_state[e].u_prime);
        elements_[e]->compute_deviation(full_state[e].v, baro_state[e].V_bar,
                                          clinic_state[e].v_prime);

        clinic_state[e].HT = full_state[e].HT;
        clinic_state[e].HS = full_state[e].HS;
        clinic_state[e].omega = full_state[e].omega;
    }
}

void ModeSplittingSolver::recombine(
    const std::vector<BarotropicState>& baro_state,
    const std::vector<BaroclinicState>& clinic_state,
    std::vector<PrimitiveState>& full_state) const {

    size_t num_elems = baro_state.size();

    #pragma omp parallel for
    for (size_t e = 0; e < num_elems; ++e) {
        int n_horiz = (order_ + 1) * (order_ + 1);
        int n_vert = order_ + 1;
        int n_total = clinic_state[e].u_prime.size();

        // u = U_bar + u'
        for (int i_h = 0; i_h < n_horiz; ++i_h) {
            for (int k = 0; k < n_vert; ++k) {
                int idx = i_h * n_vert + k;
                full_state[e].u(idx) = baro_state[e].U_bar(i_h) + clinic_state[e].u_prime(idx);
                full_state[e].v(idx) = baro_state[e].V_bar(i_h) + clinic_state[e].v_prime(idx);
            }
        }

        // Update eta from barotropic
        for (int i_h = 0; i_h < n_horiz; ++i_h) {
            for (int k = 0; k < n_vert; ++k) {
                int idx = i_h * n_vert + k;
                full_state[e].eta(idx) = baro_state[e].eta(i_h);
            }
        }

        // Tracers from baroclinic
        full_state[e].HT = clinic_state[e].HT;
        full_state[e].HS = clinic_state[e].HS;
        full_state[e].omega = clinic_state[e].omega;
    }
}

void ModeSplittingSolver::step(Real dt_3d, std::vector<PrimitiveState>& full_state) {
    std::vector<BarotropicState> baro_state;
    std::vector<BaroclinicState> clinic_state;

    // 1. Decompose into barotropic and baroclinic
    decompose(full_state, baro_state, clinic_state);

    // 2. Compute baroclinic forcing for barotropic equations
    compute_baroclinic_forcing(clinic_state, full_state);

    // 3. Subcycle barotropic equations
    std::vector<VecX> eta_avg(full_state.size());
    subcycle_barotropic(dt_3d, baro_state, eta_avg);

    // 4. Update baroclinic state
    update_baroclinic(dt_3d, eta_avg, clinic_state, full_state);

    // 5. Recombine into full state
    recombine(baro_state, clinic_state, full_state);
}

void ModeSplittingSolver::compute_baroclinic_forcing(
    const std::vector<BaroclinicState>& clinic_state,
    const std::vector<PrimitiveState>& full_state) {

    size_t num_elems = clinic_state.size();

    #pragma omp parallel for
    for (size_t e = 0; e < num_elems; ++e) {
        int n_horiz = (order_ + 1) * (order_ + 1);
        baroclinic_forcing_x_[e].resize(n_horiz);
        baroclinic_forcing_y_[e].resize(n_horiz);

        // Baroclinic pressure gradient forcing
        elements_[e]->baroclinic_pressure_forcing(
            full_state[e].rho, full_state[e].H,
            baroclinic_forcing_x_[e], baroclinic_forcing_y_[e]);

        // Could add wind stress, bottom friction here
    }
}

void ModeSplittingSolver::subcycle_barotropic(
    Real dt_3d, std::vector<BarotropicState>& baro_state,
    std::vector<VecX>& eta_avg) {

    size_t num_elems = baro_state.size();

    #pragma omp parallel for
    for (size_t e = 0; e < num_elems; ++e) {
        elements_[e]->subcycle(dt_3d, params_.subcycles,
                                baroclinic_forcing_x_[e],
                                baroclinic_forcing_y_[e],
                                baro_state[e], eta_avg[e]);
    }
}

void ModeSplittingSolver::update_baroclinic(
    Real dt_3d, const std::vector<VecX>& eta_avg,
    std::vector<BaroclinicState>& clinic_state,
    std::vector<PrimitiveState>& full_state) {

    // Update baroclinic state using the time-averaged eta from barotropic subcycling
    // This implements the slow (3D) part of the mode-split time stepping

    size_t num_elems = full_state.size();
    int n_horiz = (order_ + 1) * (order_ + 1);
    int n_vert = order_ + 1;

    #pragma omp parallel for
    for (size_t e = 0; e < num_elems; ++e) {
        const ModeSplittingElement& elem = *elements_[e];
        const VecX& h = elements_[e]->bathymetry();

        // 1. Update eta in full_state using time-averaged value from barotropic
        // Expand 2D eta_avg to 3D state (same eta at all vertical levels)
        for (int i_h = 0; i_h < n_horiz; ++i_h) {
            for (int k = 0; k < n_vert; ++k) {
                int idx = i_h * n_vert + k;
                full_state[e].eta(idx) = eta_avg[e](i_h);
            }
        }

        // 2. Update water depth H = eta + h
        for (int i_h = 0; i_h < n_horiz; ++i_h) {
            Real H_val = eta_avg[e](i_h) + h(i_h);
            for (int k = 0; k < n_vert; ++k) {
                int idx = i_h * n_vert + k;
                full_state[e].H(idx) = H_val;
            }
        }

        // 3. Update conserved momentum Hu, Hv from baroclinic velocities
        // Hu = H * (U_bar + u')
        VecX U_bar(n_horiz), V_bar(n_horiz);
        elem.depth_average(full_state[e].u, U_bar);
        elem.depth_average(full_state[e].v, V_bar);

        for (int i_h = 0; i_h < n_horiz; ++i_h) {
            for (int k = 0; k < n_vert; ++k) {
                int idx = i_h * n_vert + k;
                Real u_total = U_bar(i_h) + clinic_state[e].u_prime(idx);
                Real v_total = V_bar(i_h) + clinic_state[e].v_prime(idx);
                full_state[e].u(idx) = u_total;
                full_state[e].v(idx) = v_total;
                full_state[e].Hu(idx) = full_state[e].H(idx) * u_total;
                full_state[e].Hv(idx) = full_state[e].H(idx) * v_total;
            }
        }

        // 4. Simple tracer update (advection with forward Euler)
        // For tracers HT and HS, we apply simple advection:
        // d(HT)/dt = -u * dT/dx - v * dT/dy - omega * dT/dsigma (+ diffusion)
        // Here we use a simplified explicit update

        // Extract T, S from conserved HT, HS
        int n_total = clinic_state[e].HT.size();
        for (int idx = 0; idx < n_total; ++idx) {
            Real H_val = full_state[e].H(idx);
            Real H_inv = (H_val > 1e-10) ? 1.0 / H_val : 0.0;
            full_state[e].T(idx) = clinic_state[e].HT(idx) * H_inv;
            full_state[e].S(idx) = clinic_state[e].HS(idx) * H_inv;
        }

        // Update HT, HS with new H (after eta change)
        for (int idx = 0; idx < n_total; ++idx) {
            clinic_state[e].HT(idx) = full_state[e].H(idx) * full_state[e].T(idx);
            clinic_state[e].HS(idx) = full_state[e].H(idx) * full_state[e].S(idx);
        }

        // 5. Diagnose omega from continuity equation
        // omega(sigma) = -(1/H) * integral_{-1}^{sigma} [d(Hu)/dx + d(Hv)/dy] dsigma
        // For now, use a simplified estimate based on depth-averaged divergence
        VecX HU_bar(n_horiz), HV_bar(n_horiz);
        elem.depth_integrate(full_state[e].u, full_state[e].H, HU_bar);
        elem.depth_integrate(full_state[e].v, full_state[e].H, HV_bar);

        // Compute horizontal divergence using 2D operators
        // Note: Full implementation would use DG gradient operators
        VecX div_HU = elem.compute_2d_divergence(HU_bar, HV_bar);

        // Simple omega diagnosis: linear profile from 0 at bottom to div/H at surface
        for (int i_h = 0; i_h < n_horiz; ++i_h) {
            Real H_val = full_state[e].H(i_h * n_vert);
            Real H_inv = (H_val > 1e-10) ? 1.0 / H_val : 0.0;

            for (int k = 0; k < n_vert; ++k) {
                int idx = i_h * n_vert + k;
                // Linear interpolation: omega = 0 at bottom (k=0), varies to top
                Real sigma_frac = static_cast<Real>(k) / static_cast<Real>(n_vert - 1);
                clinic_state[e].omega(idx) = -sigma_frac * div_HU(i_h) * H_inv;
            }
        }

        // Copy omega to full state
        full_state[e].omega = clinic_state[e].omega;

        // 6. Update density from equation of state
        // rho = rho(T, S, p) using linear approximation
        for (int idx = 0; idx < n_total; ++idx) {
            Real T = full_state[e].T(idx);
            Real S = full_state[e].S(idx);
            // Linear EOS: rho = rho_0 * (1 - alpha*(T-T0) + beta*(S-S0))
            Real T0 = 10.0, S0 = 35.0;
            full_state[e].rho(idx) = constants_.rho_0 * (
                1.0 - constants_.alpha * (T - T0) + constants_.beta * (S - S0));
        }
    }
}

Real ModeSplittingSolver::compute_barotropic_dt(
    const std::vector<BarotropicState>& states, Real dx_min) const {

    // CFL for surface gravity waves: c = sqrt(g * H_max)
    Real H_max = 0.0;
    for (const auto& state : states) {
        H_max = std::max(H_max, state.H.maxCoeff());
    }

    Real c_max = std::sqrt(constants_.g * H_max);
    return params_.barotropic_cfl * dx_min / c_max;
}

Real ModeSplittingSolver::compute_baroclinic_dt(
    const std::vector<PrimitiveState>& states, Real dx_min) const {

    // CFL for advection: max velocity
    Real u_max = 0.0;
    for (const auto& state : states) {
        Real u_elem = std::max(state.u.cwiseAbs().maxCoeff(),
                                state.v.cwiseAbs().maxCoeff());
        u_max = std::max(u_max, u_elem);
    }

    if (u_max < 1e-10) u_max = 1.0;  // Fallback
    return 0.8 * dx_min / u_max;
}

}  // namespace drifter
