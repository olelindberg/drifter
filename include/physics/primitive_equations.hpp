#pragma once

// Hydrostatic Primitive Equations for Ocean Modeling
//
// The 3D hydrostatic primitive equations in sigma-coordinates:
//
// Momentum (horizontal):
//   d(Hu)/dt + d(Huu)/dx + d(Huv)/dy + d(Huω)/dσ = -gH*dη/dx + fHv + Fx + Dx
//   d(Hv)/dt + d(Hvu)/dx + d(Hvv)/dy + d(Hvω)/dσ = -gH*dη/dy - fHu + Fy + Dy
//
// Free surface (vertically-integrated continuity):
//   dη/dt + d/dx(∫Hu dσ) + d/dy(∫Hv dσ) = 0
//
// Tracers (temperature, salinity):
//   d(HT)/dt + d(HuT)/dx + d(HvT)/dy + d(HωT)/dσ = d/dσ(Kv*dT/dσ)/H + QT
//
// where:
//   H = η + h  (total water depth)
//   ω = dσ/dt  (sigma velocity, diagnosed)
//   f = Coriolis parameter
//   Fx, Fy = baroclinic pressure gradient
//   Dx, Dy = diffusion terms
//   Kv = vertical diffusivity
//   QT = tracer source/sink

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/quadrature_3d.hpp"
#include "dg/operators_3d.hpp"
#include "mesh/sigma_coordinate.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace drifter {

// Forward declarations
class VerticalVelocityDiagnosis;

/// @brief Physical constants for ocean modeling
struct OceanConstants {
    Real g = 9.81;           ///< Gravitational acceleration [m/s^2]
    Real rho_0 = 1025.0;     ///< Reference density [kg/m^3]
    Real omega = 7.2921e-5;  ///< Earth rotation rate [rad/s]
    Real R_earth = 6.371e6;  ///< Earth radius [m]
    Real cp = 3985.0;        ///< Specific heat capacity [J/(kg·K)]
    Real alpha = 2e-4;       ///< Thermal expansion coefficient [1/K]
    Real beta = 7.6e-4;      ///< Haline contraction coefficient [1/PSU]
};

/// @brief Coriolis parameter computation
class CoriolisParameter {
public:
    /// @brief Construct Coriolis parameter
    /// @param f0 Reference Coriolis parameter [1/s]
    /// @param beta Beta-plane parameter [1/(m·s)]
    /// @param y0 Reference latitude position [m]
    CoriolisParameter(Real f0 = 1e-4, Real beta = 2e-11, Real y0 = 0.0)
        : f0_(f0), beta_(beta), y0_(y0) {}

    /// @brief Compute f at a given y position (beta-plane)
    Real operator()(Real y) const { return f0_ + beta_ * (y - y0_); }

    /// @brief Compute f at each DOF location
    void compute(const VecX& y, VecX& f) const {
        f.resize(y.size());
        for (int i = 0; i < y.size(); ++i) {
            f(i) = (*this)(y(i));
        }
    }

    Real f0() const { return f0_; }
    Real beta() const { return beta_; }

private:
    Real f0_;
    Real beta_;
    Real y0_;
};

/// @brief State variables for primitive equations
struct PrimitiveState {
    VecX Hu;        ///< H * u (x-momentum)
    VecX Hv;        ///< H * v (y-momentum)
    VecX eta;       ///< Free surface elevation
    VecX HT;        ///< H * T (heat content)
    VecX HS;        ///< H * S (salt content)

    // Derived quantities (diagnosed)
    VecX H;         ///< Water depth H = eta + h
    VecX u;         ///< x-velocity
    VecX v;         ///< y-velocity
    VecX T;         ///< Temperature
    VecX S;         ///< Salinity
    VecX omega;     ///< Sigma velocity
    VecX rho;       ///< Density anomaly

    /// @brief Update derived quantities from conserved variables
    void update_derived(const VecX& h);

    /// @brief Resize all arrays
    void resize(int n);
};

/// @brief Viscosity and diffusivity parameters
struct DiffusivityParams {
    Real nu_h = 1.0;          ///< Horizontal momentum diffusivity [m^2/s]
    Real nu_v = 1e-4;         ///< Vertical momentum diffusivity [m^2/s]
    Real kappa_h_T = 1.0;     ///< Horizontal thermal diffusivity [m^2/s]
    Real kappa_v_T = 1e-5;    ///< Vertical thermal diffusivity [m^2/s]
    Real kappa_h_S = 1.0;     ///< Horizontal haline diffusivity [m^2/s]
    Real kappa_v_S = 1e-5;    ///< Vertical haline diffusivity [m^2/s]

    // Smagorinsky model parameters (if used)
    Real C_smag = 0.1;        ///< Smagorinsky coefficient
    bool use_smagorinsky = false;
};

/// @brief Right-hand side (tendencies) for primitive equations
struct PrimitiveTendencies {
    VecX dHu_dt;    ///< Tendency of H*u
    VecX dHv_dt;    ///< Tendency of H*v
    VecX deta_dt;   ///< Tendency of eta (free surface)
    VecX dHT_dt;    ///< Tendency of H*T
    VecX dHS_dt;    ///< Tendency of H*S

    void resize(int n);
    void set_zero();
};

/// @brief Hydrostatic Primitive Equations solver for a single element
class PrimitiveEquationsElement {
public:
    /// @brief Construct element solver
    PrimitiveEquationsElement(const HexahedronBasis& basis,
                               const GaussQuadrature3D& quad,
                               const OceanConstants& constants = OceanConstants());

    /// @brief Set bathymetry for this element
    void set_bathymetry(const VecX& h, const VecX& dh_dx, const VecX& dh_dy);

    /// @brief Set Coriolis parameter
    void set_coriolis(const CoriolisParameter& coriolis, const VecX& y_positions);

    /// @brief Set diffusivity parameters
    void set_diffusivity(const DiffusivityParams& params) { diffusivity_ = params; }

    // =========================================================================
    // RHS computation
    // =========================================================================

    /// @brief Compute full RHS for primitive equations
    /// @param state Current state
    /// @param[out] tendency Computed tendencies
    void compute_rhs(const PrimitiveState& state, PrimitiveTendencies& tendency) const;

    /// @brief Compute momentum RHS only (for split stepping)
    void compute_momentum_rhs(const PrimitiveState& state,
                               VecX& dHu_dt, VecX& dHv_dt) const;

    /// @brief Compute tracer RHS only
    void compute_tracer_rhs(const PrimitiveState& state,
                             VecX& dHT_dt, VecX& dHS_dt) const;

    /// @brief Compute free surface RHS (from depth-integrated continuity)
    void compute_eta_rhs(const PrimitiveState& state, VecX& deta_dt) const;

    // =========================================================================
    // Individual terms
    // =========================================================================

    /// @brief Advection terms for momentum
    void momentum_advection(const PrimitiveState& state,
                            VecX& adv_Hu, VecX& adv_Hv) const;

    /// @brief Coriolis terms
    void coriolis_terms(const PrimitiveState& state,
                        VecX& cor_Hu, VecX& cor_Hv) const;

    /// @brief Barotropic pressure gradient (-gH*grad(eta))
    void barotropic_pressure_gradient(const PrimitiveState& state,
                                       VecX& pg_u, VecX& pg_v) const;

    /// @brief Horizontal diffusion
    void horizontal_diffusion(const VecX& field, Real nu,
                               VecX& diff_x, VecX& diff_y) const;

    /// @brief Vertical diffusion in sigma coordinates
    void vertical_diffusion(const VecX& field, Real kappa,
                            const VecX& H, VecX& diff) const;

    /// @brief Advection for tracer field
    void tracer_advection(const PrimitiveState& state, const VecX& HT,
                           VecX& adv_HT) const;

private:
    const HexahedronBasis& basis_;
    const GaussQuadrature3D& quad_;
    OceanConstants constants_;
    DiffusivityParams diffusivity_;

    // Cached bathymetry and derivatives
    VecX h_;
    VecX dh_dx_;
    VecX dh_dy_;

    // Coriolis parameter at DOFs
    VecX f_;

    // DG operators
    DG3DElementOperator elem_op_;
    SigmaCoordinateOperators sigma_op_;

    // 2D horizontal operators for eta equation
    int n_horiz_;
    int n_vert_;
    MatX D_x_2d_;
    MatX D_y_2d_;
    VecX sigma_weights_;

    /// @brief Build 2D differentiation matrices for horizontal operations
    void build_2d_operators();
};

/// @brief Global primitive equations solver for entire mesh
class PrimitiveEquationsSolver {
public:
    /// @brief Construct solver
    /// @param order Polynomial order
    /// @param constants Physical constants
    PrimitiveEquationsSolver(int order, const OceanConstants& constants = OceanConstants());

    /// @brief Initialize solver with mesh
    /// @param num_elements Number of elements
    /// @param bathymetry Bathymetry at each element's DOFs
    /// @param coriolis Coriolis parameter calculator
    /// @param y_positions Y-coordinates at DOFs for Coriolis
    void initialize(int num_elements,
                    const std::vector<VecX>& bathymetry,
                    const std::vector<VecX>& dh_dx,
                    const std::vector<VecX>& dh_dy,
                    const CoriolisParameter& coriolis,
                    const std::vector<VecX>& y_positions);

    /// @brief Set boundary conditions
    void set_boundary_conditions(const std::vector<BoundaryCondition>& bc);

    /// @brief Set face connections (for DG flux computation)
    void set_face_connections(const std::vector<std::vector<FaceConnection>>& conn);

    /// @brief Compute RHS for all elements
    void compute_rhs(const std::vector<PrimitiveState>& states,
                     std::vector<PrimitiveTendencies>& tendencies) const;

    /// @brief Apply interface fluxes (DG coupling between elements)
    void apply_interface_fluxes(const std::vector<PrimitiveState>& states,
                                 std::vector<PrimitiveTendencies>& tendencies) const;

    /// @brief Apply boundary conditions
    void apply_boundary_conditions(const std::vector<PrimitiveState>& states,
                                    std::vector<PrimitiveTendencies>& tendencies,
                                    Real time) const;

    /// @brief Get DG integration manager
    const DG3DIntegration& dg_integration() const { return dg_; }

private:
    int order_;
    OceanConstants constants_;

    HexahedronBasis basis_;
    GaussQuadrature3D quad_;
    DG3DIntegration dg_;

    std::vector<std::unique_ptr<PrimitiveEquationsElement>> elements_;
    std::vector<BoundaryCondition> boundary_conditions_;
    std::vector<std::vector<FaceConnection>> face_connections_;
};

/// @brief Wind stress forcing
struct WindStress {
    std::function<Vec2(Real, Real, Real)> stress_func;  ///< (x, y, t) -> (tau_x, tau_y)

    /// @brief Compute wind stress at DOF locations
    void compute(const VecX& x, const VecX& y, Real time,
                 VecX& tau_x, VecX& tau_y) const;
};

/// @brief Bottom friction parameterization
class BottomFriction {
public:
    enum class Type { Linear, Quadratic, ManningN };

    /// @brief Construct with quadratic drag coefficient
    explicit BottomFriction(Real Cd = 2.5e-3, Type type = Type::Quadratic)
        : Cd_(Cd), type_(type) {}

    /// @brief Compute bottom stress
    void compute(const VecX& u_bot, const VecX& v_bot, const VecX& H,
                 VecX& tau_x, VecX& tau_y) const;

private:
    Real Cd_;
    Type type_;
};

}  // namespace drifter
