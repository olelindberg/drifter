#pragma once

#include "core/types.hpp"
#include "mesh/element.hpp"
#include <functional>

namespace drifter {

// Physical constants
struct PhysicalConstants {
    Real gravity = 9.81; // m/s^2
    Real omega = 7.2921e-5; // Earth's angular velocity (rad/s)
    Real rho_0 = 1025.0; // Reference density (kg/m^3)
    Real nu = 1.0e-6; // Kinematic viscosity (m^2/s)
};

// Shallow water equations flux computation
class ShallowWaterFlux {
public:
    explicit ShallowWaterFlux(const PhysicalConstants &constants);

    // Compute physical flux F(U) in x-direction
    void flux_x(const VecX &U, VecX &F) const;

    // Compute physical flux G(U) in y-direction
    void flux_y(const VecX &U, VecX &G) const;

    // Compute source terms (bathymetry, Coriolis, friction)
    void source(const VecX &U, const Vec3 &position, Real bathymetry, const Vec3 &grad_bathymetry,
                VecX &S) const;

    // Numerical flux (Riemann solver interface)
    void numerical_flux(const VecX &UL, const VecX &UR, const Vec3 &normal, VecX &F_star) const;

    // Wave speed estimate for CFL
    Real max_wave_speed(const VecX &U) const;

    // Eigenvalues of flux Jacobian
    Vec3 eigenvalues(const VecX &U, const Vec3 &normal) const;

private:
    PhysicalConstants constants_;
};

// Riemann solvers for shallow water
namespace riemann {

// Local Lax-Friedrichs (Rusanov)
void lax_friedrichs(const VecX &UL, const VecX &UR, const Vec3 &normal, Real g, VecX &F_star);

// Roe solver with entropy fix
void roe(const VecX &UL, const VecX &UR, const Vec3 &normal, Real g, VecX &F_star);

// HLLC solver
void hllc(const VecX &UL, const VecX &UR, const Vec3 &normal, Real g, VecX &F_star);

} // namespace riemann

// Wetting and drying treatment
class WettingDrying {
public:
    explicit WettingDrying(Real threshold = 1.0e-4);

    // Check if element is wet
    bool is_wet(Real depth) const { return depth > threshold_; }

    // Apply positivity-preserving limiter
    void apply_limiter(Element &elem);

    // Compute modified flux for thin layers
    void modify_flux(const VecX &U, VecX &F) const;

private:
    Real threshold_;
};

// Coriolis force computation
class CoriolisForce {
public:
    // f-plane approximation
    void set_f_plane(Real f0);

    // Beta-plane approximation
    void set_beta_plane(Real f0, Real beta, Real y0);

    // Full spherical
    void set_spherical(Real omega);

    // Compute Coriolis parameter at given latitude
    Real coriolis_parameter(Real latitude) const;

    // Add Coriolis source term
    void add_source(const VecX &U, Real f, VecX &S) const;

private:
    enum class Mode { FPlane,
                      BetaPlane,
                      Spherical };
    Mode mode_ = Mode::FPlane;
    Real f0_ = 0.0, beta_ = 0.0, y0_ = 0.0, omega_ = 7.2921e-5;
};

// Bottom friction models
class BottomFriction {
public:
    enum class Model { None,
                       Linear,
                       Quadratic,
                       Manning };

    void set_model(Model model, Real coefficient);

    // Compute friction source term
    void add_source(const VecX &U, Real depth, VecX &S) const;

private:
    Model model_ = Model::None;
    Real coefficient_ = 0.0;
};

} // namespace drifter
