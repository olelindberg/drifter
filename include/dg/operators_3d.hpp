#pragma once

// 3D DG integration operators for hexahedral elements
// Provides gradient, divergence, and Laplacian operators with
// support for non-conforming interfaces via mortar elements
//
// Adapted from wobbler: DG2DIntegration, DG1DIntegration

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/face_connection.hpp"
#include "dg/mortar.hpp"
#include "dg/quadrature_3d.hpp"
#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace drifter {

// Forward declarations
class Mesh;
class Element;

/// @brief Boundary condition type for DG operators
enum class BCType : uint8_t {
    Dirichlet, ///< Specified value
    Neumann,   ///< Specified flux
    Periodic,  ///< Periodic boundary
    Outflow,   ///< Free outflow (extrapolation)
    Inflow,    ///< Specified inflow
    NoSlip,    ///< No-slip wall (velocity = 0)
    FreeSlip   ///< Free-slip wall (normal velocity = 0)
};

/// @brief Boundary condition specification
struct BoundaryCondition {
    BCType type = BCType::Dirichlet;
    int boundary_id = -1;                               ///< Boundary marker/ID
    std::function<VecX(const Vec3 &, Real)> value_func; ///< BC value at (x, t)
};

/// @brief Physical flux function signature for advection-like equations
/// F(U) returns the flux tensor (3 components for 3D)
using PhysicalFluxFunc = std::function<Tensor3(const VecX &U)>;

/// @brief Solution state at a point (for flux evaluation)
struct ElementState {
    VecX U;        ///< Solution values (num_vars)
    MatX grad_U;   ///< Solution gradients (num_vars x 3)
    Vec3 position; ///< Physical position
    Vec3 normal;   ///< Outward normal (for face evaluations)
};

/// @brief 3D DG integration operators for a single element
/// @details Provides local element-wise operators for DG discretization
class DG3DElementOperator {
public:
    /// @brief Construct element operator
    /// @param basis Hexahedron basis
    /// @param quad Volume quadrature
    explicit DG3DElementOperator(
        const HexahedronBasis &basis, const GaussQuadrature3D &quad);

    /// Polynomial order
    int order() const { return basis_.order(); }

    /// Number of DOFs per element (velocity grid)
    int num_dofs_velocity() const { return basis_.num_dofs_velocity(); }

    /// Number of DOFs per element (tracer grid)
    int num_dofs_tracer() const { return basis_.num_dofs_tracer(); }

    // =========================================================================
    // Mass matrix operations
    // =========================================================================

    /// Apply mass matrix: M * u
    void mass(const VecX &u, VecX &Mu, bool use_lgl = true) const;

    /// Apply inverse mass matrix: M^{-1} * u
    void mass_inv(const VecX &u, VecX &Minv_u, bool use_lgl = true) const;

    // =========================================================================
    // Derivative operations in reference space
    // =========================================================================

    /// Compute gradient in reference coordinates: (du/dxi, du/deta, du/dzeta)
    /// @param u Solution at DOFs
    /// @param grad_u Output: gradient at DOFs (3 columns for xi, eta, zeta)
    void
    gradient_reference(const VecX &u, MatX &grad_u, bool use_lgl = true) const;

    /// Compute divergence in reference coordinates: du/dxi + dv/deta + dw/dzeta
    /// @param flux Flux components (3 VecX for xi, eta, zeta components)
    /// @param div_flux Output: divergence at DOFs
    void divergence_reference(
        const std::array<VecX, 3> &flux, VecX &div_flux,
        bool use_lgl = true) const;

    // =========================================================================
    // Volume integral contribution
    // =========================================================================

    /// @brief Compute volume integral contribution to RHS
    /// @details For conservation law: dU/dt + div(F) = S
    ///          Volume term: integral of (grad_phi . F) over element
    /// @param U Solution at DOFs
    /// @param physical_flux Function computing F(U)
    /// @param jacobian Jacobian matrix (3x3) at each quadrature point
    /// @param det_J Jacobian determinant at each quadrature point
    /// @param[out] rhs Volume contribution to RHS
    void volume_integral(
        const VecX &U, const PhysicalFluxFunc &physical_flux,
        const std::vector<Mat3> &jacobian, const VecX &det_J, VecX &rhs,
        bool use_lgl = true) const;

    // =========================================================================
    // Face integral contribution
    // =========================================================================

    /// @brief Compute face flux contribution for a single face
    /// @param face_id Local face ID (0-5)
    /// @param U_interior Interior solution at face DOFs
    /// @param U_exterior Exterior solution at face DOFs
    /// @param normal Outward normal
    /// @param numerical_flux Numerical flux function
    /// @param face_jacobian Surface Jacobian (det of face metric)
    /// @param[out] rhs Face contribution to RHS
    void face_integral(
        int face_id, const VecX &U_interior, const VecX &U_exterior,
        const Vec3 &normal, const NumericalFluxFunc &numerical_flux,
        const VecX &face_jacobian, VecX &rhs, bool use_lgl = true) const;

    // =========================================================================
    // Access to basis and quadrature
    // =========================================================================

    const HexahedronBasis &basis() const { return basis_; }
    const GaussQuadrature3D &quadrature() const { return quad_; }

private:
    const HexahedronBasis &basis_;
    const GaussQuadrature3D &quad_;

    // Face quadratures
    std::array<FaceQuadrature, 6> face_quads_;

    // Precomputed basis values at quadrature points
    MatX phi_at_quad_lgl_; // (num_quad x num_dofs)
    MatX phi_at_quad_gl_;
    std::array<MatX, 3>
        dphi_at_quad_lgl_; // Gradients (num_quad x num_dofs) for each dir
    std::array<MatX, 3> dphi_at_quad_gl_;
};

/// @brief Global DG integration manager for entire mesh
/// @details Handles assembly of element contributions and interface fluxes
class DG3DIntegration {
public:
    /// @brief Construct integration manager
    /// @param order Polynomial order
    /// @param use_mortar Use mortar elements for non-conforming interfaces
    DG3DIntegration(int order, bool use_mortar = true);

    /// Polynomial order
    int order() const { return order_; }

    /// Get element operator
    const DG3DElementOperator &element_operator() const { return elem_op_; }

    /// Get basis
    const HexahedronBasis &basis() const { return basis_; }

    // =========================================================================
    // Gradient operator
    // =========================================================================

    /// @brief Compute weak gradient of scalar field
    /// @details Uses DG formulation: (grad_h u, v) = -(u, div v) + <u, v.n>
    /// @param U Scalar field at DOFs (per element)
    /// @param face_connections Face connectivity information
    /// @param bc Boundary conditions
    /// @param[out] grad_U_x X-component of gradient at DOFs
    /// @param[out] grad_U_y Y-component of gradient at DOFs
    /// @param[out] grad_U_z Z-component of gradient at DOFs
    void gradient(
        const std::vector<VecX> &U,
        const std::vector<std::vector<FaceConnection>> &face_connections,
        const std::vector<BoundaryCondition> &bc, std::vector<VecX> &grad_U_x,
        std::vector<VecX> &grad_U_y, std::vector<VecX> &grad_U_z) const;

    // =========================================================================
    // Divergence operator
    // =========================================================================

    /// @brief Compute weak divergence of vector field
    /// @details Uses DG formulation: (div_h F, v) = -(F, grad v) + <F.n, v>
    /// @param F_x, F_y, F_z Vector field components at DOFs
    /// @param face_connections Face connectivity
    /// @param numerical_flux Numerical flux for interface
    /// @param bc Boundary conditions
    /// @param[out] div_F Divergence at DOFs
    void divergence(
        const std::vector<VecX> &F_x, const std::vector<VecX> &F_y,
        const std::vector<VecX> &F_z,
        const std::vector<std::vector<FaceConnection>> &face_connections,
        const NumericalFluxFunc &numerical_flux,
        const std::vector<BoundaryCondition> &bc,
        std::vector<VecX> &div_F) const;

    // =========================================================================
    // Full RHS computation for conservation laws
    // =========================================================================

    /// @brief Compute RHS for conservation law: dU/dt = -div(F(U)) + S
    /// @param U Solution state (per element)
    /// @param physical_flux Physical flux function F(U)
    /// @param numerical_flux Numerical flux function F*(U_L, U_R, n)
    /// @param source_term Source term function S(U, x)
    /// @param mesh Mesh geometry
    /// @param face_connections Face connectivity
    /// @param bc Boundary conditions
    /// @param[out] rhs Right-hand side at DOFs
    void compute_rhs(
        const std::vector<VecX> &U, const PhysicalFluxFunc &physical_flux,
        const NumericalFluxFunc &numerical_flux,
        const std::function<VecX(const VecX &, const Vec3 &)> &source_term,
        const std::vector<std::vector<FaceConnection>> &face_connections,
        const std::vector<BoundaryCondition> &bc, std::vector<VecX> &rhs) const;

    // =========================================================================
    // Laplacian operator (for diffusion)
    // =========================================================================

    /// @brief Compute DG Laplacian using BR2 or LDG method
    /// @param U Scalar field at DOFs
    /// @param diffusivity Diffusion coefficient
    /// @param face_connections Face connectivity
    /// @param bc Boundary conditions
    /// @param[out] lap_U Laplacian at DOFs
    void laplacian(
        const std::vector<VecX> &U, Real diffusivity,
        const std::vector<std::vector<FaceConnection>> &face_connections,
        const std::vector<BoundaryCondition> &bc,
        std::vector<VecX> &lap_U) const;

    // =========================================================================
    // Non-conforming interface handling
    // =========================================================================

    /// Register a non-conforming face connection
    void register_nonconforming_interface(const FaceConnection &conn);

    /// Build mortar operators (call after registering all interfaces)
    void build_mortar_operators();

    /// Check if mortar elements are being used
    bool uses_mortar() const { return use_mortar_; }

    /// Get mortar interface manager (for external flux computation)
    const MortarInterfaceManager *mortar_manager() const {
        return mortar_manager_.get();
    }

private:
    int order_;
    bool use_mortar_;

    HexahedronBasis basis_;
    GaussQuadrature3D quad_;
    DG3DElementOperator elem_op_;

    // Mortar interface manager
    std::unique_ptr<MortarInterfaceManager> mortar_manager_;

    // Helper methods
    void compute_interface_flux_conforming(
        int elem_left, int face_left, int elem_right, int face_right,
        const VecX &U_left, const VecX &U_right, const Vec3 &normal,
        const NumericalFluxFunc &numerical_flux, VecX &rhs_left,
        VecX &rhs_right) const;

    void compute_interface_flux_nonconforming(
        const FaceConnection &conn, const std::vector<VecX> &U,
        const NumericalFluxFunc &numerical_flux, std::vector<VecX> &rhs) const;

    void compute_boundary_flux(
        int elem, int face_id, const VecX &U_interior, const Vec3 &normal,
        const BoundaryCondition &bc, Real time, const Vec3 &face_center,
        const NumericalFluxFunc &numerical_flux, VecX &rhs) const;
};

// =============================================================================
// Metric terms for curvilinear coordinates
// =============================================================================

/// @brief Compute geometric factors for curvilinear element
/// @details Given physical coordinates at nodes, compute Jacobian and metric
/// terms
struct GeometricFactors {
    std::vector<Mat3> jacobian;     ///< Jacobian matrix at each quad point
    std::vector<Mat3> jacobian_inv; ///< Inverse Jacobian
    VecX det_J;                     ///< Jacobian determinant
    std::array<VecX, 6> face_det_J; ///< Face Jacobian determinants

    /// Compute geometric factors from physical node positions
    static GeometricFactors compute(
        const HexahedronBasis &basis, const GaussQuadrature3D &quad,
        const std::vector<Vec3> &physical_nodes);
};

/// @brief Transform gradient from reference to physical coordinates
/// grad_physical = J^{-T} * grad_reference
inline Mat3 transform_gradient(const Mat3 &grad_ref, const Mat3 &jacobian_inv) {
    return jacobian_inv.transpose() * grad_ref;
}

/// @brief Compute contravariant metric tensor for flux transformation
/// G = J^{-1} * J^{-T} * det(J)
inline Mat3 contravariant_metric(const Mat3 &jacobian_inv, Real det_J) {
    return det_J * jacobian_inv * jacobian_inv.transpose();
}

// =============================================================================
// Sigma-coordinate specific operators
// =============================================================================

/// @brief Operators specialized for sigma-coordinate ocean modeling
/// @details Handles terrain-following coordinates with moving free surface
class SigmaCoordinateOperators {
public:
    /// @brief Construct sigma coordinate operators
    /// @param basis Hexahedron basis
    /// @param quad Volume quadrature
    SigmaCoordinateOperators(
        const HexahedronBasis &basis, const GaussQuadrature3D &quad);

    /// @brief Compute horizontal gradient in sigma coordinates
    /// @details Accounts for sigma-coordinate metric terms:
    ///          du/dx|_z = du/dx|_sigma - (dsigma/dx) * du/dsigma
    void horizontal_gradient(
        const VecX &U,
        const VecX &eta,     // Free surface elevation
        const VecX &h,       // Bathymetry
        const VecX &deta_dx, // Free surface gradient
        const VecX &deta_dy,
        const VecX &dh_dx, // Bathymetry gradient
        const VecX &dh_dy, VecX &dU_dx, VecX &dU_dy) const;

    /// @brief Compute vertical gradient in sigma coordinates
    /// @details du/dz = (1/H) * du/dsigma where H = eta + h
    void vertical_gradient(
        const VecX &U, const VecX &eta, const VecX &h, VecX &dU_dz) const;

    /// @brief Compute sigma-coordinate divergence
    /// @details Accounts for time-varying water column depth
    void sigma_divergence(
        const VecX &Hu,    // H * u (x-transport)
        const VecX &Hv,    // H * v (y-transport)
        const VecX &omega, // Sigma velocity
        const VecX &H,     // Water column depth
        VecX &div) const;

    /// @brief Update metrics when free surface changes (steady state, no time
    /// derivative)
    void update_sigma_metrics(
        const VecX &eta, const VecX &h, const VecX &deta_dx,
        const VecX &deta_dy, const VecX &dh_dx, const VecX &dh_dy);

    /// @brief Update metrics with time derivative for ALE formulation
    /// @param deta_dt Time derivative of free surface elevation
    void update_sigma_metrics_with_time(
        const VecX &eta, const VecX &h, const VecX &deta_dx,
        const VecX &deta_dy, const VecX &dh_dx, const VecX &dh_dy,
        const VecX &deta_dt);

    /// @brief Compute ALE correction term for material derivative
    /// @details In ALE formulation, the material derivative becomes:
    ///          D/Dt|_moving = D/Dt|_fixed - w_mesh * d/dz
    ///          This computes the correction: -w_mesh * dU/dz
    /// @param U Field values at DOFs
    /// @param w_mesh Mesh velocity (dz/dt at constant sigma)
    /// @param[out] correction ALE correction term to add to time derivative
    void
    ale_correction(const VecX &U, const VecX &w_mesh, VecX &correction) const;

    /// @brief Compute full material derivative in moving coordinates
    /// @details DU/Dt = dU/dt + u*dU/dx + v*dU/dy + (omega -
    /// w_mesh/H)*dU/dsigma
    /// @param U Field at DOFs
    /// @param dU_dt Partial time derivative
    /// @param u, v Horizontal velocities
    /// @param omega Sigma velocity (dsigma/dt following fluid)
    /// @param w_mesh Mesh velocity
    /// @param[out] material_deriv Full material derivative
    void material_derivative(
        const VecX &U, const VecX &dU_dt, const VecX &u, const VecX &v,
        const VecX &omega, const VecX &w_mesh, VecX &material_deriv) const;

    /// @brief Get cached dsigma/dt values
    const VecX &dsigma_dt() const { return dsigma_dt_; }

    /// @brief Get cached H values
    const VecX &H() const { return H_; }

    /// @brief Get cached H_inv values
    const VecX &H_inv() const { return H_inv_; }

private:
    const HexahedronBasis &basis_;
    const GaussQuadrature3D &quad_;

    // Sigma-coordinate metric terms (cached for efficiency)
    VecX H_;         // Total water depth H = eta + h
    VecX H_inv_;     // 1/H
    VecX dsigma_dx_; // Metric term dsigma/dx
    VecX dsigma_dy_; // Metric term dsigma/dy
    VecX dsigma_dt_; // Metric term dsigma/dt (for ALE)
    VecX sigma_;     // Sigma values at DOFs
};

} // namespace drifter
