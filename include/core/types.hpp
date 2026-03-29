#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cstdint>
#include <vector>

namespace drifter {

// Floating point precision (double for ocean modeling)
using Real = double;

// Vector and matrix types
using Vec2  = Eigen::Vector2d;
using Vec3  = Eigen::Vector3d;
using VecX  = Eigen::VectorXd;
using Mat2  = Eigen::Matrix2d;
using Mat3  = Eigen::Matrix3d;
using MatX  = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<Real>;

// 3D tensor type (array of 3 matrices for x, y, z components)
using Tensor3 = std::array<MatX, 3>;

// Reference element coordinates (xi, eta, zeta) in [-1, 1]^3
using RefCoord = Vec3;

// Index types
using Index      = std::int64_t;
using LocalIndex = std::int32_t;

// Element types for unstructured mesh
enum class ElementType : std::uint8_t {
  Triangle, // 2D triangular element
  Quadrilateral, // 2D quad element
  Tetrahedron, // 3D tet element
  Hexahedron, // 3D hex element
  Prism, // 3D prismatic element (common in ocean models)
  Pyramid // 3D pyramid element
};

// Hexahedron geometry constants
namespace Hex {
constexpr int VERTICES = 8;
constexpr int FACES    = 6;
constexpr int EDGES    = 12;

// Face node ordering (VTK convention)
// Face 0: xi = -1, Face 1: xi = +1
// Face 2: eta = -1, Face 3: eta = +1
// Face 4: zeta = -1, Face 5: zeta = +1
constexpr std::array<std::array<int, 4>, 6> FACE_NODES = {{
    {0, 3, 7, 4}, // xi = -1
    {1, 2, 6, 5}, // xi = +1
    {0, 1, 5, 4}, // eta = -1
    {2, 3, 7, 6}, // eta = +1
    {0, 1, 2, 3}, // zeta = -1 (bottom)
    {4, 5, 6, 7} // zeta = +1 (top)
}};

// Outward normals in reference space
inline Vec3 face_normal(int face_id) {
  switch (face_id) {
  case 0:
    return Vec3{-1, 0, 0};
  case 1:
    return Vec3{+1, 0, 0};
  case 2:
    return Vec3{0, -1, 0};
  case 3:
    return Vec3{0, +1, 0};
  case 4:
    return Vec3{0, 0, -1};
  case 5:
    return Vec3{0, 0, +1};
  default:
    return Vec3{0, 0, 0};
  }
}

// DOF index for tensor-product basis at node (i, j, k) with order p
inline constexpr int dof_index(int i, int j, int k, int order) { return i + (order + 1) * (j + (order + 1) * k); }

// Number of DOFs for order p
inline constexpr int num_dofs(int order) { return (order + 1) * (order + 1) * (order + 1); }
} // namespace Hex

// Polynomial orders for p-adaptivity
constexpr int MAX_POLYNOMIAL_ORDER = 8;

// Sigma-coordinate types for terrain-following vertical
struct SigmaCoord {
  Real sigma; // Normalized vertical: -1 (bottom) to 0 (surface)
  Real eta; // Free surface elevation
  Real h; // Bathymetry depth (positive downward)

    // Total water column depth
  Real H() const { return eta + h; }

    // Physical z-coordinate from sigma
  Real z() const { return eta + sigma * H(); }

    // Metric term dz/dsigma
  Real dz_dsigma() const { return H(); }
};

// Nodal grid type for staggered arrangement
enum class NodalGrid : std::uint8_t {
  LGL, // Legendre-Gauss-Lobatto (includes boundary points) - for velocity
  GL // Gauss-Legendre (interior only) - for tracers
};

// State vector indices for shallow water equations
struct SWE {
  static constexpr int H     = 0; // Water depth
  static constexpr int HU    = 1; // x-momentum
  static constexpr int HV    = 2; // y-momentum
  static constexpr int NVARS = 3;
};

// State vector indices for 3D baroclinic equations in sigma coordinates
// Velocities (u, v, omega) on LGL nodes, tracers (T, S) on GL nodes
struct Baroclinic {
    // Prognostic variables (on LGL grid)
  static constexpr int HU             = 0; // H * u (x-momentum per unit area)
  static constexpr int HV             = 1; // H * v (y-momentum per unit area)
  static constexpr int NVARS_VELOCITY = 2;

    // Tracers (on GL grid)
  static constexpr int HT           = 0; // H * T (temperature)
  static constexpr int HS           = 1; // H * S (salinity)
  static constexpr int NVARS_TRACER = 2;

    // Diagnostic variables
    // omega (sigma velocity) - diagnosed from continuity
    // w (physical vertical velocity) - computed from omega
    // eta (free surface) - from barotropic mode
    // p (pressure) - hydrostatic from density
};

// Physical constants for ocean modeling
struct PhysicalConstants {
  static constexpr Real g       = 9.81; // Gravitational acceleration [m/s^2]
  static constexpr Real omega   = 7.2921e-5; // Earth's angular velocity [rad/s]
  static constexpr Real rho_0   = 1025.0; // Reference density [kg/m^3]
  static constexpr Real cp      = 3985.0; // Specific heat capacity [J/(kg·K)]
  static constexpr Real T_0     = 10.0; // Reference temperature [°C]
  static constexpr Real S_0     = 35.0; // Reference salinity [PSU]
  static constexpr Real alpha_T = 2.0e-4; // Thermal expansion coeff [1/K]
  static constexpr Real beta_S  = 7.6e-4; // Haline contraction coeff [1/PSU]
};

} // namespace drifter
