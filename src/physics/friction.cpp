// Bottom Friction - Stub implementation
// Seabed drag parameterization

#include "core/types.hpp"

namespace drifter {

// Placeholder for bottom friction
//
// Quadratic drag law:
//   τ_b = ρ C_d |u_b| u_b
//
// where C_d is the drag coefficient (typically 0.001 - 0.003)
//       u_b is the near-bottom velocity
//
// Options:
// - Constant C_d
// - Roughness-dependent C_d via log-law:
//   C_d = [κ / ln(z_b/z_0)]²
//   where z_0 is bottom roughness length

}  // namespace drifter
