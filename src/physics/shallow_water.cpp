// Shallow Water Equations - Stub implementation
// 2D depth-averaged shallow water equations

#include "core/types.hpp"

namespace drifter {

// Placeholder for shallow water equations
// Used for barotropic mode in mode-split time stepping
//
// Equations:
//   ∂η/∂t + ∂(Hu)/∂x + ∂(Hv)/∂y = 0
//   ∂(Hu)/∂t + ∂(Hu² + gh²/2)/∂x + ∂(Huv)/∂y = fHv - gH∂η/∂x + τ_x/ρ - C_d|u|u
//   ∂(Hv)/∂t + ∂(Huv)/∂x + ∂(Hv² + gh²/2)/∂y = -fHu - gH∂η/∂y + τ_y/ρ - C_d|u|v
//
// where H = η + h is total water depth

} // namespace drifter
