// Physical Flux Functions - Stub implementation
// Flux vectors for hyperbolic systems

#include "core/types.hpp"

namespace drifter {

// Placeholder for flux function evaluation
//
// For shallow water equations:
//   F(U) = [Hu, Hu² + gH²/2, Huv]ᵀ  (x-direction)
//   G(U) = [Hv, Huv, Hv² + gH²/2]ᵀ  (y-direction)
//
// For primitive equations (sigma coords):
//   Includes metric terms from coordinate transformation
//
// Key functions:
// - compute_physical_flux(U) -> (F, G, W)
// - compute_max_wavespeed(U) -> |u| + c (for CFL)

} // namespace drifter
