// Riemann Solvers - Stub implementation
// Approximate Riemann solvers for numerical flux

#include "core/types.hpp"

namespace drifter {

// Placeholder for Riemann solver implementations
// These compute numerical flux at element interfaces
//
// Implemented solvers:
// 1. Local Lax-Friedrichs (Rusanov)
//    F* = 0.5*(F_L + F_R) - 0.5*λ_max*(U_R - U_L)
//
// 2. HLL (Harten-Lax-van Leer)
//    Two-wave approximation
//
// 3. HLLC (HLL with Contact)
//    Adds middle wave for better contact resolution
//
// 4. Roe (linearized)
//    Exact for linear problems
//
// For ocean models, HLLC or LLF typically used

}  // namespace drifter
