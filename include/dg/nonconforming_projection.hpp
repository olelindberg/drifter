#pragma once

// Non-conforming interface projection utilities
//
// For bounded (Bernstein) interpolation to be continuous across non-conforming
// interfaces, fine elements must use values interpolated from the coarse
// element's polynomial at boundary faces, rather than independently sampled
// values.
//
// This module provides utilities to project coarse element data onto fine
// elements at non-conforming interfaces.

#include "core/types.hpp"
#include "dg/bernstein_basis.hpp"
#include "mesh/octree_adapter.hpp"
#include <vector>

namespace drifter {

/// @brief Project coarse element data onto fine elements at non-conforming
/// interfaces
///
/// For each fine element that has a coarser neighbor, this function overwrites
/// the DOF values on the shared face with values interpolated from the coarse
/// element's polynomial. This ensures continuity when using Bernstein
/// interpolation.
///
/// The data is assumed to be 2D (horizontal) bathymetry values, stored at
/// (order+1)^2 DOF positions per element (one layer, not full 3D).
///
/// @param mesh The octree mesh with element connectivity
/// @param element_data Per-element 2D data at DOFs (modified in-place)
/// @param order Polynomial order
/// @param method Interpolation method to use (Lagrange or Bernstein)
void project_coarse_to_fine_2d(
    const OctreeAdapter &mesh, std::vector<VecX> &element_data, int order,
    SeabedInterpolation method);

/// @brief Project coarse element data onto fine elements at non-conforming
/// interfaces (3D)
///
/// Similar to project_coarse_to_fine_2d but for full 3D element data with
/// (order+1)^3 DOFs per element. Only the DOFs on the shared face are modified.
///
/// @param mesh The octree mesh with element connectivity
/// @param element_data Per-element 3D data at DOFs (modified in-place)
/// @param order Polynomial order
/// @param method Interpolation method to use (Lagrange or Bernstein)
void project_coarse_to_fine_3d(
    const OctreeAdapter &mesh, std::vector<VecX> &element_data, int order,
    SeabedInterpolation method);

} // namespace drifter
