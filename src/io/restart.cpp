// Restart I/O - Stub implementation
// Checkpoint/restart functionality

#include "core/types.hpp"

namespace drifter {

// Placeholder for restart file handling
//
// Key features:
// 1. Save complete state for restart:
//    - Mesh (including AMR tree structure)
//    - Solution (all variables at all DOFs)
//    - Time and timestep info
//    - Random state (if stochastic forcing used)
//
// 2. Load and resume simulation:
//    - Verify mesh consistency
//    - Interpolate if polynomial order changed
//    - Handle parallel redistribution
//
// Format options:
// - Zarr (preferred for parallel)
// - HDF5
// - Binary (fastest but not portable)

}  // namespace drifter
