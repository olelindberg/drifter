# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DRIFTER is a 3D Discontinuous Galerkin (DG) adaptive multi-resolution coastal ocean circulation model written in C++20. It implements the primitive equations for ocean modeling with terrain-following sigma coordinates.

## Project Conventions

- Always separate I/O code from algorithmic code
- Never add "Co-Authored-By" lines to git commit messages
- Always output per-element interpolations to VTK (never uniform global mesh)
- Never use fallback methods of any kind

## Build Commands

```bash
# Configure (from project root)
cmake -B build

# Build (use max 6 parallel jobs to avoid memory issues)
LD_LIBRARY_PATH=/home/ole/.local/lib cmake --build build --parallel 6

# Run all tests
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build --output-on-failure

# Run only unit tests
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -L unit --output-on-failure

# Run only integration tests
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -L integration --output-on-failure

# Run a single test by name
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -R "TestName" --output-on-failure

# List available test names
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -N

# Run test executable directly (shows all logging output)
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_unit_tests
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_integration_tests

# Run specific test(s) with gtest filter
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_unit_tests --gtest_filter="TestName*"

# Run tests with verbose output
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -V
```

The `LD_LIBRARY_PATH` is needed for GDAL and other libraries installed in `/home/ole/.local/lib`.

## Code Formatting

The project uses clang-format for code style. CI checks formatting with clang-format-15:
```bash
# Check formatting (dry run)
find src include -name '*.cpp' -o -name '*.hpp' | xargs clang-format-15 --dry-run --Werror

# Apply formatting
find src include -name '*.cpp' -o -name '*.hpp' | xargs clang-format-15 -i
```

## Architecture

### Core Data Types (`include/core/types.hpp`)

- `Real` = double precision
- `VecX`, `MatX` = Eigen dynamic vectors/matrices
- `Vec2`, `Vec3`, `Mat2`, `Mat3` = Fixed-size 2D/3D vectors/matrices
- `SpMat` = Eigen sparse matrix (used in FEM assembly and solvers)
- `Index` = int64_t for element indexing
- Hexahedron geometry constants in `Hex::` namespace
- `SigmaCoord` for terrain-following vertical coordinates

### Module Structure

**dg/** - Discontinuous Galerkin basis functions and operators
- `HexahedronBasis` - 3D tensor-product Lagrange basis on LGL/GL nodes
- `LagrangeBasis1D` - 1D basis with derivative matrices
- `BernsteinBasis` - Bernstein polynomials for bounded interpolation (used in SeabedVTKWriter)
- Staggered grids: LGL (Legendre-Gauss-Lobatto) for velocities, GL (Gauss-Legendre) for tracers

**mesh/** - 3D mesh representation and generation
- `OctreeAdapter` - Directional (anisotropic) AMR octree with per-axis refinement levels
- `ElementBounds` - Physical bounds of hexahedral elements
- `FaceConnection` - Face connectivity with conforming/non-conforming support
- `GeoTiffReader` - Bathymetry loading via GDAL
- `CoastlineRefinement` - R-tree based coastline-adaptive mesh refinement

**bathymetry/** - 2D bathymetry surface fitting (uses `QuadtreeAdapter`, not `OctreeAdapter`)
- `QuadtreeAdapter` - 2D AMR quadtree for bathymetry mesh refinement
- `CGCubicBezierBathymetrySmoother` - Cubic Bezier (C¹ continuity)
- `CGLinearBezierBathymetrySmoother` - Linear Bezier (C⁰ continuity)
- See detailed documentation in the "CG Bezier Bathymetry Smoother" section below

**physics/** - Ocean physics
- `PrimitiveEquations` - 3D baroclinic equations in sigma coordinates
- `ModeSplitting` - Barotropic/baroclinic mode splitting
- `EquationOfState` - Linear/UNESCO density from T,S
- Prognostic variables: Hu, Hv (momentum), HT, HS (tracers), eta (surface elevation)

**solver/** - Time integration
- `SimulationDriver` - Main simulation loop
- `TimeStepper` - RK3-SSP, RK4 time integration with adaptive CFL

**io/** - Input/output
- `VTKWriter` - VTU output with high-order Lagrange hexahedra support
- `SeabedVTKWriter` - High-resolution seabed surface visualization
- `ZarrWriter` - Zarr v3 output (optional, requires zarrs_ffi)

**amr/** - Adaptive mesh refinement
- `Refinement` - Error estimation, element marking, solution projection
- Directional (anisotropic) refinement following octree structure
- L2 projection for solution transfer during refinement/coarsening

**flux/** - Numerical fluxes for DG
- `NumericalFlux` - Interface flux computations (Lax-Friedrichs, HLLC, Roe, Central)
- Provides upwinding and stability at element interfaces

**parallel/** - MPI parallelization
- `DomainDecomposition` - Spatial decomposition for MPI
- `HaloExchange` - Ghost cell communication at partition boundaries

### Key Patterns

1. **Element data storage**: Per-element `VecX` vectors containing all DOFs
   - DOF indexing: `i + (order+1) * (j + (order+1) * k)` for tensor-product basis

2. **Face numbering** (hexahedra):
   - 0: xi=-1, 1: xi=+1, 2: eta=-1, 3: eta=+1, 4: zeta=-1 (bottom), 5: zeta=+1 (top)

3. **Non-conforming interfaces**: Mortar projection for h-adaptivity at 2:1 size ratios

4. **VTK output**: Uses VTK 5.1 format with `VTK_LAGRANGE_HEXAHEDRON` (type 72) for high-order elements

5. **Dirichlet boundary conditions**: Use row/column elimination in the stiffness matrix Q:
   - For each Dirichlet DOF `i`: move coupling terms to RHS, then set `Q(i,:) = 0`, `Q(:,i) = 0`, `Q(i,i) = 1`, `c(i) = -value`
   - For C² constraints with Dirichlet DOFs: compute `b_c2 = -A_c2 * x_dir`, then zero Dirichlet columns in A_c2

### Bezier Bathymetry Smoother (`bathymetry/`)

CG (Continuous Galerkin) Bezier smoothers fit Bezier surfaces to bathymetry data. DOFs at element boundaries are shared for automatic C⁰ continuity.

**Available Smoothers:**
| Class | Degree | Continuity | DOFs/element |
|-------|--------|------------|--------------|
| `CGCubicBezierBathymetrySmoother` | Cubic (3) | C¹ | 16 |
| `CGLinearBezierBathymetrySmoother` | Linear (1) | C⁰ | 4 |

**Adaptive variants:** `AdaptiveCGCubicBezierSmoother`, `AdaptiveCGLinearBezierSmoother` - error-driven mesh refinement.

**Optimization (ShipMesh Formulation):**

Uses smoothness-first formulation: `Q = α·H + λ·(BᵀWB + εI)` where H is thin plate Hessian, BᵀWB is data fitting, and α normalizes scales. This avoids boundary oscillations from standard least-squares + regularization.

- `λ = 0`: Pure thin plate (soap film, ignores data)
- `λ → ∞`: Approaches least-squares fit
- Typical: `λ = 0.01` (smooth) to `λ = 100` (close to data)

Solved via KKT system with constraint projection for exact satisfaction.

**Key Configuration (`CGCubicBezierSmootherConfig`):**
- `lambda` - Data fitting weight
- `enable_edge_constraints` / `edge_ngauss` - C¹ edge derivative constraints
- `enable_natural_bc` - Zero normal curvature at boundaries (default, preferred)
- `enable_boundary_dirichlet` - Pin corners to data (can cause oscillations)

**Non-conforming meshes:** Hanging node constraints via de Casteljau subdivision ensure continuity at 2:1 T-junctions.

**Multigrid Preconditioner:** `BezierMultigridPreconditioner` accelerates CG solves using geometric multigrid with:
- V-cycle on natural quadtree hierarchy (coarsening via Morton code parent grouping)
- Colored multiplicative Schwarz smoother (4-8 colors via graph coloring)
- MG levels = max_tree_depth - min_tree_level + 1

**Transfer Operator Strategies (`TransferOperatorStrategy`):**
| Strategy | P (prolongation) | R (restriction) | Weights | Use case |
|----------|------------------|-----------------|---------|----------|
| `L2Projection` | R^T | M_c^{-1} P^T M_f | ±large (up to ±4000) | Default, symmetric |
| `BezierSubdivision` | de Casteljau | P^T normalized | Non-negative [0,1] | Adaptive meshes |

Use `BezierSubdivision` for adaptive meshes where L2 projection's large negative weights cause instability.

**Coarse Grid Strategies (`CoarseGridStrategy`):**
| Strategy | Method | Use case |
|----------|--------|----------|
| `Galerkin` | A_c = R * A_f * P | Default, automatic |
| `CachedRediscretization` | Direct assembly from cached element matrices | With BezierSubdivision |

**Recommended configuration for adaptive meshes:**
```cpp
MultigridConfig config;
config.min_tree_level = 0;  // Coarsest level (1x1 element)
config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
config.coarse_grid_strategy = CoarseGridStrategy::CachedRediscretization;
```

**Verification Report:** See `docs/cg_bezier_solver_verification.md` for solver comparison results (Direct vs Iterative+LU vs Iterative+MG) and iterative method benchmarks. Regenerate figures with `./docs/regenerate_figures.sh`.

## Testing

Tests use GoogleTest framework with fixtures in `tests/test_utils.hpp`:
- `TOLERANCE = 1e-12` for strict floating-point comparisons
- `LOOSE_TOLERANCE = 1e-8` for physics tests

Integration test fixture `SimulationTest` in `tests/integration/test_integration_fixtures.hpp` provides:
- Temporary output directory management
- VTK point ordering utilities for Lagrange hexahedra

Test organization:
- `tests/unit/` - Component-level tests
- `tests/integration/` - Full pipeline tests (mesh → physics → output)

### Integration Tests

Test files in `tests/integration/`:

| File | Purpose |
|------|---------|
| `test_cg_cubic_bezier_bathymetry_smoother.cpp` | CG cubic Bezier (C¹), edge constraints, non-conforming meshes |
| `test_cg_linear_bezier_bathymetry_smoother.cpp` | CG linear Bezier (C⁰), DOF sharing |
| `test_adaptive_cg_cubic_bezier_smoother.cpp` | Adaptive refinement for CG cubic smoother |
| `test_adaptive_cg_linear_bezier_smoother.cpp` | Adaptive refinement for CG linear smoother |
| `test_multi_source_bathymetry.cpp` | Multi-source bathymetry loading and blending |
| `test_bathymetry.cpp` | GeoTIFF loading, coastline-adaptive refinement, seabed VTK output |
| `test_mesh.cpp` | Octree mesh creation, face connectivity |
| `test_dg_operators.cpp` | Gradient/divergence operators, mass matrix, face interpolation, discrete Green's identity |
| `test_initial_conditions.cpp` | Quiescent, Kelvin wave, lock exchange initial conditions |
| `test_conservation.cpp` | Mass, flux, energy, enstrophy, tracer conservation |
| `test_simulation_full.cpp` | Full simulation pipeline with diagnostics and VTK output |
| `test_vtk_output.cpp` | VTK writer, PVD time series |

Run specific tests with gtest filter: `./build/tests/drifter_integration_tests --gtest_filter="CG*Bezier*"`

## Dependencies

Required: Eigen3, Boost (Geometry), GDAL
Optional (default ON): MPI, OpenMP, GTest (for tests)
Optional: VTK, zarrs_ffi (Zarr output), netCDF, HDF5

CMake build options (set with `-D`):
- `DRIFTER_USE_MPI=ON` - MPI parallelization (default ON)
- `DRIFTER_USE_OPENMP=ON` - OpenMP threading (default ON)
- `DRIFTER_USE_CUDA=OFF` - CUDA GPU acceleration (default OFF)
- `DRIFTER_USE_ZARR=ON` - Zarr v3 output via zarrs_ffi (default ON)
- `DRIFTER_USE_VTK=ON` - VTK output (default ON)
- `ZARRS_FFI_DIR=/path` - Custom path to zarrs_ffi library