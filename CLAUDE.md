# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DRIFTER is a 3D Discontinuous Galerkin (DG) adaptive multi-resolution coastal ocean circulation model written in C++20. It implements the primitive equations for ocean modeling with terrain-following sigma coordinates.

## Build Commands

```bash
# Configure (from project root)
cmake -B build

# Build
LD_LIBRARY_PATH=/home/ole/.local/lib cmake --build build --parallel

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
```

The `LD_LIBRARY_PATH` is needed for GDAL and other libraries installed in `/home/ole/.local/lib`.

## Architecture

### Core Data Types (`include/core/types.hpp`)

- `Real` = double precision
- `VecX`, `MatX` = Eigen dynamic vectors/matrices
- `Vec3`, `Mat3` = 3D vectors/matrices
- `Index` = int64_t for element indexing
- Hexahedron geometry constants in `Hex::` namespace
- `SigmaCoord` for terrain-following vertical coordinates

### Module Structure

**dg/** - Discontinuous Galerkin basis functions and operators
- `HexahedronBasis` - 3D tensor-product Lagrange basis on LGL/GL nodes
- `LagrangeBasis1D` - 1D basis with derivative matrices
- `BernsteinBasis` - Bernstein polynomials for bounded interpolation (used in SeabedVTKWriter)
- Staggered grids: LGL (Legendre-Gauss-Lobatto) for velocities, GL (Gauss-Legendre) for tracers

**mesh/** - Mesh representation and generation
- `OctreeAdapter` - Directional (anisotropic) AMR octree with per-axis refinement levels
- `ElementBounds` - Physical bounds of hexahedral elements
- `FaceConnection` - Face connectivity with conforming/non-conforming support
- `GeoTiffReader` - Bathymetry loading via GDAL
- `CoastlineRefinement` - R-tree based coastline-adaptive mesh refinement

**physics/** - Ocean physics
- `PrimitiveEquations` - 3D baroclinic equations in sigma coordinates
- `ModeSplitting` - Barotropic/baroclinic mode splitting
- `EquationOfState` - Linear/UNESCO density from T,S
- Prognostic variables: Hu, Hv (momentum), HT, HS (tracers), eta (surface elevation)

**solver/** - Time integration
- `SimulationDriver` - Main simulation loop
- `TimeStepper` - RK3-SSP, RK4 time integration with adaptive CFL

**io/** - Input/output
- `VTKWriter` - VTU/Legacy VTK output with high-order Lagrange hexahedra support
- `SeabedVTKWriter` - High-resolution seabed surface visualization
- `ZarrWriter` - Zarr v3 output (optional, requires zarrs_ffi)

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

Fits quintic Bezier surfaces (36 DOFs per element, 6×6 control points) to bathymetry data with C² continuity at element interfaces.

**Key files:**
- `BezierBathymetrySmoother` - Main solver using constrained QP (KKT system)
- `BezierBasis2D` - Quintic tensor-product Bernstein basis evaluation
- `ThinPlateHessian` - Curvature regularization energy `E = ∫[(z_uu + z_vv)² + 2z_uv²]`
- `BezierC2ConstraintBuilder` - C² continuity constraints at shared vertices
- `BezierDataFittingAssembler` - Least-squares data fitting term

**Configuration (`BezierSmootherConfig`):**
- `lambda` - Data fitting weight (0 = pure thin plate/soap film, higher = closer to data)
- `gradient_weight` - First derivative penalty (slope smoothing)
- `enable_boundary_dirichlet` - Pin boundary corner DOFs to input data

**Critical design decisions:**
1. **Dirichlet BCs only at corners**: Apply Dirichlet constraints only at corner DOFs (vertices), not interior edge DOFs. This preserves C² constraints at shared boundary vertices, which involve derivative terms depending on multiple edge DOFs.

2. **Thin plate scaling for physical coordinates**: The energy must account for element size (dx, dy):
   - `scale_uu_uu = dy/dx³` (for z_uu² term)
   - `scale_vv_vv = dx/dy³` (for z_vv² term)
   - `scale_uu_vv = 1/(dx·dy)` (for cross term)
   - `scale_uv_uv = 2/(dx·dy)` (for z_uv² term)

3. **C² constraints at vertices (9 per shared vertex)**: z, z_u, z_v, z_uu, z_uv, z_vv, z_uuv, z_uvv, z_uuvv - with proper scaling by element size.

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

## Dependencies

Required: Eigen3, Boost (Geometry), MPI, OpenMP, GTest
Optional: GDAL (geospatial), VTK, zarrs_ffi (Zarr output)


## User notes:

Always separate i/o code from algorithmic code