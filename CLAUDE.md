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

**mesh/** - 3D mesh representation and generation
- `OctreeAdapter` - Directional (anisotropic) AMR octree with per-axis refinement levels
- `ElementBounds` - Physical bounds of hexahedral elements
- `FaceConnection` - Face connectivity with conforming/non-conforming support
- `GeoTiffReader` - Bathymetry loading via GDAL
- `CoastlineRefinement` - R-tree based coastline-adaptive mesh refinement

**bathymetry/** - 2D bathymetry surface fitting (uses `QuadtreeAdapter`, not `OctreeAdapter`)
- `QuadtreeAdapter` - 2D AMR quadtree for bathymetry mesh refinement
- `CGBezierBathymetrySmoother` - Quintic Bezier (C² continuity) with shared DOFs at element boundaries
- `CGCubicBezierBathymetrySmoother` - Cubic Bezier (C¹ continuity)
- `CGLinearBezierBathymetrySmoother` - Linear Bezier (C⁰ continuity)
- `CGTriharmonicBezierBathymetrySmoother` - Quintic Bezier with triharmonic energy
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

The CG (Continuous Galerkin) Bezier bathymetry smoothers fit Bezier surfaces to bathymetry data with configurable continuity. DOFs at element boundaries are shared, providing automatic C⁰ continuity.

**Available Smoothers:**
- `CGBezierBathymetrySmoother` - Quintic (degree 5), C² continuity, 36 DOFs/element
- `CGCubicBezierBathymetrySmoother` - Cubic (degree 3), C¹ continuity, 16 DOFs/element
- `CGLinearBezierBathymetrySmoother` - Linear (degree 1), C⁰ continuity, 4 DOFs/element
- `CGTriharmonicBezierBathymetrySmoother` - Quintic with triharmonic (3rd order) energy

**Optimization Problem (ShipMesh Formulation):**

The smoother uses the ShipMesh-style formulation where smoothness is the primary objective and data fitting is weighted by λ. This approach prioritizes smooth surfaces and avoids boundary oscillations that occur with standard least-squares + regularization.

```
minimize    ½ xᵀQx + cᵀx
subject to  Ax = b       (C² continuity constraints)
```

where:
- `x` = vector of all Bezier control points (36 per element)
- `Q = α·H + λ·(BᵀWB + εI)` where:
  - `H` = thin plate Hessian (smoothness)
  - `BᵀWB` = weighted least-squares normal matrix
  - `ε = 1e-4` = ridge regularization (Tikhonov)
  - `α = ||BᵀWB + εI|| / ||H||` = scale normalization factor (see below)
- `c = -λ·BᵀWd` where `d` is the data vector
- `A` = constraint matrix (edge derivative constraints + hanging node constraints for non-conforming meshes)
- `b` = constraint RHS (zero for interior continuity)

**Scale normalization for λ:**

The thin plate Hessian H scales with physical element size (e.g., ~1e-8 for km-scale elements),
while the data fitting matrix BᵀWB is typically ~1. Without normalization, λ would need to be
adjusted based on coordinate magnitude. The scale factor α normalizes H to have comparable
magnitude to BᵀWB, making λ scale-invariant:

```
α = ||BᵀWB + εI||_F / ||H||_F
Q = α·H + λ·(BᵀWB + εI)
```

This ensures λ controls the actual balance regardless of whether coordinates are in meters or kilometers.

**Key difference from standard regularization:**

Standard approach (causes oscillations):
```
Q = BᵀWB + λ·H    (data fitting + λ×smoothness)
```

ShipMesh approach (used here):
```
Q = α·H + λ·BᵀWB    (normalized smoothness + λ×data fitting)
```

This means:
- `λ = 0`: Pure thin plate energy (soap film surface, ignores data)
- `λ → ∞`: Approaches least-squares fit to data
- Typical values: `λ = 0.01` (smooth) to `λ = 100` (close to data)
- The scale factor α ensures these λ values work regardless of physical coordinate scale

The objective combines:
1. **Thin plate energy** (primary): `E_tp = ∫∫ (z_uu + z_vv)² + 2z_uv² du dv`
2. **Gradient penalty** (optional): `E_grad = γ·∫∫ z_u² + z_v² du dv`
3. **Data fitting** (weighted by λ): `E_data = λ·Σᵢ wᵢ(z(uᵢ,vᵢ) - dᵢ)²`
4. **Ridge regularization**: `E_ridge = λε·||x||²` (stabilizes ill-conditioned BᵀWB)

Solved via KKT system:
```
[ Q   Aᵀ ] [ x ]   [ -c ]
[ A   0  ] [ μ ] = [  b ]
```

After KKT solve, the solution is projected onto the constraint manifold to ensure exact constraint satisfaction:
```
x = x - Aᵀ(AAᵀ)⁻¹(Ax - b)
```
This corrects numerical drift when λ is large and the data fitting term dominates.

**Key files:**
- `CGBezierBathymetrySmoother` - CG quintic solver with C² continuity
- `CGCubicBezierBathymetrySmoother` - CG cubic solver with C¹ continuity
- `CGLinearBezierBathymetrySmoother` - CG linear solver with C⁰ continuity
- `BezierBasis2D` - Quintic tensor-product Bernstein basis evaluation
- `CubicBezierBasis2D` - Cubic tensor-product Bernstein basis evaluation
- `LinearBezierBasis2D` - Linear tensor-product Bernstein basis evaluation
- `ThinPlateHessian` - Curvature regularization energy `E = ∫[(z_uu + z_vv)² + 2z_uv²]`
- `BezierDataFittingAssembler` - Least-squares data fitting term

**Configuration (`CGBezierSmootherConfig`):**
- `lambda` - Data fitting weight (0 = pure thin plate/soap film, higher = closer to data)
- `gradient_weight` - First derivative penalty (slope smoothing)
- `enable_natural_bc` - Enable natural boundary conditions (default: true)
- `enable_boundary_dirichlet` - Pin boundary corner DOFs to input data (default: false, mutually exclusive with natural BC)

**Boundary Conditions:**

The biharmonic (thin plate) equation requires two boundary conditions. Two options are available:

1. **Natural boundary conditions** (default, `enable_natural_bc = true`):
   - Enforces zero normal curvature at all boundary DOFs: z_nn = 0
   - Formula: z_nn = nx² · z_xx + 2·nx·ny · z_xy + ny² · z_yy = 0
   - For axis-aligned boundaries:
     - Left/right edges (n = (±1, 0)): z_xx = 0
     - Bottom/top edges (n = (0, ±1)): z_yy = 0
   - Domain corners get TWO constraints (z_xx = 0 AND z_yy = 0) since they lie on two edges
   - This is the standard "natural spline" or "free edge" boundary condition
   - Results in smoother boundary behavior without forced values
   - The surface is determined by data fitting + C² continuity + smoothness energy

2. **Dirichlet boundary conditions** (`enable_boundary_dirichlet = true`, `enable_natural_bc = false`):
   - Pins boundary corner DOFs to match input bathymetry data
   - Applied only at 4 domain corner vertices (not interior edge DOFs)
   - Preserves C² constraints at shared boundary vertices
   - May introduce boundary oscillations if data is noisy

Natural BCs are preferred for most applications as they produce smoother surfaces and avoid boundary artifacts.

**DOF Structure (36 per element):**

Control point indexing: `dof = i + 6*j` where i,j ∈ {0,1,2,3,4,5}

```
Parameter space (u,v) ∈ [0,1]²:
  j=5: [5]  [11] [17] [23] [29] [35]   ← top edge (v=1)
  j=4: [4]  [10] [16] [22] [28] [34]
  j=3: [3]  [9]  [15] [21] [27] [33]
  j=2: [2]  [8]  [14] [20] [26] [32]
  j=1: [1]  [7]  [13] [19] [25] [31]
  j=0: [0]  [6]  [12] [18] [24] [30]   ← bottom edge (v=0)
       ↑                         ↑
     left                      right
    (u=0)                      (u=1)
```

- Edge 0 (left, u=0): DOFs [0, 1, 2, 3, 4, 5]
- Edge 1 (right, u=1): DOFs [30, 31, 32, 33, 34, 35]
- Edge 2 (bottom, v=0): DOFs [0, 6, 12, 18, 24, 30]
- Edge 3 (top, v=1): DOFs [5, 11, 17, 23, 29, 35]

### Non-conforming Mesh Constraints (Hanging Nodes)

For adaptive mesh refinement with 2:1 size ratios, "hanging nodes" occur where fine element edges meet coarse element edges. The DOF manager automatically generates hanging node constraints to ensure C⁰ continuity across non-conforming interfaces using de Casteljau subdivision.

```
Non-conforming T-junction:
+---------------+-------+
|               |   F1  |
|     Coarse    *-------+    * = hanging node
|               |   F0  |
+---------------+-------+
```

The fine element edge DOFs are constrained to interpolate the coarse element's Bezier curve via de Casteljau subdivision weights.

**Thin plate scaling for physical coordinates**: The energy must account for element size (dx, dy):
- `scale_uu_uu = dy/dx³` (for z_uu² term)
- `scale_vv_vv = dx/dy³` (for z_vv² term)
- `scale_uu_vv = 1/(dx·dy)` (for cross term)
- `scale_uv_uv = 2/(dx·dy)` (for z_uv² term)

### CG Bezier Bathymetry Smoother (`bathymetry/cg_bezier_bathymetry_smoother.hpp`)

Shares DOFs at element boundaries for automatic C⁰ continuity. Explicit constraints enforce C¹/C² derivative continuity.

**Edge Derivative Constraints (`CGBezierSmootherConfig`):**

Edge constraints enforce C²/C¹ continuity along shared element edges at Gauss quadrature points:

| Option | Description |
|--------|-------------|
| `enable_edge_constraints` | Enable edge derivative constraints (z_n, z_nn matching at Gauss points) |
| `edge_ngauss` | Number of Gauss points per edge (default: 4) |

**Recommended Configuration:**

```cpp
CGBezierSmootherConfig config;
config.lambda = 1.0;
config.enable_edge_constraints = true;  // 896 constraints for 8×8 mesh
config.edge_ngauss = 4;                 // 4 Gauss points per edge
```

Edge constraints directly address normal derivative discontinuities along element boundaries where kinks occur.

**DOF Sharing:**
```
For quintic Bezier (6×6 = 36 DOFs per element):
  - Corner DOFs (4): shared by all elements meeting at vertex
  - Edge DOFs (4 per edge, excluding corners): shared by 2 elements
  - Interior DOFs (16): unique per element

For 8×8 uniform mesh: 1681 total DOFs (vs 2304 without sharing)
```

### CG Cubic Bezier Bathymetry Smoother (`bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp`)

Continuous Galerkin variant using cubic (degree 3) Bezier with C¹ continuity. Lower DOF count than quintic, suitable when C² continuity is not required.

**Key Properties:**
- Cubic Bezier: 4×4 = 16 DOFs per element (vs 36 for quintic)
- C¹ continuity via edge derivative constraints
- Shared DOFs at element boundaries (CG assembly)

**Configuration (`CGCubicBezierSmootherConfig`):**
- `lambda` - Data fitting weight
- `gradient_weight` - First derivative penalty
- `enable_c1_edge_constraints` - Enable C¹ edge derivative constraints (z_n matching at Gauss points)
- `edge_ngauss` - Number of Gauss points per edge (default: 4)

**Usage:**
```cpp
CGCubicBezierSmootherConfig config;
config.lambda = 1.0;
config.enable_c1_edge_constraints = true;

CGCubicBezierBathymetrySmoother smoother(quadtree, config);
smoother.set_bathymetry_data(source);
smoother.solve();
```

### Adaptive CG Bezier Smoother

Error-driven adaptive mesh refinement coupled with CG Bezier smoothing. Iteratively refines elements where fitting error exceeds threshold while maintaining continuity.

**Available Variants:**
- `AdaptiveCGBezierSmoother` - Adaptive quintic (C² continuity)
- `AdaptiveCGCubicBezierSmoother` - Adaptive cubic (C¹ continuity)

**Algorithm:**
1. Solve CG Bezier smoothing on current mesh
2. Estimate L2 error per element: `||z_data - z_bezier||_L2`
3. Mark elements where `normalized_error > threshold`
4. Refine marked elements (with 2:1 balancing)
5. Re-solve until `error < tolerance` everywhere OR max iterations/elements reached

**Configuration (`AdaptiveCGBezierConfig`):**
- `error_threshold` - Stop when max error < threshold (default: 0.1m)
- `max_iterations` - Maximum adaptation iterations (default: 10)
- `max_elements` - Maximum number of elements (default: 10000)
- `refine_fraction` - Fraction of elements to refine per iteration (default: 0.2)
- `smoother_config` - Underlying CGBezierSmootherConfig

**Usage:**
```cpp
AdaptiveCGBezierConfig config;
config.error_threshold = 0.5;  // 0.5 meter threshold

AdaptiveCGBezierSmoother smoother(xmin, xmax, ymin, ymax, 4, 4, config);
smoother.set_bathymetry_data(geotiff_source);
auto result = smoother.solve_adaptive();
```

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
| `test_cg_bezier_bathymetry_smoother.cpp` | CG quintic Bezier (C²), DOF sharing, edge constraints, non-conforming meshes |
| `test_cg_cubic_bezier_bathymetry_smoother.cpp` | CG cubic Bezier (C¹), edge constraints, non-conforming meshes |
| `test_cg_linear_bezier_bathymetry_smoother.cpp` | CG linear Bezier (C⁰), DOF sharing |
| `test_cg_triharmonic_bezier_bathymetry_smoother.cpp` | CG quintic with triharmonic energy |
| `test_adaptive_cg_bezier_smoother.cpp` | Adaptive refinement for CG quintic smoother |
| `test_adaptive_cg_cubic_bezier_smoother.cpp` | Adaptive refinement for CG cubic smoother |
| `test_bathymetry.cpp` | GeoTIFF loading, coastline-adaptive refinement, seabed VTK output |
| `test_mesh.cpp` | Octree mesh creation, face connectivity |
| `test_dg_operators.cpp` | Gradient/divergence operators, mass matrix, face interpolation, discrete Green's identity |
| `test_initial_conditions.cpp` | Quiescent, Kelvin wave, lock exchange initial conditions |
| `test_conservation.cpp` | Mass, flux, energy, enstrophy, tracer conservation |
| `test_simulation_full.cpp` | Full simulation pipeline with diagnostics and VTK output |
| `test_vtk_output.cpp` | VTK writer, PVD time series |

Run specific tests with gtest filter: `./build/tests/drifter_integration_tests --gtest_filter="CG*Bezier*"`

## Dependencies

Required: Eigen3, Boost (Geometry), MPI, OpenMP, GTest
Optional: GDAL (geospatial), VTK, zarrs_ffi (Zarr output)


## Instructions for claude from user:

- Always separate i/o code from algorithmic code
- Never add "Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>" to git commit messages
- Always output per element intepolations to vtk
- Never output to an uniform global mesh to vtk 