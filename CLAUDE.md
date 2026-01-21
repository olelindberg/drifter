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
- `A` = C² constraint matrix (continuity of z, ∂z/∂u, ∂z/∂v, ∂²z/∂u², ∂²z/∂u∂v, ∂²z/∂v², and higher derivatives at shared vertices)
- `b` = constraint RHS (zero for interior continuity, nonzero when Dirichlet BCs are eliminated)

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
- `BezierBathymetrySmoother` - Main solver using constrained QP (KKT system)
- `BezierBasis2D` - Quintic tensor-product Bernstein basis evaluation
- `ThinPlateHessian` - Curvature regularization energy `E = ∫[(z_uu + z_vv)² + 2z_uv²]`
- `BezierC2ConstraintBuilder` - C² continuity constraints at shared vertices
- `BezierDataFittingAssembler` - Least-squares data fitting term

**Configuration (`BezierSmootherConfig`):**
- `lambda` - Data fitting weight (0 = pure thin plate/soap film, higher = closer to data)
- `gradient_weight` - First derivative penalty (slope smoothing)
- `enable_boundary_dirichlet` - Pin boundary corner DOFs to input data

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

### Bezier Continuity Constraints

This section describes the general theory for enforcing continuity between tensor-product Bezier elements of any degree. The constraint counts and DOF layouts vary by polynomial degree.

**Vertex constraints**: Enforce continuity at element corners where multiple elements meet.

The number of constraints per shared vertex depends on the polynomial degree and desired continuity:
- **C⁰** (degree 1): 1 constraint (value only)
- **C¹** (degree 3): 4 constraints (value + first derivatives)
- **C²** (degree 5): 9 constraints (value + first + second derivatives)

Each constraint has the form: `φ₁/scale₁ = φ₂/scale₂` where `scale = dx^nu · dy^nv`

For vertices shared by n elements, use the "star" pattern: pick a reference element and constrain (n-1) others to it. This produces (n-1) × k independent constraints, where k is the number of derivative conditions (1, 4, or 9).

**Example: C² constraints for quintic elements (9 per shared vertex pair):**

   For each pair of elements sharing a vertex, 9 derivative constraints are enforced:
   ```
   Constraint    Derivative order (nu,nv)    Scale factor
   ─────────────────────────────────────────────────────────
   z             (0,0)                       1
   z_u           (1,0)                       dx
   z_v           (0,1)                       dy
   z_uu          (2,0)                       dx²
   z_uv          (1,1)                       dx·dy
   z_vv          (0,2)                       dy²
   z_uuv         (2,1)                       dx²·dy
   z_uvv         (1,2)                       dx·dy²
   z_uuvv        (2,2)                       dx²·dy²
   ```

   The derivative values at corners depend on nearby control points:
   - `z` at corner (0,0): depends only on DOF 0
   - `z_u` at corner (0,0): depends on DOFs 0, 6 (and 12 for quintic)
   - `z_v` at corner (0,0): depends on DOFs 0, 1 (and 2 for quintic)
   - Higher derivatives involve more control points along each direction

   **Example: Total DOFs and constraints for 2×2 quintic mesh (no AMR):**
   ```

   DOFs: 
   - Degree 1 elements (linear) : 4 elements × (2 x 2) DOFs/element =  16 total DOFs
   - Degree 3 elements (cubic)  : 4 elements × (4 x 4) DOFs/element =  64 total DOFs
   - Degree 5 elements (quintic): 4 elements × (6 x 6) DOFs/element = 144 total DOFs

   Vertex type              Count    Elements sharing    Constraints each    Total
   ─────────────────────────────────────────────────────────────────────────────────
   Interior (center)        1        4                   (4-1)×9 = 27        27
   Edge (boundary midpts)   4        2                   (2-1)×9 = 9         36
   Corner (domain corners)  4        1                   0                   0
   ─────────────────────────────────────────────────────────────────────────────────
   Total C² constraints:                                                     63

   KKT system size: (144 + 63) × (144 + 63) = 207 × 207
   ```

   **Degree 1 (bilinear) elements - C⁰ continuity only:**

   For degree 1 elements, only C⁰ continuity (value matching) is possible since higher
   derivatives don't exist. Each element has 4 DOFs at corners:
   ```
   DOF layout (degree 1):
     j=1: [1]  [3]   ← top (v=1)
     j=0: [0]  [2]   ← bottom (v=0)
          ↑    ↑
        left  right
       (u=0) (u=1)
   ```

   Corner-to-DOF mapping: `dof = i + 2*j` where i,j ∈ {0,1}
   - Corner 0 (u=0, v=0): DOF 0
   - Corner 1 (u=1, v=0): DOF 2
   - Corner 2 (u=0, v=1): DOF 1
   - Corner 3 (u=1, v=1): DOF 3

   For a 2×2 mesh with degree 1 elements:
   ```
   Physical layout:           Vertex sharing:
   +-------+-------+          Corner vertices: 1 element each (4 vertices)
   |       |       |          Edge vertices: 2 elements each (4 vertices)
   |   2   |   3   |          Center vertex: 4 elements (1 vertex)
   |       |       |
   +-------+-------+          Total vertices: 9
   |       |       |
   |   0   |   1   |
   |       |       |
   +-------+-------+

   C⁰ constraints (1 per shared vertex pair):
   ─────────────────────────────────────────────────────────────────────────────────
   Vertex type              Count    Elements sharing    Constraints each    Total
   ─────────────────────────────────────────────────────────────────────────────────
   Interior (center)        1        4                   (4-1)×1 = 3         3
   Edge (boundary midpts)   4        2                   (2-1)×1 = 1         4
   Corner (domain corners)  4        1                   0                   0
   ─────────────────────────────────────────────────────────────────────────────────
   Total C⁰ constraints:                                                    7

   DOFs: 16,  Constraints: 7
   Free DOFs after constraints: 16 - 7 = 9 (matches 9 unique vertices)
   ```

   This shows that C⁰ constraints for bilinear elements exactly reduce the
   duplicated corner DOFs to unique vertex values, as expected.

   **Degree 3 (bicubic) elements - C¹ continuity:**

   For degree 3 elements, C¹ continuity (value + first derivatives) can be enforced.
   Each element has 16 DOFs in a 4×4 grid:
   ```
   DOF layout (degree 3):
     j=3: [3]  [7]  [11] [15]   ← top (v=1)
     j=2: [2]  [6]  [10] [14]
     j=1: [1]  [5]  [9]  [13]
     j=0: [0]  [4]  [8]  [12]   ← bottom (v=0)
          ↑              ↑
        left           right
       (u=0)          (u=1)
   ```

   Corner-to-DOF mapping: `dof = i + 4*j` where i,j ∈ {0,1,2,3}
   - Corner 0 (u=0, v=0): DOF 0
   - Corner 1 (u=1, v=0): DOF 12
   - Corner 2 (u=0, v=1): DOF 3
   - Corner 3 (u=1, v=1): DOF 15

   C¹ constraints at each vertex (4 per shared vertex pair):
   ```
   Constraint    Derivative order (nu,nv)    Scale factor
   ─────────────────────────────────────────────────────────
   z             (0,0)                       1
   z_u           (1,0)                       dx
   z_v           (0,1)                       dy
   z_uv          (1,1)                       dx·dy
   ```

   For a 2×2 mesh with degree 3 elements:
   ```
   Physical layout:           Vertex sharing:
   +-------+-------+          Corner vertices: 1 element each (4 vertices)
   |       |       |          Edge vertices: 2 elements each (4 vertices)
   |   2   |   3   |          Center vertex: 4 elements (1 vertex)
   |       |       |
   +-------+-------+          Total vertices: 9
   |       |       |
   |   0   |   1   |
   |       |       |
   +-------+-------+

   C¹ constraints (4 per shared vertex pair):
   ─────────────────────────────────────────────────────────────────────────────────
   Vertex type              Count    Elements sharing    Constraints each    Total
   ─────────────────────────────────────────────────────────────────────────────────
   Interior (center)        1        4                   (4-1)×4 = 12        12
   Edge (boundary midpts)   4        2                   (2-1)×4 = 4         16
   Corner (domain corners)  4        1                   0                   0
   ─────────────────────────────────────────────────────────────────────────────────
   Total C¹ constraints:                                                    28

   DOFs: 64,  Constraints: 28
   Free DOFs after constraints: 64 - 28 = 36

   Unique vertex DOFs: 9 vertices × 4 derivatives = 36 ✓
   ```

   The derivative constraints at corners depend on nearby control points:
   - `z` at corner (0,0): DOF 0
   - `z_u` at corner (0,0): DOFs 0, 4 (linear combination)
   - `z_v` at corner (0,0): DOFs 0, 1 (linear combination)
   - `z_uv` at corner (0,0): DOFs 0, 1, 4, 5 (bilinear combination)

   **Degree 5 (quintic) elements - C² continuity:**

   For degree 5 elements, full C² continuity (value + first + second derivatives) is enforced.
   Each element has 36 DOFs in a 6×6 grid:
   ```
   DOF layout (degree 5):
     j=5: [5]  [11] [17] [23] [29] [35]   ← top (v=1)
     j=4: [4]  [10] [16] [22] [28] [34]
     j=3: [3]  [9]  [15] [21] [27] [33]
     j=2: [2]  [8]  [14] [20] [26] [32]
     j=1: [1]  [7]  [13] [19] [25] [31]
     j=0: [0]  [6]  [12] [18] [24] [30]   ← bottom (v=0)
          ↑                         ↑
        left                      right
       (u=0)                      (u=1)
   ```

   Corner-to-DOF mapping: `dof = i + 6*j` where i,j ∈ {0,1,2,3,4,5}
   - Corner 0 (u=0, v=0): DOF 0
   - Corner 1 (u=1, v=0): DOF 30
   - Corner 2 (u=0, v=1): DOF 5
   - Corner 3 (u=1, v=1): DOF 35

   C² constraints at each vertex (9 per shared vertex pair):
   ```
   Constraint    Derivative order (nu,nv)    Scale factor
   ─────────────────────────────────────────────────────────
   z             (0,0)                       1
   z_u           (1,0)                       dx
   z_v           (0,1)                       dy
   z_uu          (2,0)                       dx²
   z_uv          (1,1)                       dx·dy
   z_vv          (0,2)                       dy²
   z_uuv         (2,1)                       dx²·dy
   z_uvv         (1,2)                       dx·dy²
   z_uuvv        (2,2)                       dx²·dy²
   ```

   For a 2×2 mesh with degree 5 elements:
   ```
   Physical layout:           Vertex sharing:
   +-------+-------+          Corner vertices: 1 element each (4 vertices)
   |       |       |          Edge vertices: 2 elements each (4 vertices)
   |   2   |   3   |          Center vertex: 4 elements (1 vertex)
   |       |       |
   +-------+-------+          Total vertices: 9
   |       |       |
   |   0   |   1   |
   |       |       |
   +-------+-------+

   C² constraints (9 per shared vertex pair):
   ─────────────────────────────────────────────────────────────────────────────────
   Vertex type              Count    Elements sharing    Constraints each    Total
   ─────────────────────────────────────────────────────────────────────────────────
   Interior (center)        1        4                   (4-1)×9 = 27        27
   Edge (boundary midpts)   4        2                   (2-1)×9 = 9         36
   Corner (domain corners)  4        1                   0                   0
   ─────────────────────────────────────────────────────────────────────────────────
   Total C² constraints:                                                    63

   DOFs: 144,  Constraints: 63
   Free DOFs after constraints: 144 - 63 = 81

   Unique vertex DOFs: 9 vertices × 9 derivatives = 81 ✓
   ```

   The derivative constraints at corners depend on nearby control points:
   - `z` at corner (0,0): DOF 0
   - `z_u` at corner (0,0): DOFs 0, 6, 12 (quadratic in u)
   - `z_v` at corner (0,0): DOFs 0, 1, 2 (quadratic in v)
   - `z_uu` at corner (0,0): DOFs 0, 6, 12 (second derivative in u)
   - `z_uv` at corner (0,0): DOFs 0, 1, 6, 7 and neighbors (mixed partials)
   - `z_vv` at corner (0,0): DOFs 0, 1, 2 (second derivative in v)
   - Higher mixed derivatives involve 3×3 stencils of control points

   **Interior non-conforming vertex (T-junction) constraints:**

   At a non-conforming 2:1 interface, a "hanging node" or T-junction vertex occurs
   where two fine elements share a corner that lies on the edge of a coarse element:
   ```
   Non-conforming T-junction:
   +---------------+-------+
   |               |   F1  |
   |     Coarse    *-------+    * = hanging node (T-junction vertex)
   |               |   F0  |
   +---------------+-------+
   ```

   The hanging node (*) lies at the midpoint of the coarse element's edge. At this
   vertex, 3 elements meet: 1 coarse + 2 fine. Constraints must ensure:
   1. The fine element corners match the coarse Bezier surface at that point
   2. Derivatives of fine elements match derivatives of coarse surface

   **Degree 1 (bilinear) - C⁰ at T-junction:**

   For degree 1, only value matching is needed. The hanging node's value is
   interpolated from the coarse element's edge DOFs:
   ```
   Coarse edge DOFs:  [C0]----[C1]    (2 DOFs on edge)
   Fine corners:           *          (hanging node at midpoint)

   Constraint: z_fine = 0.5·z_C0 + 0.5·z_C1

   T-junction constraints (1 per fine element):
   - F0 corner at *: 1 constraint (value interpolation)
   - F1 corner at *: 1 constraint (value interpolation)
   Total: 2 constraints for the T-junction
   ```

   **Degree 3 (bicubic) - C¹ at T-junction:**

   For degree 3, value and first derivatives must match. The fine element's
   edge DOFs must interpolate the coarse element's Bezier surface:
   ```
   Coarse element (right edge):        Fine elements (left edges):

   v=1   [C3]--+                        +--[g3]     F1: v ∈ [0.5, 1]
               |                        |
               |                        +--[g2]
               |                        |
   v=2/3 [C2]--+                        +--[g1]
               |                        |
   v=1/2       |                  [f3]--+--[g0]  ←  Hanging nodes (F0 top-left, F1 bottom-left)
               |                        |
   v=1/3 [C1]--+                  [f2]--+
               |                        |
               |                  [f1]--+
               |                        |
   v=0   [C0]--+                  [f0]--+           F0: v ∈ [0, 0.5]

   Coarse edge: 4 DOFs at v = 0, 1/3, 2/3, 1
   F0's left edge: 4 DOFs at v = 0, 1/6, 1/3, 1/2 (in F0's local coords: 0, 1/3, 2/3, 1)
   F1's left edge: 4 DOFs at v = 1/2, 2/3, 5/6, 1 (in F1's local coords: 0, 1/3, 2/3, 1)
   The hanging node * is at v=0.5 where f3=g0 (same physical point, different elements)
   ```

   At the hanging node, we constrain 4 derivatives per fine element:
   - z: value matches coarse Bezier at parameter (1, 0.5)
   - z_u: ∂z/∂u matches coarse
   - z_v: ∂z/∂v matches coarse
   - z_uv: mixed partial matches

   Each fine element contributes 4 constraints at its corner.
   Total: 2 × 4 = 8 constraints for the T-junction

   Additionally, interior edge DOFs of fine elements are constrained to
   interpolate the coarse Bezier curve:
    - 2 interior DOFs per fine edge × 2 fine elements = 4 edge constraints

   **Degree 5 (quintic) - C² at T-junction:**

   For degree 5, full C² continuity requires matching up to second derivatives:
   ```
   Coarse element (right edge):        Fine elements (left edges):

   v=1   [C5]--+                        +--[g5]     F1: v ∈ [0.5, 1]
               |                        |
               |                        +--[g4]
               |                        |
   v=4/5 [C4]--+                        +--[g3]
               |                        |
               |                        +--[g2]
               |                        |
   v=3/5 [C3]--+                        +--[g1]
               |                        |
   v=1/2       |                  [f5]--+--[g0]  ←  Hanging nodes (F0 top-left, F1 bottom-left)
               |                        |
   v=2/5 [C2]--+                  [f4]--+
               |                        |
               |                  [f3]--+
               |                        |
   v=1/5 [C1]--+                  [f2]--+
               |                        |
               |                  [f1]--+
               |                        |
   v=0   [C0]--+                  [f0]--+           F0: v ∈ [0, 0.5]

   Coarse edge: 6 DOFs at v = 0, 1/5, 2/5, 3/5, 4/5, 1
   F0's left edge: 6 DOFs at v = 0, 1/10, 2/10, 3/10, 4/10, 1/2
   F1's left edge: 6 DOFs at v = 1/2, 6/10, 7/10, 8/10, 9/10, 1
   The hanging node * is at v=0.5 where f5=g0 (same physical point)
   ```

   At the hanging node, we constrain all 9 derivatives per fine element:
   - z, z_u, z_v           (value + first derivatives)
   - z_uu, z_uv, z_vv      (second derivatives)
   - z_uuv, z_uvv, z_uuvv  (mixed higher derivatives)

   Each derivative is computed from the coarse Bezier surface evaluated at
   the midpoint parameter (u=1, v=0.5 for a right edge).

   Each fine element contributes 9 constraints at its corner.
   Total: 2 × 9 = 18 constraints for the T-junction

   Additionally, interior edge DOFs of fine elements are constrained to
   interpolate the coarse Bezier curve:
   - 4 interior DOFs per fine edge × 2 fine elements = 8 edge constraints

   **Summary of T-junction constraints:**
   ```
   Degree    Continuity    Constraints per fine corner    Total at T-junction
   ───────────────────────────────────────────────────────────────────────────
   1         C⁰            1                              2 + 0 edge = 2
   3         C¹            4                              8 + 4 edge = 12
   5         C²            9                              18 + 8 edge = 26
   ```

   Note: Edge constraints ensure the interior DOFs along the fine-coarse interface
   interpolate the coarse Bezier curve, preventing gaps or overlaps.

   **Detailed Example: Non-conforming 1+4 domain with degree 1 (C⁰ continuity)**

   This example shows a coarse element (E0) adjacent to a 2×2 grid of fine elements
   (E1-E4). The T-junction at V3 is where fine elements meet the coarse element's edge.

   ```
   Physical layout and global DOF numbering:

   y
   ↑
   1 V1[1]────────V4[3,13]────V7[15,17]───V10[19]
     │              │            │            │
     │              │     E3     │     E4     │
     │      E0      │            │            │
  0.5│            V3*[5,12]───V6[7,9,14,16]─V9[11,18]
     │              │            │            │
     │              │     E1     │     E2     │
     │              │            │            │
   0 V0[0]────────V2[2,4]─────V5[6,8]──────V8[10]
     0              1          1.5            2  → x

   * = T-junction (hanging node on E0's right edge)
   [n,m,...] = global DOFs from different elements at this vertex

   Element bounds:
   - E0: [0,1] × [0,1], E1: [1,1.5] × [0,0.5], E2: [1.5,2] × [0,0.5]
   - E3: [1,1.5] × [0.5,1], E4: [1.5,2] × [0.5,1]

   Local DOF layout (degree 1):  [1]─[3]  where local_dof = i + 2*j
                                  │   │
                                 [0]─[2]
   ```

   **C⁰ Constraints (10 total):**

   Conforming vertices (8 constraints) - value matching using star pattern:
   ```
   V2: x[4] = x[2]           V5: x[8] = x[6]         V7: x[17] = x[15]
   V4: x[13] = x[3]          V6: x[9] = x[7]         V9: x[18] = x[11]
                                 x[14] = x[7]
                                 x[16] = x[7]
   ```

   T-junction at V3 (2 constraints) - interpolation from E0's edge:
   ```
   V3 at (1, 0.5) lies at midpoint of E0's right edge (DOFs 2 and 3).
   Fine element corners must match interpolated value:

   x[5]  = 0.5·x[2] + 0.5·x[3]    (E1's top-left corner)
   x[12] = 0.5·x[2] + 0.5·x[3]    (E3's bottom-left corner)
   ```

   **Verification:**
   ```
   DOFs: 5 elements × 4 = 20
   Constraints: 8 conforming + 2 T-junction = 10
   Free DOFs: 20 - 10 = 10 = 11 vertices - 1 hanging node ✓
   ```

   **Detailed Example: Non-conforming 1+4 domain with degree 3 (C¹ continuity)**

   Same mesh as degree 1, but with bicubic elements (16 DOFs each, 4×4 control points).
   C¹ continuity requires matching value and first derivatives at vertices.

   ```
   Physical layout and global DOF numbering (80 total DOFs):

   y
   ↑
   1 V1[3]────────V4[15,51]──V7[63,67]───V10[79]
     │              │            │            │
     │              │     E3     │     E4     │
     │      E0      │            │            │
  0.5│            V3*[7,48]───V6[15,19,60,64]─V9[31,78]
     │              │            │            │
     │              │     E1     │     E2     │
     │              │            │            │
   0 V0[0]────────V2[12,16]───V5[28,32]────V8[44]
     0              1          1.5            2  → x

   * = T-junction (hanging node on E0's right edge)

   Local DOF layout (degree 3):  [3]─[7]─[11]─[15]   where local_dof = i + 4*j
                                  │    │    │    │
                                 [2]─[6]─[10]─[14]
                                  │    │    │    │
                                 [1]─[5]─[9]─[13]
                                  │    │    │    │
                                 [0]─[4]─[8]─[12]

   Corner DOFs: (0,0)→0, (1,0)→12, (0,1)→3, (1,1)→15
   ```

   **C¹ Constraints at each vertex (4 per shared vertex pair):**

   For C¹ continuity, match z, z_u, z_v, z_uv at each shared vertex.
   Scale factors account for element size: z_u uses dx, z_v uses dy, z_uv uses dx·dy.

   ```
   Derivative    Corner (0,0) DOFs involved    Bernstein coefficients
   ─────────────────────────────────────────────────────────────────────
   z             [0]                           1
   z_u           [0, 4]                        3·[-1, +1]/dx
   z_v           [0, 1]                        3·[-1, +1]/dy
   z_uv          [0, 1, 4, 5]                  9·[+1, -1, -1, +1]/(dx·dy)
   ```

   **Conforming vertex constraints (32 total):**
   ```
   Vertex    Elements      Constraints (4 each: z, z_u, z_v, z_uv)
   ─────────────────────────────────────────────────────────────────────
   V2        E0, E1        4 constraints (E1 matches E0)
   V4        E0, E3        4 constraints (E3 matches E0)
   V5        E1, E2        4 constraints (E2 matches E1)
   V6        E1,E2,E3,E4   12 constraints (E2,E3,E4 match E1)
   V7        E3, E4        4 constraints (E4 matches E3)
   V9        E2, E4        4 constraints (E4 matches E2)
   ─────────────────────────────────────────────────────────────────────
   Total: (1+1+1+3+1+1) × 4 = 32 conforming constraints
   ```

   **T-junction at V3 (8 constraints):**

   V3 at (1, 0.5) lies at midpoint of E0's right edge. Fine element corners
   must match the coarse Bezier surface value AND derivatives at this point.

   ```
   For each fine element (E1 and E3), constrain 4 derivatives:
   - z:    interpolate from E0's edge curve at v=0.5
   - z_u:  match ∂z/∂x of E0's surface at (u=1, v=0.5)
   - z_v:  match ∂z/∂y of E0's surface at (u=1, v=0.5)
   - z_uv: match mixed partial of E0's surface

   The coarse edge is a cubic Bezier with control points [C0, C1, C2, C3].
   At t=0.5, cubic Bernstein basis values are [1/8, 3/8, 3/8, 1/8].

   z at V3 = (1/8)·C0 + (3/8)·C1 + (3/8)·C2 + (1/8)·C3

   T-junction constraints: 2 elements × 4 derivatives = 8 constraints
   ```

   **Edge DOF constraints (4 total):**

   Interior DOFs along fine element edges at the coarse-fine interface must
   interpolate the coarse Bezier curve to prevent gaps:

   ```
   E0's right edge: cubic curve with 4 control points
   E1's left edge (lower half): 2 interior DOFs at v = 1/6, 2/6 (local: 1/3, 2/3)
   E3's left edge (upper half): 2 interior DOFs at v = 4/6, 5/6 (local: 1/3, 2/3)

   Each interior DOF constrained to lie on E0's cubic edge curve.
   Total: 2 interior DOFs × 2 fine elements = 4 edge constraints
   ```

   **Verification:**
   ```
   DOFs: 5 elements × 16 = 80
   Constraints: 32 conforming + 8 T-junction + 4 edge = 44
   Free DOFs: 80 - 44 = 36

   Expected: 11 vertices × 4 derivatives - 1 hanging node × 4 = 40
   But edge interior DOFs are also constrained...

   Detailed count:
   - 4 domain corners × 4 derivatives = 16 free
   - 6 conforming vertices × 4 derivatives = 24 free (but shared)
   - Actually: unique vertex derivatives = 10 × 4 = 40
   - Minus edge DOF constraints = 40 - 4 = 36 ✓
   ```

   **Detailed Example: Non-conforming 1+4 domain with degree 5 (C² continuity)**

   Same mesh as above, but with quintic elements (36 DOFs each, 6×6 control points).
   C² continuity requires matching value, first, and second derivatives at vertices.

   ```
   Physical layout (180 total DOFs):

   y
   ↑
   1 V1[5]────────V4[35,77]──V7[113,149]──V10[185]
     │              │            │            │
     │              │     E3     │     E4     │
     │      E0      │            │            │
  0.5│            V3*[11,72]──V6[35,41,108,144]─V9[71,180]
     │              │            │            │
     │              │     E1     │     E2     │
     │              │            │            │
   0 V0[0]────────V2[30,36]───V5[66,72]────V8[102]
     0              1          1.5            2  → x

   * = T-junction (hanging node on E0's right edge)

   Local DOF layout (degree 5):  [5]─[11]─[17]─[23]─[29]─[35]  where local_dof = i + 6*j
                                  │    │    │    │    │    │
                                 [4]─[10]─[16]─[22]─[28]─[34]
                                  │    │    │    │    │    │
                                 [3]─[9]─[15]─[21]─[27]─[33]
                                  │    │    │    │    │    │
                                 [2]─[8]─[14]─[20]─[26]─[32]
                                  │    │    │    │    │    │
                                 [1]─[7]─[13]─[19]─[25]─[31]
                                  │    │    │    │    │    │
                                 [0]─[6]─[12]─[18]─[24]─[30]

   Corner DOFs: (0,0)→0, (1,0)→30, (0,1)→5, (1,1)→35
   ```

   **C² Constraints at each vertex (9 per shared vertex pair):**

   For C² continuity, match z and all derivatives up to order 2 at each shared vertex:
   z, z_u, z_v, z_uu, z_uv, z_vv, z_uuv, z_uvv, z_uuvv

   ```
   Derivative    Corner (0,0) DOFs involved    Scale factor
   ─────────────────────────────────────────────────────────────────────
   z             [0]                           1
   z_u           [0, 6]                        dx
   z_v           [0, 1]                        dy
   z_uu          [0, 6, 12]                    dx²
   z_uv          [0, 1, 6, 7]                  dx·dy
   z_vv          [0, 1, 2]                     dy²
   z_uuv         [0, 1, 6, 7, 12, 13]          dx²·dy
   z_uvv         [0, 1, 2, 6, 7, 8]            dx·dy²
   z_uuvv        [0, 1, 2, 6, 7, 8, 12, 13, 14] dx²·dy²
   ```

   **Conforming vertex constraints (72 total):**
   ```
   Vertex    Elements      Constraints (9 each)
   ─────────────────────────────────────────────────────────────────────
   V2        E0, E1        9 constraints (E1 matches E0)
   V4        E0, E3        9 constraints (E3 matches E0)
   V5        E1, E2        9 constraints (E2 matches E1)
   V6        E1,E2,E3,E4   27 constraints (E2,E3,E4 match E1)
   V7        E3, E4        9 constraints (E4 matches E3)
   V9        E2, E4        9 constraints (E4 matches E2)
   ─────────────────────────────────────────────────────────────────────
   Total: (1+1+1+3+1+1) × 9 = 72 conforming constraints
   ```

   **T-junction at V3 (18 constraints):**

   V3 at (1, 0.5) lies at midpoint of E0's right edge. Fine element corners
   must match all 9 derivatives of E0's surface at this point.

   ```
   For each fine element (E1 and E3), constrain 9 derivatives:
   - z, z_u, z_v:        value and first derivatives
   - z_uu, z_uv, z_vv:   second derivatives
   - z_uuv, z_uvv, z_uuvv: mixed higher derivatives

   The coarse edge is a quintic Bezier with control points [C0, C1, C2, C3, C4, C5].
   At t=0.5, quintic Bernstein basis values are [1/32, 5/32, 10/32, 10/32, 5/32, 1/32].

   T-junction constraints: 2 elements × 9 derivatives = 18 constraints
   ```

   **Edge DOF constraints (8 total):**

   Interior DOFs along fine element edges at the coarse-fine interface must
   interpolate the coarse Bezier curve:

   ```
   E0's right edge: quintic curve with 6 control points
   E1's left edge (lower half): 4 interior DOFs (local j = 1,2,3,4)
   E3's left edge (upper half): 4 interior DOFs (local j = 1,2,3,4)

   Each interior DOF constrained to lie on E0's quintic edge curve.
   Total: 4 interior DOFs × 2 fine elements = 8 edge constraints
   ```

   **Verification:**
   ```
   DOFs: 5 elements × 36 = 180
   Constraints: 72 conforming + 18 T-junction + 8 edge = 98
   Free DOFs: 180 - 98 = 82

   Expected: 10 independent vertices × 9 derivatives = 90
   Minus edge DOF constraints = 90 - 8 = 82 ✓
   ```

**Critical design decisions:**
1. **Dirichlet BCs only at corners**: Apply Dirichlet constraints only at corner DOFs (vertices), not interior edge DOFs. This preserves C² constraints at shared boundary vertices, which involve derivative terms depending on multiple edge DOFs.

2. **Thin plate scaling for physical coordinates**: The energy must account for element size (dx, dy):
   - `scale_uu_uu = dy/dx³` (for z_uu² term)
   - `scale_vv_vv = dx/dy³` (for z_vv² term)
   - `scale_uu_vv = 1/(dx·dy)` (for cross term)
   - `scale_uv_uv = 2/(dx·dy)` (for z_uv² term)

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


## Instructions for claude from user:

- Always separate i/o code from algorithmic code
- Never add "Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>" to git commit messages
- Always output per element intepolations to vtk
- Never output to an uniform global mesh to vtk 