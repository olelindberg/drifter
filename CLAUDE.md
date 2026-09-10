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

Every command that builds or runs a binary needs `LD_LIBRARY_PATH=/home/ole/.local/lib` — see the
note below for why. Shell state does not persist between Bash tool calls, so prefix each
invocation rather than relying on an earlier `export`.

```bash
# Configure (from project root) - no prefix needed, nothing is executed
cmake -B build

# Build (use max 6 parallel jobs to avoid memory issues)
LD_LIBRARY_PATH=/home/ole/.local/lib cmake --build build --parallel 6

# Run all tests
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build --output-on-failure

# Run only unit / integration tests
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -L unit --output-on-failure
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -L integration --output-on-failure

# Run a single test by name
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -R "TestName" --output-on-failure

# List available test names (no binary is run)
ctest --test-dir build -N

# Run test executable directly (shows all logging output)
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_unit_tests
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_integration_tests

# Run specific test(s) with gtest filter
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_unit_tests --gtest_filter="TestName*"

# Run tests with verbose output
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -V
```

**Why the `LD_LIBRARY_PATH` prefix is required** (README.md and the older notes are right about
this; a previous revision of this file wrongly claimed the path did not exist and the prefix was
unnecessary):

GDAL and PROJ are local builds in `/home/ole/.local/lib`, not system packages — the system
`ldconfig` only knows `libproj.so.15` / `.22`, and the build needs `libproj.so.25`. CMake gives
the binaries `RUNPATH=/home/ole/.local/lib`, which is why `libgdal.so.37` resolves without help,
but **`RUNPATH` is not inherited by transitive dependencies**: `libgdal.so.37` carries no
`RUNPATH` of its own, so *its* dependency `libproj.so.25` is not found. Hence the prefix.

This bites at **build** time, not just at run time: `gtest_discover_tests` executes the freshly
linked test binaries, so without the prefix `cmake --build` fails with

```
build/tests/drifter_unit_tests: error while loading shared libraries: libproj.so.25: ...
CMake Error at .../GoogleTestAddTests.cmake:83 (message): Error running test executable.
```

which looks like a compile failure but is purely a loader path problem.

Two further test executables exist beyond unit/integration:
- `./build/tests/drifter_benchmarks` (ctest label `benchmark`) from `tests/benchmarks/` -
  long-running, excluded from CI.
- `./build/tests/drifter_tools` (ctest label `tool`) from `tests/tools/` - developer
  diagnostics, currently matrix sparsity dumps (Matrix Market + plots).

## Applications

There is no `drifter` executable. The library `drifter_lib` backs **two** apps under `apps/`,
both of which are bathymetry mesh tools, not ocean simulations:

| Target | Source | What it does | Config |
|--------|--------|--------------|--------|
| `highrider` | `apps/highrider/main.cpp` → `Drifter` (`src/core/drifter.cpp`) | High-order path: adaptive smoothing of GeoTIFF bathymetry. `"smoother": {"type": ...}` selects **CG cubic Bezier** (default, the KKT/multigrid machinery) or **CG Hermite** C⁰/C¹ (direct SPD solve) | `config/highrider_example.json`, `config/highrider_hermite_example.json` |
| `lowrider` | `apps/lowrider/main.cpp` → `Lowrider` (`src/core/lowrider.cpp`) | Low-order path: adaptive **bilinear** mesh generation — no smoothing solve, just refine-and-sample | `config/lowrider_example.json` |

```bash
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/apps/highrider/highrider config/highrider_example.json
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/apps/highrider/highrider config/highrider_hermite_example.json
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/apps/lowrider/lowrider  config/lowrider_example.json
```

Each app has its own config struct and reader — `DrifterConfig`/`ConfigReader` vs
`LowriderConfig`/`LowriderConfigReader` (`include/core/lowrider_config.hpp`). They are
independent; adding a knob to one does not add it to the other.

**Lowrider pipeline** (`LinearMeshGenerator`, `bathymetry/linear_mesh_generator.hpp`) runs in
two stages, which is why its config has both a `coastline` and a `refinement` section:
1. **Coastline refinement** — refine near land boundaries, driven by coastline curvature radius.
2. **Error-driven seabed refinement** — Dörfler marking (`dorfler_theta`) on per-element error
   until `error_threshold`, `max_elements`, `max_level`, or the GeoTIFF pixel resolution limit
   is hit (`ConvergenceReason` records which).

`LinearBezierSurface` holds 4 corner DOFs per element shared for C⁰ continuity, and is fitted
by *sampling* the bathymetry at corners — refinement re-fits only new DOFs incrementally.

`BUILD_SHARED_LIBS=ON` additionally builds `libdrifter.so` for external projects linking to
DRIFTER.

## Python Scripts

`scr/` holds the plotting/report scripts used to regenerate the docs figures (matplotlib,
numpy, pandas, scipy; managed with `uv`, see `scr/pyproject.toml`). Two driver scripts run the
relevant gtest filters and then the plotters:
- `docs/regenerate_figures.sh` - solver verification report and figures
- `scr/run_hierarchical_benchmark.sh` - hierarchical ordering benchmark
  (`scr/plot_hierarchical_benchmark.py`)

`scr/plot_matrix_sparsity.py` renders the `.mtx` dumps produced by `tests/tools/`.

## Code Formatting

**Do not run clang-format.** Match the surrounding code by hand instead. The `clang-format` on
`PATH` here is a different version from the one CI uses and reformats whole files rather than the
lines you touched.

## CI

`.github/workflows/ci.yml` builds a gcc-12 / clang-15 × Debug / Release matrix with Ninja, plus a
clang-format-15 check. **CI configures with `DRIFTER_USE_VTK=OFF`, `DRIFTER_USE_ZARR=OFF` and no
GDAL**, so anything guarded behind those options — most of the bathymetry app path — is not
compiled there. A local build with the defaults exercises far more code than CI does; do not treat
a green CI run as coverage for GeoTIFF/VTK work. `nightly.yml` runs the extended suite on a cron.

## Logging and Profiling

- `core/logger.hpp` — singleton `Logger` behind `LOG_DEBUG/INFO/WARNING/ERROR(msg)` macros that
  capture the enclosing function name and short-circuit before building the message string when
  the level is disabled. Default level is INFO (DEBUG off). Use these rather than `std::cout`.
- `core/scoped_timer.hpp` — `ScopedTimer` accumulates elapsed ms into a `double&`;
  `OptionalScopedTimer` takes a pointer and is a true no-op when null, which is how the smoothers
  carry per-phase timings without paying for them when profiling is off.

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
- `GeoTiffReader` - Bathymetry loading via GDAL. Exposes `pixel_size_x/y()` and
  `min_element_size()`, the resolution floor below which refinement gains nothing
- `MultiSourceBathymetry` - Primary GeoTIFF plus higher-resolution tiles, with per-source
  `source_id` carried into VTK output for provenance
- `CoastlineRefinement` - R-tree based coastline-adaptive mesh refinement. Loading and indexing
  are **domain-filtered** (bounds passed to `load()`/`build_index()`) — required for global
  datasets. Also provides `has_circumradius_below()` over a box, which drives both apps'
  coastline stage, and circumradius-comb VTK debug output
- `Hilbert2D` (`mesh/hilbert.hpp`) - 2D Hilbert encode/decode, alongside the Morton codes used
  for quadtree leaf ordering

**bathymetry/** - 2D bathymetry surface fitting (uses `QuadtreeAdapter`, not `OctreeAdapter`)
- `QuadtreeAdapter` - 2D AMR quadtree. Point location is by **Morton code** over sorted leaves;
  the former Boost R-tree PIMPL is gone. Supports `refine(elements)` / `refine_where(pred)`,
  which rebalance to 2:1 and report the newly created leaves so surfaces can re-fit incrementally
- `CGCubicBezierBathymetrySmoother` - Cubic Bezier (C¹ continuity)
- `CGLinearBezierBathymetrySmoother` - Linear Bezier (C⁰ continuity)
- `LinearBezierSurface` + `LinearMeshGenerator` - the lowrider low-order path (see Applications)
- `ElementErrorEstimator` - polymorphic per-element error, built by `create_error_estimator()`
  from an `ErrorMetricType`. Two families: Gauss-quadrature
  (`NormalizedError`, `MeanDifference`, `VolumeChange`) and pixel-based
  (`PixelRMSE`, `PixelMaxError`) which evaluate at GeoTIFF pixel centers rather than quadrature
  points. Add a metric by adding an estimator + a factory case, not by editing the refiners
- `CGHermiteBathymetrySmoother` - Hermite corner value/derivative DOFs (structural C⁰/C¹, SPD)
- See detailed documentation in the "Bathymetry Smoothers" section below

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
- `SeabedVTKWriter` - High-resolution seabed surface visualization (also in `io/vtk_writer.hpp`,
  alongside `HighOrderVTKWriter`, `OceanVTKWriter`, `XDMFWriter`)
- `BathymetryVTKWriter` (`io/bathymetry_vtk_writer.hpp`) - quadtree/octree bathymetry surface output
- `QuadtreeVTKWriter` - linear quad cells (VTK type 9) for the lowrider mesh, with per-element
  level/error/depth cell data
- `WaterVTKWriter` - writes only elements whose depth > 0, via an injected depth query
- `write_matrix_market()` (`io/matrix_market_writer.hpp`) - dump a `SpMat` to `.mtx` for the
  sparsity tooling in `tests/tools/` and `scr/plot_matrix_sparsity.py`
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

5. **Dirichlet boundary conditions** (legacy `CGDofManager` only - NOT used by the CG Bezier smoothers): row/column elimination in the stiffness matrix Q:
   - For each Dirichlet DOF `i`: move coupling terms to RHS, then set `Q(i,:) = 0`, `Q(:,i) = 0`, `Q(i,i) = 1`, `c(i) = -value`
   - The CG Bezier smoothers apply no Dirichlet elimination and have no C² constraints. See `docs/cg_bezier_matrix_system.md` section 10.

### Bathymetry Smoothers (`bathymetry/`)

Two families of CG (Continuous Galerkin) surface smoother fit a smooth surface to bathymetry
data, sharing the assembly/solve base classes (`CGSmootherBase`, `CGSurfaceDofManagerBase`,
`AdaptiveCGSmootherBase` — note these names are family-neutral, *not* `cg_bezier_*`).

**Bezier family** — Bernstein control-value DOFs, shared at element boundaries so C⁰ is
structural; C¹ is imposed by collocation, which makes the system an indefinite KKT saddle point.

| Class | Degree | Continuity | DOFs/element |
|-------|--------|------------|--------------|
| `CGCubicBezierBathymetrySmoother` | Cubic (3) | C¹ (approximate, ~1e-8) | 16 |
| `CGLinearBezierBathymetrySmoother` | Linear (1) | C⁰ | 4 |

**Hermite family** (`CGHermiteBathymetrySmoother`, `CGHermiteDofManager`, `HermiteBasis1D/2D`,
`HermiteHessian`) — DOFs are elevation *and derivatives* at element corners, so C^r continuity is
**structural**: no edge constraints, no KKT system. `continuity_order` selects 0 (bilinear, 4
DOFs/element) or 1 (bicubic Bogner–Fox–Schmit, 16 DOFs/element). Hanging nodes and zero-gradient
BCs are pure master/slave substitutions, so the condensed system `Q_red = TᵀQT` is **SPD** and is
factorized directly — no Schur complement, no iterative path, no multigrid.
Which direct backend runs is the **runtime** config key `"smoother": {"solver": ...}` →
`HermiteSolverKind`, one of nine (`SimplicialLDLT` default, `SimplicialLDLTMetis`,
`SimplicialLLT`, `PardisoLDLT`, `PardisoLLT`, `UmfPackLU`, `CholmodSimplicialLDLT`,
`CholmodSupernodalLLT`, `CholmodSupernodalNesdis`). The CMake options decide only which are
*available*; naming an uncompiled one throws from `solve()` rather than substituting another.
On 66k DOFs the supernodal/multifrontal backends beat the simplicial ones by ~7x
(`PardisoLDLT` 540 ms vs `SimplicialLDLT` 3743 ms) because they reach dense BLAS3 kernels;
the default stays `SimplicialLDLT` only because it needs no optional dependency. The table is
at the top of `src/bathymetry/cg_hermite_bathymetry_smoother.cpp`, reproduced by
`tests/benchmarks/test_hermite_solver_comparison.cpp`.
`constraint_violation()` is identically zero by construction. The trade-off vs Bezier is the loss
of the convex-hull / variation-diminishing property: a Hermite fit can overshoot in steep regions,
which is why both families are kept. See `docs/hermite_bathymetry_system.md`.

Hermite-specific config (`CGHermiteSmootherConfig`): `continuity_order`, `lambda`,
`ridge_epsilon`, `ngauss_data` (2 for C⁰, 4 for C¹), `enable_zero_gradient_bc` (silently inactive
at r=0), `use_equilibration` (symmetric scaling of the mixed-unit DOFs inside `solve()` only),
`boundary_relaxation`.

`AdaptiveCGHermiteConfig` additionally stops refinement at the data resolution (the Bezier path
does not read these from JSON): `enforce_pixel_limit` (default true) with `min_element_size`
(0 = auto from the data resolution), and `min_data_points_per_element` (default 4) — the minimum
number of raster pixels a *child* element must cover. The resolution is per element, supplied by
`AdaptiveCGHermiteSmoother::set_resolution_func()`, which `Drifter::run()` wires to
`MultiSourceBathymetry::get_min_element_size_meters()` so high-resolution tiles allow finer
refinement than the primary raster. Hitting either limit reports
`ConvergenceReason::PixelResolution`.

**Adaptive variants:** `AdaptiveCGCubicBezierSmoother`, `AdaptiveCGLinearBezierSmoother`,
`AdaptiveCGHermiteSmoother` - error-driven mesh refinement (solve → estimate → Dörfler-mark →
refine).

**Selecting a family from config:** the JSON key `"smoother": { "type": ... }` maps to
`BathySmootherKind` (`core/enum_strings.hpp`) — `CubicBezier` (default), `HermiteC0`,
`HermiteC1`. `DrifterConfig` parses both `adaptive` (cubic Bezier) and `hermite_adaptive` from
the same JSON `"adaptive"` section (both are always populated so `save()` stays lossless), and
`Drifter::run()` (`src/core/drifter.cpp`) dispatches on `smoother_kind` through the
`run_adaptive_smoother<>` template — the two adaptive smoothers share the surface it needs, so
one body serves both. `print_config()` reports only the selected family's settings.
See `config/highrider_example.json` (cubic) and `config/highrider_hermite_example.json` (C¹).

**Optimization (ShipMesh Formulation):**

Uses smoothness-first formulation: `Q = α·H + λ·(BᵀWB + εI)` where H is the smoothness operator, BᵀWB is data fitting, and `α = ‖BᵀWB‖_F / ‖H‖_F` normalizes scales. This avoids boundary oscillations from standard least-squares + regularization.

H differs by smoother: the **cubic** uses thin plate energy `∫[(z_xx+z_yy)² + 2z_xy²]`, the **linear** uses gradient/membrane energy `∫[z_x² + z_y²]` (thin plate energy vanishes for bilinear surfaces).

- `λ = 0`: Pure smoothness (ignores data)
- `λ → ∞`: Approaches least-squares fit
- Typical: `λ = 0.01` (smooth) to `λ = 100` (close to data)

Solved via a KKT saddle-point system `[[Q, Aᵀ], [A, -εI]] [x; μ] = [b; 0]`; all constraints are homogeneous. The cubic smoother builds the KKT system; the linear smoother solves `Q_reduced x = b_reduced` directly after hanging-node condensation.

**Key Configuration (`CGCubicBezierSmootherConfig`):**
- `lambda` - Data fitting weight (default 0.01; linear smoother default 1.0)
- `edge_ngauss` - Collocation points per edge for C¹ and boundary constraints (default 4). C¹ edge derivative constraints are always built - there is no enable flag.
- `enable_natural_bc` - Zero normal curvature at boundaries (default: **false**)
- `enable_zero_gradient_bc` - Zero normal gradient / symmetry BC at boundaries (default: false)
- `ngauss_data` - Data fitting quadrature (default 4; **values above 4 are silently clamped to 4**)
- `lower_bound` / `upper_bound` - parsed from config but currently **unused** by any solve path

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

Note: cached element matrices are stored as `H + λB + λεI` with **α = 1** (α is not known until after assembly), so `CachedRediscretization` coarse operators differ from the true Q by the α factor. This is the likely cause of the MG (L2+Cached) data residual outlier in the verification report.

**Recommended configuration for adaptive meshes:**
```cpp
MultigridConfig config;
config.min_tree_level = 0;  // Coarsest level (1x1 element)
config.transfer_strategy = TransferOperatorStrategy::BezierSubdivision;
config.coarse_grid_strategy = CoarseGridStrategy::CachedRediscretization;
```

**Iterative solver stack (`bathymetry/`):** the smoothers share a small solver framework, all
under `include/bathymetry/`:
- `IIterativeMethod` (`iterative_method.hpp`) - common interface for smoothers and standalone
  solvers, built via `IterativeMethodFactory`. Implementations: `JacobiMethod`, `SchwarzMethod`
  (block builder in `schwarz_block_builder.hpp`), selected by `SmootherType` (Jacobi,
  MultiplicativeSchwarz, AdditiveSchwarz, ColoredMultiplicativeSchwarz).
- `FlexibleCG` - flexible CG, needed because the MG preconditioner is non-stationary.
- Schur preconditioners for the KKT system: `SchurPreconditioner` base with
  `DiagonalApproxCGSchurPreconditioner` and `BlockDiagApproxCGSchurPreconditioner`.
- `ConstraintCondenser` - hanging-node/constraint condensation shared by the smoothers.
- Basis and energy operators are split into base + concrete pairs
  (`basis_2d_base.hpp` / `cubic_bezier_basis_2d.hpp` / `linear_bezier_basis_2d.hpp`;
  `hessian_base.hpp` / `thin_plate_hessian.hpp` / `cubic_thin_plate_hessian.hpp`), so a
  new smoother degree means adding a basis + Hessian pair, not editing the assembler.

**DOF ordering and static condensation (both default OFF — leave them off):**
`CGCubicBezierSmootherConfig` exposes two experimental switches:
- `use_hierarchical_ordering` - orders DOFs by (level, is_constrained, type, Hilbert key)
  instead of the default Morton order. Benchmarked at **0.99x average speedup** — the multigrid
  preconditioner already exploits the quadtree hierarchy, so reordering only adds overhead.
  See `docs/hierarchical_ordering_benchmark.md`. May still help direct solvers (untested).
- `use_static_condensation` - eliminates the 4 interior/bubble DOFs of each cubic element via an
  element Schur complement (`StaticCondensationManager`), a 25% DOF reduction. **Only works on
  single-element meshes**; it is auto-disabled for multi-element meshes because the C¹ edge
  constraints couple interior DOFs across elements.

The smoother also exposes `H_global()`, `BtWB_global()`, `Q_global()`, and `condensed_system()`
for diagnostics — these are what the sparsity tooling and the docs figures consume.

**Documentation index (`docs/`):**
| File | Contents |
|------|----------|
| `cg_bezier_matrix_system.md` | Full derivation of the assembled CG Bezier system |
| `hermite_smoothness_operator.md` | Derivation of H in the Hermite basis |
| `cg_bezier_solver_verification.md` | Direct vs iterative vs MG solver comparison |
| `cg_cubic_bezier_uniform_evaluation.md` | Uniform-grid accuracy/timing tables |
| `uniform_vs_adaptive_convergence.md` | Uniform vs AMR convergence on synthetic bathymetry |
| `hierarchical_ordering_benchmark.md` | Hierarchical vs Morton DOF ordering — concludes no benefit |
| `hermite_bathymetry_system.md` | Derivation of the Hermite system now implemented by `CGHermiteBathymetrySmoother` — corner elevation/derivative DOFs, structural C^r, no KKT. Covers C⁰/C¹/C² (only C⁰ and C¹ are implemented). Section numbering parallels `cg_bezier_matrix_system.md` |
| `sea_water_equation_of_state.md` | The forms of ρ(T,S,p) in the literature — linear/quadratic (cabbeling, thermobaricity), UNESCO 1983 with full coefficients, the potential-temperature fits (Jackett & McDougall 1995, MDJWF 2003, Wright 1997, Brydon 1999), TEOS-10, and the Roquet polynomial approximations. Theory document, literature citations only; `governing_equations.md` §6 links to it |
| `coastline_adaptivity.md` | Theory of the coastline pre-pass and the water mask — discrete curvature as a circumradius, the R-tree query, the criterion `R(B) < min(Δx,Δy)` and its fixed point, then the Water/Beach/Inland element classification that decides what is assembled |

**Matrix System:** See `docs/cg_bezier_matrix_system.md` for the full derivation of the assembled system - least-squares data term, thin plate/membrane energy, C⁰/C¹ continuity, hanging-node constraints, boundary conditions, and KKT layout.

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
| `test_cg_hermite_bathymetry_smoother.cpp` | CG Hermite (structural C⁰/C¹), SPD condensed system |
| `test_adaptive_cg_hermite_smoother.cpp` | Adaptive refinement for CG Hermite; continuity is exact across refinement-induced T-junctions |
| `test_multi_source_bathymetry.cpp` | Multi-source bathymetry loading and blending |
| `test_mesh.cpp` | Octree mesh creation, face connectivity |
| `test_dg_operators.cpp` | Gradient/divergence operators, mass matrix, face interpolation, discrete Green's identity |
| `test_initial_conditions.cpp` | Quiescent, Kelvin wave, lock exchange initial conditions |
| `test_conservation.cpp` | Mass, flux, energy, enstrophy, tracer conservation |
| `test_simulation_full.cpp` | Full simulation pipeline with diagnostics and VTK output |
| `test_vtk_output.cpp` | VTK writer, PVD time series |

Run specific tests with gtest filter:
`LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_integration_tests --gtest_filter="CG*Bezier*"`

The lowrider path is covered by unit tests instead: `test_linear_mesh.cpp`,
`test_pixel_error_estimator.cpp`, `test_pixel_max_error_estimator.cpp`, and
`test_hierarchical_ordering.cpp` (Morton/Hilbert ordering and static condensation).

Hermite unit tests: `test_hermite_basis.cpp` (1D/2D basis, change of basis to Bernstein),
`test_hermite_energy.cpp` (element energy matrices vs the symbolic derivation), and
`test_hermite_constraints.cpp` (hanging-node and BC substitutions).

## Dependencies

Required: Eigen3, Boost (Geometry), GDAL
Optional (default ON): MPI, OpenMP, GTest (for tests)
Optional: VTK, zarrs_ffi (Zarr output), netCDF, HDF5

CMake build options (set with `-D`):
- `DRIFTER_USE_MPI=ON` - MPI parallelization (default ON)
- `DRIFTER_USE_OPENMP=ON` - OpenMP threading. (An earlier revision of this file claimed the
  option was unconditionally overridden to OFF; there is no such line, and `OpenMP::OpenMP_CXX`
  is linked PUBLIC, so `-fopenmp` is on every compile line.)
- `DRIFTER_USE_CUDA=OFF` - CUDA GPU acceleration (default OFF)
- `DRIFTER_USE_ZARR=ON` - Zarr v3 output via zarrs_ffi (default ON)
- `DRIFTER_USE_VTK=ON` - VTK output (default ON)
- `DRIFTER_USE_METIS=ON` - METIS ordering for sparse solvers (default ON, found QUIET)
- `DRIFTER_USE_CHOLMOD=OFF` / `DRIFTER_USE_UMFPACK=OFF` / `DRIFTER_USE_MKL=OFF` - make extra
  `HermiteSolverKind` backends available (see the Hermite section above). CHOLMOD and UMFPACK
  need `libsuitesparse-dev`; MKL needs `libmkl-dev` (~1.8 GB with its dependencies). Unlike the
  METIS option none of them degrade quietly - if the option is ON and the library is missing, the
  configure fails. MKL links the **GNU** threading layer (`libmkl_gnu_thread` + `-fopenmp` as a
  link flag) rather than `libiomp5`, so the process holds one OpenMP runtime and not two;
  `ldd build/... | grep -E "iomp5|gomp"` should show only `gomp`. Set `MKL_NUM_THREADS` — PARDISO
  single-threaded gives up its whole advantage
- `DRIFTER_BUILD_TESTS=ON` / `DRIFTER_BUILD_DOCS=OFF`
- `BUILD_SHARED_LIBS=OFF` - also build `libdrifter.so` alongside the static `drifter_lib`
- `ZARRS_FFI_DIR=/path` - Custom path to zarrs_ffi library

GDAL is a hard `REQUIRED` dependency regardless of options (CI passes a `DRIFTER_USE_GDAL`
flag that no longer exists).