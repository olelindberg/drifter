# Paper Audit — Bathymetry Program (single-core)

This document audits the DRIFTER **bathymetry program** against the three reference
papers in this folder, and judges whether the implementation is *optimal* in the
light of their findings.

- **Program under audit:** [src/main.cpp](../../src/main.cpp) →
  `Drifter::run()` ([src/core/drifter.cpp](../../src/core/drifter.cpp)) →
  `AdaptiveCGCubicBezierSmoother` → `CGCubicBezierBathymetrySmoother`. The program
  fits a continuous, C¹ cubic-Bézier surface to multi-source bathymetry by minimizing
  thin-plate energy plus a weighted least-squares data term, on an adaptively refined
  quadtree.
- **What "the bathymetry program" actually is, numerically:** a 2-D, fixed-degree
  (cubic, 16 DOF/element) *surface-fitting* problem — a thin-plate-regularized
  least-squares solve `Q x = b` with `Q = α·H + λ·(BᵀWB + εI)` — on a 2:1-balanced
  quadtree with hanging-node and C¹ edge constraints. Default problem size is modest
  (`max_elements = 10000`, i.e. up to ~10⁴ elements ⇒ O(10⁴–10⁵) DOF), and the default
  solver is a **direct** sparse factorization.

## Scope & method

- **Single core only.** Per the request, no shared- or distributed-memory parallelism
  is considered. Every parallel-specific finding in the papers (ghost layers, MPI DOF
  enumeration, distributed constraints, repartitioning, the p4est oracle's parallel
  Balance/Ghost/Nodes/Partition communication) is **out of scope** and is marked as such
  rather than scored.
- **Method.** All three papers were read in full; every verdict below was checked
  against the current source (file:line citations are clickable). Verdicts use:
  **Match** (follows the finding), **Partial** (follows in part), **Deviation**
  (does something materially different), **Gap** (a recommended optimization is absent),
  **Appropriate** (deliberately differs and the difference is justified here),
  **N-A** (not applicable to this problem/scale).
- **Note on the third reference.** The file originally named `SKM_C284e26040808180.pdf`
  was a misfiled personal document (a marina boat-mooring lease), **not** a technical
  paper; it has been removed. The correct third reference is the **p4est** paper, added
  here as `100791634.pdf`.

### The three papers

| # | Reference | File |
|---|-----------|------|
| 1 | Bangerth, Burstedde, Heister, Kronbichler (2011), *Algorithms and Data Structures for Massively Parallel Generic Adaptive Finite Element Codes*, ACM TOMS 38(2):14 | [2049673.2049678.pdf](2049673.2049678.pdf) |
| 2 | Kronbichler & Wall (2016/2018), *A performance comparison of continuous and discontinuous Galerkin methods with fast multigrid solvers*, arXiv:1611.03029 | [A_performance_comparison_of_continuous_and_discont.pdf](A_performance_comparison_of_continuous_and_discont.pdf) |
| 3 | Burstedde, Wilcox, Ghattas (2011), *p4est: Scalable Algorithms for Parallel AMR on Forests of Octrees*, SIAM J. Sci. Comput. 33(3):1103–1133 (DOI 10.1137/100791634) | [100791634.pdf](100791634.pdf) |

Papers 1 and 3 are two halves of the same data-structure family: Paper 3 (p4est) is
the octree "oracle"; Paper 1 is the generic-FEM consumer that queries it. Their
single-core findings overlap heavily and are cross-referenced below.

---

## Paper 1 — Bangerth, Burstedde, Heister & Kronbichler (2011)

Generic adaptive-FEM algorithms and data structures. Most of the paper concerns
parallel distribution; the single-core-relevant findings are the data structures and
the *serial* algorithms it explicitly calls out.

- **A1 — Forest of trees + Morton encoding.** Each cell is identified by a tree number
  plus a Morton (z-order) path through that tree. The common coarse mesh "can be as
  small as 1" cell.
- **A2 — 2:1 mesh balance.** At most one hanging node per face/edge, enforced as a
  post-refinement step; "mostly for convenience, since it simplifies the creation of
  interpolation operators on interfaces between cells."
- **A3 — Space-filling-curve ordering for cache locality.** Enumerating cells/DOFs along
  the z-curve makes vector entries of neighboring DOFs adjacent in memory ⇒ low cache-miss
  rate.
- **A4 — Consistent global C⁰ DOF enumeration.** Vertices→lines→faces→cells, with shared
  interface DOFs deduplicated; serially this is a plain ascending enumeration with an
  O(1) "is this DOF constrained?" lookup.
- **A5 — Serial constraint elimination (the key serial recipe).** "For sequential
  computations, one can first assemble the linear system from all cell contributions
  irrespective of constraints and in a second step *eliminate* constrained degrees of
  freedom in an in-place procedure" (Bangerth & Kayser-Herold 2009, §5.2). Hanging-node
  constraints have the form `x_i = Σ_j c_ij x_j (+ b_i)`.
- **A6 — Constraint chains.** Chains arise in hp-adaptivity and with Dirichlet
  constraints; for a 2:1-balanced single-hanging-node mesh, hanging constraints do not
  chain.
- **A7 — Compressed `IndexSet`/`ConstraintMatrix`.** Sorted half-open intervals with
  prefix sums give O(log K) index→position queries. The paper is explicit that this
  compression is a *parallel* concern; serially a plain per-DOF array is the recommended
  O(1) structure.
- **A8 — Pre-compute the sparsity pattern.** Build the CRS pattern once before assembly
  to avoid repeated reallocation.
- **A9 — Marking via `nth_element`.** Find the refinement threshold with `nth_element`
  (O(N) average), not a full sort.
- **A10 — `SolutionTransfer` between meshes.** On refine/coarsen, interpolate/project the
  old solution onto the new mesh (relevant to time-dependent / nonlinear problems, to
  avoid recomputing from scratch).
- **A11 — (OUT OF SCOPE) Parallel machinery.** Ghost cells, parallel DOF enumeration via
  `MPI_Allgather`, distributed constraints, sparsity-pattern communication,
  repartitioning, the p4est oracle interface. Excluded by the single-core constraint.

---

## Paper 2 — Kronbichler & Wall (2016/2018)

A fair node-level performance comparison of CG, DG-SIP and HDG for the Poisson equation
with state-of-the-art multigrid solvers. The findings are per-core (single-node) and so
are in scope; the §4.5 massively-parallel scaling study is not.

- **B1 — Matrix-free ≫ matrix-based (headline).** Modern matrix-free operator evaluation
  with sum factorization is up to an order of magnitude faster *time-to-solution* than
  sparse matrix-vector products, because matrix-based iterative solvers are
  memory-bandwidth bound (a sparse mat-vec sits at the CRS bandwidth ceiling and cannot
  be improved).
- **B2 — Sum factorization.** Reduces per-element evaluation from O((k+1)^{2d}) to
  O(d·(k+1)^{d+1}) for tensor-product bases.
- **B3 — Even-odd decomposition.** Halves the 1-D kernel cost at higher degree.
- **B4 — Affine/Cartesian Jacobian reuse.** On Cartesian/affine cells the Jacobian is
  constant and is stored/precomputed once; reference-element data is reused.
- **B5 — CG static condensation.** Eliminate the (k−1)^d cell-interior DOFs via a Schur
  complement before the global solve; statically-condensed CG is the best matrix-based
  scheme (but still 3–20× slower than matrix-free CG).
- **B6 — Multigrid recipe.** A geometric-multigrid V-cycle preconditioning a CG Krylov
  outer iteration is the method of choice. A Chebyshev-accelerated point-Jacobi smoother
  (needs only the diagonal + a λ_max estimate) is ideal **for the Laplacian**; the paper
  is explicit that **non-diagonally-dominant / higher-order operators need block-relaxation
  or ILU smoothers instead** (its HDG trace operator is the example).
- **B7 — CG matrix-free wins across degrees.** Throughput per DOF is ~constant in degree
  for matrix-free; matrix-based throughput falls as rows get denser at higher degree.
- **B8 — Roofline.** Confirms matrix-free is compute-bound/favourable and sparse mat-vec
  is at the memory-bandwidth limit.
- **B9 — Scale/operator caveat (audit note, not a paper claim).** This study targets the
  **Laplacian** at **10⁶–10⁹ DOF solved iteratively**, where matrix-free + multigrid wins.
  The bathymetry smoother is a **≤~10⁵-DOF thin-plate + least-squares fit solved by a
  direct factorization**, a regime where a sparse direct solve is typically the *fastest
  and most robust* choice. This caveat frames the B1/B5 verdicts.

---

## Paper 3 — Burstedde, Wilcox & Ghattas (2011) — p4est

The forest-of-octrees AMR algorithms underlying `OctreeAdapter`/`QuadtreeAdapter`. The
paper is mostly parallel, but its encoding and local algorithms are exactly the
single-core foundation of DRIFTER's mesh layer.

- **P1 — Linear octree storage.** Store only the leaves, in a flat array; each octant is
  a fixed-length bit-interleaved Morton index + level (24 bytes), with no parent/child
  pointers.
- **P2 — Morton encoding + O(1) arithmetic octant operations.** A total order is obtained
  by sorting the Morton index; `Child_id`, `Parent`, `Descendants`, and
  `Face/Edge/Corner_neighbor` are O(1) bit operations (Algorithms 1–7), and neighbor
  lookup is a binary search in the sorted leaf array.
- **P3 — Integer-only topology, no floating point.** "No floating-point arithmetic is
  used, avoiding topological errors due to roundoff" — all neighbor/connectivity relations
  use integer coordinates.
- **P4 — Forest + interoctree connectivity.** Multiple trees with arbitrary relative
  orientations (exterior-octant transformations, Algorithms 8–13) support complex
  geometries; a single tree (K=1) is explicitly allowed.
- **P5 — 2:1 balance via insulation layer.** A two-stage balance (local single-tree
  balance, then an insulation-neighborhood pass) rather than an iterative ripple;
  O(N_p log N_p).
- **P6 — Refine/Coarsen as in-place array operations.** Refine splices children into the
  leaf array at the traversal point; Coarsen uses a sliding window; both O(N_p).
- **P7 — `Nodes`: globally unique node numbering.** Classify nodes as *independent* vs
  *hanging*; only independent nodes carry unknowns; nodes are canonicalized to the
  lowest participating tree and z-ordered.
- **P8 — Nonrecursive coarsen for field transfer.** Preferred "in order to interpolate a
  numerical field to the new mesh more easily."
- **P9 — Many small trees can beat one big tree** on cache performance.
- **(OUT OF SCOPE)** Partition/load-balance, Ghost layer, the parallel insulation
  communication in Balance, `Find_owners`, parallel `Nodes`/`Checksum`.

---

## Finding → code path → verdict

References to the code are clickable; line numbers were current at the time of writing.

### Against Paper 1 (deal.II generic AFEM)

| Finding | Code path | Verdict |
|---|---|---|
| **A1** forest + Morton | Single-root octree/quadtree, Morton encode (`Morton3D::encode`, `MortonUtil::refine`) — [octree_adapter.cpp:500](../../src/mesh/octree_adapter.cpp#L500), [quadtree_adapter.cpp:461](../../src/bathymetry/quadtree_adapter.cpp#L461) | **Match.** A single macro-element (K=1) is exactly what Paper 1/p4est permit for a simple rectangular domain. |
| **A2** 2:1 balance | `OctreeAdapter::balance()` auto-run after every refine — [octree_adapter.cpp:138](../../src/mesh/octree_adapter.cpp#L138), called at [:202](../../src/mesh/octree_adapter.cpp#L202); quadtree exposes `EdgeNeighborInfo` Coarse/FineToCoarse — [quadtree_adapter.hpp](../../include/bathymetry/quadtree_adapter.hpp) | **Match.** Same one-hanging-node invariant, per-axis. (Algorithm differs — see P5.) |
| **A3** z-curve ordering | `collect_leaves` pre-order DFS over Morton-ordered children ⇒ leaves in z-order — [quadtree_adapter.cpp:392](../../src/bathymetry/quadtree_adapter.cpp#L392); DOF numbering follows element order — [cg_bezier_dof_manager.cpp:190](../../src/bathymetry/cg_bezier_dof_manager.cpp#L190) | **Match.** Cells/DOFs lie on the z-curve. No extra bandwidth renumbering, but SparseLU applies its own fill-reducing reorder (AMD/METIS). |
| **A4** global C⁰ DOF enumeration | 3-pass vertex→edge→interior with shared-DOF dedup — [cg_bezier_dof_manager.cpp:190-249](../../src/bathymetry/cg_bezier_dof_manager.cpp#L190); `is_constrained` O(1) via hash set — [:66](../../src/bathymetry/cg_bezier_dof_manager.cpp#L66) | **Match in outcome.** Correct C⁰ sharing and O(1) constraint test. *Method deviation*: sharing is by geometric position hashing, not topology — see **P3**. |
| **A5** serial in-place elimination | `condense_matrix_and_rhs` assembles full Q then expands slave→master into the reduced free system — [constraint_condenser.cpp:44](../../src/bathymetry/constraint_condenser.cpp#L44); driven by `build_condensed_system` — [cg_cubic_bezier_bathymetry_smoother.cpp:135](../../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L135) | **Strong match.** This is precisely the Bangerth & Kayser-Herold §5.2 recipe. (The C¹ *edge* constraints are not of the form `x_i=Σc x_j` and are handled with KKT Lagrange multipliers — a sound choice outside A5's scope.) |
| **A6** constraint chains | 2:1 balance ⇒ single hanging node; no Dirichlet (data-fit problem) | **Match / N-A.** No hanging-node chains arise. |
| **A7** compressed IndexSet | `global_to_free_`/`free_to_global_` are plain `std::vector` arrays — [cg_bezier_dof_manager.cpp:479-492](../../src/bathymetry/cg_bezier_dof_manager.cpp#L479) | **Match (serial).** Paper 1 states the compressed interval structure is a parallel concern; serial arrays are the right call. |
| **A8** pre-computed sparsity | Triplets + `setFromTriplets` on every assembly — [cg_bezier_smoother_base.cpp:301](../../src/bathymetry/cg_bezier_smoother_base.cpp#L301), [:381](../../src/bathymetry/cg_bezier_smoother_base.cpp#L381) | **Gap (minor).** No reuse of a precomputed pattern, but each AMR iteration rebuilds the mesh and DOF numbering, so the pattern changes anyway; within a single solve `setFromTriplets` is standard. |
| **A9** `nth_element` marking | Full `std::stable_sort` then greedy Dörfler accumulation — [adaptive_cg_cubic_bezier_smoother.cpp:643](../../src/bathymetry/adaptive_cg_cubic_bezier_smoother.cpp#L643) | **Deviation (minor).** O(N log N) sort instead of O(N) `nth_element`; negligible at ≤10⁴ elements. |
| **A10** SolutionTransfer | Adaptive loop re-solves from raw data each iteration; `prev_solutions_` feeds only the coarsening metric — [adaptive_cg_cubic_bezier_smoother.cpp:537-555](../../src/bathymetry/adaptive_cg_cubic_bezier_smoother.cpp#L537) | **N-A / appropriate.** For a data-fitting problem the solution is fully determined by the fit; there is nothing to "transfer", and a direct solve needs no warm start. |
| **A11** parallel machinery | — | **Out of scope** (single-core). |

### Against Paper 2 (CG-vs-DG multigrid)

| Finding | Code path | Verdict |
|---|---|---|
| **B1** matrix-free ≫ matrix-based | Everything assembled (`H_global_`, `BtWB_global_`, `Q`) + direct **SparseLU** by default — [cg_bezier_smoother_base.cpp:396](../../src/bathymetry/cg_bezier_smoother_base.cpp#L396), [cg_cubic_bezier_bathymetry_smoother.cpp:261](../../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L261) | **Deviation, justified at scale.** This is the matrix-based path the paper argues against — but the paper's argument is about *large iterative* solves. At ≤~10⁵ DOF a direct sparse factorization is typically optimal, so the **default is well-chosen**. B1 becomes relevant only if the problem is scaled up or the optional iterative path is used at scale. |
| **B2** sum factorization | Data fitting forms dense 16×16 BᵀWB per Gauss point; `basis().evaluate(u,v)` recomputed at fixed reference points inside the per-element loop — [cg_bezier_smoother_base.cpp:336-364](../../src/bathymetry/cg_bezier_smoother_base.cpp#L336) | **Gap (minor here).** The reference basis tables are element-independent and could be precomputed once; sum factorization itself only matters if the operator goes matrix-free. |
| **B3** even-odd decomposition | not used | **N-A** (only relevant with sum-factorized matrix-free kernels). |
| **B4** affine Jacobian reuse | Thin-plate Hessian computed once, reused via `scaled_hessian(dx,dy)` — [cg_bezier_smoother_base.cpp:281](../../src/bathymetry/cg_bezier_smoother_base.cpp#L281); element-matrix cache keyed on (morton, level) — [:431](../../src/bathymetry/cg_bezier_smoother_base.cpp#L431) | **Partial match.** The reference-Hessian reuse *is* the affine optimization; the data-fitting basis evaluations are not hoisted (see B2). |
| **B5** CG static condensation | Not done — the 4 cubic interior DOFs/element stay in the global system; only hanging-node condensation + edge KKT | **Gap.** Interior-DOF static condensation would shrink the system; benefit is modest given the small direct solve. |
| **B6** MG smoother choice | `BezierMultigridPreconditioner` uses Colored Multiplicative Schwarz (block) smoother + SparseLU coarse — config in [cg_cubic_bezier_bathymetry_smoother.cpp:338-360](../../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L338) | **Appropriate.** The operator is a 4th-order thin-plate (biharmonic-like) energy, not the Laplacian; a block/Schwarz smoother is the right family here, matching Paper 2's own caveat that non-diagonally-dominant/high-order operators need block relaxation rather than point-Jacobi-Chebyshev. |
| **B7** high-degree throughput | Fixed cubic (k=3) | **N-A** (degree is not a variable). |
| **B8** roofline | — | covered by B1. |
| **B9** scale/operator caveat | Default direct solve at ≤~10⁵ DOF | frames B1/B5: the matrix-based default is **near-optimal for the actual size**. |

### Against Paper 3 (p4est)

| Finding | Code path | Verdict |
|---|---|---|
| **P1** linear octree storage | Pointer-based tree (`unique_ptr` children) + recursive `collect_leaves` — [quadtree_adapter.cpp:392](../../src/bathymetry/quadtree_adapter.cpp#L392); octree similarly | **Deviation.** A pointer tree, not a flat Morton-sorted leaf array. Fine at this scale; linear storage is leaner and more cache-friendly. The DFS still yields z-ordered leaves, so the downstream ordering benefit (A3) is retained. |
| **P2** Morton + O(1) arithmetic ops | Morton encoding present; but neighbor lookup uses a Boost R-tree + `precompute_neighbors` — [quadtree_adapter.cpp:405-419](../../src/bathymetry/quadtree_adapter.cpp#L405); octree uses `find_neighbor_same_or_coarser` tree-walk | **Partial.** Encoding matches; octant *navigation* is geometric (R-tree, O(log n)) rather than arithmetic Morton + binary search (O(1)/O(log N)). Comparable complexity, more memory, and uses floating-point geometry (see P3). |
| **P3** integer-only topology | DOF/edge identity via floating-point `quantize_position` (scale 1e8) and quantized edge midpoints — [cg_bezier_dof_manager.cpp:116](../../src/bathymetry/cg_bezier_dof_manager.cpp#L116), [:519-545](../../src/bathymetry/cg_bezier_dof_manager.cpp#L519) | **Deviation (robustness) — the most notable flag.** p4est deliberately avoids floating point for topology to prevent roundoff errors; DRIFTER's DOF sharing and edge matching rest on a quantization tolerance. It works for axis-aligned quadtrees with clean coordinates, but topological (index-based) sharing would be strictly more robust and also better matches the project's "never use fallback methods" rule. |
| **P4** forest + interoctree transforms | Single root tree — [adaptive_cg_cubic_bezier_smoother.cpp:26-31](../../src/bathymetry/adaptive_cg_cubic_bezier_smoother.cpp#L26) | **N-A by design.** K=1 suffices for one rectangular bathymetry domain; no interoctree transformations needed. |
| **P5** insulation-layer 2:1 balance | Iterative ripple loop (scan leaves, refine any neighbor >1 level coarser, repeat) — [octree_adapter.cpp:138-182](../../src/mesh/octree_adapter.cpp#L138) | **Match (invariant), simpler algorithm.** Enforces the identical 2:1 constraint; the insulation-layer approach mainly saves *parallel communication*, which is out of scope. |
| **P6** array-splice refine/coarsen | `create_children` in the pointer tree + full `rebuild_leaf_list` — [octree_adapter.cpp:184](../../src/mesh/octree_adapter.cpp#L184), [quadtree_adapter.cpp:380](../../src/bathymetry/quadtree_adapter.cpp#L380) | **Deviation (minor).** O(N) leaf-list rebuild per refine vs in-place splice; immaterial at this scale. |
| **P7** independent/hanging node numbering | 3-pass numbering + hanging classification via FineToCoarse edges and de Casteljau extraction — [cg_bezier_dof_manager.cpp:375-477](../../src/bathymetry/cg_bezier_dof_manager.cpp#L375) | **Match conceptually.** The independent-vs-hanging distinction and z-ordered numbering align with `Nodes`; the identification is geometric rather than topological (P3). |
| **P8** nonrecursive coarsen for transfer | re-fit from data each iteration | **N-A** (no field to interpolate; see A10). |
| **P9** many small trees vs one | Single tree | **N-A / minor** at this scale. |
| Parallel `Partition`/`Ghost`/`Find_owners`/`Nodes` comms | — | **Out of scope** (single-core). |

---

## Prioritized single-core fix list

Ordered by value/effort. None changes the program's correctness today; items 1–4 are
small and worthwhile, item 5 only matters if the problem grows, item 6 is optional.

1. **(Robustness, medium) Make DOF/edge sharing topological, not floating-point.**
   Replace `quantize_position`-based DOF and edge identification with index-based sharing
   derived from quadtree connectivity (the p4est P3 / deal.II A4 approach). This removes
   the roundoff-tolerance dependence and aligns with the project rule "never use fallback
   methods of any kind." Touches
   [cg_bezier_dof_manager.cpp](../../src/bathymetry/cg_bezier_dof_manager.cpp).
2. **(Perf, low) Hoist the reference basis tables.** `basis().evaluate(u,v)` at the fixed
   reference Gauss points is element-independent; precompute it (and the per-element-size
   BᵀWB structure) once instead of recomputing inside the per-element loop —
   [cg_bezier_smoother_base.cpp:336-364](../../src/bathymetry/cg_bezier_smoother_base.cpp#L336)
   (Paper 2 B2/B4).
3. **(Perf, medium) Optional interior-DOF static condensation.** Eliminate the 4 cubic
   interior DOFs per element via a local Schur complement before the global direct solve
   (Paper 2 B5). Modest benefit given the small system, but it shrinks `Q`.
4. **(Perf, low) Use `std::nth_element` for the Dörfler threshold** instead of a full
   `std::stable_sort` —
   [adaptive_cg_cubic_bezier_smoother.cpp:643](../../src/bathymetry/adaptive_cg_cubic_bezier_smoother.cpp#L643)
   (Paper 1 A9).
5. **(Scalability, large — only if problem size grows or the iterative path is used at
   scale) Matrix-free, sum-factorized operator evaluation** for the hot `Q`-matvec /
   smoother apply (Paper 2 B1/B2/B3), keeping direct SparseLU as the small-problem
   default. The existing Colored-Schwarz multigrid smoother is already appropriate for
   the 4th-order operator (B6), so this is purely about the operator-apply kernel.
6. **(Optional, low priority) Linear (flat Morton-sorted) octree storage** (p4est P1) if
   memory/cache ever becomes a concern — likely unnecessary at current sizes.

---

## Overall verdict

The bathymetry program implements the **same conceptual structures** the three papers
advocate: a forest-of-octrees mesh with Morton/z-ordering (A1/P1–P2), 2:1 balance
(A2/P5), in-place hanging-node condensation exactly per Bangerth & Kayser-Herold §5.2
(A5), and de Casteljau hanging-node constraints with an independent/hanging-node
distinction (A4/P7). For the program's regime — **single core, single domain, ≤~10⁵
DOF, a thin-plate + least-squares fit** — its two biggest engineering choices are
sound:

- **Direct SparseLU rather than matrix-free + multigrid (Paper 2 B1).** Justified: at
  this size a direct sparse factorization is typically the fastest and most robust
  option. The matrix-free thesis would only pay off at much larger scale or on the
  optional iterative path.
- **Block (Colored Schwarz) multigrid smoother (Paper 2 B6).** Appropriate for the
  4th-order thin-plate operator — the paper itself recommends block relaxation over
  point-Jacobi-Chebyshev for non-diagonally-dominant / high-order operators.

The choices that genuinely *differ* from the papers are concentrated in the mesh/DOF
layer and are reasonable-but-improvable rather than wrong: a pointer tree instead of
linear octree storage (P1), geometric R-tree navigation instead of arithmetic Morton
(P2), and — the one substantive flag — **floating-point position quantization for
topology (P3)**, which p4est deliberately avoids and which the prioritized fix list
addresses first.

**Bottom line:** the implementation is close to optimal *for its stated single-core,
modest-scale problem*. It is not optimal in the asymptotic sense the papers care about
(matrix-free at millions of DOF), but those findings do not apply at this scale. The
most worthwhile single-core changes are the small robustness/efficiency items 1–4
above; the matrix-free rework (item 5) should be deferred until the problem size
demands it.
