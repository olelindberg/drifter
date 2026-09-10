# Coastline Adaptivity — Circumradius-Driven Refinement and the Water Mask

This document derives the two stages that stand between a vector coastline dataset and the linear
system the bathymetry smoothers assemble. The first is *geometric*: a discrete circumradius is
computed along the shoreline, indexed spatially, and used to drive a refinement pre-pass that
resolves the stretches of coast that turn sharply. The second is *classificatory*: the elements
that pre-pass produces are sorted into water, beach and inland, and only the water elements carry
a problem worth solving.

The two stages meet at the mesh and nowhere else. The first knows nothing about depth values; the
second knows nothing about vector geometry. What connects them is that the coastline is exactly
where the classification changes, so resolving it geometrically is what makes the classification
sharp.

Every claim is anchored to source. Three results are load-bearing and stated as such: the
refinement criterion is a shape comparison and not a distance comparison (§7.1), the sweep has a
geometric fixed point at the circumradius of the features an element contains but no floor beneath
it, so the resolution limits decide where it actually stops (§7.2), and inland elements can be
dropped from assembly only because adjacency is defined by shared node rather than shared edge
(§12).

**Scope:** `CoastlineReader` and `CoastlineIndex` (`mesh/coastline_refinement.hpp`), the
`refine_coastline()` pre-pass as implemented in both `AdaptiveCGHermiteSmoother` and
`LinearMeshGenerator`, and `ElementDataMask` (`bathymetry/element_data_mask.hpp`) with its effect
on the assembled system. The system itself is derived in
[hermite_bathymetry_system.md](hermite_bathymetry_system.md); the error-driven refinement that
follows this pre-pass is studied in
[uniform_vs_adaptive_convergence.md](uniform_vs_adaptive_convergence.md).

---

## 1. Overview

The chain has six links, three per stage:

$$
\underbrace{\text{vector file} \;\to\; \mathcal{P} \;\to\; \mathcal{C} \;\to\; \{(v, R(v))\}}
_{\text{geometric (§2–§5)}}
\;\to\;
\underbrace{\big(\exists\, v \in B_E : R(v) < h_E\big) \;\to\; \text{mesh}}_{\text{criterion (§6–§9)}}
\;\to\;
\underbrace{\{\text{Water},\ \text{Beach},\ \text{Inland}\}}_{\text{classification (§10–§13)}}
$$

Reading left to right: a georeferenced vector dataset is filtered to the domain, reprojected and
flattened into ordered two-dimensional polylines $\mathcal{P}$; polylines that continue one another
are stitched into chains $\mathcal{C}$, because the circumradius is a three-point quantity; each
interior vertex receives a discrete circumradius $R(v)$; the vertices and their
radii are put into a spatial index; the mesh is refined while any element is larger than the
tightest feature it contains; and the resulting elements are classified by whether they hold data.

The pre-pass runs **once**, before the Dörfler-marked error loop, and is idempotent.

### Symbols and dimensions

| Symbol | Meaning | Built at |
|---|---|---|
| $\mathcal{P}$ | Ordered polylines in the working SRS, stored as concatenated vertices plus offsets | [coastline_refinement.cpp:279-375](../src/mesh/coastline_refinement.cpp#L279-L375) |
| $\mathcal{S}$ | The $n-1$ segments implied by each polyline of $\mathcal{P}$; never stored | [:132-141](../src/mesh/coastline_refinement.cpp#L132-L141) |
| $\mathcal{C}$ | Chains: maximal runs of polylines joined end to end; $\lvert\text{chain}\rvert \geq 3$ | [:526-608](../src/mesh/coastline_refinement.cpp#L526-L608) |
| $\mathcal{V}$ | Interior vertices of $\mathcal{C}$ with finite radius, inside the domain — the circumradius samples | [:855-884](../src/mesh/coastline_refinement.cpp#L855-L884) |
| $R(v)$ | Discrete circumradius at vertex $v$; curvature is its reciprocal $\kappa = 1/R$ and is never computed | [:611-636](../src/mesh/coastline_refinement.cpp#L611-L636) |
| $S(B,\tau)$ | Whether an element box $B$ holds a vertex with $R(v) < \tau$ | [:937-947](../src/mesh/coastline_refinement.cpp#L937-L947) |
| $h_E$ | Element size $\min(\Delta x_E, \Delta y_E)$; the criterion's threshold | [quadtree_adapter.hpp](../include/bathymetry/quadtree_adapter.hpp) |
| $\ell_E,\ \ell_{\max}$ | Element refinement level and its cap | `coastline.max_level` |
| $\rho(x,y)$ | Local raster resolution in metres | [multi_source_bathymetry.hpp](../include/mesh/multi_source_bathymetry.hpp) |
| $r$ | Hermite continuity order; $(r+1)^2$ DOFs per corner | [cg_hermite_dof_manager.hpp:63](../include/bathymetry/cg_hermite_dof_manager.hpp#L63) |

---

## 2. Vector input and the domain filter

The reader opens the dataset through GDAL's vector driver, selects a layer by name or takes the
first, and builds a coordinate transform from the layer's spatial reference to the working SRS —
[coastline_refinement.cpp:279-375](../src/mesh/coastline_refinement.cpp#L279-L375).

Both spatial references are put into traditional GIS axis order —
[:321-322](../src/mesh/coastline_refinement.cpp#L321-L322). This matters for geographic
coordinate systems: EPSG:4326 is officially defined as latitude-then-longitude, while every
shapefile written in practice stores longitude first. Forcing traditional order makes the stored
ordering the authoritative one, so a coastline in EPSG:4326 reprojects to a projected working SRS
such as EPSG:3034 without a transposition.

### 2.1 The four-corner envelope

Reading a global coastline dataset in full is not viable, so the layer is filtered on the GDAL
side before any feature is materialized. The filter rectangle has to be expressed in the *source*
SRS, which requires mapping the domain box backwards through the inverse transform.

A rectangle in the target SRS is not a rectangle in the source SRS — projection curves its edges.
Transforming two opposite corners is therefore insufficient. All four corners are transformed and
their envelope taken — [:337-346](../src/mesh/coastline_refinement.cpp#L337-L346):

$$
\left[\min_i x_i,\ \max_i x_i\right] \times \left[\min_i y_i,\ \max_i y_i\right],
\qquad
(x_i, y_i) = T^{-1}\big(\text{corner}_i\big),\quad i = 0,\dots,3
$$

This is a conservative outer bound: the envelope of the transformed corners contains the
transformed rectangle whenever the projection's edge curvature does not exceed the corner spread,
which holds for any domain small enough that the transform is close to affine over it. The
envelope is then handed to `SetSpatialFilterRect` — [:348](../src/mesh/coastline_refinement.cpp#L348).

### 2.2 Flattening to polylines

Each surviving feature is cloned, flattened to two dimensions, and walked recursively —
[:168-193](../src/mesh/coastline_refinement.cpp#L168-L193). Line strings contribute their vertices
in order; polygons contribute the exterior ring **and every interior ring**, so lakes and inner
harbours are coastline too; multi-geometries and geometry collections recurse. Coordinates are
transformed in bulk per line string rather than per point.

The output is $\mathcal{P}$, held in CSR form — every vertex once in a flat array, plus one offset
per polyline — [:54-59](../src/mesh/coastline_refinement.cpp#L54-L59). Segments are *implied*:
segment $j$ of polyline $k$ runs from vertex $\text{offsets}[k]+j$ to its successor, so the shared
vertex between consecutive segments is stored once rather than twice. Nothing downstream stores a
segment; the consumers that want segment semantics unpack them on the fly through a single
`for_each_segment` helper — [:132-141](../src/mesh/coastline_refinement.cpp#L132-L141).

---

## 3. Chain stitching

The circumradius at a vertex needs its two neighbours, so it needs *ordering*. That ordering is
**preserved from the input** rather than rebuilt: an OSM split line and a polygon ring are already
ordered, and §2.2 keeps them that way. What remains is only to join polylines that continue one
another, so a vertex where two features meet still gets a sample.

`build_chains()` — [:526-608](../src/mesh/coastline_refinement.cpp#L526-L608) — therefore hashes
**only the two endpoints of each polyline** onto a quantized integer grid, giving an adjacency map
from endpoint to incident polylines. A greedy walk then starts at any unused polyline and
repeatedly hops to an unused polyline sharing the current end, recording for each whether it is
traversed forwards or backwards, until no continuation remains. A chain is thus a run of polyline
ids, and its vertices are enumerated without ever being copied — consecutive links share a vertex,
which is emitted once.

This is what keeps the pass affordable at continental scale. A 2000 km domain over the OSM
coastline holds ~16.2 M segments across ~260 k polylines, so the endpoint map has ~520 k entries
rather than the ~32 M a per-segment adjacency would need.

The walk yields maximal chains under the adjacency relation. Chains of fewer than three vertices
are discarded — [:602](../src/mesh/coastline_refinement.cpp#L602) — because they contain no
interior vertex and therefore contribute no circumradius sample.

The result is

$$
\mathcal{C} = \Big\{ (p_0, p_1, \dots, p_{n-1}) \;:\; n \geq 3 \Big\},
$$

whose interior vertices $p_1, \dots, p_{n-2}$ are the sample points of §4.

> **The decomposition into chains is not unique, and does not have to be.** At a junction where
> three or more polylines meet, the walk picks one continuation and the others start chains of
> their own. The criterion in §7 depends only on the *set* of (vertex, radius) pairs, never on
> which chain a vertex belongs to or on the direction of traversal, so any consistent decomposition
> produces the same refinement. Only the circumradius-comb diagnostic (§4.1) is sensitive to
> orientation, and there only in the sign of the drawn normal.

---

## 4. Discrete circumradius

> **Circumradius, not curvature.** The quantity computed, indexed and compared everywhere below is
> the circumradius $R$ — a *length*. Curvature is its reciprocal, $\kappa = 1/R$, and is never
> computed or stored anywhere in the code. The two are mutually exclusive names for reciprocal
> quantities, so this document and the implementation use "circumradius" throughout, and reserve
> "curvature" for the geometric notion it names.

For three consecutive vertices $p_0, p_1, p_2$ define the leg vectors and side lengths

$$
v_1 = p_1 - p_0, \qquad v_2 = p_2 - p_1, \qquad
a = \lVert v_1 \rVert, \quad b = \lVert v_2 \rVert, \quad c = \lVert p_2 - p_0 \rVert .
$$

The discrete circumradius at $p_1$ is the **circumradius of the triangle** $(p_0, p_1, p_2)$ —
[:611-636](../src/mesh/coastline_refinement.cpp#L611-L636). Starting from the classical
circumradius identity and substituting twice the signed area of the triangle,
$2A = \lvert v_1 \times v_2 \rvert$ where $v_1 \times v_2 = v_{1x} v_{2y} - v_{1y} v_{2x}$:

$$
R(p_1) \;=\; \frac{abc}{4A}
\;=\; \frac{a\,b\,c}{2\,\lvert v_1 \times v_2 \rvert},
\qquad
\kappa(p_1) \;=\; \frac{1}{R(p_1)} \;=\; \frac{2\,\lvert v_1 \times v_2 \rvert}{a\,b\,c} .
$$

A collinear triple has no circumcircle, and the cross product vanishes with it. The guard is
written relative to the leg lengths — [:629-631](../src/mesh/coastline_refinement.cpp#L629-L631):

$$
\lvert v_1 \times v_2 \rvert < 10^{-12}\, a\, b
\quad\Longleftrightarrow\quad
\lvert \sin \theta \rvert < 10^{-12}
\qquad\Longrightarrow\qquad
R(p_1) = +\infty ,
$$

with $\theta$ the turning angle between the legs. Writing the test on $\sin\theta$ rather than on
the raw cross product makes it scale-invariant: a straight coast sampled at 10 m spacing and the
same coast sampled at 10 km spacing are both recognised as straight.

Infinite radii are never inserted into the index —
[:859-863](../src/mesh/coastline_refinement.cpp#L859-L863). A straight shoreline therefore exerts
no refinement pressure at all, however long it is and however many vertices describe it.

> **Why the circumradius is the right discrete measure here.** It is exact on the case that
> matters: if $p_0, p_1, p_2$ are sampled from a circle of radius $\varrho$, the circumradius is
> $\varrho$ regardless of the sample spacing. So the quantity does not drift as the coastline
> dataset's vertex density varies, which it does substantially between digitized survey sections.
> It is also positively homogeneous of degree one — scaling the geometry by $s$ scales $R$ by $s$
> — which is what makes §7 a comparison of a length against a length, with the configured ceiling
> the only constant in it.

### 4.1 Normal toward the circumcentre

The refinement criterion needs only $R$, but the diagnostic output also needs a direction. The
normal at $p_1$ is built from the averaged unit tangent —
[:639-683](../src/mesh/coastline_refinement.cpp#L639-L683):

$$
\hat{t} = \frac{\hat{v}_1 + \hat{v}_2}{\lVert \hat{v}_1 + \hat{v}_2 \rVert},
\qquad
\hat{n} = \operatorname{sign}(v_1 \times v_2)\; \mathsf{R}_{90}\, \hat{t},
\qquad
\mathsf{R}_{90} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} .
$$

The sign flip points $\hat{n}$ at the concave side, i.e. toward the circumcentre. Where the two
legs are antiparallel — a hairpin, at which the averaged tangent degenerates — the normal falls
back to the rotated first leg.

The circumradius comb draws one tooth per interior vertex, of length $\min(R,\ 10^4\,\text{m})$,
and tags each tooth with its untruncated radius as cell data —
[:687-781](../src/mesh/coastline_refinement.cpp#L687-L781). Long teeth mark flat coast, short
teeth mark tight features; the clamp keeps a nearly straight section from drawing a tooth the size
of the domain.

The tooth carries **no scale factor**: it is drawn in the same length units as the mesh, which
makes the criterion of §7 readable straight off the picture — a tooth longer than the element it
sits in is a feature the element already resolves, and one shorter than the element is a feature
that should have refined it, unless a limit of §8 held it back. This is purely a visualization:
nothing in §6–§9 reads it.

---

## 5. Spatial indexing

Two Boost.Geometry R\*-trees of node capacity 16 hold the results —
[:46-52](../src/mesh/coastline_refinement.cpp#L46-L52):

| Tree | Key | Value | Query used | Built |
|---|---|---|---|---|
| Segment tree | segment bounding box | — | box overlap (`intersects`) | on first query |
| Circumradius tree | vertex point | radius $R(v)$ | box overlap (`intersects`) | by `build_index()` |

The segment tree answers "does the coastline pass through this box"; the circumradius tree answers
"does this box hold a feature sharper than a given length". **Only the second drives refinement**,
and that asymmetry is reflected in what `build_index()` —
[:790-908](../src/mesh/coastline_refinement.cpp#L790-L908) — actually does: it records which
segments fall inside the domain, but defers building a tree over them until something calls
`intersects()`. The pre-pass never does, so on a continental domain that tree is never built.

Both trees are **bulk-loaded** through the rtree's packing constructor rather than by repeated
`insert()`. The distinction is not a micro-optimization: `insert()` runs the R\* insertion path,
forced reinsertion and split heuristic included, once per element, whereas the packing constructor
performs a single STR-style build. On the 2000 km domain (~14.4 M samples) this is the difference
between minutes and seconds. Note that `rtree::insert(first, last)` does *not* pack — only the
constructor does.

Samples are filtered to the domain box before packing. The GDAL filter of §2.1 works on the
source-SRS envelope, which is strictly larger than the target-SRS domain box, so without this a
few percent of samples sit outside the domain and slow every query.

The index exposes no nearest-neighbour and no distance query. Every downstream question is
answered by box overlap against an element's bounds, which is what keeps the per-element cost at
one R-tree descent and makes the criterion in §7 a purely local test.

The index shares the reader's CSR vertex storage rather than copying it, and keeps only a list of
in-domain segment ids of its own, so it stays valid after the reader is destroyed without
duplicating hundreds of megabytes of geometry.

---

## 6. The element circumradius query

For an element with bounding box $B$ and a threshold length $\tau$,
`has_circumradius_below()` reports —
[:937-947](../src/mesh/coastline_refinement.cpp#L937-L947):

$$
S(B, \tau) \;=\;
\begin{cases}
\text{true} & \exists\, v \in \mathcal{V} \cap B : R(v) < \tau, \\[4pt]
\text{false} & \text{otherwise.}
\end{cases}
$$

The threshold is supplied by the caller; §7 passes the element's own size. Two properties of this
definition carry the whole scheme.

**It is a predicate, not a reduction.** The query answers whether the box holds a feature tighter
than $\tau$, and nothing is clamped, floored or averaged on the way out. That matters because the
sharpest vertex in a box is the one that should decide: a box holding one tight inlet among a
hundred gentle vertices is refined on the strength of the inlet alone. The implementation exploits
this by stopping at the first qualifying sample rather than gathering every sample in the box —
which is what keeps a coarse-level query, whose box may cover the entire coastline, cheap.

There is **no floor** on the radius. An earlier revision clamped the reported radius from below by
a configured $R_{\min}$, which silently converted a noise floor into a target coastal element size;
the criterion now sees the true circumradius, and the limits of §8 are the only thing that bounds
how far a very tight feature is chased.

**An empty box reports false.** No coastline vertex inside the element means no refinement
pressure, with no special case and no distance cutoff to calibrate. This is what makes the
criterion self-limiting away from the coast, and it is the reason the pre-pass can be run over
every element of the mesh on every sweep without a proximity pre-filter.

---

## 7. The refinement criterion

An element $E$ is marked for refinement exactly when it holds a coastline feature tighter than the
element itself — the threshold of §6 is the element's own size:

$$
\boxed{\;\text{refine } E \iff \exists\, v \in \mathcal{V} \cap B_E : R(v) \;<\; h_E,
\qquad h_E = \min(\Delta x_E,\, \Delta y_E)\;}
$$

The comparison is strict, and $h_E$ is the **shorter** side, so an anisotropic element is sized by
its narrow dimension. Implemented identically in both pipelines —
[adaptive_cg_hermite_smoother.cpp:449-462](../src/bathymetry/adaptive_cg_hermite_smoother.cpp#L449-L462)
and
[linear_mesh_generator.cpp:133-144](../src/bathymetry/linear_mesh_generator.cpp#L133-L144).

Both sides are lengths in the working SRS, so the test is dimensionless and scale-covariant, and it
carries **no tuning constant at all** — the coastline section names no parameter beyond `max_level`.
Because $h_E$ halves with every sweep while $R(v)$ is fixed by the data, an element stops as soon as
it is smaller than the tightest feature it contains (§7.2).

### 7.1 Why it is not a distance criterion

> **Distance to the coastline never appears.** Not in the index (§5), not in the query (§6), not
> in the criterion. Refinement is driven by the *shape* of the shoreline inside an element, not by
> the element's proximity to it.

The three cases follow directly from §6:

| Configuration of $E$ | Sharpest $R(v)$ in $B_E$ | Marked |
|---|---|---|
| No coastline vertex inside $E$ | none | no |
| Straight coast crossing $E$ | $+\infty$, never indexed (collinear, §4) | no |
| Fjord, inlet or headland inside $E$ | finite, small | yes, while it is $< h_E$ |

The second row is the substantive one. A long straight coast is resolved by the coarsest element
that contains it, at zero refinement cost, because the geometry there is already exactly
representable. Effort is spent only where the shoreline turns — which is where an element that
straddles land and water would otherwise carry a boundary too complex for its size, and where the
classification of §11 would otherwise be ambiguous.

A distance criterion would behave the opposite way: it would refine uniformly along a straight
coast, spending its budget on the sections that need it least.

### 7.2 Fixed point of the sweep

Refinement halves the element while $R(v)$ is a property of the data, so the criterion is monotone
in the right direction: an element that fires at size $h$ becomes children of size $h/2$, and it
stops as soon as $h_E \leq R_E^\star$, where

$$
R_E^\star \;=\; \min_{v \,\in\, \mathcal{V} \cap B_E} R(v)
$$

is the tightest feature the element still contains. The sweep therefore leaves coastal elements at

$$
h_E \;\in\; \left(\tfrac{1}{2} R_E^\star,\; R_E^\star\right],
$$

with the upper endpoint attained exactly when the base mesh size is a power-of-two multiple of
$R_E^\star$. Away from the coast $R_E^\star = \infty$ and the base size is untouched.

The fixed point is geometric, but it is **not by itself a bound on cost**: $R_E^\star$ is whatever
the dataset contains, and a digitized shoreline can carry sub-metre circumradii. Nothing floors it —
the clamp that used to (§6) is gone — so the admissibility limits of §8 are what stop the sweep on
real data, and the pre-pass reports which one held:

| Stop | Test | Reported as |
|---|---|---|
| Element budget | `num_elements >= max_elements`, at the top of each sweep | "maximum elements reached" |
| Level cap | $\ell_E \geq \ell_{\max}$ | "maximum refinement level reached" |
| Data resolution | children would fall below the raster limit (§8) | "data resolution limit reached" |
| Fixed point reached | every element is smaller than the features it holds | "coastline resolved to the circumradius of its features" |

This fixed point is verified analytically in the integration suite. The test coastline is a
sawtooth with legs $a = b = \sqrt{125}$ and chord $c = 10$, giving

$$
R = \frac{abc}{2\,\lvert v_1 \times v_2 \rvert} = \frac{125 \cdot 10}{200} = 6.25 ,
$$

one radius shared by the whole coastline. `CoastlinePrePassRefinesTowardTheCoastline`
(`tests/integration/test_adaptive_cg_hermite_smoother.cpp:422`) asserts that the smallest element
side after the pre-pass equals $6.25$ to within $10^{-10}$ — that the sweep converges *to* the
circumradius and, because the comparison is strict, does not step below it. Its companion
`CoastlinePrePassLeavesAnAlreadyFineMeshAlone` starts from a $32\times32$ mesh of $3.125$-wide
elements and asserts zero sweeps, which is what pins the threshold to $h_E$ rather than to any
configured length.

### 7.3 Sweep structure

The outer loop repeats until a full pass marks nothing —
[adaptive_cg_hermite_smoother.cpp:449-513](../src/bathymetry/adaptive_cg_hermite_smoother.cpp#L449-L513).
Each pass rescans the mesh from element zero, because refinement rebalances the quadtree to a 2:1
level ratio and invalidates every element index.

That rebalancing also means the marked set is a *lower bound* on what is refined: enforcing 2:1
across a newly created T-junction can refine a neighbour that the criterion did not select. The
fixed point of §7.2 is unaffected — rebalancing only ever makes elements smaller, and small
elements do not fire.

When a pass marks nothing, the elements it skipped at a limit are re-tested with the same criterion
so the reported reason names the limit that actually held, rather than claiming the coast was
resolved.

---

## 8. Admissibility limits

The criterion says what *should* be refined; a separate predicate says what *may* be. Since
refinement halves an element, every resolution test is written on the would-be children —
[adaptive_cg_hermite_smoother.cpp:196-235](../src/bathymetry/adaptive_cg_hermite_smoother.cpp#L196-L235):

| Limit | Test | Bound from |
|---|---|---|
| Level cap | $\ell_E \geq \ell_{\max}$ | `coastline.max_level` |
| Pixel resolution | $\min\!\big(\tfrac{\Delta x_E}{2},\, \tfrac{\Delta y_E}{2}\big) < s_{\min}$ | `min_element_size`, or $\rho$ when it is zero |
| Data density | $\dfrac{\Delta x_E}{2\rho} \cdot \dfrac{\Delta y_E}{2\rho} < N_{\min}$ | `min_data_points_per_element` |
| Element budget | $\lvert \mathcal{E} \rvert \geq$ `max_elements` | checked once per sweep |
| Pinned element | `mask.is_pinned(E)` | §11 |

The pixel and density limits express the same idea from two directions: refining below the raster
resolution adds degrees of freedom that no measurement constrains. The first bounds the child's
side, the second its pixel count.

The resolution $\rho$ is **per element**, sampled at the element centre through
`MultiSourceBathymetry::get_min_element_size_meters()`. A high-resolution survey tile therefore
permits finer coastal refinement inside its footprint than the primary raster allows outside it,
without any per-region configuration.

The element budget is checked once per sweep rather than per element, so a sweep completes the
marking it started; the bound is honoured at sweep granularity. It exists because §7 has no floor
on the circumradius: coastline detail far below the mesh scale would otherwise drive the entire
coast to the pixel limit before the first solve.

The pinned-element test is shared with the error-driven loop and is **inert during the pre-pass**,
since no smoother — and hence no mask — exists yet. That is the intended behaviour: the coast is
exactly where the pinned beach elements will be, and resolving them is the point of the pre-pass.

### 8.1 A floor per element, not a stop for the mesh

Because $\rho$ is per element, so is every limit built on it: an element reaching its floor says
nothing about the rest of the mesh. The error-driven loop therefore applies the predicate **at
marking time** — `select_elements_for_refinement` marks only among elements passing `can_refine`,
so an element parked at its floor consumes no Dörfler budget and the greedy walk keeps descending
the error list to elements that still have room.

Two consequences follow, both visible in the reported result:

- `ConvergenceReason::PixelResolution` is raised only when **no** element in the mesh may be
  refined. A run whose steepest features sit on the coarsest data keeps adapting everywhere else.
- The `error_threshold` test is against `max_refinable_error`, not `max_error`. Elements at their
  floor can hold error above the threshold indefinitely; testing the mesh maximum would spend
  every remaining iteration refining elsewhere for nothing. When the two differ, the log says so
  and reports both — a converged run may legitimately leave large residual error on elements the
  data cannot resolve, and that is a signal to supply finer data rather than to refine harder.

The floor bounds *marking*, not element size. Where $\rho$ varies sharply in space, 2:1 balancing
across the jump can still split an element below its own floor; a valid quadtree takes precedence.

---

## 9. The two pipelines

Both applications run the same criterion over their own mesh generator:

| | highrider (`AdaptiveCGHermiteSmoother`) | lowrider (`LinearMeshGenerator`) |
|---|---|---|
| Entry point | [`refine_coastline()`:440-530](../src/bathymetry/adaptive_cg_hermite_smoother.cpp#L440-L530) | [`refine_coastline()`:127-220](../src/bathymetry/linear_mesh_generator.cpp#L127-L220) |
| Element | Hermite, $(r+1)^2$ DOFs per corner | bilinear, 4 DOFs per element |
| Criterion | $\exists\, v \in B_E : R(v) < h_E$ | identical |
| After refining | `refine_octree_only()`, then discards the smoother | `refine_elements()`, then rebuilds the surface |
| Budget bound | `max_elements`, per sweep | not applied in the pre-pass |
| Idempotence | `coastline_refined_` flag, cleared by `set_coastline()` | re-entrant; called once by `Lowrider::run()` |

The structural difference is what each pipeline must rebuild after the mesh moves. The highrider
refines the quadtree only and drops any smoother built from the previous mesh, deferring all
assembly until the first solve; the lowrider's surface is fitted by sampling rather than solving,
so it can rebuild immediately and cheaply.

The pre-pass is wired only for the Hermite family —
[drifter.cpp:61-67](../src/core/drifter.cpp#L61-L67) — through an `if constexpr (requires ...)`
test on `set_coastline`, so the Bézier smoothers compile against the same template and simply do
not receive a coastline.

Both run the pre-pass to completion **before** the first solve, and the error-driven loop then
starts from the coastline-resolved mesh. The ordering is not incidental: error-driven refinement
measures the misfit of a *fitted* surface, and a surface fitted across an unresolved shoreline
misfits everywhere along it, which would spend the error budget rediscovering geometry that the
vector data already states exactly.

---

## 10. Why the geometric stage is not enough

The pre-pass sizes elements to the shoreline. It does not say what to solve inside them, and the
naive answer is wrong in a way that is worth stating precisely.

Land, missing data and out-of-coverage all evaluate to depth $0$ through the raster path. If those
zeros enter the least-squares term as ordinary observations, the data term asks the surface to
interpolate a discontinuity: depth $0$ on one side of the shoreline, tens or hundreds of metres a
few pixels away, with no intermediate measurements. The smoothness operator resists, the data term
insists, and the minimizer of

$$
J(x) \;=\; \alpha\, x^\top H x \;+\; \lambda\left(x^\top B^\top W B x - 2 x^\top B^\top W z + z^\top W z\right)
$$

settles on an oscillation between them. A Hermite element has no convex-hull property to bound
that oscillation — unlike a Bernstein basis, its coefficients are values and derivatives, and
nothing constrains the surface to the range of its DOFs. The excursion is therefore unbounded in
principle and large in practice.

The fix is to change the granularity of the decision. A single sample is the wrong unit: whether
one quadrature point has a measurement says nothing about whether the *region* around it poses a
well-posed fitting problem. The decision moves from the point to the element.

---

## 11. Element classification

`ElementDataMask` sorts every element of the mesh into three classes —
[element_data_mask.hpp:40-44](../include/bathymetry/element_data_mask.hpp#L40-L44):

| Class | Definition | Treatment |
|---|---|---|
| Water | at least one sample is a real depth reading | assembled and solved |
| Beach | no water data, but shares a node with a Water element | all corner DOFs pinned to $0$ |
| Inland | no water data, and no Water element shares a node | dropped from assembly and from output |

"Not water" covers both land and missing data. The interior of a landmass and the interior of a
survey hole pose the same problem — no data, no boundary influence — and are treated the same.

The beach class is the essential one: it is the pinned rim that supplies the water region its
Dirichlet boundary. Without it the water elements would have a free boundary at the shoreline and
nothing to hold the surface down.

The mask is rebuilt from scratch on every re-fit, which is what keeps it correct across adaptive
refinement — an element that was Beach at one resolution may split into a Water child and an
Inland child at the next.

### 11.1 The sampling pattern

An element is Water if any probe finds a measurement —
[element_data_mask.cpp:49-98](../src/bathymetry/element_data_mask.cpp#L49-L98). The probe set is
the four corners inset by $\varepsilon = 10^{-3}$ of the side, plus the centre, plus an
$n \times n$ interior grid at

$$
(u_i, v_j) = \left(\frac{i + \tfrac{1}{2}}{n},\; \frac{j + \tfrac{1}{2}}{n}\right),
\qquad i, j = 0, \dots, n-1, \qquad n = 3 \text{ by default},
$$

for up to $4 + 1 + 9 = 14$ probes. Corners and centre are tested first, so an element well inside
the water exits after one probe; the raster lookups are where the per-element cost lies.

> **Why the corners are inset.** A corner is shared by up to four elements. Probing it exactly
> would let a coastline passing through that single point decide the class of all four at once —
> the classification would turn on a set of measure zero, and would flip discontinuously under an
> arbitrarily small perturbation of the vector data. Insetting by a fixed fraction of the side
> makes each element's probe set interior to itself, so neighbouring elements are classified
> independently and the classification is stable.

The interior grid catches the case the corner-and-centre set misses: a channel narrow enough to
pass between them. Raising $n$ makes the classifier more conservative about calling an element
non-water, at linear cost in probes.

### 11.2 Point classification

Each probe is resolved by `MultiSourceBathymetry::classify()` —
[multi_source_bathymetry.cpp:230-264](../src/mesh/multi_source_bathymetry.cpp#L230-L264) — into
three outcomes rather than two:

$$
\text{SampleKind} \;\in\; \{\ \text{Water},\ \text{Land},\ \text{NoData}\ \}
$$

with NoData for a point that no source covers or whose value is a fill sentinel, Land for a real
reading at depth $\leq 0$, and Water otherwise. The classifier walks the sources — primary first,
then the higher-resolution tiles — in exactly the order `evaluate()` uses, so the class of a point
and the value fitted at it can never come from different sources and can never disagree.

Sentinel detection covers the three forms present in the datasets in use —
[geotiff_reader.hpp:41-49](../include/mesh/geotiff_reader.hpp#L41-L49): NaN, the $-9999$ default,
and a large positive float near $3.4 \times 10^{38}$. NaN needs its own branch because both
$v = v_{\text{nd}}$ and $\lvert v - v_{\text{nd}} \rvert < \tau$ are false for it. Bilinear
interpolation propagates the sentinel: if any of the four corner pixels is NoData, the
interpolated value is NoData, rather than a blend of a real depth with a fill value.

---

## 12. Node adjacency and why Inland can be dropped

Skipping the assembly of an element is only safe if none of its degrees of freedom needs the
operator support that assembly would have provided. For the Inland class this holds, and the
reason is the choice of adjacency relation.

> **Claim.** Every DOF at a corner of an Inland element is pinned. Omitting the element from
> assembly therefore cannot leave a free DOF with no operator support.

**Proof.** Let $E$ be Inland and let $\nu$ be one of its corners. Suppose some Water element $E'$
also touches $\nu$. Then $E$ and $E'$ share a node, so by the definition of the Beach class $E$
would have been promoted to Beach — contradicting $E$ Inland. Hence every element touching $\nu$
is non-water, so every element touching $\nu$ is pinned, so all $(r+1)^2$ DOFs at $\nu$ are pinned
(§13). As $\nu$ was arbitrary among the corners of $E$, every DOF $E$ owns is pinned, and the rows
assembly would have produced are exactly the rows condensation removes. $\blacksquare$

The proof consumes the hypothesis that adjacency is **by shared node — corners included**, and
fails without it. Under edge-only adjacency, an element touching water only diagonally would be
classified Inland while sharing a corner node with a Water element; that node's derivative DOFs
are left free by the water element, and dropping the diagonal element's assembly would strand
them.

### 12.1 Two-pass construction

[element_data_mask.cpp:100-177](../src/bathymetry/element_data_mask.cpp#L100-L177):

1. **Classify.** Every element failing `sample_is_water()` is provisionally Inland; the rest are
   Water. If neither data mask is supplied, every element is Water and nothing is allocated —
   the analytic-function path is untouched.
2. **Promote.** Corners are quantized to integer keys by a `PositionQuantizer` whose tolerance is
   relative to the smallest element and whose origin is mesh-relative, matching
   `CGHermiteDofManager` so that large projected coordinates do not lose resolution. This builds a
   node → elements map; any node touched by a Water element promotes all its non-water elements to
   Beach.
3. **Repair T-junctions.** The map is then supplemented by an edge-neighbour walk. A 2:1 T-junction
   places a fine element's corner at the *midpoint* of a coarse element's edge, where the coarse
   element has no corner of its own — so corner-position matching alone would miss that pair, and
   the fine element could be called Inland while abutting water.

---

## 13. Effect on the assembled system

**Assembly skips pinned elements.** Both `assemble_hessian_global()`
([cg_smoother_base.cpp:307](../src/bathymetry/cg_smoother_base.cpp#L307)) and
`assemble_data_fitting_global()`
([:364](../src/bathymetry/cg_smoother_base.cpp#L364)) continue past any element the mask reports as
pinned. §12 is what licenses this for Inland; for Beach it is licensed by the pins themselves,
since every DOF the element owns is constrained. The saving is real rather than cosmetic: the data
term's raster lookups are the dominant per-element cost, and they are skipped with the element.

**Pins are whole nodes, not just values.** For a pinned element, all $(r+1)^2$ DOFs at each of its
four corners are pinned — value *and* derivatives —
[cg_hermite_dof_manager.cpp:322-355](../src/bathymetry/cg_hermite_dof_manager.cpp#L322-L355). The
alternative — pinning the value and leaving the derivatives free — fails for the reason given in
§10: a shared corner would carry the water side's downward slope into the pinned element, and a
bicubic extrapolating that slope across the element has nothing bounding it. With the full node
pinned the element is identically zero, so the value the solver holds, the value written to VTK
and the value handed to `SeabedSurface` all agree.

**The formulation is unchanged.** A pin is expressed as a slave with an empty master list,

$$
x_{\text{slave}} \;=\; \sum_{m} w_m\, x_{\text{master}_m} \;=\; 0 ,
$$

which is the same master/slave substitution the hanging-node machinery already uses. It enters
the transfer operator $T$ as a zero column, so the condensed operator

$$
Q_{\text{red}} \;=\; T^\top Q\, T
$$

remains symmetric positive definite, the direct Cholesky factorization still applies, and
`constraint_violation()` stays identically zero. No side condition, no KKT block, no Lagrange
multiplier — see §10 of [hermite_bathymetry_system.md](hermite_bathymetry_system.md).

**Beach and Inland differ only in output.** Both are pinned and both are skipped in assembly. The
writers emit an `element_class` cell array and filter Inland elements out of the surface, so an
inland region leaves a hole rather than a flat patch at sea level that a reader might mistake for
a measurement.

**Elements are pinned, not deleted.** A pinned DOF is already outside the solved system, so
removing its element from the quadtree would not shrink the factorization. It would only cost a
removal path the quadtree does not have, plus 2:1 rebalancing, octree synchronization, Morton
reordering and multigrid sibling coarsening — all to remove rows that condensation removes for
free.

> **One case is not exactly zero.** A fine Beach element bordering a coarse Water element across a
> T-junction retains a hanging node that is slaved to the coarse element's trace. Structural
> continuity has to win over the pin there, or the surface would tear along the junction, so the
> hanging node follows the water side rather than sitting at zero. The affected DOFs are confined
> to the T-junction itself.

---

## 14. Configuration reference

One schema, shared by both applications —
[coastline_config.hpp](../include/core/coastline_config.hpp), with
`using LowriderCoastlineConfig = CoastlineConfig;` so each app keeps its own name for it. Parsed
at [config_reader.cpp:373-376](../src/core/config_reader.cpp#L373-L376) and
[lowrider_config_reader.cpp:96-98](../src/core/lowrider_config_reader.cpp#L96-L98).

| Key | Default | Effect | Derived in |
|---|---|---|---|
| `file` | `""` | Vector dataset; empty disables the pre-pass entirely | §2 |
| `layer` | `""` | Layer name; empty takes the first layer | §2 |
| `srs` | `""` | Working SRS, e.g. `"EPSG:3034"`; empty applies no transform and no spatial filter | §2 |
| `max_level` | `10` | Level cap $\ell_{\max}$ for the pre-pass, independent of the error loop's cap | §8 |

The mask has one knob of its own, `nsamples` (default 3, §11.1), set in code rather than from
JSON, and inherits the resolution limits `min_element_size`, `min_data_points_per_element` and
`enforce_pixel_limit` from `AdaptiveCGHermiteConfig` (§8).

The coastline section carries no length parameter at all: the criterion compares the circumradius
against the element's own size (§7). The shipped configurations set `max_level: 12` — a deliberate
figure, since the fixed point of §7.2 has no floor under it and a densely digitized shoreline will
otherwise run to the data-resolution limit of §8. See
[config/highrider_hermite_example.json](../config/highrider_hermite_example.json) and
[config/lowrider_example.json](../config/lowrider_example.json).

---

## See also

- [hermite_bathymetry_system.md](hermite_bathymetry_system.md) — the system assembled on the mesh this document produces; §10 there covers the condensation the pins of §13 rely on
- [cg_bezier_matrix_system.md](cg_bezier_matrix_system.md) — the alternative DOF choice, its KKT system, and the convex-hull property §10 invokes
- [uniform_vs_adaptive_convergence.md](uniform_vs_adaptive_convergence.md) — the error-driven refinement stage that runs after this pre-pass
