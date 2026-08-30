# CG Bézier Bathymetry Smoother — The Matrix System

This document derives every block of the linear system assembled by the CG Bézier bathymetry
smoothers: the least-squares data term, the thin-plate smoothness term, the continuity
constraints, the non-conforming (hanging-node) constraints, and the boundary conditions.

Every claim is anchored to source. Section 10 lists components that are *documented or
configured but not actually assembled* — read it before assuming a term is present.

**Scope:** `CGCubicBezierBathymetrySmoother` (degree 3, C¹) and
`CGLinearBezierBathymetrySmoother` (degree 1, C⁰). The adaptive variants
(`AdaptiveCGCubicBezierSmoother`, `AdaptiveCGLinearBezierSmoother`) rebuild the mesh and then
assemble exactly the same system — they add nothing to the matrix.

---

## 1. Overview

The smoother finds Bézier control values $x$ minimizing a weighted combination of a
thin-plate (or membrane) smoothness energy and a data-fitting residual, subject to linear
continuity and boundary constraints.

**Primal operator and right-hand side** — [cg_smoother_base.cpp:396-405](../src/bathymetry/cg_smoother_base.cpp#L396-L405):

$$
Q \;=\; \alpha H \;+\; \lambda\left(B^\top W B + \varepsilon I\right),
\qquad
b \;=\; \lambda\, B^\top W z
$$

**Constrained system** (KKT saddle point) — [constraint_condenser.cpp:5-42](../src/bathymetry/constraint_condenser.cpp#L5-L42):

$$
\begin{bmatrix} Q & A^\top \\ A & -\epsilon_c I \end{bmatrix}
\begin{bmatrix} x \\ \mu \end{bmatrix}
=
\begin{bmatrix} b \\ 0 \end{bmatrix}
$$

This corresponds to the objective

$$
J(x) \;=\; \alpha\, x^\top H x \;+\; \lambda\left(x^\top B^\top W B x - 2 x^\top B^\top W z + z^\top W z\right)
\quad\text{subject to}\quad A x = 0 .
$$

Note the constraint right-hand side is **zero**: every constraint in the live code is
homogeneous. There is no inhomogeneous constraint anywhere in the assembled system.

### Symbols and dimensions

| Symbol | Meaning | Size | Built at |
|---|---|---|---|
| $n_g$ | Global (shared) DOFs | — | [cg_cubic_bezier_dof_manager.cpp:11-30](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L11-L30) |
| $n_f$ | Free DOFs ($n_g - m_h$) | — | [cg_surface_dof_manager_base.cpp:98-110](../src/bathymetry/cg_surface_dof_manager_base.cpp#L98-L110) |
| $H$ | Smoothness (thin-plate / membrane) operator | $n_g \times n_g$ | [cg_smoother_base.cpp:260-302](../src/bathymetry/cg_smoother_base.cpp#L260-L302) |
| $B^\top W B$ | Data-fitting normal equations | $n_g \times n_g$ | [cg_smoother_base.cpp:308-390](../src/bathymetry/cg_smoother_base.cpp#L308-L390) |
| $B^\top W z$ | Data-fitting RHS | $n_g$ | same |
| $z^\top W z$ | Data energy scalar | — | same |
| $Q$ | Primal operator | $n_g \times n_g$ | [:396-403](../src/bathymetry/cg_smoother_base.cpp#L396-L403) |
| $Q_{\text{red}}$ | Hanging-node condensed operator | $n_f \times n_f$ | [constraint_condenser.cpp:83-84](../src/bathymetry/constraint_condenser.cpp#L83-L84) |
| $A_h$ | Hanging-node constraints ($m_h$ rows) | $m_h \times n_g$ | [cg_cubic_bezier_bathymetry_smoother.cpp:672-691](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L672-L691) |
| $A_e$ | C¹ edge derivative constraints ($m_e$ rows) | $m_e \times n_f$ | [:798-836](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L798-L836) |
| $A_b$ | Boundary curvature (natural BC, $m_b$ rows) | $m_b \times n_f$ | [:838-871](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L838-L871) |
| $A_g$ | Boundary gradient (zero-gradient BC, $m_g$ rows) | $m_g \times n_f$ | [:873-906](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L873-L906) |
| $A$ | Stacked constraint matrix | $(m_e{+}m_b{+}m_g) \times n_f$ | [:200-239](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L200-L239) |

Per-element DOF count is 16 for cubic (4×4 control lattice) and 4 for linear (2×2).

---

## 2. DOF numbering and C⁰ continuity

**C⁰ continuity is structural, not a constraint row.** It is enforced entirely by the global
DOF numbering: control points that coincide geometrically across an element boundary are
assigned the *same* global index, so the fitted surface is automatically continuous. No
matrix row is ever generated for C⁰.

The cubic DOF manager assigns indices in four passes
([cg_cubic_bezier_dof_manager.cpp:22-26](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L22-L26)):

1. `assign_vertex_dofs()` — the 4 corner control points per element, deduplicated by position.
2. `assign_edge_dofs()` — the 2 interior control points of each edge ($k = 1, 2$), deduplicated.
3. `assign_interior_dofs()` — the 4 remaining interior points get fresh indices.
4. `assign_edge_dofs_nonconforming()` — un-shares coarse-edge interior DOFs at 2:1 T-junctions
   and re-points the fine element's shared endpoint at the coarse DOF.

Deduplication uses a quantized physical position as a hash key
([:36-39](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L36-L39)):

$$
\text{key}(x, y) = \left(\left\lfloor 10^{8} x \right\rceil,\; \left\lfloor 10^{8} y \right\rceil\right)
$$

> **Note.** The cubic manager uses a **global absolute** scale of $10^{8}$, so the effective
> merge tolerance is ~$10^{-8}$ in whatever units the domain uses. The linear manager instead
> uses a **mesh-relative** scale — origin-shifted and normalized by the minimum element size
> ([cg_linear_bezier_dof_manager.cpp:39](../src/bathymetry/cg_linear_bezier_dof_manager.cpp#L39),
> [:54-61](../src/bathymetry/cg_linear_bezier_dof_manager.cpp#L54-L61)). The two are not
> equivalent for projected coordinates with large offsets.

### Free vs. slave DOFs

`build_dof_mappings()` marks a DOF **free** iff it is not a hanging-node slave; slaves get
`global_to_free[g] = -1`. All constraint matrices in the default solve path are assembled on
free DOFs.

### Local DOF index conventions differ between bases

| Basis | `dof_index(i,j)` | Corners |
|---|---|---|
| Cubic ([cubic_bezier_basis_2d.hpp:65-71](../include/bathymetry/cubic_bezier_basis_2d.hpp#L65-L71)) | $i + 4j$ | 0, 3, 12, 15 |
| Linear ([linear_bezier_basis_2d.hpp:66](../include/bathymetry/linear_bezier_basis_2d.hpp#L66)) | $j + 2i$ | 0, 1, 2, 3 |

These are transposed relative to each other. Code that assumes one convention while working
with the other basis will silently transpose the control lattice.

---

## 3. Least-squares data-fitting term

Assembled by `assemble_data_fitting_global()`
([cg_smoother_base.cpp:308-390](../src/bathymetry/cg_smoother_base.cpp#L308-L390)).

The fitted surface on element $e$ with bounds $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$ is

$$
z(u,v) \;=\; \sum_{k} x_{g(e,k)}\, B_k(u,v), \qquad (u,v) \in [0,1]^2
$$

where $B_k$ are the tensor-product Bernstein basis functions and $g(e,k)$ maps a local DOF to
its global index. The data misfit functional is

$$
\mathcal{R}(x) \;=\; \int_\Omega r(x,y)\,\bigl(z(x,y) - d(x,y)\bigr)^2 \,\mathrm{d}x\,\mathrm{d}y
$$

discretized with tensor-product Gauss–Legendre quadrature on each element.

### There is no explicit $B$ or $W$ matrix

The normal equations are formed **directly** at quadrature points — no rectangular $B$
($n_{\text{pts}} \times n_g$) is ever materialized. Each element contributes

$$
\left(B^\top W B\right)_{IJ} \mathrel{+}= \sum_q w_q\, B_i(u_q,v_q)\, B_j(u_q,v_q),
\qquad
\left(B^\top W z\right)_{I} \mathrel{+}= \sum_q w_q\, B_i(u_q,v_q)\, d_q
$$

with $I = g(e,i)$, $J = g(e,j)$. The scalar $z^\top W z = \sum_q w_q d_q^2$ is accumulated in
parallel ([:350](../src/bathymetry/cg_smoother_base.cpp#L350)) and used only by the
`data_residual()` diagnostic.

The weight matrix is therefore **implicitly diagonal**
([:346](../src/bathymetry/cg_smoother_base.cpp#L346)):

$$
W = \operatorname{diag}(w_q), \qquad
w_q = \hat{w}_i\, \hat{w}_j \cdot \underbrace{(\Delta x\, \Delta y)}_{\text{Jacobian}} \cdot\; r(x_q, y_q)
$$

Reference weights on $[0,1]$ sum to 1, so $\sum_q w_q = |\Omega_e|$ per element when $r \equiv 1$.

### Data points are not binned

There is **no scattered-point binning** in this path. The data is a callback
`std::function<Real(Real,Real)> bathy_func` evaluated *at the element's own quadrature points* —
the mesh defines the sample locations, so binning is trivial by construction.
`set_scattered_points()` ([:83-102](../src/bathymetry/cg_smoother_base.cpp#L83-L102))
converts a point cloud into a brute-force nearest-neighbour lookup function (O(N) per
evaluation) which is then sampled the same way.

Default `ngauss_data` is 4 (cubic) and 2 (linear), giving $4^2 = 16$ or $2^2 = 4$ samples per element.

### Boundary relaxation — the only inhomogeneous part of $W$

`compute_relaxation_factor()`
([:445-474](../src/bathymetry/cg_smoother_base.cpp#L445-L474)) reduces data fitting near
selected domain edges, letting the surface relax toward pure smoothness there. With $s$ the
distance to the nearest enabled boundary and $\delta$ the relaxation width:

$$
r(x,y) = r_{\min} + (1 - r_{\min})\, t^2(3 - 2t),
\qquad t = \operatorname{clamp}(s/\delta,\, 0,\, 1)
$$

The smoothstep $t^2(3-2t)$ has zero derivative at both ends, so $r$ is C¹. Disabled by default
(`BoundaryRelaxationConfig::enabled = false`) and configurable per edge via `edge_enabled`.

---

## 4. Smoothness term $H$

Assembled by `assemble_hessian_global()`
([cg_smoother_base.cpp:260-302](../src/bathymetry/cg_smoother_base.cpp#L260-L302)),
which scatters a per-element matrix `hessian.scaled_hessian(dx, dy)` into the global CG pattern,
dropping entries below $10^{-16}$.

### 4.1 Cubic — thin-plate energy

The reference bilinear form on $[0,1]^2$
([cubic_thin_plate_hessian.cpp:98-114](../src/bathymetry/cubic_thin_plate_hessian.cpp#L98-L114)) is

$$
E = \int_0^1\!\!\int_0^1 \left[ (z_{uu} + z_{vv})^2 + 2 z_{uv}^2 \right] \mathrm{d}u\,\mathrm{d}v
$$

realized as $H_{\text{ref}} = S^\top W S + 2\, D_{uv}^\top W D_{uv}$ with $S = D_{uu} + D_{vv}$,
then symmetrized.

The matrix actually assembled is the **physically scaled** one
([:131-166](../src/bathymetry/cubic_thin_plate_hessian.cpp#L131-L166)). Substituting
$z_{xx} = z_{uu}/\Delta x^2$, $z_{yy} = z_{vv}/\Delta y^2$, $z_{xy} = z_{uv}/(\Delta x \Delta y)$
into the physical energy $\int [(z_{xx}+z_{yy})^2 + 2z_{xy}^2]\,\mathrm{d}x\,\mathrm{d}y$ with
Jacobian $\Delta x \Delta y$ gives the anisotropic form

$$
H(\Delta x, \Delta y) \;=\;
\frac{\Delta y}{\Delta x^3} H_{uu,uu}
\;+\; \frac{\Delta x}{\Delta y^3} H_{vv,vv}
\;+\; \frac{1}{\Delta x \Delta y}\left(H_{uu,vv} + H_{uu,vv}^\top\right)
\;+\; \frac{2}{\Delta x \Delta y} H_{uv,uv}
$$

where $H_{ab,cd} = D_{ab}^\top W D_{cd}$, followed by a final symmetrization.

> **On the cross-term factor of 2.** The mathematical expansion of $(z_{xx}+z_{yy})^2$ contains
> $2 z_{xx} z_{yy}$, but the code multiplies the *symmetrized pair* by 1, not 2. These are the
> same thing: $H_{uu,vv} = D_{uu}^\top W D_{vv}$ is not symmetric, and for any $c$,
> $2\,c^\top H_{uu,vv}\, c = c^\top\!\left(H_{uu,vv} + H_{uu,vv}^\top\right) c$. The $z_{uv}^2$
> term, by contrast, keeps an explicit factor 2 because $H_{uv,uv}$ is already symmetric.

The anisotropic scaling matters: on a directionally refined element ($\Delta x \neq \Delta y$)
the $\Delta y/\Delta x^3$ and $\Delta x/\Delta y^3$ factors differ by orders of magnitude, which
is what makes the smoother behave correctly on stretched cells.

$D_{uu}$, $D_{vv}$, $D_{uv}$ are each $n_q^2 \times 16$ and cached in the constructor.
`ngauss_energy` defaults to 4 and accepts 2–6.

### 4.2 Linear — membrane (Dirichlet) energy

Thin-plate energy vanishes identically for a bilinear surface, so the linear smoother uses
**gradient energy** instead
([dirichlet_hessian.cpp:79-100](../src/bathymetry/dirichlet_hessian.cpp#L79-L100)):

$$
E = \int_0^1\!\!\int_0^1 \left[ z_u^2 + z_v^2 \right] \mathrm{d}u\,\mathrm{d}v,
\qquad H_{\text{ref}} = D_u^\top W D_u + D_v^\top W D_v
$$

with physical scaling ([:116-133](../src/bathymetry/dirichlet_hessian.cpp#L116-L133)):

$$
H(\Delta x, \Delta y) = \frac{\Delta y}{\Delta x} H_{u,u} + \frac{\Delta x}{\Delta y} H_{v,v}
$$

This is a *soap film*, not a thin plate: it minimizes surface area rather than curvature.
`ngauss_energy` defaults to 2 and accepts 1–4.

---

## 5. The scaling factors $\alpha$, $\lambda$, $\varepsilon$

### $\alpha$ — automatic scale normalization

$H$ and $B^\top W B$ have completely different physical units and magnitudes. $\alpha$
normalizes them so that $\lambda$ is a dimensionless, mesh-independent knob
([cg_smoother_base.cpp:386-389](../src/bathymetry/cg_smoother_base.cpp#L386-L389)):

$$
\alpha = \frac{\lVert B^\top W B \rVert_F}{\lVert H \rVert_F}
\quad\text{(when } \lVert H \rVert_F > 10^{-14}\text{, else } 0)
$$

This is a **Frobenius-norm ratio, not a trace ratio** (Eigen's `SpMat::norm()` is Frobenius
over stored nonzeros).

> **Ordering dependency.** $\alpha$ is computed at the *end* of the data-fitting assembly and
> reads `H_global_`. The Hessian must therefore be assembled first. Both smoothers do this —
> [cg_cubic_bezier_bathymetry_smoother.cpp:89,93](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L89)
> and [cg_linear_bezier_bathymetry_smoother.cpp:66,70](../src/bathymetry/cg_linear_bezier_bathymetry_smoother.cpp#L66)
> — but reordering them would silently zero $\alpha$.

### $\lambda$ — the smoothness/data trade-off

$\lambda$ multiplies the data block, the ridge term, **and** the right-hand side:

$$
Q = \alpha H + \lambda\left(B^\top W B + \varepsilon I\right), \qquad b = \lambda B^\top W z
$$

| $\lambda$ | Behaviour |
|---|---|
| $0$ | Pure smoothness — soap film / thin plate, ignores data entirely ($b = 0$) |
| $0.01$ | Strongly smoothed (cubic default) |
| $1.0$ | Balanced (linear default) |
| $\to \infty$ | Approaches an unregularized least-squares fit |

### $\varepsilon I$ — ridge regularization

$H$ is singular: its null space is the constants and linears for the cubic thin plate, and the
constants for the linear membrane energy. The ridge term is what makes $Q$ symmetric positive
definite, and it is the **only** null-space handling in the code — there is no explicit
null-space projection or pinning.

Applied uniformly to every diagonal entry
([:399-401](../src/bathymetry/cg_smoother_base.cpp#L399-L401)):

$$
Q_{ii} \mathrel{+}= \lambda \varepsilon, \qquad \varepsilon = 10^{-4} \text{ (default)}
$$

> **Caveat.** $\lambda\varepsilon$ is added per-DOF without any element-area weighting, so the
> ridge term is **not mesh-size-consistent**: its relative influence grows as the mesh is
> refined. On strongly adaptive meshes this biases fine regions more than coarse ones.

---

## 6. C¹ edge derivative constraints

Generated by `build_edge_derivative_constraints()`
([cg_cubic_bezier_dof_manager.cpp:319-433](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L319-L433)),
cubic smoother only. C⁰ is already structural (§2); these rows add matching of the **normal
first derivative** across element interfaces.

Interfaces are found by hashing quantized edge midpoints; an entry with exactly two
`(element, edge)` pairs is a conforming interior interface. The constrained derivative follows
the edge orientation: horizontal edges (2, 3) constrain $z_v$, vertical edges (0, 1) constrain $z_u$.

For each Gauss point $t$ on the shared edge, one row enforces

$$
\frac{1}{h_1} \sum_k \partial_n B_k\!\left(\xi_1(t)\right) x_{g(e_1,k)}
\;-\;
\frac{1}{h_2} \sum_k \partial_n B_k\!\left(\xi_2(t)\right) x_{g(e_2,k)}
\;=\; 0
$$

where $h = \Delta x^{n_u} \Delta y^{n_v}$ converts the parametric derivative to physical units.
Assembly ([:798-836](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L798-L836))
**divides** by the scale — note this is opposite to the boundary constraints of §8, which
multiply. Entries below $10^{-14}$ are dropped.

> **These are collocation, not weak, constraints.** The generator produces Gauss *abscissae
> only* — the weights are discarded ([:349-365](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L349-L365)).
> Each Gauss point yields an independent equation rather than contributing to a single
> integrated condition. With `edge_ngauss = 4` (default) on a cubic edge this is generally
> over-determined; the redundant rows are absorbed by the $-\epsilon_c I$ block of §9.

### Non-conforming C¹

At a 2:1 T-junction the size check fails and the interface is handled separately by
`build_nonconforming_edge_derivative_constraints()`
([:435-493](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L435-L493)). The fine edge
parameter maps into the coarse edge as

$$
t_{\text{coarse}} = t_{\text{start}} + \tfrac{1}{2} t_{\text{fine}},
\qquad
t_{\text{start}} = \begin{cases} 0 & \text{sub-edge } 0 \\ \tfrac{1}{2} & \text{sub-edge } 1 \end{cases}
$$

The row form is otherwise identical, with element 1 = fine and element 2 = coarse. Because the
fine and coarse elements have different $h$, the physical scaling is what makes the two sides
comparable.

---

## 7. Non-conforming (hanging-node) constraints

At a 2:1 interface the fine element's edge control points do not have coarse counterparts to be
shared with, so they are made **slaves** of the coarse edge's control points. The weights come
from exact Bézier subdivision, which guarantees the fine curve reproduces the coarse curve
restricted to the sub-edge — i.e. C⁰ is exact, not approximated.

### De Casteljau subdivision matrices

[cubic_bezier_basis_2d.cpp:397-428](../src/bathymetry/cubic_bezier_basis_2d.cpp#L397-L428).
For degree $n$, restricting to the left or right half:

$$
S^{\text{left}}_{kj} = \frac{\binom{k}{j}}{2^{k}} \;\; (j \le k),
\qquad
S^{\text{right}}_{kj} = \frac{\binom{n-k}{\,j-k\,}}{2^{\,n-k}} \;\; (j \ge k)
$$

Only the exact halves $[0, \tfrac12]$ and $[\tfrac12, 1]$ are implemented; any other interval
throws. All weights lie in $[0,1]$ and each row sums to 1 — a partition of unity, which is why
`BezierSubdivision` transfer is stable where L2 projection (with weights up to $\pm 4000$) is not.

### Constraint form

For each non-shared fine edge DOF $k$
([cg_cubic_bezier_dof_manager.cpp:209-279](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L209-L279)):

$$
x_{\text{slave}} \;=\; \sum_{m} S_{km}\, x_{\text{master}_m}
$$

Endpoints already shared through the DOF numbering are skipped ($k=0$ for sub-edge 0, $k=3$ for
sub-edge 1), and each DOF is constrained at most once.

### Two ways it enters the system

**(a) Condensation — the default** (`use_condensation = true`). The constraint never becomes a
matrix row. With $T$ the $n_g \times n_f$ prolongation implied by the constraints,

$$
Q_{\text{red}} = T^\top Q\, T, \qquad b_{\text{red}} = T^\top b
$$

$T$ is **matrix-free**: `condense_matrix_and_rhs`
([constraint_condenser.cpp:44-85](../src/bathymetry/constraint_condenser.cpp#L44-L85)) walks the
nonzeros of $Q$ and scatters $Q_{IJ} w_I w_J$ using an `expand_dof` callback that maps a global
DOF to its `(free index, weight)` pairs
([cg_cubic_bezier_bathymetry_smoother.cpp:161-180](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L161-L180)).
Slave values are recovered afterward by back-substitution.

**(b) Explicit KKT rows** (`use_condensation = false`). `assemble_A_hanging`
([:672-691](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L672-L691)) emits rows
$+1 \cdot x_{\text{slave}} - \sum_m S_{km} x_{\text{master}_m} = 0$ on global DOFs.

Condensation is preferred: it produces a smaller, still-SPD system and satisfies the constraint
exactly by construction rather than through a multiplier.

---

## 8. Boundary conditions

Both are applied on domain boundary edges via `for_each_boundary_edge`, use the same
`edge_ngauss` collocation abscissae, are homogeneous, and **both default to `false`**.

### Natural BC — zero normal curvature (`enable_natural_bc`)

[cg_cubic_bezier_dof_manager.cpp:495-549](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L495-L549).
Suppresses the boundary oscillations a thin-plate fit otherwise produces at a free edge:

$$
\frac{\partial^2 z}{\partial n^2} = 0
\quad\Longrightarrow\quad
\frac{1}{\Delta x^2}\sum_k \partial^2_u B_k\, x_k = 0
\;\;\text{or}\;\;
\frac{1}{\Delta y^2}\sum_k \partial^2_v B_k\, x_k = 0
$$

for vertical (edges 0, 1) and horizontal (edges 2, 3) boundaries respectively.

### Zero-gradient / symmetry BC (`enable_zero_gradient_bc`)

[:551-605](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L551-L605). A reflecting or
symmetry condition:

$$
\frac{\partial z}{\partial n} = 0
\quad\Longrightarrow\quad
\frac{1}{\Delta x}\sum_k \partial_u B_k\, x_k = 0
\;\;\text{or}\;\;
\frac{1}{\Delta y}\sum_k \partial_v B_k\, x_k = 0
$$

Both assemble by **multiplying** the basis coefficients by the scale
([:838-906](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L838-L906)) — the
reciprocal convention to the C¹ edge rows of §6, because here the scale is already $1/\Delta x^k$.

---

## 9. Assembly and solve

### KKT layout — exact signs

`assemble_kkt` ([constraint_condenser.cpp:5-42](../src/bathymetry/constraint_condenser.cpp#L5-L42))
places **$+A$ in both off-diagonal blocks** (no negation anywhere) and subtracts a small
regularization from the multiplier diagonal:

$$
\begin{bmatrix} Q & A^\top \\ A & -\epsilon_c I \end{bmatrix}
\begin{bmatrix} x \\ \mu \end{bmatrix}
=
\begin{bmatrix} b \\ 0 \end{bmatrix},
\qquad \epsilon_c = 10^{-10}
$$

The $-\epsilon_c I$ block regularizes redundant constraint rows (§6) that would otherwise make
the saddle system singular. It is the reason the direct solver tolerates an over-determined $A$.

### Constraint stacking order

**Condensed path** (default) — `build_condensed_system`
([cg_cubic_bezier_bathymetry_smoother.cpp:135-243](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L135-L243)).
Primal block is $Q_{\text{red}}$ ($n_f \times n_f$); hanging nodes are **absent from $A$**,
having already been eliminated:

$$
A = \begin{bmatrix} A_e \\ A_b \\ A_g \end{bmatrix} \in \mathbb{R}^{(m_e + m_b + m_g) \times n_f},
\qquad
\text{KKT size} = n_f + m_e + m_b + m_g
$$

**Full-KKT path** (`use_condensation = false`) stacks hanging → edge → curvature → gradient on
global DOFs, for a KKT size of $n_g + m_h + m_e + m_b + m_g$.

If there are no constraints at all, the smoother falls back to a plain `SparseLU` solve of
$Qx = b$.

> The struct field `sys.num_edge` holds the sum of *all three* constraint counts, not just the
> edge count ([:140-142](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L140-L142)).

### Schur complement (iterative path)

`solve_with_constraints_iterative`
([:305-576](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L305-L576)) eliminates the
primal block:

$$
S \mu = A Q^{-1} b, \quad S = A Q^{-1} A^\top,
\qquad\text{then}\qquad
x = Q^{-1}\left(b - A^\top \mu\right)
$$

$Q^{-1}$ is applied either exactly (`SparseLU`) or approximately (one multigrid V-cycle); the
latter requires flexible CG since the preconditioner then varies between iterations.

> **This path solves the *exact* saddle system** — it never forms the $-\epsilon_c I$ block.
> That is why the iterative solvers report lower constraint violation than the direct solver in
> [cg_bezier_solver_verification.md](cg_bezier_solver_verification.md).

### The linear smoother

`CGLinearBezierBathymetrySmoother` never builds a KKT system at all. It has no C¹ constraints
and no boundary conditions — only hanging-node condensation, followed by
`SparseLU` on $Q_{\text{red}} x = b_{\text{red}}$
([cg_linear_bezier_bathymetry_smoother.cpp:94-173](../src/bathymetry/cg_linear_bezier_bathymetry_smoother.cpp#L94-L173)).

---

## 10. Components **not** in the assembled system

Several terms are referenced in documentation, exposed in configuration, or present in the
source tree but play **no part in the solve**. Verified by grep over `src/` and `include/`.

| Claimed / configured | Reality |
|---|---|
| **C² constraints** (`A_c2`, `b_c2 = −A_c2·x_dir`) | No live code path. The only `build_c2_constraints()` is in the dead `CGDofManager` ([cg_dof_manager.cpp:313](../src/bathymetry/cg_dof_manager.cpp#L313)), and despite its name that routine builds **C⁰ value** constraints. The highest continuity actually enforced is C¹. |
| **Dirichlet row/column elimination** | Not applied. `assemble_Q`/`assemble_b` never eliminate rows or columns. The recipe exists only in legacy `CGDofManager` ([:489-563](../src/bathymetry/cg_dof_manager.cpp#L489-L563)). Boundary behaviour is controlled instead by the natural BC, the zero-gradient BC, and boundary relaxation of $W$. |
| **Bound constraints** (`lower_bound`, `upper_bound`, `max_bound_iterations`) | Parsed by `config_reader.cpp:71-75` and stored, but **never read by any solve path**. `set_bounds`/`clear_bounds` only mutate the config. Inert. |
| **"Constraint projection for exact satisfaction"** | `constraint_projection_ms` is declared and copied between profile structs but **never assigned a nonzero value**. Exact constraint satisfaction comes from the KKT/Schur multipliers, not from a projection step. |
| **Land masking** | `land_mask_func_` gates *refinement decisions* only. It does not mask $W$, remove rows, or otherwise touch the matrix. |
| **Multi-source blending weights** | `MultiSourceBathymetry::evaluate` returns the first valid source by priority; there is no weighted blend. Multi-source affects only the sampled value $d$. |
| **Explicit null-space handling** | None. The $\lambda\varepsilon I$ ridge alone makes $Q$ SPD (§5). |
| **Membrane term in the cubic smoother** | The cubic smoother has *only* thin-plate energy; the linear smoother has *only* membrane energy. Neither combines them. |

### Known inconsistency: $\alpha$ missing from cached element matrices

[cg_smoother_base.cpp:367-377](../src/bathymetry/cg_smoother_base.cpp#L367-L377)
caches per-element matrices as

$$
Q_e^{\text{cached}} = H_e + \lambda B_e + \lambda \varepsilon I
$$

with an implicit $\alpha = 1$, because $\alpha$ is not known until after the assembly loop
finishes. The true operator uses $\alpha H$, and $\alpha$ can span many orders of magnitude.
`CoarseGridStrategy::CachedRediscretization` therefore builds coarse operators for a materially
different problem. This is the likely cause of the outlier in
[cg_bezier_solver_verification.md](cg_bezier_solver_verification.md), where MG (L2+Cached)
reports a data residual of 0.1941 against 0.1398 for every other solver. The cached path also
over-counts $\varepsilon$ on shared DOFs, adding it once per adjacent element.

### Dead modules

Present in the tree but referenced by nothing outside themselves — do not mistake these for the
live path: `bezier_data_fitting.*` (degree-5 DG assembler; only its `BathymetryPoint` struct is
used), `biharmonic_assembler.*` (IPDG C¹ penalty alternative; only its `BathymetrySource` base
class is used), `cg_dof_manager.*`, `cg_bezier_dof_manager.*`, `thin_plate_hessian.*`,
`bezier_basis_2d.*` (the quintic 36-DOF variant).

---

## 11. Quadrature reference

Six independent Gauss–Legendre tables exist in the codebase, with **differing supported ranges
and out-of-range behaviour**. Requesting an unsupported order silently clamps in some and throws
in others.

| Location | Range | Out of range | Used for |
|---|---|---|---|
| [cg_smoother_base.cpp:28-59](../src/bathymetry/cg_smoother_base.cpp#L28-L59) | 1–4 | **clamps to 4** | Data fitting ($B^\top W B$) |
| [cubic_thin_plate_hessian.cpp:18-69](../src/bathymetry/cubic_thin_plate_hessian.cpp#L18-L69) | 2–6 | throws | Cubic thin-plate $H$ |
| [dirichlet_hessian.cpp:17-52](../src/bathymetry/dirichlet_hessian.cpp#L17-L52) | 1–4 | throws | Linear membrane $H$ |
| [cg_cubic_bezier_dof_manager.cpp:349-365](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L349-L365) (×3) | 2–4 | **clamps to 4** | C¹ edge, curvature BC, gradient BC (**abscissae only**) |
| [adaptive_cg_smoother_base.cpp:47-84](../src/bathymetry/adaptive_cg_smoother_base.cpp#L47-L84) | 1–6 | — | Adaptive error estimation |
| [bezier_data_fitting.cpp:124-170](../src/bathymetry/bezier_data_fitting.cpp#L124-L170) | 1–6 | — | *(dead code)* |

Setting `ngauss_data = 6` therefore does **not** give 6-point quadrature — it silently gives 4.

---

## 12. Configuration reference (cubic)

| Parameter | Default | Effect on the system |
|---|---|---|
| `lambda` | `0.01` | Scales $B^\top W B$, $\varepsilon I$, and $b$ |
| `ridge_epsilon` | `1e-4` | $\lambda\varepsilon$ on every diagonal entry |
| `ngauss_data` | `4` | Quadrature for $B^\top W B$ (max 4, see §11) |
| `ngauss_energy` | `4` | Quadrature for $H$ |
| `edge_ngauss` | `4` | Collocation points per edge → rows in $A_e$, $A_b$, $A_g$ |
| `enable_natural_bc` | `false` | Adds $A_b$ (zero normal curvature) |
| `enable_zero_gradient_bc` | `false` | Adds $A_g$ (zero normal gradient) |
| `use_condensation` | `true` | Hanging nodes condensed into $Q_{\text{red}}$ vs. explicit KKT rows |
| `lower_bound` / `upper_bound` | unset | **Inert** — parsed but never used (§10) |

The linear smoother's defaults differ: `lambda = 1.0`, `ngauss_data = 2`, `ngauss_energy = 2`.

---

## 13. Diagnostic identities

Useful as sanity checks when validating an implementation change
([cg_smoother_base.cpp:237-254](../src/bathymetry/cg_smoother_base.cpp#L237-L254)):

$$
\text{data\_residual} = x^\top B^\top W B x - 2 x^\top B^\top W z + z^\top W z
$$

$$
\text{regularization\_energy} = x^\top H x
$$

$$
\text{objective\_value} = \alpha \cdot \text{regularization\_energy} + \lambda \cdot \text{data\_residual}
$$

$$
\text{constraint\_violation} = \lVert A x \rVert_2
$$

Two caveats: `objective_value` **omits the $\lambda\varepsilon\lVert x\rVert^2$ ridge
contribution**, so it is not exactly the quantity the solver minimizes; and
`constraint_violation`
([cg_cubic_bezier_bathymetry_smoother.cpp:908-924](../src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp#L908-L924))
uses the **full global** $A$ including hanging-node rows, so it is a valid check even in the
condensed path where those rows never entered the solve.

---

## See also

- [hermite_bathymetry_system.md](hermite_bathymetry_system.md) — an alternative DOF choice that makes C^r structural and removes the KKT system entirely
- [cg_bezier_solver_verification.md](cg_bezier_solver_verification.md) — solver and multigrid benchmarks
- [cg_cubic_bezier_uniform_evaluation.md](cg_cubic_bezier_uniform_evaluation.md) — λ and mesh-resolution sweeps
- [uniform_vs_adaptive_convergence.md](uniform_vs_adaptive_convergence.md) — AMR convergence study
- [coastline_adaptivity.md](coastline_adaptivity.md) — curvature-driven refinement toward a vector coastline, and the element classification that precedes assembly
