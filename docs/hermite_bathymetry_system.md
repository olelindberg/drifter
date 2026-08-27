# Hermite Bathymetry Smoother — Avoiding the KKT System

This document assesses an alternative formulation for the CG bathymetry smoothers. Instead of
Bernstein control values, the degrees of freedom become **elevation and derivatives at the four
element corners**. The claim under test is that this makes C^r continuity *structural*, removes
the edge-derivative constraints, allows boundary conditions to be assigned directly to corner
DOFs, and therefore eliminates the indefinite KKT saddle-point system.

**Verdict: the formulation works.** Three of the four claims hold outright and one needs
correcting (§1). Every matrix in this document was derived symbolically over $\mathbb{Q}[h]$ and
cross-checked against a second, independent derivation (§5, §8); the two load-bearing theorems
(§3, §8) were verified as polynomial identities rather than argued informally.

**Scope:** all three continuity orders — C⁰ (bilinear, 4 DOF/element), C¹ (bicubic
Bogner–Fox–Schmit, 16 DOF/element), and C² (biquintic, 36 DOF/element). Read alongside
[cg_bezier_matrix_system.md](cg_bezier_matrix_system.md), which derives the system this one
replaces; section numbering here deliberately parallels it.

---

## 1. Overview and verdict

The existing cubic smoother represents the surface in a **Bernstein control-point basis**.
Coincident control points share a global index, so C⁰ is structural — it costs no equations. But
Bernstein DOFs say nothing about derivatives, so C¹ must be imposed *externally*: `edge_ngauss`
collocation rows per interior edge, stacked into a constraint matrix $A$ and solved as

$$
\begin{bmatrix} Q & A^\top \\ A & -\epsilon_c I \end{bmatrix}
\begin{bmatrix} x \\ \mu \end{bmatrix}
=
\begin{bmatrix} b \\ 0 \end{bmatrix}
$$

That single decision pulls in `assemble_kkt`, three separate solve paths, the Schur-complement
machinery and two approximate-CG Schur preconditioners. It is also only *approximately* C¹: the
collocation system is over-determined and its residual is absorbed by the $-\epsilon_c I$ block
(measured violation $\approx 2.45\times10^{-8}$, per
[cg_bezier_solver_verification.md](cg_bezier_solver_verification.md)).

The Hermite formulation attacks the seam directly.

| Claim | Assessment |
|---|---|
| Avoids the edge derivative constraints | **Yes — and exactly, not approximately.** C^r holds pointwise along every conforming edge by construction (§3). |
| Boundary conditions assigned directly to corner DOFs | **Yes.** Zero normal gradient becomes elimination of $z_n, z_{nt}$ at boundary nodes; no constraint rows (§9). |
| Avoids the KKT system | **Yes.** The reduced operator is SPD, so Cholesky or plain CG replaces the Schur-complement path (§10). |
| Eliminates hanging-node constraints | **No — but it does not matter.** See below. |

> **The one correction.** Hanging nodes still require constraints. What matters is their *form*:
> they remain pure master/slave **substitutions** $x_{\text{slave}} = \sum_m w_m x_{\text{master}_m}$,
> never side conditions. A substitution is a change of basis $x = Tx_f$, giving $T^\top Q T$ —
> smaller, still SPD, and never a saddle point. Only the C¹ edge constraints were irreducibly
> side conditions, because a normal-derivative match cannot be written as a per-DOF substitution
> in the Bernstein basis. Removing *those* is what removes the KKT system. The existing
> [`condense_matrix_and_rhs`](../include/bathymetry/constraint_condenser.hpp#L49) /
> [`back_substitute_slaves`](../include/bathymetry/constraint_condenser.hpp#L77) path already
> does exactly this — it is how `CGLinearBezierBathymetrySmoother` avoids KKT today
> ([cg_linear_bezier_bathymetry_smoother.cpp:94-173](../src/bathymetry/cg_linear_bezier_bathymetry_smoother.cpp#L94-L173)).

The final system is

$$
Q_{\text{red}}\, x_f = b_{\text{red}}, \qquad Q_{\text{red}} = T^\top Q T \;\succ\; 0
$$

with no multipliers, no constraint block, and $\texttt{constraint\_violation}() \equiv 0$ by
construction.

> **This is a change of DOFs, not a change of approximation space.** Bernstein and Hermite span
> the *same* polynomial space $Q_p$ on each element. They are related by a fixed invertible
> matrix $M_e$ (§5), so the existing element matrices, quadrature loops, `SeabedSurface`
> interface and VTK writers all survive unchanged. What changes is which linear functionals are
> called degrees of freedom — and that is precisely what decides whether continuity is
> structural or constrained.

---

## 2. The Hermite family

Let $r$ be the continuity order. The element is the tensor-product Hermite rectangle of degree

$$
p = 2r + 1
$$

with corner degrees of freedom given by the **tensor-product derivative set**

$$
\left\{\ \frac{\partial^{\,a+b} z}{\partial x^{a}\, \partial y^{b}}\ :\ a, b \in \{0,\dots,r\}\ \right\}
\qquad\Longrightarrow\qquad (r+1)^2 \ \text{DOFs per corner}
$$

Four corners give $4(r+1)^2 = (2r+2)^2 = (p+1)^2$ DOFs per element, exactly the dimension of the
tensor-product polynomial space $Q_p$. The element is unisolvent.

| $r$ | $p$ | Element | DOFs / corner | DOFs / element | Corner DOF set |
|---|---|---|---|---|---|
| 0 | 1 | bilinear | 1 | 4 | $z$ |
| 1 | 3 | bicubic (Bogner–Fox–Schmit) | 4 | 16 | $z,\ z_x,\ z_y,\ z_{xy}$ |
| 2 | 5 | biquintic | 9 | 36 | $z,\ z_x,\ z_y,\ z_{xx},\ z_{xy},\ z_{yy},\ z_{xxy},\ z_{xyy},\ z_{xxyy}$ |

> **Note on the C² set.** The nine C² corner DOFs are the *mixed* derivatives $z_{xxy}$,
> $z_{xyy}$, $z_{xxyy}$ — **not** the pure third derivatives $z_{xxx}$, $z_{yyy}$. The DOF set is
> the Cartesian product $\{0,1,2\}\times\{0,1,2\}$ of per-axis derivative orders, which is what
> makes the element a tensor product and what makes §3 work. Including $z_{xxx}$ would break
> unisolvency: $\partial_x^3$ is not of the form $\partial_x^a\partial_y^b$ with $a,b \le 2$.

Throughout, $n$ and $t$ denote the directions normal and tangential to an edge, so a corner DOF
may equivalently be written $\partial_n^a \partial_t^b z$.

---

## 3. Why C^r is structural

This is the theorem the whole formulation rests on.

**Claim.** Let two elements share an edge, with all $2(r+1)^2$ corner DOFs at the edge's two
endpoints identified between them. Then for every $a = 0,\dots,r$, the traces of
$\partial_n^a z$ along the edge agree *identically* — for arbitrary values of all remaining DOFs,
and for arbitrary (unequal) element sizes on the two sides.

**Proof.** Fix $a$ and restrict to the edge $x = x_0$, parametrised by the tangential coordinate.
Because the element is a tensor product of degree $p = 2r+1$ in each direction, the restriction of
$\partial_x^a z$ to that edge is a **polynomial of degree $p$ in the tangential variable alone**.
That polynomial is uniquely determined by the $2(r+1)$ Hermite data

$$
\partial_n^a \partial_t^b z \Big|_{\text{endpoint } 0,1}, \qquad b = 0,\dots,r
$$

by uniqueness of Hermite interpolation ($2(r+1) = p+1$ conditions on a degree-$p$ polynomial).
Every one of those values is a shared corner DOF. Both elements therefore interpolate the same
data with the same unique polynomial, so the traces coincide. Interior DOFs and the normal
element sizes never enter, because they do not appear in the interpolation data. $\blacksquare$

Verified symbolically in 2D for $r = 1$ with two elements of *different* widths
$h_x^A \neq h_x^B$ and all 24 remaining DOFs left free: the differences $t_A^{(a)} - t_B^{(a)}$
reduce identically to zero for $a = 0$ and $a = 1$.

> **This is exactly what §6 of [cg_bezier_matrix_system.md](cg_bezier_matrix_system.md) is trying
> to achieve and cannot.** There, the normal-derivative match is imposed at `edge_ngauss = 4`
> collocation points on a cubic edge — over-determined, inconsistent in general, and reconciled
> only by the $-\epsilon_c I$ regularisation. Here the same condition holds pointwise along the
> entire edge, for free, as a consequence of DOF identification.

---

## 4. The 1D Hermite bases

Everything tensorises from the 1D bases on the reference interval $[0,1]$. Write $H_{s,m}(t)$ for
the basis function dual to the $m$-th derivative at node $s \in \{0,1\}$:

$$
\frac{\mathrm{d}^{m'}}{\mathrm{d}t^{m'}} H_{s,m}\big|_{t = s'} = \delta_{ss'}\,\delta_{mm'},
\qquad s, s' \in \{0,1\},\quad m, m' \in \{0,\dots,r\}
$$

**$r = 0$ (linear).**

$$
H_{0,0} = 1 - t, \qquad H_{1,0} = t
$$

> **These are the degree-1 Bernstein polynomials.** $B_{0,1} = 1-t$, $B_{1,1} = t$
> ([linear_bezier_basis_2d.cpp:29-40](../src/bathymetry/linear_bezier_basis_2d.cpp#L29-L40)).
> The C⁰ Hermite element *is* `CGLinearBezierBathymetrySmoother` — same basis, same DOFs, same
> matrices. Nothing in this document changes the C⁰ case; it is included because the general
> formulas must reduce to it, and they do (§5, §8 both collapse to the identity and to de
> Casteljau respectively).

**$r = 1$ (cubic).**

$$
H_{0,0} = 2t^3 - 3t^2 + 1, \quad
H_{0,1} = t^3 - 2t^2 + t, \quad
H_{1,0} = -2t^3 + 3t^2, \quad
H_{1,1} = t^3 - t^2
$$

**$r = 2$ (quintic).**

$$
\begin{aligned}
H_{0,0} &= -6t^5 + 15t^4 - 10t^3 + 1 & H_{1,0} &= 6t^5 - 15t^4 + 10t^3 \\
H_{0,1} &= -3t^5 + 8t^4 - 6t^3 + t & H_{1,1} &= -3t^5 + 7t^4 - 4t^3 \\
H_{0,2} &= -\tfrac12 t^5 + \tfrac32 t^4 - \tfrac32 t^3 + \tfrac12 t^2 \quad & H_{1,2} &= \tfrac12 t^5 - t^4 + \tfrac12 t^3
\end{aligned}
$$

### Parametric versus physical derivatives

Global DOFs are **physical** derivatives. This is not a stylistic choice: a node shared by
elements of different sizes has only one unambiguous set of derivative values, and parametric
derivatives $\partial_t^m$ would disagree between neighbours by the size ratio. With
$x = x_e + h_x u$,

$$
\frac{\partial^m}{\partial u^m} = h_x^{\,m}\, \frac{\partial^m}{\partial x^m}
\qquad\Longrightarrow\qquad
\hat z^{(m)} = h_x^{\,m}\, z^{(m)}
$$

so the basis function dual to the *physical* $m$-th derivative is $H_{s,m}(u)\,h_x^{\,m}$. Each
element applies its own $h_x, h_y$; the shared nodal values stay element-independent. In 2D the
element basis dual to $\partial_x^a \partial_y^b z$ at corner $(s_x, s_y)$ is

$$
N_{(s_x,a),(s_y,b)}(u,v) \;=\; H_{s_x,a}(u)\, H_{s_y,b}(v)\; h_x^{\,a}\, h_y^{\,b}
$$

---

## 5. Bernstein control points in terms of corner DOFs

*The central section.* The two bases are related by an explicit matrix, which is what makes this
a refactor of the DOF map rather than a rewrite of the element library.

For a degree-$p$ Bézier curve the endpoint derivative identities give $\hat z(0) = c_0$,
$\hat z'(0) = p(c_1 - c_0)$, $\hat z''(0) = p(p-1)(c_2 - 2c_1 + c_0)$, and mirrored at $t=1$.
Inverting these and converting to physical derivatives yields, for $k = 0,\dots,r$:

$$
c_k \;=\; \sum_{m=0}^{k} \frac{\binom{k}{m}}{\binom{p}{m}}\, \frac{h^m}{m!}\; z_0^{(m)},
\qquad\qquad
c_{p-k} \;=\; \sum_{m=0}^{k} \frac{\binom{k}{m}}{\binom{p}{m}}\, \frac{(-h)^m}{m!}\; z_1^{(m)}
$$

Since $p = 2r+1$, these two families supply all $p+1 = 2(r+1)$ control points. Write $c = M(h)\,q$
with $q = (z_0, z_0', \dots, z_0^{(r)},\ z_1, z_1', \dots, z_1^{(r)})^\top$.

**$M(h)$ is block-diagonal by node.** The first $r+1$ control points depend only on the left
node's DOFs; the last $r+1$ only on the right node's. This is §3 restated in Bernstein language:
two elements sharing a node generate *identical* control points near that node, and those are
precisely the control points that determine the trace derivatives up to order $r$. The classical
"C^r joint conditions" on Bézier control points are not imposed here — they are automatically
satisfied because each block is generated by a single node's data.

$$
M_{r=0} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
\qquad
M_{r=1}(h) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & \tfrac{h}{3} & 0 & 0 \\
0 & 0 & 1 & -\tfrac{h}{3} \\
0 & 0 & 1 & 0
\end{bmatrix}
\qquad
M_{r=2}(h) = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
1 & \tfrac{h}{5} & 0 & 0 & 0 & 0 \\
1 & \tfrac{2h}{5} & \tfrac{h^2}{20} & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & -\tfrac{2h}{5} & \tfrac{h^2}{20} \\
0 & 0 & 0 & 1 & -\tfrac{h}{5} & 0 \\
0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}
$$

**Invertibility.** Each node block is triangular with diagonal
$h^k / \big(k!\,\binom{p}{k}\big) \neq 0$, so $M(h)$ is invertible for every $h > 0$:

$$
\det M_{r=0} = 1, \qquad \det M_{r=1}(h) = \frac{h^2}{9}, \qquad \det M_{r=2}(h) = \frac{h^6}{10^4}
$$

The inverse recovers corner DOFs from control points — needed to initialise a Hermite solve from
an existing Bézier solution. For $r=1$:

$$
z_0 = c_0, \quad z_0' = \frac{3(c_1 - c_0)}{h}, \quad z_1 = c_3, \quad z_1' = \frac{3(c_3 - c_2)}{h}
$$

**Two dimensions.** $M_e = M(h_x) \otimes M(h_y)$, of size $4\times4$ (identity), $16\times16$, or
$36\times36$. Because it already carries the $h^m$ factors, $M_e$ subsumes the
parametric-to-physical scaling entirely; there is no separate scaling matrix.

> **Implementation consequence.** $M_e$ is an explicit per-element congruence, so every element
> matrix already in the codebase is reused verbatim:
> $$Q_e = M_e^\top \hat Q_e M_e, \qquad b_e = M_e^\top \hat b_e$$
> where $\hat Q_e, \hat b_e$ are the Bernstein element matrices assembled today.
> `CubicBezierBasis2D`, `CubicThinPlateHessian`, `DirichletHessian`, the $B^\top W B$ quadrature
> loop, `SeabedSurface` and both VTK writers stay untouched. The genuinely new code is $M_e$, a
> DOF manager, and the hanging-node and boundary-condition logic.

---

## 6. The data-fitting term

Structurally unchanged from
[cg_smoother_base.cpp:308-390](../src/bathymetry/cg_smoother_base.cpp#L308-L390).
As in the Bézier system there is no explicit $B$ or $W$; the normal equations accumulate directly
at quadrature points. With $\hat B(u_q, v_q)$ the existing Bernstein evaluation,

$$
B_e^\top W B_e \;=\; M_e^\top \left[\, \sum_q w_q\, \hat B\, \hat B^\top \right] M_e,
\qquad
b_e \;=\; \lambda\, M_e^\top \left[\, \sum_q w_q\, \hat B\, d_q \right]
$$

$$
w_q = \hat w_i\, \hat w_j \cdot \underbrace{(h_x h_y)}_{\text{Jacobian}} \cdot\; r(x_q, y_q)
$$

with $r$ the boundary relaxation factor
([:445](../src/bathymetry/cg_smoother_base.cpp#L445)), the only inhomogeneous part of $W$.
$M_e$ is applied once to the accumulated element block, not per quadrature point.

> **Quadrature adequacy — the 4-point clamp matters only at $r = 2$.**
> [`gauss_legendre_01`](../src/bathymetry/cg_smoother_base.cpp#L28-L59) silently truncates
> any request of 4 or more points to exactly 4 (`pts.resize(4)` at
> [:48](../src/bathymetry/cg_smoother_base.cpp#L48)), which is exact to degree
> $2\cdot4-1 = 7$. The data-term integrand $N_i N_j$ has degree $2p$ per direction:
>
> | $r$ | $p$ | integrand degree | points needed | 4-point clamp |
> |---|---|---|---|---|
> | 0 | 1 | 2 | 2 | sufficient |
> | 1 | 3 | 6 | 4 | **exactly sufficient** |
> | 2 | 5 | 10 | 6 | **insufficient** |
>
> So the existing clamp is adequate for the C¹ element but must be lifted before C² is attempted.
> The same arithmetic applies to the smoothness term of §7.

---

## 7. The smoothness operator $H$

> Derived in full — element basis, quadrature and closed Kronecker form, assembly, null spaces —
> in [hermite_smoothness_operator.md](hermite_smoothness_operator.md), for all three orders. This
> section states the result.

The energies are unchanged: membrane (Dirichlet) for $r=0$, thin plate for $r \ge 1$. Only the
congruence is new. Writing $\hat H_{ab,cd} = \hat D_{ab}^\top \hat W \hat D_{cd}$ for the
reference-element blocks that
[`CubicThinPlateHessian::scaled_hessian`](../src/bathymetry/cubic_thin_plate_hessian.cpp#L131-L166)
already builds:

$$
H_e \;=\; M_e^\top \left[
\frac{h_y}{h_x^3}\hat H_{uu,uu}
+ \frac{h_x}{h_y^3}\hat H_{vv,vv}
+ \frac{1}{h_x h_y}\big(\hat H_{uu,vv} + \hat H_{uu,vv}^\top\big)
+ \frac{2}{h_x h_y}\hat H_{uv,uv}
\right] M_e
$$

and for $r = 0$ the membrane form
$H_e = M_e^\top\big[\tfrac{h_y}{h_x}\hat H_{u,u} + \tfrac{h_x}{h_y}\hat H_{v,v}\big]M_e$, with
$M_e = I$.

> **The implemented energy is not the standard thin plate, and its null space is larger.** The
> code integrates $\int\!\!\int\big[(z_{uu}+z_{vv})^2 + 2z_{uv}^2\big]$, whereas the standard
> thin-plate energy is $\int\!\!\int\big[z_{uu}^2 + 2z_{uv}^2 + z_{vv}^2\big]$. Expanding the
> first gives an extra $2z_{uu}z_{vv}$ cross term. Computing both 16×16 element Hessians on the
> reference element and taking null spaces:
>
> $$\ker H_{\text{implemented}} = \operatorname{span}\{1,\ u,\ v,\ u^2 - v^2\}\ (\dim 4),
> \qquad \ker H_{\text{standard}} = \operatorname{span}\{1,\ u,\ v\}\ (\dim 3)$$
>
> The mode $u^2 - v^2$ is annihilated because it is harmonic ($z_{uu} + z_{vv} = 0$) *and* has
> $z_{uv} = 0$. Globally on a conforming mesh the C¹ patching forces the coefficient of
> $x^2 - y^2$ to be the same on every element, so the global null space is also 4-dimensional.
> This is inherited unchanged from the Bézier smoother — it is a property of the energy, not of
> the basis — and $\lambda \varepsilon I$ covers it either way. Worth knowing before interpreting
> $\lambda \to 0$ behaviour.

---

## 8. Hanging-node constraints

These persist, but as substitutions (§1). At a 2:1 T-junction the new midpoint node's DOFs are
fully determined by the coarse edge's trace, because that trace is already fixed by the coarse
element's own corner DOFs.

### The constraint matrices $G_r$

Evaluate the coarse edge's Hermite interpolant and its first $r$ derivatives at $t = \tfrac12$.
In **parametric** form (coarse edge of unit length), $[G_r]_{ij} = H_j^{(i)}(\tfrac12)$:

$$
G_0 = \begin{bmatrix} \tfrac12 & \tfrac12 \end{bmatrix}
\qquad
G_1 = \begin{bmatrix}
\tfrac12 & \tfrac18 & \tfrac12 & -\tfrac18 \\[4pt]
-\tfrac32 & -\tfrac14 & \tfrac32 & -\tfrac14
\end{bmatrix}
\qquad
G_2 = \begin{bmatrix}
\tfrac12 & \tfrac{5}{32} & \tfrac{1}{64} & \tfrac12 & -\tfrac{5}{32} & \tfrac{1}{64} \\[4pt]
-\tfrac{15}{8} & -\tfrac{7}{16} & -\tfrac{1}{32} & \tfrac{15}{8} & -\tfrac{7}{16} & \tfrac{1}{32} \\[4pt]
0 & -\tfrac32 & -\tfrac14 & 0 & \tfrac32 & -\tfrac14
\end{bmatrix}
$$

In **physical** derivatives, with $h_t$ the *coarse* edge length, $i$ the slave derivative order
and $m_j$ the master DOF's derivative order:

$$
\big[G_r^{\text{phys}}(h_t)\big]_{ij} \;=\; h_t^{\,m_j - i}\,[G_r]_{ij}
$$

giving, for example,

$$
G_1^{\text{phys}}(h) = \begin{bmatrix}
\tfrac12 & \tfrac{h}{8} & \tfrac12 & -\tfrac{h}{8} \\[4pt]
-\tfrac{3}{2h} & -\tfrac14 & \tfrac{3}{2h} & -\tfrac14
\end{bmatrix}
$$

### The full nodal block

A hanging node carries $(r+1)^2$ slave DOFs indexed $(a,b)$ — normal order $a$, tangential order
$b$. The constraint acts **only on the tangential index**; the normal index passes through
untouched, because the normal derivative is the same physical quantity on both sides. Hence

$$
x_{\text{slave}} \;=\; \Big( I_{r+1} \otimes G_r^{\text{phys}}(h_t) \Big)\, x_{\text{master}}
$$

$(r+1)^2$ slaves, each a combination of $2(r+1)$ masters, drawn from the coarse edge's two
endpoint nodes. Pure substitution — no multipliers.

### Two theorems

**C^r is exact across a 2:1 T-junction.** The fine element's trace on $[0, h_t/2]$ is the unique
degree-$p$ Hermite interpolant of the data $\{$shared endpoint DOFs, constrained midpoint
DOFs$\}$. The coarse trace restricted to $[0, h_t/2]$ is a degree-$p$ polynomial matching exactly
that data. By uniqueness they are the same polynomial. Verified symbolically for $r = 0, 1, 2$
against an arbitrary coarse trace with $p+1$ free coefficients: the difference reduces
identically to zero. This is strictly stronger than the Bézier code's Gauss-point collocation.

**Under 2:1 balance, a node is hanging with respect to at most one edge.** A node cannot be the
midpoint of both a horizontal and a vertical coarse edge: the vertical coarse edge spans one
entire coarse element face, which a crossing horizontal coarse edge would contradict. Constraints
therefore never conflict, and no consistency reconciliation is needed.

### Cross-check against de Casteljau

$G_r$ can be derived a second way, entirely inside the Bernstein basis: subdivide the coarse
curve, then read off the fine element's corner DOFs. With $S^{\text{left}}_{kj} = \binom{k}{j}/2^k$
the de Casteljau restriction to $[0,\tfrac12]$
([cubic_bezier_basis_2d.cpp:397-428](../src/bathymetry/cubic_bezier_basis_2d.cpp#L397-L428)),

$$
G_r^{\text{phys}}(h) \;=\; \Big[\, M(h/2)^{-1}\, S^{\text{left}}\, M(h) \,\Big]_{\text{lower block}}
$$

Verified symbolically for $r = 0, 1, 2$: the bottom $(r+1)$ rows agree with $G_r^{\text{phys}}$
entry for entry. Two independent derivations — Hermite evaluation at $t=\tfrac12$, and de
Casteljau composed with the change of basis of §5 — produce the same matrix. This identity is a
ready-made unit test.

### Reproduction property

$G_r$ reproduces **every polynomial of degree $\le 2r+1$** exactly: taking master DOFs from any
such polynomial yields its true value and derivatives at the midpoint. Verified symbolically for
$r = 0,1,2$ with $p+1$ symbolic coefficients. In particular a constant $z \equiv 1$ maps to
$(1, 0, \dots, 0)$, and the **value-DOF columns** of row 0 sum to 1.

> **Caveat — row 0 does not sum to 1 in general.** For $r = 2$ the sum of *all* entries in the
> value row is $1 + h^2/32$, not 1. Only the value-DOF columns form a partition of unity; the
> derivative columns must not be included. Unlike de Casteljau weights, Hermite constraint
> weights are neither non-negative nor normalised — $G_1$ already contains $-3/2h$, which grows
> without bound as $h \to 0$. This is a genuine difference from the `BezierSubdivision` transfer
> operator and feeds directly into the conditioning discussion of §11.

> **Caveat — chained constraints.** A master node may itself be a slave of a coarser edge: 2:1
> balance permits an element at level $L$ to neighbour elements at $L-1$ and $L+1$ simultaneously,
> so a coarse-element corner can be the midpoint of an even coarser edge. The constraint set must
> therefore be closed transitively to a fixpoint before condensation (deal.II's
> `AffineConstraints::close()` is the reference implementation). Anisotropic refinement
> ($\text{level}_x \neq \text{level}_y$) widens the case analysis further; the adaptive drivers
> currently only issue `RefineMask::XY`, which keeps elements isotropic.

---

## 9. Boundary conditions — strong, no rows

Because the trace of $\partial_n^a z$ along a boundary edge is a Hermite interpolant of the
endpoint DOFs (§3), fixing those DOFs fixes the entire trace. Boundary conditions become
row/column eliminations, not constraint rows.

| Condition | $r = 0$ | $r = 1$ | $r = 2$ |
|---|---|---|---|
| Dirichlet $z = g$ | fix $z$ | fix $z,\ z_t$ | fix $z,\ z_t,\ z_{tt}$ |
| Zero normal gradient $\partial_n z = 0$ | *not expressible* | fix $z_n,\ z_{nt}$ | fix $z_n,\ z_{nt},\ z_{ntt}$ |
| Zero normal curvature $\partial_n^2 z = 0$ | — | *natural only* | fix $z_{nn},\ z_{nnt},\ z_{nntt}$ |
| Free / natural | nothing | nothing | nothing |

For a constant Dirichlet value the tangential derivatives are zero, so only $z$ carries data; for
a varying $g$ the tangential DOFs take $\partial_t^b g$.

Two observations against the current configuration surface:

- **`enable_zero_gradient_bc`** becomes two DOF eliminations per boundary node instead of
  collocation rows — exact, and free.
- **`enable_natural_bc`** is vacuous at $r = 1$: $z_{nn}$ is not a DOF, and a free edge already
  *is* the variational natural boundary condition of the thin-plate functional. Imposing nothing
  is the correct discretisation, which is what the current default `enable_natural_bc = false`
  already delivers. At $r = 2$ the condition becomes strongly enforceable for the first time.

---

## 10. Assembly and solve

The operator and right-hand side are unchanged in form
([cg_smoother_base.cpp:396-405](../src/bathymetry/cg_smoother_base.cpp#L396-L405)):

$$
Q = \alpha H + \lambda\big(B^\top W B + \varepsilon I\big),
\qquad
b = \lambda\, B^\top W z,
\qquad
\alpha = \frac{\lVert B^\top W B\rVert_F}{\lVert H \rVert_F}
$$

Partition the global DOFs into free $f$, hanging-slave $s$, and BC-fixed $d$. Build the
prolongation $T$ (identity on free rows, $I_{r+1}\otimes G_r^{\text{phys}}$ on slave rows, zero on
fixed rows) and the lift $x_g$ carrying the prescribed values. Then

$$
Q_{\text{red}} = T^\top Q T, \qquad b_{\text{red}} = T^\top\big(b - Q x_g\big), \qquad
x = T x_f + x_g
$$

$T$ has full column rank and $Q \succ 0$ (from $\varepsilon > 0$), so **$Q_{\text{red}}$ is
symmetric positive definite**. Consequences:

| Bézier system | Hermite system |
|---|---|
| indefinite KKT, size $n_f + m_e + m_b + m_g$ | SPD, size $n_f$ |
| `SparseLU` (+ METIS ordering) | `SimplicialLDLT` / `SimplicialLLT` |
| Schur-complement CG on $S = AQ^{-1}A^\top$ | plain PCG on $Q_{\text{red}}$ |
| `ISchurPreconditioner` + 2 approximate-CG variants | not needed |
| `FlexibleCG` (variable preconditioner) | not needed |
| three solve paths (direct / iterative / full-KKT) | one |
| `constraint_violation()` $\approx 2.45\times10^{-8}$ | $\equiv 0$ by construction |

What disappears from the codebase: `assemble_A_edge{,_free}`, `assemble_A_boundary_free`,
`assemble_A_gradient_free`, `assemble_kkt`, `CondensedSystem::A_edge`,
`solve_with_constraints_{direct,iterative,full_kkt}`, all of
[`schur_preconditioner.hpp`](../include/bathymetry/schur_preconditioner.hpp), both
`*_approx_cg_schur_preconditioner` classes, `flexible_cg.*`, and the entire
`SchurPreconditionerType` configuration surface. The cubic solve path collapses to the shape of
[`CGLinearBezierBathymetrySmoother::solve_with_constraints`](../src/bathymetry/cg_linear_bezier_bathymetry_smoother.cpp#L94-L173).

> **One gap in the reusable machinery.** `condense_matrix_and_rhs`
> ([constraint_condenser.hpp:49](../include/bathymetry/constraint_condenser.hpp#L49)) assumes
> homogeneous constraints — it computes $T^\top Q T$ and $T^\top c$ with no lift term. Every
> constraint in the current code is homogeneous, so this is fine as-is for hanging nodes and for
> zero-gradient BCs. **Inhomogeneous Dirichlet** ($z = g \neq 0$) needs the $-Q x_g$ correction
> added before condensation.

---

## 11. Conditioning — the price of physical derivative DOFs

Hermite DOFs are dimensionally inhomogeneous: $z$ has units of length, $z_x$ is dimensionless,
$z_{xy}$ has units of $1/\text{length}$, $z_{xxyy}$ of $1/\text{length}^3$. Rows of $Q$ scale like
$h^{-|\alpha|}$ where $|\alpha|$ is the DOF's total derivative order, so on an adaptive mesh

$$
\kappa(Q) \;\sim\; \left(\frac{h_{\max}}{h_{\min}}\right)^{2r} \times \kappa_{\text{intrinsic}}
$$

At $r = 2$ with four refinement levels ($h_{\max}/h_{\min} = 16$) that is $16^4 \approx 6.5\times10^4$
of pure scaling, before any intrinsic ill-conditioning. The same factor appears in $G_r$, whose
entries carry $h^{m_j - i}$ (§8).

**Remedy: symmetric equilibration.** Choose a nodal length scale $\ell_i$ (e.g. the smallest
adjacent element size) and set $S = \operatorname{diag}\big(\ell_i^{|\alpha_i|}\big)$, then solve

$$
\big(S Q S\big)\big(S^{-1}x\big) = S b
$$

This is a similarity transform, so it preserves symmetry and definiteness, and it restores the DOF
types to comparable magnitude. It is standard practice for Hermite elements and should be built in
from the start rather than retrofitted.

> **The ridge term needs the same treatment.** $\lambda\varepsilon I$ added uniformly across DOFs
> of different physical units is dimensionally inconsistent: it penalises $z_{xxyy}$ and $z$ with
> the same weight despite their entries differing by many orders of magnitude. It should be
> applied in the equilibrated space, i.e. $\lambda\varepsilon S^{-2}$ in original coordinates.
> The Bézier ridge is already not mesh-size-consistent (§5 of
> [cg_bezier_matrix_system.md](cg_bezier_matrix_system.md)); this is a sharper version of the same
> defect.

---

## 12. Interfacing with the existing pipeline

Because the per-element approximation space is *identical* ($Q_p$ either way, §1), the downstream
interfaces are untouched. Conversion is $c_e = M_e q_e$ at the output boundary only.

| Component | Status |
|---|---|
| `SeabedSurface::set_element_coefficients` ([seabed_surface.hpp:58](../include/mesh/seabed_surface.hpp#L58)) | unchanged — still receives Bernstein coefficients |
| `io::write_cg_bezier_surface_vtk` / `write_bezier_control_points_vtk` | unchanged (callback-based) |
| `CubicBezierBasis2D`, `LinearBezierBasis2D` | unchanged — used for $\hat B$ and for $M_e$ |
| `CubicThinPlateHessian`, `DirichletHessian` | unchanged — reference blocks reused under congruence |
| $B^\top W B$ quadrature loop | unchanged in structure; $M_e$ applied to the element block |
| `condense_matrix_and_rhs`, `back_substitute_slaves` | reused as-is (modulo the lift, §10) |
| `QuadtreeAdapter` neighbour / hanging-node detection | unchanged |
| DOF manager | **new** — position key must carry a derivative multi-index |
| $M_e$, $G_r$, strong-BC elimination | **new** |
| KKT / Schur machinery | **deleted** |

---

## 13. Comparison

Conforming $N \times N$ mesh, `edge_ngauss = 4`, interior edges $= 2N(N-1)$:

| | C⁰ Bernstein *(current)* | C⁰ Hermite | C¹ Bernstein *(current)* | C¹ Hermite (BFS) | C² Hermite |
|---|---|---|---|---|---|
| DOFs | $(N+1)^2$ | $(N+1)^2$ | $(3N+1)^2$ | $4(N+1)^2$ | $9(N+1)^2$ |
| Constraint rows | 0 | 0 | $8N(N-1)$ | 0 | 0 |
| System | SPD | SPD | **indefinite KKT** | **SPD** | **SPD** |
| Continuity | exact C⁰ | exact C⁰ | **approximate C¹** | **exact C¹** | **exact C²** |
| Boundedness | convex hull | convex hull | convex hull | none | none |
| Conditioning | benign | benign | benign | needs §11 | needs §11 |

| $N$ | C¹ Bernstein KKT | C¹ Hermite | ratio | C² Hermite |
|---|---|---|---|---|
| 16 | $2401 + 1920 = 4321$ | $1156$ | $3.74\times$ | $2601$ |
| 64 | $37\,249 + 32\,256 = 69\,505$ | $16\,900$ | $4.11\times$ | $38\,025$ |

The $N = 16$ row is the configuration used in
[cg_bezier_solver_verification.md](cg_bezier_solver_verification.md). Note that **C² Hermite
(2601 DOFs) is a smaller system than the current C¹ Bézier KKT (4321)** — and it is symmetric
positive definite while delivering a strictly higher order of continuity.

Per-row sparsity is comparable (a BFS node couples to the $3\times3$ node patch, $\approx 36$
entries; a Bézier corner control point couples across four elements, up to 49), so the DOF-count
ratio carries through to total non-zeros.

---

## 14. What is lost

**The convex-hull and variation-diminishing properties.** Bernstein control values bound the
surface they generate; Hermite DOFs do not, and a Hermite interpolant can overshoot in steep
regions. For bathymetry this is a real concern — an overshoot can put the fitted seabed above sea
level between data points.

Today this costs nothing: `lower_bound` / `upper_bound` are parsed into
`CGCubicBezierSmootherConfig` but never read by any solve path (§10 of
[cg_bezier_matrix_system.md](cg_bezier_matrix_system.md)). But it is the strongest reason to keep
the Bézier smoother alongside rather than replace it.

> **The coefficients are not lost, though.** Because $c_e = M_e q_e$ is explicit and cheap, the
> Bernstein coefficients remain inspectable at every step, and bound enforcement can be posed as
> $\ell \le M_e q_e \le u$ — inequality constraints on the corner DOFs. That is a QP rather than a
> linear solve, so it is not free, but it is a well-posed formulation rather than an impossibility.

**The `BezierSubdivision` multigrid transfer.** Its selling point is weights in $[0,1]$ (versus
$\pm4000$ for L2 projection). Hermite prolongation is *exact nested interpolation* — a coarse BFS
function restricted to a child quadrant is exactly representable in the child's BFS space, so $P$
is built from the same $G_r$ evaluation as the hanging-node constraints. That is arguably a better
transfer operator, but its weights are not in $[0,1]$ and it requires the §11 scaling to be
numerically sound.

**Position-hash DOF identity.** `quantize_position`
([cg_cubic_bezier_dof_manager.cpp:36-39](../src/bathymetry/cg_cubic_bezier_dof_manager.cpp#L36-L39))
maps a physical position to a DOF index. With Hermite, 4 or 9 DOFs share a node, so position alone
no longer disambiguates $z$ from $z_x$ from $z_{xy}$; the key must carry a derivative multi-index.
This incidentally forces a revisit of finding **P3** in
[references/paper-audit.md](references/paper-audit.md), which flags floating-point position
quantization for topology as the implementation's main deviation from p4est practice.

**Multigrid genericity.** `CompositeGridNode::dof_indices` is a hard-coded
`std::array<Index,16>`, and 16/64 appear as literals throughout
`bezier_multigrid_preconditioner.cpp`. The C¹ element is also 16 DOFs so the cubic case fits
as-is, but C⁰ (4) and C² (36) would require templating. `MultigridPreconditioner::setup` also takes
`const CGCubicBezierDofManager&` concretely rather than the base class.

---

## 15. Summary

The Hermite formulation replaces an over-determined collocation system and an indefinite saddle
point with an exactly-continuous, symmetric positive definite system roughly four times smaller,
while reusing the existing element matrices through a per-element congruence $M_e$. The
hanging-node constraints survive but as substitutions, which condense through machinery already in
the codebase. The costs are the loss of Bernstein's boundedness guarantee and a conditioning
problem that must be handled by equilibration from the outset.

Recommended target: **C¹ (bicubic BFS)**, implemented as a new smoother alongside the Bézier one
so the two can be cross-validated against the existing polynomial-reproduction and
interface-continuity tests. C⁰ requires no work — it is the existing linear smoother. C² is a
clean generalisation of the same formulas but needs the quadrature clamp lifted (§6) and the
equilibration of §11 to be in place.

---

## See also

- [hermite_smoothness_operator.md](hermite_smoothness_operator.md) — full discretization of the smoothness operator $H$ (§7)
- [cg_bezier_matrix_system.md](cg_bezier_matrix_system.md) — the system this one replaces
- [cg_bezier_solver_verification.md](cg_bezier_solver_verification.md) — solver and multigrid benchmarks
- [references/paper-audit.md](references/paper-audit.md) — single-core audit against the AMR literature
