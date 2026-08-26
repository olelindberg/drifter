# The Smoothness Operator $H$ — Energy Discretization in the Hermite Basis

This document derives the discrete smoothness operator $H$ for the Hermite bathymetry smoother of
[hermite_bathymetry_system.md](hermite_bathymetry_system.md), whose §7 states the result in three
paragraphs. Here the derivation is written out: the element basis, its derivatives, the element
matrix in both quadrature and closed Kronecker form, the physical derivative scaling, the global
assembly, and the properties of the resulting operator.

The route throughout is the one the code takes — energy functional, differentiated and integrated
over each element to give a quadratic form in the DOFs — not a Galerkin discretization of a weak
form. $H$ is the matrix of that quadratic form.

**Scope:** all three continuity orders — $r = 0$ (bilinear, 4 DOF/element), $r = 1$ (bicubic
Bogner–Fox–Schmit, 16 DOF/element), $r = 2$ (biquintic, 36 DOF/element). Notation follows the
parent document: $p = 2r+1$ is the polynomial degree per direction, $h_x, h_y$ are the element
dimensions, $(u,v) \in [0,1]^2$ are reference coordinates, $H_{s,m}$ is the 1D Hermite basis of §4
there, and $M(h)$ / $M_e$ is the Bernstein↔Hermite change of basis of §5 there.

---

## 1. Where $H$ sits

$H$ is the discrete smoothness (regularization) operator in the smoothness-first system assembled
by [`assemble_Q`](../src/bathymetry/cg_bezier_smoother_base.cpp#L396-L405):

$$
Q = \alpha H + \lambda\big(B^\top W B + \varepsilon I\big),
\qquad
\alpha = \frac{\lVert B^\top W B \rVert_F}{\lVert H \rVert_F}
$$

$H$ is the Gram matrix of an energy bilinear form $a(\cdot,\cdot)$ in the chosen basis: for a
surface $z_h = \sum_I x_I N_I$ with global DOF vector $x$,

$$
a(z_h, z_h) \;=\; x^\top H\, x, \qquad H_{IJ} = a(N_I, N_J)
$$

Everything below is the construction of $H_{IJ}$ when the $N_I$ are Hermite shape functions and
the $x_I$ are **nodal values and physical derivatives**, rather than Bernstein control values.

> **The approximation space is unchanged.** Bernstein and Hermite span the same tensor-product
> space $Q_p$ per element (§1, §5 of the parent document). The bilinear form is unchanged, so $H$
> in the two bases is the *same operator in different coordinates* — a congruence, derived
> independently in §6 below. Nothing here alters what is being minimized; it alters which linear
> functionals carry the coefficients.

---

## 2. The energies

Two energies are in play, selected by continuity order.

### 2.1 Membrane (Dirichlet) energy — $r = 0$

$$
a_{\text{mem}}(z, w) \;=\; \int_\Omega \big( z_x w_x + z_y w_y \big)\, \mathrm{d}x\,\mathrm{d}y,
\qquad
a_{\text{mem}}(z,z) = \int_\Omega \big( z_x^2 + z_y^2 \big)
$$

Used at $r = 0$ because the thin-plate energy **degenerates** on $Q_1$. A bilinear function
$z = c_0 + c_1 x + c_2 y + c_3 xy$ has $z_{xx} = z_{yy} = 0$, so both thin-plate forms of §2.2
collapse to the pure twist term $2\int z_{xy}^2 = 2 c_3^2\, h_x h_y$, which sees only one
coefficient per element. That is not a usable regularizer: on a conforming $N \times N$ grid of
continuous bilinear elements, zero twist energy means the nodal values satisfy
$z_{i,j} - z_{i+1,j} - z_{i,j+1} + z_{i+1,j+1} = 0$ on every element, i.e. $z_{ij} = f_i + g_j$, a
null space of dimension $2N+1$ out of $(N+1)^2$ DOFs — it **grows with the mesh**. The membrane
energy has null space $\{1\}$ at every resolution. This is a soap film: it minimizes area, not
curvature.

> **Small correction to the existing docs.** `CLAUDE.md` and §4.2 of
> [cg_bezier_matrix_system.md](cg_bezier_matrix_system.md) state that "thin plate energy vanishes
> for bilinear surfaces". Strictly, only the *bending* terms $z_{xx}^2, z_{yy}^2, z_{xx}z_{yy}$
> vanish; the twist term $2z_{xy}^2$ survives. The conclusion — that the membrane energy is the
> right choice at $r = 0$ — is unaffected, and the reason is the mesh-dependent null space above
> rather than an identically zero integrand.

### 2.2 Thin-plate energy — $r \ge 1$

The form implemented today
([cubic_thin_plate_hessian.cpp:98-114](../src/bathymetry/cubic_thin_plate_hessian.cpp#L98-L114))
is

$$
a_{\text{impl}}(z,z) \;=\; \int_\Omega \Big[ \big( z_{xx} + z_{yy} \big)^2 + 2 z_{xy}^2 \Big]
\;=\; \int_\Omega \Big[ z_{xx}^2 + z_{yy}^2 + 2 z_{xx} z_{yy} + 2 z_{xy}^2 \Big]
$$

whereas the standard thin-plate (bending) energy is

$$
a_{\text{std}}(z,z) \;=\; \int_\Omega \Big[ z_{xx}^2 + 2 z_{xy}^2 + z_{yy}^2 \Big]
$$

The two differ by the cross term $2\int z_{xx} z_{yy}$.

> **The implemented form is a third energy, not a variant of the standard one.** The classical
> fact is that the standard bending energy and the squared Laplacian differ by a null Lagrangian:
> $a_{\text{std}} = \int (\Delta z)^2 - 2\int (z_{xx}z_{yy} - z_{xy}^2)$, and
> $\int (z_{xx}z_{yy} - z_{xy}^2)$ depends only on boundary data. The implemented form is
> $\int (\Delta z)^2$ **plus** an extra twist penalty $2\int z_{xy}^2$, so it is separated from
> $a_{\text{std}}$ by $2\int z_{xy}^2 + 2\int(z_{xx}z_{yy} - z_{xy}^2)$ — a genuine change of
> energy, not a boundary term. Its most visible consequence is the **null space**:
>
> $$\ker a_{\text{impl}} = \operatorname{span}\{1,\ x,\ y,\ x^2 - y^2\}\ (\dim 4),
> \qquad \ker a_{\text{std}} = \operatorname{span}\{1,\ x,\ y\}\ (\dim 3)$$
>
> The mode $x^2 - y^2$ is harmonic ($z_{xx} + z_{yy} = 0$) *and* has $z_{xy} = 0$, so it costs
> nothing under $a_{\text{impl}}$. This holds for every $r \ge 1$ and both element and global
> operator (§9), and is inherited from the Bézier smoother — it is a property of the energy, not
> of the basis.

Both energies are symmetric and positive semi-definite; both are written below.

---

## 3. The Hermite element basis

### 3.1 Shape functions

Let the element be $[x_e, x_e + h_x] \times [y_e, y_e + h_y]$ with $x = x_e + h_x u$,
$y = y_e + h_y v$. The DOFs are the physical corner derivatives

$$
q_{(s_x,a),(s_y,b)} \;=\; \frac{\partial^{\,a+b} z}{\partial x^a\, \partial y^b}
\bigg|_{\text{corner } (s_x, s_y)}, \qquad s_x, s_y \in \{0,1\},\quad a, b \in \{0,\dots,r\}
$$

and the dual shape functions are (§4 of the parent document)

$$
N_{(s_x,a),(s_y,b)}(u,v) \;=\; \underbrace{H_{s_x,a}(u)\, H_{s_y,b}(v)}_{\hat N \ \text{(parametric)}}
\;\cdot\; h_x^{\,a}\, h_y^{\,b}
$$

The factor $h_x^a h_y^b$ is what makes the DOF a *physical* derivative, so that neighbouring
elements of different sizes share one unambiguous value per node. In matrix form,

$$
N \;=\; \hat N\, \Lambda_e,
\qquad
\Lambda_e \;=\; \operatorname{diag}\!\big( h_x^{\,a_I}\, h_y^{\,b_I} \big)_{I = 1}^{(p+1)^2}
\;=\; \Lambda(h_y) \otimes \Lambda(h_x)
$$

where $\Lambda(h) = \operatorname{diag}(h^{\mu_i})$ over the $2(r+1)$ 1D DOFs and $\mu_i$ is the
derivative order of the $i$-th 1D DOF.

### 3.2 Local DOF ordering

The Kronecker factorisation of §5 requires the ordering to be fixed explicitly — the two existing
Bernstein bases already disagree with each other ($i + 4j$ for
[cubic](../include/bathymetry/cubic_bezier_basis_2d.hpp#L65), $j + 2i$ for
[linear](../include/bathymetry/linear_bezier_basis_2d.hpp#L66), which are transposes; see §2 of
[cg_bezier_matrix_system.md](cg_bezier_matrix_system.md)).

Adopt the cubic convention. Collapse each direction's node/derivative pair into a single 1D index

$$
i \;=\; s_x (r{+}1) + a \ \in \{0, \dots, p\},
\qquad
j \;=\; s_y (r{+}1) + b \ \in \{0, \dots, p\}
$$

so the 1D DOFs run *node-major*: for $r=1$, $(z_0,\ z_0',\ z_1,\ z_1')$. The local 2D index is

$$
I \;=\; i + (p{+}1)\, j
$$

i.e. the $u$-index varies fastest. With this convention a tensor-product operator $X$ (acting in
$u$) times $Y$ (acting in $v$) assembles as the Kronecker product $Y \otimes X$ — the *rightmost*
factor is the fastest-varying index. All Kronecker products below follow that ordering.

### 3.3 The 1D bases and their derivatives

$r = 0$ (linear; these are the degree-1 Bernstein polynomials):

$$
H_{0,0} = 1 - t, \qquad H_{1,0} = t,
\qquad H_{0,0}' = -1, \qquad H_{1,0}' = 1
$$

$r = 1$ (cubic):

$$
\begin{aligned}
H_{0,0} &= 2t^3 - 3t^2 + 1, & H_{0,0}' &= 6t^2 - 6t, & H_{0,0}'' &= 12t - 6\\
H_{0,1} &= t^3 - 2t^2 + t, & H_{0,1}' &= 3t^2 - 4t + 1, & H_{0,1}'' &= 6t - 4\\
H_{1,0} &= -2t^3 + 3t^2, & H_{1,0}' &= -6t^2 + 6t, & H_{1,0}'' &= -12t + 6\\
H_{1,1} &= t^3 - t^2, & H_{1,1}' &= 3t^2 - 2t, & H_{1,1}'' &= 6t - 2
\end{aligned}
$$

$r = 2$ (quintic):

$$
\begin{aligned}
H_{0,0} &= -6t^5 + 15t^4 - 10t^3 + 1, & H_{0,0}' &= -30t^4 + 60t^3 - 30t^2, & H_{0,0}'' &= -120t^3 + 180t^2 - 60t\\
H_{0,1} &= -3t^5 + 8t^4 - 6t^3 + t, & H_{0,1}' &= -15t^4 + 32t^3 - 18t^2 + 1, & H_{0,1}'' &= -60t^3 + 96t^2 - 36t\\
H_{0,2} &= -\tfrac12 t^5 + \tfrac32 t^4 - \tfrac32 t^3 + \tfrac12 t^2, & H_{0,2}' &= -\tfrac52 t^4 + 6t^3 - \tfrac92 t^2 + t, & H_{0,2}'' &= -10t^3 + 18t^2 - 9t + 1\\
H_{1,0} &= 6t^5 - 15t^4 + 10t^3, & H_{1,0}' &= 30t^4 - 60t^3 + 30t^2, & H_{1,0}'' &= 120t^3 - 180t^2 + 60t\\
H_{1,1} &= -3t^5 + 7t^4 - 4t^3, & H_{1,1}' &= -15t^4 + 28t^3 - 12t^2, & H_{1,1}'' &= -60t^3 + 84t^2 - 24t\\
H_{1,2} &= \tfrac12 t^5 - t^4 + \tfrac12 t^3, & H_{1,2}' &= \tfrac52 t^4 - 4t^3 + \tfrac32 t^2, & H_{1,2}'' &= 10t^3 - 12t^2 + 3t
\end{aligned}
$$

These satisfy the duality relations $H_{s,m}^{(m')}(s') = \delta_{ss'}\delta_{mm'}$, which is the
only property used in §3 of the parent document and the only one needed to identify DOFs across
elements.

---

## 4. Element matrix by quadrature

This is the route the current code takes, transposed to the Hermite basis.

### 4.1 Derivative matrices

Let $\{(u_q, v_q)\}_{q=1}^{n_q^2}$ be the tensor-product Gauss–Legendre points on $[0,1]^2$ with
weights $w_q = \hat w_{q_i} \hat w_{q_j}$ (the $\tfrac12$-scaled weights of
[`compute_gauss_quadrature`](../src/bathymetry/cubic_thin_plate_hessian.cpp#L60-L68), which map
$[-1,1]$ weights to $[0,1]$). Define the **parametric** derivative matrices
$D_{ab} \in \mathbb{R}^{n_q^2 \times (p+1)^2}$ by

$$
\big[D_{ab}\big]_{qI} \;=\; \frac{\partial^{\,a+b} \hat N_I}{\partial u^a\, \partial v^b}(u_q, v_q)
\;=\; H_{s_x,\alpha}^{(a)}(u_q)\, H_{s_y,\beta}^{(b)}(v_q)
$$

for the derivative pairs each energy needs: $(u), (v)$ for the membrane form and
$(uu), (vv), (uv)$ for the thin plate. With $W = \operatorname{diag}(w_q)$ set

$$
\hat H_{ab,cd} \;=\; D_{ab}^\top\, W\, D_{cd}
$$

Note $\hat H_{ab,cd}^\top = \hat H_{cd,ab}$; the diagonal blocks $\hat H_{ab,ab}$ are symmetric,
the cross block $\hat H_{uu,vv}$ is not.

### 4.2 Physical scaling

Substituting $z_{xx} = z_{uu}/h_x^2$, $z_{yy} = z_{vv}/h_y^2$, $z_{xy} = z_{uv}/(h_x h_y)$ into
the energies of §2 and carrying the Jacobian $h_x h_y$ gives the parametric element matrix

$$
\hat H_e \;=\;
\frac{h_y}{h_x^3}\, \hat H_{uu,uu}
\;+\; \frac{h_x}{h_y^3}\, \hat H_{vv,vv}
\;+\; \frac{1}{h_x h_y}\big( \hat H_{uu,vv} + \hat H_{uu,vv}^\top \big)
\;+\; \frac{2}{h_x h_y}\, \hat H_{uv,uv}
$$

for $a_{\text{impl}}$; dropping the third term gives $a_{\text{std}}$. For $r = 0$ the membrane
counterpart is

$$
\hat H_e \;=\; \frac{h_y}{h_x}\, \hat H_{u,u} \;+\; \frac{h_x}{h_y}\, \hat H_{v,v}
$$

Both are identical in form to
[`CubicThinPlateHessian::scaled_hessian`](../src/bathymetry/cubic_thin_plate_hessian.cpp#L131-L166)
and [`DirichletHessian::scaled_hessian`](../src/bathymetry/dirichlet_hessian.cpp#L116-L133) — the
scaling depends on the energy and the geometry, not on the basis.

> **The cross term's factor of 2.** The expansion of $(z_{xx}+z_{yy})^2$ contains
> $2 z_{xx} z_{yy}$, yet the symmetrized pair above carries coefficient 1. These agree: for any
> $x$, $\;2\, x^\top \hat H_{uu,vv}\, x = x^\top(\hat H_{uu,vv} + \hat H_{uu,vv}^\top)\, x$. The
> $z_{uv}^2$ term keeps its explicit 2 because $\hat H_{uv,uv}$ is already symmetric.

Finally the DOF scaling of §3.1 converts to physical derivative DOFs:

$$
\boxed{\;H_e \;=\; \Lambda_e\, \hat H_e\, \Lambda_e\;}
$$

a diagonal congruence, so symmetry and definiteness are preserved. As in the existing code, a
final explicit symmetrization $H_e \leftarrow \tfrac12(H_e + H_e^\top)$ removes round-off
asymmetry ([:161-166](../src/bathymetry/cubic_thin_plate_hessian.cpp#L161-L166)).

The anisotropic factors matter: on a directionally refined element ($h_x \neq h_y$) the
$h_y/h_x^3$ and $h_x/h_y^3$ terms differ by orders of magnitude, and it is that imbalance that
makes the smoother behave correctly on stretched cells.

---

## 5. Element matrix in closed form — Kronecker factorisation

The quadrature route of §4 is unnecessary. Both energies are sums of *tensor-product* terms and
the Hermite basis is a tensor product, so the element matrix factorises into 1D energy matrices
whose entries are integrals of polynomials — exactly evaluable, once, for each $r$.

### 5.1 The 1D energy matrices

Define, on the reference interval,

$$
\big[K^{(m,n)}\big]_{ij} \;=\; \int_0^1 H_i^{(m)}(t)\, H_j^{(n)}(t)\, \mathrm{d}t,
\qquad i, j \in \{0,\dots,p\}
$$

using the collapsed 1D index of §3.2. Immediately $K^{(m,n)} = \big(K^{(n,m)}\big)^\top$, so
$K^{(0,0)}$ (a mass matrix), $K^{(1,1)}$ and $K^{(2,2)}$ are symmetric while $K^{(2,0)}$ is not.
Each entry is the integral of a polynomial of degree at most $2(p-\max(m,n))$ and is a rational
number; no quadrature rule is involved and no truncation error is introduced.

The **physical** 1D energy matrix on an interval of length $h$, expressed in physical-derivative
DOFs, folds in both the chain rule and the DOF scaling:

$$
K^{(m,n)}_{\text{phys}}(h) \;=\; h^{\,1 - m - n}\, \Lambda(h)\, K^{(m,n)}\, \Lambda(h)
\;=\; \int_0^h \frac{\mathrm{d}^m N_i}{\mathrm{d}x^m}\, \frac{\mathrm{d}^n N_j}{\mathrm{d}x^n}\, \mathrm{d}x
$$

(the $h^{1}$ from the Jacobian, $h^{-m-n}$ from the two chain rules, $\Lambda(h)$ twice from the
physical DOFs).

### 5.2 The element matrix

With the ordering of §3.2 ($u$ fastest, so the $v$-factor is leftmost), the thin-plate element
matrix is

$$
H_e \;=\;
K^{(0,0)}_{\text{phys}}(h_y) \otimes K^{(2,2)}_{\text{phys}}(h_x)
\;+\; K^{(2,2)}_{\text{phys}}(h_y) \otimes K^{(0,0)}_{\text{phys}}(h_x)
\;+\; \Big[ K^{(0,2)}_{\text{phys}}(h_y) \otimes K^{(2,0)}_{\text{phys}}(h_x) + \text{transpose} \Big]
\;+\; 2\, K^{(1,1)}_{\text{phys}}(h_y) \otimes K^{(1,1)}_{\text{phys}}(h_x)
$$

term by term: $\int z_{xx}w_{xx}$, $\int z_{yy}w_{yy}$, the symmetrized $2\int z_{xx}w_{yy}$, and
$2\int z_{xy}w_{xy}$. Dropping the bracketed cross term yields the standard thin plate. The
membrane matrix is

$$
H_e \;=\; K^{(0,0)}_{\text{phys}}(h_y) \otimes K^{(1,1)}_{\text{phys}}(h_x)
\;+\; K^{(1,1)}_{\text{phys}}(h_y) \otimes K^{(0,0)}_{\text{phys}}(h_x)
$$

Expanding $K_{\text{phys}}$ recovers §4 exactly: the first term carries
$h_y^{1} \cdot h_x^{1-4} = h_y / h_x^{3}$, the second $h_x/h_y^3$, the cross terms
$h_y^{-1} h_x^{-1}$, and the twist term $h_y^{-1}h_x^{-1}$ with its explicit factor 2 — the
scaling table of `scaled_hessian`, derived rather than asserted.

> **Consequences for implementation.** The six matrices
> $K^{(0,0)}, K^{(1,1)}, K^{(2,2)}, K^{(2,0)}$ (and $K^{(0,2)} = (K^{(2,0)})^\top$) are
> $(p{+}1) \times (p{+}1)$ constants per order $r$ — $4\times4$, $6\times6$ at most. The element
> matrix is then two diagonal scalings and four Kronecker products, with no basis evaluation, no
> quadrature loop, no $n_q^2 \times (p+1)^2$ derivative matrices to cache, and no quadrature
> error. This is strictly simpler than the current
> [`build_derivative_matrices`](../src/bathymetry/cubic_thin_plate_hessian.cpp#L71-L96) +
> `scaled_hessian` path, and it is what makes $r = 2$ cheap to add.

---

## 6. Consistency with the Bernstein element matrices

The two routes must agree with §7 of the parent document, which asserts
$H_e = M_e^\top \hat H_e^{\text{Bern}} M_e$. They do, and the reason is elementary.

Both bases span the same space $Q_p$ on the element, and §5 of the parent document gives the
change of basis explicitly: $c_e = M_e q_e$ with $M_e = M(h_x) \otimes M(h_y)$ (subject to the
index ordering of §3.2 — with $u$ fastest the product is $M(h_y) \otimes M(h_x)$). For any
bilinear form,

$$
a(z_h, z_h) \;=\; c_e^\top\, H_e^{\text{Bern}}\, c_e
\;=\; q_e^\top\, M_e^\top H_e^{\text{Bern}} M_e\, q_e
\qquad\Longrightarrow\qquad
H_e^{\text{Herm}} \;=\; M_e^\top\, H_e^{\text{Bern}}\, M_e
$$

This is a congruence, exact, not an approximation — provided both sides integrate the energy
exactly. If the Bernstein side is built with a quadrature rule too weak to integrate its
integrand (§7), the identity holds only up to that quadrature error, and the closed form of §5 is
the correct one.

Because $M_e$ already carries the $h^m$ factors, this route needs no separate $\Lambda_e$: the
scaling appears once, inside $M(h)$. The congruence is what licenses reusing
`CubicThinPlateHessian` and `DirichletHessian` verbatim, at the cost of one $(p+1)^2$-square
matrix product per element. §5 is the independent second derivation, and the identity between them
is a ready-made unit test.

---

## 7. Quadrature requirements

Relevant only to the §4 route; §5 needs none.

For a term $\int \partial^m \hat N_I\, \partial^m \hat N_J$, the integrand has degree $2(p-m)$ per
direction, so exact Gauss–Legendre integration needs $n_q \ge (p - m) + 1$ points per direction.
The governing (least-differentiated) factor in each energy is the one with $m = 0$ in a given
direction: the thin-plate cross block $\hat H_{uu,vv}$ is degree $p-2$ in $u$ against degree $p$
in $v$, so the binding requirement is degree $2p$ from the $\hat H_{uu,vv}$ and $\hat H_{u,u}$
type terms.

| $r$ | $p$ | energy | max integrand degree / direction | $n_q$ needed | current limit |
|---|---|---|---|---|---|
| 0 | 1 | membrane | 2 | 2 | `DirichletHessian` accepts 1–4, defaults to 2 — **sufficient** |
| 1 | 3 | thin plate | 6 | 4 | `CubicThinPlateHessian` accepts 2–6, defaults to 4 — **exactly sufficient** |
| 2 | 5 | thin plate | 10 | 6 | `CubicThinPlateHessian` caps at 6 ([:57](../src/bathymetry/cubic_thin_plate_hessian.cpp#L57)) — sufficient, but the smoother base's `gauss_legendre_01` silently truncates any request $\ge 4$ to 4 points ([:44-48](../src/bathymetry/cg_bezier_smoother_base.cpp#L44-L48)) — **insufficient there** |

Two failure modes follow from under-integration, and they differ:

- **Loss of exactness.** $H_e$ ceases to represent the energy, so $\alpha$ (which normalizes
  $\lVert H \rVert_F$) and the smoothness/data balance shift silently.
- **Spurious kernel modes.** $\hat H_{ab,cd} = D^\top W D$ has rank at most $n_q^2$. If
  $n_q^2 < (p{+}1)^2 - \dim\ker a$, the element matrix acquires zero-energy modes that the
  continuous form does not have. At $r = 2$ with the 4-point clamp, $n_q^2 = 16 < 36 - 4$, so this
  is not hypothetical.

Both disappear under the closed form of §5.

---

## 8. Global assembly

For each element, scatter $H_e$ into the global sparse operator over the element's global DOF map,
exactly as
[`assemble_hessian_global`](../src/bathymetry/cg_bezier_smoother_base.cpp#L260-L302) does today:
build triplets $(I_i, I_j, [H_e]_{ij})$, discard entries below $10^{-16}$, and set the sparse
matrix from triplets. Nothing about this loop is basis-specific; it depends only on
`num_dofs()` and `scaled_hessian(dx, dy)`, the two methods of
[`BezierHessianBase`](../include/bathymetry/bezier_hessian_base.hpp).

Two Hermite-specific points:

**The DOF map is the new work.** A node carries $(r+1)^2$ DOFs, so a position key alone no longer
identifies a DOF — it must carry the derivative multi-index $(a,b)$ as well (§14 of the parent
document). Two elements meeting at a node must agree on both the node *and* the multi-index for
the C^r identification of §3 there to hold. This is the only place where basis choice reaches the
assembly loop.

**Constraints act after assembly, by congruence.** Hanging-node substitutions and strong boundary
conditions enter as the prolongation $T$ of §10 of the parent document,

$$
H_{\text{red}} \;=\; T^\top H\, T
$$

which preserves symmetry and semi-definiteness. Since the same $T$ is applied to $B^\top W B$ and
the ridge, it is equivalent (and cheaper) to condense the assembled $Q$ once via
[`condense_matrix_and_rhs`](../include/bathymetry/constraint_condenser.hpp#L49) rather than to
condense $H$ separately — but note that $\alpha$ depends on $\lVert H \rVert_F$, which must be
computed **before** condensation for consistency with the current definition.

---

## 9. Properties

**Symmetry.** $H_e$ is symmetric by construction in both routes: $\hat H_{ab,ab}$ blocks are
symmetric, the cross block appears only as $\hat H_{uu,vv} + \hat H_{uu,vv}^\top$, and $\Lambda_e$,
$M_e$, $T$ enter as congruences.

**Positive semi-definiteness.** $x^\top H_e x = a(z_h, z_h) \ge 0$ for both energies, since both
integrands are sums of squares — $a_{\text{impl}}$ as $(z_{xx}+z_{yy})^2 + 2z_{xy}^2$ and
$a_{\text{std}}$ termwise. Congruence preserves this, so the global $H$ and the condensed
$H_{\text{red}}$ are PSD. $H$ is *never* positive definite; definiteness of $Q$ comes from
$\lambda \varepsilon I$ and the data term.

**Null spaces.**

| Energy | element kernel | global kernel (conforming C^r mesh) |
|---|---|---|
| membrane, $r=0$ | $\{1\}$, $\dim 1$ | $\{1\}$, $\dim 1$ |
| $a_{\text{impl}}$, $r \ge 1$ | $\{1, x, y, x^2 - y^2\}$, $\dim 4$ | $\dim 4$ |
| $a_{\text{std}}$, $r \ge 1$ | $\{1, x, y\}$, $\dim 3$ | $\dim 3$ |

The global kernels do not grow with the mesh: C^r patching forces the coefficients of the kernel
modes to agree across every element, so a globally zero-energy surface is a single element-kernel
polynomial extended over $\Omega$. (Element kernels are stated for the exactly-integrated
operator; see §7 for what under-integration adds.)

**Scaling and conditioning.** $H$ is dimensionally inhomogeneous, because its DOFs are. Row $I$ of
$H$ scales like $h^{-|\alpha_I|}$ where $|\alpha_I| = a_I + b_I$ is the DOF's total derivative
order, so on a mesh with size ratio $h_{\max}/h_{\min}$,

$$
\kappa(H) \;\sim\; \left( \frac{h_{\max}}{h_{\min}} \right)^{2r} \times \kappa_{\text{intrinsic}}
$$

before any intrinsic ill-conditioning of the biharmonic operator. §11 of the parent document
prescribes the fix — symmetric equilibration $S = \operatorname{diag}(\ell_I^{|\alpha_I|})$
applied as $SHS$ — and it should be built in from the start rather than retrofitted.

> **The ridge must live in the same space.** $\lambda \varepsilon I$ added uniformly across DOFs
> of different physical units penalises $z$ and $z_{xxyy}$ identically despite their entries
> differing by many orders of magnitude. Applied in the equilibrated space it becomes
> $\lambda\varepsilon S^{-2}$ in original coordinates. Since the ridge exists specifically to
> cover the null space catalogued above, getting its scaling wrong directly distorts the modes it
> is meant to control.

---

## 10. Correspondence to existing code

| Component | Status under the Hermite formulation |
|---|---|
| [`BezierHessianBase`](../include/bathymetry/bezier_hessian_base.hpp) interface (`num_dofs`, `scaled_hessian`) | unchanged — the abstraction is basis-agnostic |
| [`assemble_hessian_global`](../src/bathymetry/cg_bezier_smoother_base.cpp#L260-L302) | unchanged |
| [`CubicThinPlateHessian`](../src/bathymetry/cubic_thin_plate_hessian.cpp#L131-L166) | reusable as-is via the §6 congruence; superseded by §5 if the closed form is implemented |
| [`DirichletHessian`](../src/bathymetry/dirichlet_hessian.cpp#L116-L133) | unchanged — at $r=0$, $M_e = \Lambda_e = I$ and the Hermite element *is* the linear Bézier element |
| $\Lambda_e$ / $M_e$ per-element congruence | **new** (only if the §6 reuse route is taken) |
| $K^{(m,n)}$ tables and the Kronecker assembly | **new** (the §5 route; replaces the quadrature machinery) |
| $r = 2$ energy blocks | **new** — [`QuinticBasis2D`](../include/bathymetry/quintic_basis_2d.hpp#L28) exists (36 DOF) but has no Hessian class |
| DOF map keyed by position + derivative multi-index | **new** (§8) |
| `ngauss_energy` configuration | retained for the §4/§6 route; **vacuous** under §5 |

---

## See also

- [hermite_bathymetry_system.md](hermite_bathymetry_system.md) — the Hermite formulation; §7 is
  the summary of this document, §11 the conditioning treatment it depends on
- [cg_bezier_matrix_system.md](cg_bezier_matrix_system.md) — §4, the same operator in the
  Bernstein basis
- [cg_bezier_solver_verification.md](cg_bezier_solver_verification.md) — solver and multigrid
  benchmarks
