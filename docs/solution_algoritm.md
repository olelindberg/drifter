# Governing Equations

<!--
Conventions for this document. Prose style is in docs/CLAUDE.md; these are the notational
commitments specific to the derivation here. Follow them when editing or extending it.

- This is a theory document: it derives the equations from first principles and cites no
  source files. Do not add links into src/, class names, or config keys.
- Vertical: z points upwards along local gravity, z = 0 at the undisturbed surface, bed at
  z = -h and free surface at z = eta. g > 0 is a magnitude, so it enters the vertical
  momentum equation as -g and hydrostatic balance reads dp/dz = -rho g.
- Spherical coordinates use the physics (ISO) convention: theta is the polar angle measured
  from the rotation axis, i.e. a colatitude, not a latitude; phi is azimuthal. The triad
  (r, theta, phi) is right-handed. Geographic latitude is written varphi.
- Primed symbols (theta', phi', x') always denote the rotated spherical frame of
  §1.4. Unprimed ones denote the unrotated Earth-fixed frame.
- The spherical geopotential approximation is in force from §1.5 onwards, with the
  constant mean radius a = 6371 km.
- Symbols are reused per frame on purpose. Where a section redefines x, y, z, u, v, w, say
  so explicitly at the top of it and say where the redefinition ends, as the ECI section does.
- Every display equation carries a \tag{}, numbered consecutively from (1) in document
  order. Inserting, removing or reordering an equation renumbers every later one, so
  renumber the whole document in one pass and update the prose cross-references to match.
- Sections are numbered at both heading levels, "## 1. Coordinates systems" and
  "### 1.4 Rotated spherical coordinates", and are cited in prose as §1.4. Adding or moving
  a section renumbers the rest the same way equations do.
-->
# Solution Algorithm

## 8. Implicit Free Surface

The free surface carries external gravity waves at $c = \sqrt{g d}$ — about 200 m/s in
4000 m of water, against 2–3 m/s for the first internal mode and well under 1 m/s for
advection. An explicit scheme is therefore limited by a wave that usually carries little
of the physics of interest. Treating the barotropic terms implicitly removes that
restriction.

Within this section the equations of §7 are taken in the local Cartesian limit $r \to a$:
$x$, $y$ are the horizontal coordinates on the rotated triad, $z$ the vertical one of (39),
$u$, $v$ the corresponding velocity components, $\nabla_h = \left(\partial / \partial x, \;
\partial / \partial y\right)$, and the metric factors of §7 are unity. The redefinition holds
to the end of the document.

### 8.1 The barotropic subsystem

Collect the depth-integrated momentum equations with the barotropic pressure gradient
written out and everything else gathered into a single term,

$$
\frac{\partial \eta}{\partial t} + \nabla_h \cdot \left(d \mathbf{U}\right) = 0,
\qquad
\frac{\partial \left(d \mathbf{U}\right)}{\partial t} + g\, d\, \nabla_h \eta = \mathbf{G},
\tag{78}
$$

where $\mathbf{U} = (U, V)$ are the depth-averaged velocities and $d\mathbf{U}$ the volume
transport. The right-hand side

$$
\mathbf{G} = \underbrace{-\nabla_h \cdot \left(d \mathbf{U}\mathbf{U}
+ \int\limits_{-h}^{\eta} \tilde{\mathbf{u}} \tilde{\mathbf{u}} \, dz\right)}_{\text{advection and dispersion}}
\; \underbrace{-\, f \, \hat{\mathbf{k}} \times d \mathbf{U}}_{\text{Coriolis}}
\; \underbrace{-\, \frac{d}{\rho_0} \nabla_h p_a}_{\text{atmospheric}}
\; \underbrace{-\, \frac{g}{\rho_0} \int\limits_{-h}^{\eta} \int\limits_{z}^{\eta} \nabla_h \rho' \, dz' \, dz}_{\text{baroclinic}}
\; + \; \underbrace{\frac{\boldsymbol{\tau}_s - \boldsymbol{\tau}_b}{\rho_0}}_{\text{stresses}}
\tag{79}
$$

contains no fast waves: every term in it evolves on the advective or baroclinic time
scale, so it can be evaluated explicitly at time level $n$. Only the pair $\eta$ and
$g\,d\,\nabla_h \eta$ supports the fast mode, and only that pair is taken implicitly.

This subsystem is not posed independently — it is the depth integral of the
three-dimensional momentum equations, and $\mathbf{G}$ is correspondingly the depth
integral of their right-hand side, as made explicit in §8.4. That is what makes the barotropic and three-dimensional solutions
consistent by construction.

### 8.2 Time discretisation

Apply a $\theta$-weighted scheme to the two barotropic terms, with $\theta = 1$ giving
backward Euler and $\theta = 1/2$ the trapezoidal rule,

$$
\eta^{n+1} = \eta^{n} - \Delta t\, \nabla_h \cdot
\left[\theta \left(d \mathbf{U}\right)^{n+1} + (1 - \theta)\left(d \mathbf{U}\right)^{n}\right],
\tag{80}
$$

$$
\left(d \mathbf{U}\right)^{n+1} = \left(d \mathbf{U}\right)^{n}
- g\, \Delta t\, d \left[\theta \nabla_h \eta^{n+1} + (1 - \theta) \nabla_h \eta^{n}\right]
+ \Delta t\, \mathbf{G}^{n} .
\tag{81}
$$

The transport at the new time level appears in both, and eliminating it between them
leaves a single equation for $\eta^{n+1}$.

### 8.3 The elliptic problem

Substituting the second equation into the first gives a two-dimensional **Helmholtz
equation** for the new free surface,

$$
\boxed{\;
\eta^{n+1} - g\, \theta^{2} \Delta t^{2}\, \nabla_h \cdot \left(d\, \nabla_h \eta^{n+1}\right)
= \eta^{n} - \Delta t\, \nabla_h \cdot
\left[\left(d \mathbf{U}\right)^{n} + \theta \Delta t\, \mathbf{G}^{n}
- g\, \theta (1 - \theta) \Delta t\, d\, \nabla_h \eta^{n}\right] \;}
\tag{82}
$$

The unknown is $\eta$ on the horizontal grid alone — there is no vertical index, because
the barotropic pressure gradient is depth-independent. For $N_z$ vertical levels this
system is $N_z$ times smaller than a three-dimensional solve.

The operator $\mathcal{L} = \mathcal{I} - g \theta^{2} \Delta t^{2} \nabla_h \cdot (d \nabla_h)$
is symmetric and, since $d > 0$, positive definite, so conjugate gradients or multigrid
apply directly. The identity term keeps it non-singular even with closed boundaries
everywhere — unlike the rigid-lid formulation, which yields a pure Poisson problem with a
constant nullspace. Its condition number grows like
$1 + g \theta^{2} \Delta t^{2} d / \Delta x^{2}$, so at the large time steps that motivate
the method a multigrid preconditioner is preferable to plain CG.

Boundary conditions follow from the transport: a closed wall imposes
$\mathbf{n} \cdot (d\mathbf{U}) = 0$, which through the momentum equation becomes the
homogeneous Neumann condition $\partial \eta / \partial n = 0$; an open boundary with a
prescribed tide or surge imposes $\eta$ directly as a Dirichlet condition.

### 8.4 Stepping the three-dimensional velocities

The three-dimensional momentum equations are unchanged by the implicit treatment, and they
remain the prognostic core of the model: they are advanced **once** per time step, with no
subcycling. What the implicit free surface changes is only which part of the pressure
gradient is evaluated at which time level,

$$
\frac{1}{\rho_0}\frac{\partial p}{\partial x} =
\underbrace{g \frac{\partial \eta^{n+1}}{\partial x}}_{\text{implicit}}
+ \underbrace{\frac{1}{\rho_0}\frac{\partial p_a^{n}}{\partial x}
+ \frac{g}{\rho_0} \int\limits_{z}^{\eta} \frac{\partial \rho'^{\,n}}{\partial x} \, dz'}_{\text{explicit}} .
\tag{83}
$$

Only the barotropic term carries the fast wave, and it arrives already known from the
elliptic solve. A step proceeds in three stages.

**Stage 1 — assemble the explicit tendency.** Collect everything in the momentum equations
except the barotropic pressure gradient,

$$
\mathbf{R}^{n} = -\nabla \cdot \left(\mathbf{u}\,\mathbf{u}\right)^{n}
- f\,\hat{\mathbf{k}} \times \mathbf{u}^{n}
- \frac{1}{\rho_0}\nabla_h p_a^{n}
- \frac{g}{\rho_0} \int\limits_{z}^{\eta} \nabla_h \rho'^{\,n} \, dz'
+ \boldsymbol{\mathcal{F}}^{n},
\tag{84}
$$

whose first term contains the flux divergences $\partial (u^2)/\partial x$,
$\partial (u v)/\partial y$ and $\partial (u w)/\partial z$.

**Stage 2 — the barotropic solve.** The barotropic forcing is the *depth integral of this
same tendency*,

$$
\mathbf{G}^{n} = \int\limits_{-h}^{\eta} \mathbf{R}^{n} \, dz ,
\tag{85}
$$

which is the identity that makes the two-dimensional subsystem the depth integral of the
three-dimensional one rather than an independently posed pair. Solve the Helmholtz
equation for $\eta^{n+1}$ and recover the transport by back-substitution,

$$
\left(d\mathbf{U}\right)^{n+1} = \left(d\mathbf{U}\right)^{n}
- g \Delta t\, d \left[\theta \nabla_h \eta^{n+1} + (1-\theta)\nabla_h \eta^{n}\right]
+ \Delta t\, \mathbf{G}^{n} .
\tag{86}
$$

**Stage 3 — advance $u$ and $v$.** With $\eta^{n+1}$ known, the equation solved for the
three-dimensional velocity is

$$
u^{n+1}
- \Delta t \frac{\partial}{\partial z}\left(\nu_v \frac{\partial u^{n+1}}{\partial z}\right)
- \Delta t\, f\, v^{n+1}
= u^{n} + \Delta t \left(R_x^{n} - g \frac{\partial \eta^{n+1}}{\partial x}\right),
\tag{87}
$$

and correspondingly for $v$. Two couplings are implicit here — vertical mixing couples $u$
across levels within a column, and Coriolis couples $u$ to $v$ at a point — but **neither
couples horizontally**. In practice they are applied by splitting rather than as one block
system,

$$
u^{*} = u^{n} + \Delta t\, R_x^{n}
\;\longrightarrow\;
u^{**} = u^{*} - g \Delta t \frac{\partial \eta^{n+1}}{\partial x}
\;\longrightarrow\;
\text{Coriolis}
\;\longrightarrow\;
\text{vertical diffusion}
\;=\; u^{n+1} .
\tag{88}
$$

Whichever treatment is chosen for the Coriolis and vertical mixing terms, the same
treatment must be mirrored in $\mathbf{G}^{n}$ for the consistency identity above to remain
exact.

**The vertical diffusion solve** is a tridiagonal system per water column. For level $k$,

$$
-a_k u_{k-1}^{n+1} + \left(1 + a_k + b_k\right) u_k^{n+1} - b_k u_{k+1}^{n+1} = u_k^{**},
\qquad
a_k = \frac{\Delta t\, \nu_{v, k-1/2}}{\Delta z_k \, \Delta z_{k-1/2}},
\tag{89}
$$

with $b_k$ defined symmetrically upwards. The stress boundary conditions become the end
rows: the wind stress $\tau_s / \rho_0$ enters the right-hand side of the top row and the
bottom drag $\tau_b / \rho_0$ that of the last. Columns are independent, so the cost is
$O(N_z)$ per column with no communication, and the restriction
$\Delta t \le \Delta z^{2} / (2 \nu_v)$ — otherwise severe for thin near-surface layers —
disappears.

**Why Coriolis must be treated implicitly.** The purpose of the implicit free surface is to
permit a large $\Delta t$, but then $f \Delta t \approx 10^{-4} \times 3600 \approx 0.36$ is
not small, and forward Euler applied to pure rotation has amplification factor
$|1 + i f \Delta t| > 1$ — unconditionally unstable. Coriolis is therefore taken
semi-implicitly, which for the $2 \times 2$ rotation block inverts analytically and needs no
solver,

$$
\begin{bmatrix} 1 & -\frac{f \Delta t}{2} \\[4pt] \frac{f \Delta t}{2} & 1 \end{bmatrix}
\begin{bmatrix} u^{n+1} \\[4pt] v^{n+1} \end{bmatrix}
=
\begin{bmatrix} u^{**} + \frac{f \Delta t}{2} v^{n} \\[4pt] v^{**} - \frac{f \Delta t}{2} u^{n} \end{bmatrix} .
\tag{90}
$$

Removing the barotropic CFL exposes stability limits that were previously hidden beneath
it, and rotation is the first one encountered.

**Stage 4 — the remaining fields.** Advance the tracers, then update $\rho$, $p$ and $w$
from their diagnostics.

### 8.5 What still limits the time step

Vertical advection $\partial (u w) / \partial z$ stays inside $\mathbf{R}^{n}$, so
$w \Delta t / \Delta z \lesssim 1$ survives — and with thin layers in terrain-following
coordinates it is often the binding constraint once the barotropic wave is gone. Horizontal
advection likewise leaves $|\mathbf{u}| \Delta t / \Delta x \lesssim 1$. The implicit free
surface removes the *fastest* restriction, not all of them: the realistic gain is the ratio
between $\sqrt{g d}$ and $\max\left(|\mathbf{u}|, c_{\text{int}}\right)$, one to two orders
of magnitude rather than an unbounded time step.

### 8.6 Relation to mode splitting

The alternative is to keep the barotropic subsystem explicit and subcycle it $N$ times per
three-dimensional step, which is the split-explicit method. The two differ in more than
cost:

| | Split-explicit | Implicit free surface |
|---|---|---|
| Barotropic step | $N$ cheap explicit 2D steps | one 2D elliptic solve |
| Communication | halo exchange only, local | global, all-to-all |
| $\eta$ | prognostic, subcycled | prognostic, solved implicitly |
| Consistency | $\int \tilde{\mathbf{u}} \, dz = 0$ drifts and must be re-imposed each step | exact by construction |
| Requires the decomposition | yes | no |

The last two rows are the substantive difference. Because the implicit method never
advances a separate barotropic system, the depth integral of the three-dimensional
velocities cannot drift away from the two-dimensional solution, and the correction step
that split-explicit models apply every baroclinic step has no counterpart here. The
barotropic–baroclinic decomposition then remains useful for analysis but is no longer
required by the algorithm.

Note also that in a discontinuous Galerkin discretisation the operator $\mathcal{L}$ needs
an interior penalty or local DG formulation: the naive elementwise Laplacian is not
coercive across element faces.



