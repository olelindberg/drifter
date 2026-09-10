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

## 1. Coordinates systems

### 1.1 Earth-Centred Inertial Frame

A Cartesian frame $(x, y, z)$ with origin at the Earth's centre of mass and axes fixed
relative to the distant stars, $z$ along the rotation axis and $x$, $y$ in the equatorial
plane. It does not rotate with the Earth. The metric is trivial,

$$
ds^{2} = dx^{2} + dy^{2} + dz^{2},
\tag{1}
$$

so no metric terms arise. Being inertial, the frame carries no Coriolis or centrifugal terms,
and the only body force is the Newtonian attraction towards the centre of mass.

### 1.2 Earth-Centred Earth-Fixed Frame

A Cartesian frame $(x_e, y_e, z_e)$ sharing its origin and its $z$ axis with the inertial
frame, but rotating with the Earth: $x_e$ pierces the intersection of the equator and the
prime meridian, $y_e$ completes the right-handed set, and $z_e$ lies along the rotation axis.
The two frames differ by a rotation about $z$ through the angle $\Omega t$,

$$
\begin{bmatrix} x_e \\ y_e \\ z_e \end{bmatrix}
=
\begin{bmatrix}
\cos \Omega t & \sin \Omega t & 0 \\
-\sin \Omega t & \cos \Omega t & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix},
\qquad \Omega = 7.2921 \times 10^{-5} \; \text{rad/s} .
\tag{2}
$$

The metric is again trivial, $ds^{2} = dx_e^{2} + dy_e^{2} + dz_e^{2}$, so the frame
introduces no metric terms.

### 1.3 Spherical coordinates

Spherical coordinates $(r, \theta, \phi)$ on the Earth-fixed frame, in the physics (ISO)
convention: $r$ is the radial distance, $\theta \in [0, \pi]$ the **polar** angle measured
from the $z_e$ axis, and $\phi \in [0, 2\pi)$ the **azimuthal** angle measured in the
$x_e y_e$ plane from the $x_e$ axis. In this ordering the triad is right-handed,
$\hat{\mathbf{r}} \times \hat{\boldsymbol{\theta}} = \hat{\boldsymbol{\phi}}$. Note that
$\theta$ is a colatitude, not a latitude.

$$
x_e = r \sin\theta \cos\phi, \qquad
y_e = r \sin\theta \sin\phi, \qquad
z_e = r \cos\theta ,
\tag{3}
$$

with the inverse

$$
r = \sqrt{x_e^{2} + y_e^{2} + z_e^{2}}, \qquad
\theta = \arccos\!\left(\frac{z_e}{r}\right), \qquad
\phi = \operatorname{atan2}\left(y_e, \, x_e\right) .
\tag{4}
$$

Geographic longitude $\lambda$ and latitude $\varphi$ follow as

$$
\lambda = \phi, \qquad \varphi = \frac{\pi}{2} - \theta ,
\tag{5}
$$

so that $\sin\theta = \cos\varphi$ and $\cos\theta = \sin\varphi$: the equator is
$\theta = \pi/2$ and the north pole $\theta = 0$.

**Metric.** The Jacobian of the transformation is

$$
\mathsf{J} = \frac{\partial \left(x_e, y_e, z_e\right)}{\partial \left(r, \theta, \phi\right)}
=
\begin{bmatrix}
\dfrac{\partial x_e}{\partial r} & \dfrac{\partial x_e}{\partial \theta} & \dfrac{\partial x_e}{\partial \phi} \\[8pt]
\dfrac{\partial y_e}{\partial r} & \dfrac{\partial y_e}{\partial \theta} & \dfrac{\partial y_e}{\partial \phi} \\[8pt]
\dfrac{\partial z_e}{\partial r} & \dfrac{\partial z_e}{\partial \theta} & \dfrac{\partial z_e}{\partial \phi}
\end{bmatrix}
=
\begin{bmatrix}
\sin\theta \cos\phi & \;\;\, r \cos\theta \cos\phi & - r \sin\theta \sin\phi \\[4pt]
\sin\theta \sin\phi & \;\;\, r \cos\theta \sin\phi & \;\;\, r \sin\theta \cos\phi \\[4pt]
\cos\theta & - r \sin\theta & 0
\end{bmatrix} .
\tag{6}
$$

Its columns are the tangent vectors along the three coordinate directions. They are mutually
orthogonal, so the coordinates are orthogonal and the metric $\mathsf{J}^{T}\mathsf{J}$ is
diagonal. The scale factors are the column lengths,

$$
h_{r} = \left|\frac{\partial \mathbf{R}}{\partial r}\right| = 1,
\qquad
h_{\theta} = \left|\frac{\partial \mathbf{R}}{\partial \theta}\right| = r,
\qquad
h_{\phi} = \left|\frac{\partial \mathbf{R}}{\partial \phi}\right| = r \sin\theta ,
\tag{7}
$$

and their product is the Jacobian determinant,
$\det \mathsf{J} = h_{r} h_{\theta} h_{\phi} = r^{2}\sin\theta$, positive because the ordering
is right-handed. The line and volume elements are

$$
ds^{2} = dr^{2} + r^{2} \, d\theta^{2} + r^{2}\sin^{2}\!\theta \; d\phi^{2},
\qquad
dV = r^{2}\sin\theta \; dr \, d\theta \, d\phi .
\tag{8}
$$

The gradient and divergence follow from the scale factors as

$$
\nabla \psi = \left(
\frac{\partial \psi}{\partial r}, \;
\frac{1}{r}\frac{\partial \psi}{\partial \theta}, \;
\frac{1}{r \sin\theta}\frac{\partial \psi}{\partial \phi}
\right),
\tag{9}
$$

$$
\nabla \cdot \mathbf{F} =
\frac{1}{r^{2}}\frac{\partial \left(r^{2} F_r\right)}{\partial r}
+ \frac{1}{r \sin\theta}\frac{\partial \left(F_\theta \sin\theta\right)}{\partial \theta}
+ \frac{1}{r \sin\theta}\frac{\partial F_\phi}{\partial \phi} .
\tag{10}
$$


### 1.4 Rotated spherical coordinates

The same spherical coordinates, but on axes rotated so that the model domain straddles the
rotated equator $\theta' = \pi/2$. The coordinates are $(r, \theta', \phi')$ with $\theta'$
the rotated polar angle and $\phi'$ the rotated azimuth. Writing the polar angle of the
rotated pole as $\theta_p$ and its azimuth as $\phi_p$, the transformation is the spherical
law of cosines together with its companion for the azimuth,

$$
\theta' = \arccos\!\Big(
\cos \theta \, \cos \theta_p + \sin \theta \, \sin \theta_p \, \cos(\phi - \phi_p)
\Big),
\tag{11}
$$

$$
\phi' = \operatorname{atan2}\!\Big(
\sin \theta \, \sin(\phi - \phi_p), \;\;
\cos \theta_p \, \sin \theta \, \cos(\phi - \phi_p) - \sin \theta_p \, \cos \theta
\Big),
\tag{12}
$$

$$
r = a + z .
\tag{13}
$$

Equivalently, the rotation carrying the Earth-fixed axes onto the rotated ones is

$$
\mathsf{Q} =
\begin{bmatrix}
\cos\theta_p & 0 & -\sin\theta_p \\
0 & 1 & 0 \\
\sin\theta_p & 0 & \;\;\,\cos\theta_p
\end{bmatrix}
\begin{bmatrix}
\;\;\,\cos\phi_p & \sin\phi_p & 0 \\
-\sin\phi_p & \cos\phi_p & 0 \\
0 & 0 & 1
\end{bmatrix} .
\tag{14}
$$

**Metric.** Because $\mathsf{Q}$ is a rigid rotation, the position in the rotated Cartesian
axes $(x', y', z')$ is spherical in exactly the same way as before,

$$
x' = r \sin\theta' \cos\phi', \qquad
y' = r \sin\theta' \sin\phi', \qquad
z' = r \cos\theta' ,
\tag{15}
$$

with Jacobian

$$
\mathsf{J}' = \frac{\partial \left(x', y', z'\right)}{\partial \left(r, \theta', \phi'\right)}
=
\begin{bmatrix}
\sin\theta' \cos\phi' & \;\;\, r \cos\theta' \cos\phi' & - r \sin\theta' \sin\phi' \\[4pt]
\sin\theta' \sin\phi' & \;\;\, r \cos\theta' \sin\phi' & \;\;\, r \sin\theta' \cos\phi' \\[4pt]
\cos\theta' & - r \sin\theta' & 0
\end{bmatrix} .
\tag{16}
$$

Since $\mathsf{Q}^{T}\mathsf{Q} = \mathsf{I}$, the metric is unchanged by the rotation,
$\mathsf{J}^{T}\mathsf{J} = \mathsf{J}'^{T}\mathsf{J}'$, so the coordinates remain orthogonal
and every metric quantity keeps its unrotated form with $\theta, \phi$ replaced by
$\theta', \phi'$. The scale factors are again the column lengths,

$$
h_{r} = 1, \qquad h_{\theta'} = r, \qquad h_{\phi'} = r \sin\theta' ,
\tag{17}
$$

with determinant $\det \mathsf{J}' = h_{r} h_{\theta'} h_{\phi'} = r^{2}\sin\theta'$, giving

$$
ds^{2} = dr^{2} + r^{2} \, d\theta'^{2} + r^{2}\sin^{2}\!\theta' \; d\phi'^{2},
\qquad
dV = r^{2}\sin\theta' \; dr \, d\theta' \, d\phi' .
\tag{18}
$$


### 1.5 Vertical

The vertical coordinate $z$ points **upwards** along the local gravity direction. The water
column is bounded below by the bed and above by the free surface,

$$
z = -h(x, y) \quad \text{(bed)}, \qquad z = \eta(x, y, t) \quad \text{(free surface)},
\tag{19}
$$

with $z = 0$ at the undisturbed surface, so that $\eta$ measures the departure from rest
and the total water depth is $d = \eta + h$.

Surfaces of constant $z$ are geopotential surfaces — level surfaces of the combined
gravitational and centrifugal field — not surfaces of constant geometric height. Gravity is
therefore perpendicular to them by construction, which is what allows the horizontal metric
to be treated as spherical (the *spherical geopotential approximation*) with a constant mean
radius $a = 6371$ km. It is also why $\eta$ is $O(1)$ m rather than the $O(100)$ m that
separates the geoid from a reference ellipsoid.

Gravity acts downwards and $g > 0$ denotes its magnitude. It therefore enters the vertical
momentum equation (38) as $-g$, and hydrostatic balance reads
$\partial p / \partial z = -\rho g$. This convention is used throughout the document.

**Bed.** In the rotated spherical coordinates of the previous section the bathymetry is
$h(\theta', \phi')$ and the bed is the fixed, impermeable surface $r = r_b$ with
$r_b = a - h$. A fluid particle on it remains on it,

$$
\frac{D}{Dt}\left(r - a + h(\theta', \phi')\right) = 0
\qquad \text{at} \quad r = r_b ,
\tag{20}
$$

which, with
$D / Dt = \partial_t + w\,\partial_r + \left(v / r\right) \partial_{\theta'}
+ \left(u / r \sin\theta'\right) \partial_{\phi'}$
and $h$ independent of time, expands to

$$
w_b
+ \frac{v_b}{r_b}\frac{\partial h}{\partial \theta'}
+ \frac{u_b}{r_b \sin\theta'}\frac{\partial h}{\partial \phi'} = 0 .
\tag{21}
$$

Here $u$, $v$ and $w$ are the azimuthal, polar and radial velocity components on the rotated
triad, resolved as in §4 below, and the subscript $b$ denotes evaluation at $r = r_b$.

**Free surface.** The free surface is the material surface $r = r_s$ with
$r_s = a + \eta(\theta', \phi', t)$, across which no fluid passes,

$$
\frac{D}{Dt}\left(r - a - \eta(\theta', \phi', t)\right) = 0
\qquad \text{at} \quad r = r_s ,
\tag{22}
$$

which expands to

$$
w_s = \frac{\partial \eta}{\partial t}
+ \frac{v_s}{r_s}\frac{\partial \eta}{\partial \theta'}
+ \frac{u_s}{r_s \sin\theta'}\frac{\partial \eta}{\partial \phi'} ,
\tag{23}
$$

the subscript $s$ denoting evaluation at $r = r_s$. Unlike the bed, this surface moves, so
the time derivative survives.


## 2. Navier–Stokes Equations in Earth-Centred Inertial Frame

This section is written in an Earth-Centred Inertial (ECI) frame, which does **not** rotate
with the Earth: its axes are fixed relative to the distant stars. In such a frame no Coriolis
or centrifugal terms exist. They are not physical forces but inertial terms, and they arise
only when the equations are referred to a rotating frame.


The ECI frame is a plain Cartesian frame $(x, y, z)$ with its origin at the Earth's centre of
mass and axes fixed relative to the distant stars: $z$ along the rotation axis, $x$ and $y$
in the equatorial plane. Write $\mathbf{r} = (x, y, z)$ for position,
$r = |\mathbf{r}| = \sqrt{x^{2} + y^{2} + z^{2}}$, and $\mathbf{u} = (u, v, w)$ for the
velocity measured in this frame.

Within this section only, $x$, $y$, $z$ and $u$, $v$, $w$ refer to the inertial frame. From
the next section onwards the same symbols denote the local frame of §1,
with $z$ vertical.

### 2.1 Equations of motion

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0,
\tag{24}
$$

$$
\frac{\partial u}{\partial t}
+ \frac{\partial}{\partial x}\left(u^{2} + \frac{p}{\rho}\right)
+ \frac{\partial}{\partial y}\left(u v\right)
+ \frac{\partial}{\partial z}\left(u w\right)
= g_{*x} + \mathcal{F}_x,
\tag{25}
$$

$$
\frac{\partial v}{\partial t}
+ \frac{\partial}{\partial x}\left(u v\right)
+ \frac{\partial}{\partial y}\left(v^{2} + \frac{p}{\rho}\right)
+ \frac{\partial}{\partial z}\left(v w\right)
= g_{*y} + \mathcal{F}_y,
\tag{26}
$$

$$
\frac{\partial w}{\partial t}
+ \frac{\partial}{\partial x}\left(u w\right)
+ \frac{\partial}{\partial y}\left(v w\right)
+ \frac{\partial}{\partial z}\left(w^{2} + \frac{p}{\rho}\right)
= g_{*z} + \mathcal{F}_z .
\tag{27}
$$

The inertial frame is where this form is most natural. Both source terms are genuine: the
gravitational term is a true body force, and $\boldsymbol{\mathcal{F}}$ is a stress
divergence that can itself be absorbed into $\mathsf{F}$. No fictitious force appears, so
away from boundaries the flux divergence alone governs the momentum budget and a
discretisation telescopes to exact global momentum conservation. Referring the equations to
a rotating frame breaks this: the Coriolis term is not a flux divergence and cannot be
written as one.

### 2.2 Gravitation

The body force is the **true Newtonian gravitational attraction**, not the effective gravity
of the rotating frame,

$$
\mathbf{g}_{*} = -\nabla \Phi_g, \qquad \Phi_g = -\frac{G M}{r},
\tag{28}
$$

so that $\mathbf{g}_{*} = -\left(G M / r^{3}\right)\mathbf{r}$ for a spherically symmetric
Earth. It points at the centre of mass. It is therefore **not** perpendicular to the geoid,
and the constant $g \approx 9.81$ m/s² of the following sections is not this quantity. A
real Earth adds the oblateness terms of the geopotential expansion, of which $J_2$ dominates.


## 3. Navier–Stokes Equations in Earth-Centred Earth-Fixed Frame

The Earth-fixed frame shares its origin with the inertial one, so position is common, but it
rotates at $\boldsymbol{\Omega} = \Omega\hat{\mathbf{z}}$. Velocities therefore differ by the
frame velocity, and accelerations by two further terms,

$$
\mathbf{u}_I = \mathbf{u} + \boldsymbol{\Omega}\times\mathbf{r},
\qquad
\left.\frac{D\mathbf{u}_I}{Dt}\right|_{\text{inertial}}
= \frac{D\mathbf{u}}{Dt}
+ \underbrace{2\,\boldsymbol{\Omega}\times\mathbf{u}}_{\text{Coriolis}}
+ \underbrace{\boldsymbol{\Omega}\times\left(\boldsymbol{\Omega}\times\mathbf{r}\right)}_{\text{centrifugal}} ,
\tag{29}
$$

where $\mathbf{u} = (u, v, w)$ is now the velocity relative to the rotating frame. Since
$\nabla\cdot(\boldsymbol{\Omega}\times\mathbf{r}) = 0$, continuity is unchanged.

**Effective gravity.** The centrifugal term depends on position alone, so it is a gradient
and is absorbed into the gravitational potential rather than carried as a separate term,

$$
\mathbf{g} = \mathbf{g}_{*} - \boldsymbol{\Omega}\times\left(\boldsymbol{\Omega}\times\mathbf{r}\right)
= -\nabla\Phi,
\qquad
\Phi = -\frac{G M}{r} - \tfrac{1}{2}\Omega^{2}\left(x^{2} + y^{2}\right),
\tag{30}
$$

using $\boldsymbol{\Omega}\times(\boldsymbol{\Omega}\times\mathbf{r}) = -\Omega^{2}(x, y, 0)$
for $\boldsymbol{\Omega}$ along $z$. This $\mathbf{g}$ is the effective gravity, of magnitude
$g \approx 9.81$ m/s², used from here on. Surfaces of constant $\Phi$ are the geopotential
surfaces of *Coordinates systems*, and the centrifugal contribution is what makes them oblate
rather than spherical.

With the centrifugal term absorbed and
$2\boldsymbol{\Omega}\times\mathbf{u} = 2\Omega\left(-v, \, u, \, 0\right)$, the equations in
conservation form are

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0,
\tag{31}
$$

$$
\frac{\partial u}{\partial t}
+ \frac{\partial}{\partial x}\left(u^{2} + \frac{p}{\rho}\right)
+ \frac{\partial}{\partial y}\left(u v\right)
+ \frac{\partial}{\partial z}\left(u w\right)
- 2\Omega v
= g_{x} + \mathcal{F}_x,
\tag{32}
$$

$$
\frac{\partial v}{\partial t}
+ \frac{\partial}{\partial x}\left(u v\right)
+ \frac{\partial}{\partial y}\left(v^{2} + \frac{p}{\rho}\right)
+ \frac{\partial}{\partial z}\left(v w\right)
+ 2\Omega u
= g_{y} + \mathcal{F}_y,
\tag{33}
$$

$$
\frac{\partial w}{\partial t}
+ \frac{\partial}{\partial x}\left(u w\right)
+ \frac{\partial}{\partial y}\left(v w\right)
+ \frac{\partial}{\partial z}\left(w^{2} + \frac{p}{\rho}\right)
= g_{z} + \mathcal{F}_z .
\tag{34}
$$

The frame is still Cartesian, so the metric remains trivial and no metric terms appear. The
flux divergences are unchanged from the inertial form, and the only visible new term is the
Coriolis one.

The Coriolis term admits no treatment like the centrifugal one: it depends on velocity, is not
a gradient, and is not a flux divergence. It is the one term that breaks the pure conservation
form of the inertial equations, and it survives into every frame built on this one.

## 4. Navier–Stokes Equations in Rotated Spherical Coordinates

The frame does not change here: these are still the Earth-fixed equations of the previous
section, so the Coriolis term and the effective gravity carry over unchanged. Only the
*coordinates* change, from the Cartesian $(x_e, y_e, z_e)$ to the rotated spherical
$(r, \theta', \phi')$ of *Coordinates systems*. Because those coordinates have
position-dependent scale factors, two things happen: the derivatives acquire the factors
$h_{\theta'}^{-1}$ and $h_{\phi'}^{-1}$ and the divergences acquire the weight
$\det\mathsf{J}' = r^{2}\sin\theta'$, and the turning of the basis vectors generates metric
terms that had no counterpart in the Cartesian frame. The Coriolis term is also resolved on
the new axes, which splits its horizontal part in two.

The velocity components are taken along the coordinate unit vectors in their right-handed
order,
$\mathbf{u} = w\,\hat{\mathbf{r}} + v\,\hat{\boldsymbol{\theta}}' + u\,\hat{\boldsymbol{\phi}}'$,
so $u$ is azimuthal (eastward on an unrotated grid), $v$ is along increasing polar angle and
therefore positive equatorward, and $w$ is radial. In conservation form,

$$
\frac{1}{r^{2}}\frac{\partial \left(r^{2} w\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial u}{\partial \phi'} = 0,
\tag{35}
$$

$$
\frac{\partial u}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} u w\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u v \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u u\right)}{\partial \phi'}
+ \frac{u w}{r}
+ \frac{u v \cot\theta'}{r}
+ f v - \tilde{f}_{\theta'} w
+ \frac{1}{\rho \, r \sin\theta'}\frac{\partial p}{\partial \phi'}
= \mathcal{F}_{\phi'},
\tag{36}
$$

$$
\frac{\partial v}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} v w\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v v \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u v\right)}{\partial \phi'}
+ \frac{v w}{r}
- \frac{u^{2} \cot\theta'}{r}
- f u + \tilde{f}_{\phi'} w
+ \frac{1}{\rho \, r}\frac{\partial p}{\partial \theta'}
= \mathcal{F}_{\theta'},
\tag{37}
$$

$$
\frac{\partial w}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} w w\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v w \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u w\right)}{\partial \phi'}
- \frac{u^{2} + v^{2}}{r}
+ \tilde{f}_{\theta'} u - \tilde{f}_{\phi'} v
+ \frac{1}{\rho}\frac{\partial p}{\partial r}
= -g + \mathcal{F}_r .
\tag{38}
$$

Here $p$ is the pressure, $\rho$ the density and $g$ the magnitude of the **effective**
gravity of the previous section, which already carries the centrifugal contribution. The
radial coordinate is

$$
r = a + z, \qquad a = 6371 \; \text{km},
\tag{39}
$$

$z$ being the height above the undisturbed surface used elsewhere in this document; the two
differ by a constant, so $\partial / \partial r = \partial / \partial z$ and the free surface
and bed sit at $r = a + \eta$ and $r = a - h$. No approximation has been made: these are the
complete equations, and every term is retained.


**Rotation.** The Coriolis terms are the components of
$2\boldsymbol{\Omega} \times \mathbf{u}$ resolved on the rotated axes, evaluated in the
right-handed triad $(\hat{\mathbf{r}}, \hat{\boldsymbol{\theta}}', \hat{\boldsymbol{\phi}}')$.
Each parameter is twice a projection of $\boldsymbol{\Omega}$,

$$
f = 2\,\boldsymbol{\Omega}\cdot\hat{\mathbf{r}} = 2\Omega\cos\theta,
\qquad
\tilde{f}_{\theta'} = 2\,\boldsymbol{\Omega}\cdot\hat{\boldsymbol{\theta}}' = -2\Omega\sin\theta\,\cos\alpha,
\qquad
\tilde{f}_{\phi'} = 2\,\boldsymbol{\Omega}\cdot\hat{\boldsymbol{\phi}}' = 2\Omega\sin\theta\,\sin\alpha,
\tag{40}
$$

with $\Omega = 7.2921 \times 10^{-5}$ rad/s and $\alpha$ the rotation about $\hat{\mathbf{r}}$
carrying the geographic $\hat{\boldsymbol{\phi}}$ onto $\hat{\boldsymbol{\phi}}'$. The
vertical projection is unaffected by a rotation about the vertical, so $f$ is evaluated at the
**true** polar angle $\theta$, never at $\theta'$; in latitude terms $f = 2\Omega\sin\varphi$.
The horizontal part of $\boldsymbol{\Omega}$ points towards the geographic pole and so
projects onto **both** rotated axes; on an unrotated grid $\alpha = 0$, leaving
$\tilde{f}_{\theta'} = -2\Omega\sin\theta$ and $\tilde{f}_{\phi'} = 0$. The Coriolis force is
perpendicular to the flow and does no work: it redistributes momentum between the components
without changing the kinetic energy.


**Forcing.** The terms $\boldsymbol{\mathcal{F}} = (\mathcal{F}_r, \mathcal{F}_{\theta'}, \mathcal{F}_{\phi'})$
are the divergence of the turbulent stress, carrying the internal mixing together with the
surface wind stress and bottom drag applied at the boundaries. They are written out in §7.5
below.

To keep the derivations that follow readable, the next several sections set
$\boldsymbol{\mathcal{F}} = 0$, $\tilde{f} = 0$ and $f = 0$ and reinstate rotation and
forcing once the closed system is assembled.

### 4.1 Vertical velocity

Continuity (35) fixes the radial velocity, given the horizontal components and the bed
condition (21). Multiplying it by $r^{2}$ and integrating in the radial direction from the
bed $r_b = a - h$ to a level $r$ gives

$$
r^{2} w - r_b^{2} w_b = - \frac{1}{\sin\theta'} \int\limits_{r_b}^{r} r'
\left[\frac{\partial \left(v \sin\theta'\right)}{\partial \theta'}
+ \frac{\partial u}{\partial \phi'}\right] dr' ,
\tag{41}
$$

where $r'$ is the radial integration variable. The lower limit depends on $\theta'$ and
$\phi'$, so taking the horizontal derivatives outside the integral by Leibniz' rule leaves a
boundary term at the bed in each,

$$
\int\limits_{r_b}^{r} r' \frac{\partial \left(v \sin\theta'\right)}{\partial \theta'} \, dr'
= \frac{\partial}{\partial \theta'} \int\limits_{r_b}^{r} r' v \sin\theta' \, dr'
- r_b v_b \sin\theta' \frac{\partial h}{\partial \theta'},
\tag{42}
$$

$$
\int\limits_{r_b}^{r} r' \frac{\partial u}{\partial \phi'} \, dr'
= \frac{\partial}{\partial \phi'} \int\limits_{r_b}^{r} r' u \, dr'
- r_b u_b \frac{\partial h}{\partial \phi'} ,
\tag{43}
$$

using $\partial r_b / \partial \theta' = -\partial h / \partial \theta'$ and
$\partial r_b / \partial \phi' = -\partial h / \partial \phi'$. By the bed condition (21)
the two boundary terms contribute $-r_b^{2} w_b$, which cancels the bed term on the left,
leaving the diagnostic

$$
w\left(r, \theta', \phi', t\right) = - \frac{1}{r^{2} \sin\theta'}\left[
\frac{\partial}{\partial \theta'} \int\limits_{r_b}^{r} r' v \sin\theta' \, dr'
+ \frac{\partial}{\partial \phi'} \int\limits_{r_b}^{r} r' u \, dr'
\right] ,
\tag{44}
$$

for every level $r \in \left[a - h, \; a + \eta\right]$.

### 4.2 Depth-integrated continuity

Integrating (35) over the whole water column instead, from the bed $r_b = a - h$ to the free
surface $r_s = a + \eta$, gives

$$
r_s^{2} w_s - r_b^{2} w_b = - \frac{1}{\sin\theta'} \int\limits_{r_b}^{r_s} r'
\left[\frac{\partial \left(v \sin\theta'\right)}{\partial \theta'}
+ \frac{\partial u}{\partial \phi'}\right] dr' .
\tag{45}
$$

Both limits now depend on $\theta'$ and $\phi'$, so Leibniz' rule leaves a boundary term at
each,

$$
\int\limits_{r_b}^{r_s} r' \frac{\partial \left(v \sin\theta'\right)}{\partial \theta'} \, dr'
= \frac{\partial}{\partial \theta'} \int\limits_{r_b}^{r_s} r' v \sin\theta' \, dr'
- r_s v_s \sin\theta' \frac{\partial \eta}{\partial \theta'}
- r_b v_b \sin\theta' \frac{\partial h}{\partial \theta'},
\tag{46}
$$

$$
\int\limits_{r_b}^{r_s} r' \frac{\partial u}{\partial \phi'} \, dr'
= \frac{\partial}{\partial \phi'} \int\limits_{r_b}^{r_s} r' u \, dr'
- r_s u_s \frac{\partial \eta}{\partial \phi'}
- r_b u_b \frac{\partial h}{\partial \phi'} .
\tag{47}
$$

By the bed condition (21) the bed terms cancel $-r_b^{2} w_b$, and by the kinematic condition
(23) the surface terms cancel every part of $r_s^{2} w_s$ except $r_s^{2} \, \partial \eta / \partial t$,
leaving

$$
\frac{\partial \eta}{\partial t}
+ \frac{1}{r_s^{2} \sin\theta'}\left[
\frac{\partial}{\partial \theta'} \int\limits_{r_b}^{r_s} r' v \sin\theta' \, dr'
+ \frac{\partial}{\partial \phi'} \int\limits_{r_b}^{r_s} r' u \, dr'
\right] = 0 .
\tag{48}
$$

This is the prognostic equation for the free surface: the local rate of change of $\eta$
against the horizontal divergence of the depth-integrated volume transport.

### 4.3 Pressure from radial momentum

Rearranging (38) for the radial pressure gradient gives

$$
\frac{\partial p}{\partial r} = -\rho \left[
g
+ \frac{\partial w}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} w w\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v w \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u w\right)}{\partial \phi'}
- \frac{u^{2} + v^{2}}{r}
+ \tilde{f}_{\theta'} u - \tilde{f}_{\phi'} v
- \mathcal{F}_r \right],
\tag{49}
$$

the bracket holding the radial acceleration together with the metric, Coriolis and friction
terms.

Integrating (49) from an arbitrary level $r$ to the free surface $r_s = a + \eta$, where the
pressure equals the applied atmospheric pressure $p_a(\theta', \phi', t)$, gives

$$
p\left(r, \theta', \phi', t\right) = p_a\left(\theta', \phi', t\right)
+ \int\limits_{r}^{r_s} \rho \left[
g
+ \frac{\partial w}{\partial t}
+ \frac{1}{r'^{2}}\frac{\partial \left(r'^{2} w w\right)}{\partial r'}
+ \frac{1}{r'\sin\theta'}\frac{\partial \left(v w \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r'\sin\theta'}\frac{\partial \left(u w\right)}{\partial \phi'}
- \frac{u^{2} + v^{2}}{r'}
+ \tilde{f}_{\theta'} u - \tilde{f}_{\phi'} v
- \mathcal{F}_r \right] dr' ,
\tag{50}
$$

for every level $r \in \left[a - h, \; a + \eta\right]$, with $\rho$, $u$, $v$, $w$ and
$\mathcal{F}_r$ evaluated at $r'$. The integrand carries $\partial w / \partial t$, so this
fixes $p$ only jointly with the radial momentum equation (38) it came from.

**Linearization.** Taking $u$, $v$ and $w$ as first-order perturbations about a state of
rest and retaining $\rho$ in full, the three advective terms and the centrifugal term
$-\left(u^{2} + v^{2}\right)/r'$ of (50) are second order and are discarded, so that to
first order

$$
p\left(r, \theta', \phi', t\right) = p_a\left(\theta', \phi', t\right)
+ \int\limits_{r}^{r_s} \rho \left(
g + \frac{\partial w}{\partial t}
+ \tilde{f}_{\theta'} u - \tilde{f}_{\phi'} v - \mathcal{F}_r \right) dr' .
\tag{51}
$$

Neglecting in addition the vertical acceleration $\partial w / \partial t$ and the radial
friction $\mathcal{F}_r$ leaves the quasi-hydrostatic balance, in which the radial Coriolis
terms are the only departure from hydrostatic,

$$
p\left(r, \theta', \phi', t\right) = p_a\left(\theta', \phi', t\right)
+ \int\limits_{r}^{r_s} \rho \left(
g + \tilde{f}_{\theta'} u - \tilde{f}_{\phi'} v \right) dr' .
\tag{52}
$$

**Boussinesq split.** Splitting the density in (52) about a constant reference value,
$\rho = \rho_0 + \rho'(r, \theta', \phi', t)$ with $|\rho'| \ll \rho_0$, retaining $\rho'$
only where it multiplies $g$, and using $r_s - r = \eta - z$,

$$
p = p_a + \rho_0 \, g \left(r_s - r\right)
+ g \int\limits_{r}^{r_s} \rho' \, dr'
+ \rho_0 \int\limits_{r}^{r_s} \left(
\tilde{f}_{\theta'} u - \tilde{f}_{\phi'} v
\right) dr' .
\tag{53}
$$

**Barotropic and baroclinic pressure gradients.** Differentiating along the two horizontal
coordinates at fixed $r$, with Leibniz' rule applied to the upper limit
$r_s = a + \eta(\theta', \phi', t)$ and the surface term of the Coriolis integral discarded
as second order in the perturbation, gives

$$
\frac{\partial p}{\partial \theta'} =
\underbrace{\frac{\partial p_a}{\partial \theta'}}_{\text{atmospheric}}
+ \underbrace{\rho_0 \, g \frac{\partial \eta}{\partial \theta'}}_{\text{barotropic}}
+ \underbrace{g \int\limits_{r}^{r_s} \frac{\partial \rho'}{\partial \theta'} \, dr'}_{\text{baroclinic}}
+ \underbrace{\rho_0 \int\limits_{r}^{r_s} \frac{\partial}{\partial \theta'}
\left(\tilde{f}_{\theta'} u - \tilde{f}_{\phi'} v\right) dr'}_{\text{Coriolis}}
+ g \, \rho'_s \frac{\partial \eta}{\partial \theta'},
\tag{54}
$$

and correspondingly in $\phi'$,

$$
\frac{\partial p}{\partial \phi'} =
\frac{\partial p_a}{\partial \phi'}
+ \rho_0 \, g \frac{\partial \eta}{\partial \phi'}
+ g \int\limits_{r}^{r_s} \frac{\partial \rho'}{\partial \phi'} \, dr'
+ \rho_0 \int\limits_{r}^{r_s} \frac{\partial}{\partial \phi'}
\left(\tilde{f}_{\theta'} u - \tilde{f}_{\phi'} v\right) dr'
+ g \, \rho'_s \frac{\partial \eta}{\partial \phi'} ,
\tag{55}
$$

where $\rho'_s = \rho'\big|_{r = r_s}$. The baroclinic and Coriolis terms depend on $r$: both
vanish at the free surface and grow downwards. The $g \, \rho'_s$ term is $O(\rho'/\rho_0)$
relative to the barotropic one. The $r$-independence of the remaining terms belongs to
$\partial p / \partial \theta'$ and
$\partial p / \partial \phi'$ alone — in (37) and (36) these enter as
$\frac{1}{\rho \, r}\frac{\partial p}{\partial \theta'}$ and
$\frac{1}{\rho \, r \sin\theta'}\frac{\partial p}{\partial \phi'}$, whose metric factors vary
across the column by a relative $d / a$ with $d = \eta + h$, and are constant only in the
limit $r \to a$.



## 5. Tracer Equations

Let $C\left(r, \theta', \phi', t\right)$ be the mass of a constituent carried by the fluid per
unit mass of seawater, so that $\rho C$ is its mass per unit volume. The frame, the velocity
components $u$, $v$, $w$ and the metric are those of §4.

### 5.1 Conservation of a scalar

Neither heat nor salt is created in the interior, so $C$ carries no volume source and the
concentration $\rho C$ changes only by advection with the flow and by transport relative to
it,

$$
\frac{\partial \left(\rho C\right)}{\partial t}
+ \nabla \cdot \left(\rho C \, \mathbf{u}\right) = - \nabla \cdot \mathbf{q}_C ,
\tag{56}
$$

where $\mathbf{q}_C$ is the diffusive flux, that is the flux measured relative to the
barycentric motion.

**Boussinesq form.** With the density variations retained only in the buoyancy term,
continuity reduces to $\nabla \cdot \mathbf{u} = 0$ and $\rho$ is replaced by $\rho_0$ in
(56),

$$
\frac{\partial C}{\partial t} + \nabla \cdot \left(C \, \mathbf{u}\right) = \mathcal{D}_C ,
\qquad
\mathcal{D}_C = - \frac{1}{\rho_0} \nabla \cdot \mathbf{q}_C ,
\tag{57}
$$

at relative error $O\left(\rho'/\rho_0\right)$, the same order as the Boussinesq step of §4.3.

### 5.2 Rotated spherical coordinates

Resolving the divergence (10) on the rotated triad, with $u$ azimuthal, $v$ polar and $w$
radial as in §4, gives

$$
\frac{\partial C}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} w C\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v C \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u C\right)}{\partial \phi'}
= \mathcal{D}_C .
\tag{58}
$$

The metric factors are those of the continuity equation (35). A scalar has no components, so
unlike the momentum equations (36)–(38) this carries no metric or Coriolis terms.

### 5.3 Potential temperature and salinity

**Salinity.** $S$ is the mass of dissolved material per unit mass of seawater, reported in
g/kg. It is a constituent in the sense of (56) with no interior source.

**Potential temperature.** $T$ is the potential temperature: the temperature a parcel would
take if brought adiabatically, and without exchange of salt, to the reference pressure $p_r$.
The in-situ temperature does not satisfy (56) — adiabatic compression changes it with nothing
crossing the parcel boundary — whereas potential temperature removes that change by
construction. It obeys (58) up to the viscous dissipation and the pressure-work term, both
neglected here.

Both therefore take the form (58),

$$
\frac{\partial T}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} w T\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v T \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u T\right)}{\partial \phi'}
= \mathcal{D}_T ,
\tag{59}
$$

$$
\frac{\partial S}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} w S\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v S \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u S\right)}{\partial \phi'}
= \mathcal{D}_S .
\tag{60}
$$

$\mathcal{D}_T$ and $\mathcal{D}_S$ are the divergences of the turbulent tracer fluxes,
written out in §7.5. $T$ and $S$ re-enter the momentum equations only through the density,
by way of the equation of state of §6.

### 5.4 Boundary conditions

The bed and the free surface are material surfaces, so (21) and (23) already forbid advective
transport through them and only the diffusive flux is prescribed. At $r = r_s$,

$$
\rho_0 \, c_p \, \kappa_v \frac{\partial T}{\partial r}\bigg|_{r = r_s} = Q,
\qquad
\rho_0 \, \kappa_v \frac{\partial S}{\partial r}\bigg|_{r = r_s}
= \rho_0 \, S_s \left(E - P\right),
\tag{61}
$$

with $c_p$ the specific heat capacity at constant pressure, $\kappa_v$ the vertical eddy
diffusivity of §7.5, $Q$ the net surface heat flux counted positive into the ocean, $E$ and
$P$ the evaporation and precipitation rates as freshwater volume per unit area per unit time,
and $S_s = S\big|_{r = r_s}$. The salinity condition is a virtual salt flux: no salt crosses
the surface, and the flux stands for the dilution and concentration caused by a freshwater
volume that is never added to or removed from the column.

At the bed neither tracer passes,

$$
\kappa_v \frac{\partial T}{\partial r}\bigg|_{r = r_b} = 0,
\qquad
\kappa_v \frac{\partial S}{\partial r}\bigg|_{r = r_b} = 0 .
\tag{62}
$$

## 6. Equation of State

The momentum equations of §4 and the tracer equations of §5 close only once the density is
known from the state variables. The equation of state supplies it,

$$
\rho = \rho\left(T, S, p\right),
\tag{63}
$$

with $T$ the potential temperature and $S$ the salinity of §5 and $p$ the pressure of §4.3.
It is a thermodynamic relation and not a conservation law: it carries no derivatives and
introduces no unknown. Within this section $\alpha$ is the thermal expansion coefficient and
not the grid-rotation angle of §4; the redefinition ends with the section. The forms of (63)
in use are stated in
[sea_water_equation_of_state.md](sea_water_equation_of_state.md).

## 7. The Closed System

With the bathymetry $h(\theta', \phi')$ and the atmospheric pressure $p_a(\theta', \phi', t)$
prescribed, the equations of §4 to §6 close in the eight unknowns $u$, $v$, $w$, $p$, $\eta$,
$\rho$, $T$ and $S$. They are collected here in the rotated spherical frame in which they were
derived: nothing in §7.1 to §7.3 is new, and the equation cited after each heading is its
origin.

### 7.1 Prognostic equations

**Horizontal momentum**, (36) and (37):

$$
\frac{\partial u}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} u w\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u v \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u u\right)}{\partial \phi'}
+ \frac{u w}{r}
+ \frac{u v \cot\theta'}{r}
+ f v - \tilde{f}_{\theta'} w
+ \frac{1}{\rho \, r \sin\theta'}\frac{\partial p}{\partial \phi'}
= \mathcal{F}_{\phi'},
\tag{64}
$$

$$
\frac{\partial v}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} v w\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v v \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u v\right)}{\partial \phi'}
+ \frac{v w}{r}
- \frac{u^{2} \cot\theta'}{r}
- f u + \tilde{f}_{\phi'} w
+ \frac{1}{\rho \, r}\frac{\partial p}{\partial \theta'}
= \mathcal{F}_{\theta'} .
\tag{65}
$$

The Coriolis parameters $f$, $\tilde{f}_{\theta'}$ and $\tilde{f}_{\phi'}$ are those of (40),
and the horizontal pressure gradients are decomposed in (54) and (55).

**Free surface**, (48):

$$
\frac{\partial \eta}{\partial t}
+ \frac{1}{r_s^{2} \sin\theta'}\left[
\frac{\partial}{\partial \theta'} \int\limits_{r_b}^{r_s} r' v \sin\theta' \, dr'
+ \frac{\partial}{\partial \phi'} \int\limits_{r_b}^{r_s} r' u \, dr'
\right] = 0 .
\tag{66}
$$

**Tracers**, (59) and (60):

$$
\frac{\partial T}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} w T\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v T \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u T\right)}{\partial \phi'}
= \mathcal{D}_T ,
\tag{67}
$$

$$
\frac{\partial S}{\partial t}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} w S\right)}{\partial r}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(v S \sin\theta'\right)}{\partial \theta'}
+ \frac{1}{r\sin\theta'}\frac{\partial \left(u S\right)}{\partial \phi'}
= \mathcal{D}_S .
\tag{68}
$$

### 7.2 Diagnostic equations

**Radial velocity**, from continuity and the bed condition, (44):

$$
w\left(r, \theta', \phi', t\right) = - \frac{1}{r^{2} \sin\theta'}\left[
\frac{\partial}{\partial \theta'} \int\limits_{r_b}^{r} r' v \sin\theta' \, dr'
+ \frac{\partial}{\partial \phi'} \int\limits_{r_b}^{r} r' u \, dr'
\right] .
\tag{69}
$$

**Pressure**, the quasi-hydrostatic balance of §4.3 in its Boussinesq form, (53):

$$
p = p_a + \rho_0 \, g \left(r_s - r\right)
+ g \int\limits_{r}^{r_s} \rho' \, dr'
+ \rho_0 \int\limits_{r}^{r_s} \left(
\tilde{f}_{\theta'} u - \tilde{f}_{\phi'} v
\right) dr' .
\tag{70}
$$

**Density**, the equation of state (63):

$$
\rho = \rho\left(T, S, p\right), \qquad \rho' = \rho - \rho_0 .
\tag{71}
$$

### 7.3 Closure

Eight equations, eight unknowns. The radial momentum equation (38) is no longer prognostic:
§4.3 reduced it to the diagnostic (70), which demotes $p$, and continuity (35) demotes $w$ to
(69). The prognostic three-dimensional fields are $u$, $v$, $T$ and $S$, the one prognostic
two-dimensional field is $\eta$, and the stratification reaches the momentum equations only
through the baroclinic term of (70). The kinematic conditions (21) and (23) are built into
(69) and (66); the remaining boundary conditions are those of §5.4 and §7.5.


