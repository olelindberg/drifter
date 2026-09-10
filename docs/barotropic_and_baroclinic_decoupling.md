# Barotropic and Baroclinic Decoupling

The horizontal velocity is split into its depth average, the barotropic velocity, and the
departure from it, the baroclinic velocity, in the notation of Blumberg and Mellor (1987) and
Shchepetkin and McWilliams (2005). The equations are those of
[governing_equations.md](governing_equations.md) §4 to §7, written in the latitude–longitude
conventions of Vallis (2006, §2.2).

**Notation.** The frame is the rotated spherical frame of governing_equations.md §1.4, with
$\lambda$ the longitude and $\varphi$ the latitude on the rotated grid, $\lambda = \phi'$ and
$\varphi = \pi/2 - \theta'$. Within this document $\varphi$ is a latitude, not the azimuth
$\phi$ of governing_equations.md; on an unrotated grid it is the geographic latitude. The
vertical coordinate is $z = r - a$, the height above the undisturbed surface, with the bed at
$z = -h$, the free surface at $z = \eta$ and the total depth $d = \eta + h$. The velocity
components are $u$ eastward (along $\lambda$), $v$ northward (along $\varphi$) and $w$ upward;
$v$ is minus the polar component of governing_equations.md §4. $f$ is the Coriolis parameter
(40), $2\Omega\sin\varphi$ on an unrotated grid, and $\tilde{f}_{\lambda} = \tilde{f}_{\phi'}$
and $\tilde{f}_{\varphi} = -\tilde{f}_{\theta'}$ are twice the projections of
$\boldsymbol{\Omega}$ on the eastward and northward unit vectors, $0$ and
$2\Omega\cos\varphi$ on an unrotated grid. The mixing terms are
$\mathcal{F}_{\lambda} = \mathcal{F}_{\phi'}$ and $\mathcal{F}_{\varphi} = -\mathcal{F}_{\theta'}$,
$p_a$ is the atmospheric pressure, and $\rho' = \rho - \rho_0$ is the density anomaly.

## 1. Decomposition

At every $(\lambda, \varphi, t)$ the horizontal velocity components are written as the sum of
a barotropic and a baroclinic part,

$$
\begin{aligned}
u &= \bar{u} + u', \\
v &= \bar{v} + v' .
\end{aligned}
\tag{1}
$$

The barotropic velocity is the depth-integrated velocity divided by the total depth,

$$
\begin{align}
\bar{u}\left(\lambda, \varphi, t\right) &= \frac{1}{d} \int\limits_{-h}^{\eta} u \, dz,
\tag{2} \\
\bar{v}\left(\lambda, \varphi, t\right) &= \frac{1}{d} \int\limits_{-h}^{\eta} v \, dz,
\tag{3}
\end{align}
$$

and the baroclinic velocity is the remainder, $u' = u - \bar{u}$ and $v' = v - \bar{v}$. No
approximation is made: given $u$, $v$ and $\eta$, the two parts are defined uniquely. The
barotropic velocity is independent of $z$, and the baroclinic velocity carries no net
transport,

$$
\begin{align}
\int\limits_{-h}^{\eta} u' \, dz &= 0,
\tag{4} \\
\int\limits_{-h}^{\eta} v' \, dz &= 0 .
\tag{5}
\end{align}
$$

## 2. Horizontal momentum and pressure

The horizontal momentum equations (64) and (65) of
[governing_equations.md](governing_equations.md) are, with $r = a + z$,

$$
\frac{\partial u}{\partial t}
+ \frac{1}{r\cos\varphi}\frac{\partial \left(u u\right)}{\partial \lambda}
+ \frac{1}{r\cos^{2}\varphi}\frac{\partial \left(u v \cos^{2}\varphi\right)}{\partial \varphi}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} u w\right)}{\partial z}
- f v
+ \left(\frac{u}{r} + \tilde{f}_{\varphi}\right) w
+ \frac{1}{\rho \, r \cos\varphi}\frac{\partial p}{\partial \lambda}
= \mathcal{F}_{\lambda},
\tag{6}
$$

$$
\frac{\partial v}{\partial t}
+ \frac{1}{r\cos\varphi}\frac{\partial \left(u v\right)}{\partial \lambda}
+ \frac{1}{r\cos\varphi}\frac{\partial \left(v v \cos\varphi\right)}{\partial \varphi}
+ \frac{1}{r^{2}}\frac{\partial \left(r^{2} v w\right)}{\partial z}
+ \left(f + \frac{u \tan\varphi}{r}\right) u
+ \left(\frac{v}{r} - \tilde{f}_{\lambda}\right) w
+ \frac{1}{\rho \, r}\frac{\partial p}{\partial \varphi}
= \mathcal{F}_{\varphi} .
\tag{7}
$$

In each, the terms on the left are the local acceleration, the advection of momentum along
$\lambda$, $\varphi$ and $z$, the Coriolis terms and the pressure gradient, and the term on the
right is the mixing. The metric terms are collected with the advection and Coriolis terms:
$-u v \tan\varphi / r$ into the $\varphi$-advection of (6) through
$\left(r\cos^{2}\varphi\right)^{-1} \partial \left(u v \cos^{2}\varphi\right) / \partial \varphi
= \left(r\cos\varphi\right)^{-1} \partial \left(u v \cos\varphi\right) / \partial \varphi -
u v \tan\varphi / r$, $u^{2} \tan\varphi / r$ into the Coriolis term of (7) as in Vallis
(2006, (2.75)), and $u w / r$ and $v w / r$ into the $\tilde{f}$ terms. The pressure is (70),

$$
p = p_a + \rho_0 \, g \left(\eta - z\right)
+ g \int\limits_{z}^{\eta} \rho' \, dz'
- \rho_0 \int\limits_{z}^{\eta} \left(
\tilde{f}_{\varphi} u - \tilde{f}_{\lambda} v
\right) dz' .
\tag{8}
$$

Inserting

$$
\begin{aligned}
u &= \bar{u} + u', \\
v &= \bar{v} + v'
\end{aligned}
\tag{9}
$$

into (8) splits the pressure as $p = \hat{p} + p'$ with

$$
\begin{align}
\hat{p} &= p_a + \rho_0 \, g \left(\eta - z\right)
- \rho_0 \left(\eta - z\right) \left(
\tilde{f}_{\varphi} \bar{u} - \tilde{f}_{\lambda} \bar{v}
\right),
\tag{10} \\
p' &= g \int\limits_{z}^{\eta} \rho' \, dz'
- \rho_0 \int\limits_{z}^{\eta} \left(
\tilde{f}_{\varphi} u' - \tilde{f}_{\lambda} v'
\right) dz' .
\tag{11}
\end{align}
$$

The vertical velocity is the diagnostic (69), which is linear in $u$ and $v$, so it splits as
$w = \hat{w} + w'$ with

$$
\begin{align}
\hat{w} &= - \frac{1}{r^{2} \cos\varphi}\left[
\frac{\partial}{\partial \lambda} \int\limits_{-h}^{z} \left(a + z'\right) \bar{u} \, dz'
+ \frac{\partial}{\partial \varphi} \int\limits_{-h}^{z} \left(a + z'\right) \bar{v} \cos\varphi \, dz'
\right],
\tag{12} \\
w' &= - \frac{1}{r^{2} \cos\varphi}\left[
\frac{\partial}{\partial \lambda} \int\limits_{-h}^{z} \left(a + z'\right) u' \, dz'
+ \frac{\partial}{\partial \varphi} \int\limits_{-h}^{z} \left(a + z'\right) v' \cos\varphi \, dz'
\right].
\tag{13}
\end{align}
$$

$\hat{p}$ and $\hat{w}$ are the parts carried by the barotropic velocity; unlike $\bar{u}$ and
$\bar{v}$ they are not depth averages, and they vary with $z$.

The barotropic equations are the depth integrals of (6) and (7). The baroclinic equations are
(6) and (7) with the barotropic tendency replaced through the barotropic equations, as in Lan
et al. (2022). No term is neglected in
the baroclinic equations; the barotropic equations are taken in the thin-shell limit of §3.

## 3. Barotropic equations

Integrating (6) from $z = -h$ to $z = \eta$, Leibniz' rule gives
$\int_{-h}^{\eta} \partial u / \partial t \, dz
= \partial \left(d \bar{u}\right) / \partial t - u_s \, \partial \eta / \partial t$
for the local acceleration, with $h$ independent of $t$, and a boundary term at each limit
for the two horizontal advection terms. The vertical advection is written
$r^{-2} \, \partial \left(r^{2} u w\right) / \partial z
= \partial \left(u w\right) / \partial z + 2 u w / r$, whose first part integrates to
$u_s w_s - u_b w_b$. The subscripts $s$ and $b$
denote evaluation at $z = \eta$ and $z = -h$. The boundary terms at the free surface combine
into $u_s$ times the kinematic condition (23) of
[governing_equations.md](governing_equations.md), and those at the bed into $u_b$ times (21)
of that document, so both vanish; likewise for (7) with $v_s$ and $v_b$.

$\bar{u}$, $\bar{v}$, $f$, $\tilde{f}_{\lambda}$, $\tilde{f}_{\varphi}$, $p_a$ and $\eta$ are
independent of $z$ and are taken outside the integrals, and
$\int_{-h}^{\eta} f v \, dz = f d \bar{v}$ by (3). The pressure is (10) and (11),
differentiated at fixed $z$.

**Thin shell.** The depth integrals carry the weight $1/r = 1/\left(a + z\right)$ of the
spherical metric and the weight $1/\rho$ of the pressure gradient. They are evaluated with $r$
replaced by the Earth radius $a$ and $\rho$ by $\rho_0$, which neglects terms of relative order
$d/a$ and $\rho'/\rho_0$, so that
$\int_{-h}^{\eta} dz / r = \ln\left[\left(a + \eta\right) / \left(a - h\right)\right] \to d / a$,
$\int_{-h}^{\eta} dz / \left(\rho \, r\right) \to d / \left(\rho_0 \, a\right)$ and
$\int_{-h}^{\eta} \left(\eta - z\right) / \left(\rho \, r\right) dz
\to d^{2} / \left(2 \rho_0 \, a\right)$. The mixed terms linear in $u'$ or $v'$ then vanish by
(4) and (5). The terms containing only barotropic quantities are on the left and the rest on
the right,

$$
\begin{aligned}
& \frac{\partial \left(d \bar{u}\right)}{\partial t}
+ \frac{1}{a \cos\varphi}\frac{\partial \left(d \bar{u} \bar{u}\right)}{\partial \lambda}
+ \frac{1}{a \cos^{2}\varphi}\frac{\partial \left(d \bar{u} \bar{v} \cos^{2}\varphi\right)}{\partial \varphi}
- f d \bar{v}
+ \left(\frac{3 \bar{u}}{a} + \tilde{f}_{\varphi}\right) \int\limits_{-h}^{\eta} \hat{w} \, dz \\
& + \frac{d}{a \cos\varphi}\left[
\frac{1}{\rho_0}\frac{\partial p_a}{\partial \lambda}
+ \left(g - \tilde{f}_{\varphi} \bar{u} + \tilde{f}_{\lambda} \bar{v}\right)
\frac{\partial \eta}{\partial \lambda}
- \frac{d}{2}\frac{\partial}{\partial \lambda}\left(
\tilde{f}_{\varphi} \bar{u} - \tilde{f}_{\lambda} \bar{v}
\right)\right] \\
& \quad = \int\limits_{-h}^{\eta} \mathcal{F}_{\lambda} \, dz
- \frac{1}{a \cos\varphi}\frac{\partial}{\partial \lambda}
\int\limits_{-h}^{\eta} u' u' \, dz
- \frac{1}{a \cos^{2}\varphi}\frac{\partial}{\partial \varphi}\left(
\cos^{2}\varphi \int\limits_{-h}^{\eta} u' v' \, dz\right) \\
& \qquad - \left(\frac{3 \bar{u}}{a} + \tilde{f}_{\varphi}\right) \int\limits_{-h}^{\eta} w' \, dz
- \frac{3}{a} \int\limits_{-h}^{\eta} u' w \, dz
- \frac{1}{\rho_0 \, a \cos\varphi} \int\limits_{-h}^{\eta}
\frac{\partial p'}{\partial \lambda} \, dz ,
\end{aligned}
\tag{14}
$$

$$
\begin{aligned}
& \frac{\partial \left(d \bar{v}\right)}{\partial t}
+ \frac{1}{a \cos\varphi}\frac{\partial \left(d \bar{u} \bar{v}\right)}{\partial \lambda}
+ \frac{1}{a \cos\varphi}\frac{\partial \left(d \bar{v} \bar{v} \cos\varphi\right)}{\partial \varphi}
+ \left(f + \frac{\bar{u} \tan\varphi}{a}\right) d \bar{u}
+ \left(\frac{3 \bar{v}}{a} - \tilde{f}_{\lambda}\right) \int\limits_{-h}^{\eta} \hat{w} \, dz \\
& + \frac{d}{a}\left[
\frac{1}{\rho_0}\frac{\partial p_a}{\partial \varphi}
+ \left(g - \tilde{f}_{\varphi} \bar{u} + \tilde{f}_{\lambda} \bar{v}\right)
\frac{\partial \eta}{\partial \varphi}
- \frac{d}{2}\frac{\partial}{\partial \varphi}\left(
\tilde{f}_{\varphi} \bar{u} - \tilde{f}_{\lambda} \bar{v}
\right)\right] \\
& \quad = \int\limits_{-h}^{\eta} \mathcal{F}_{\varphi} \, dz
- \frac{1}{a \cos\varphi}\frac{\partial}{\partial \lambda}
\int\limits_{-h}^{\eta} u' v' \, dz
- \frac{1}{a \cos\varphi}\frac{\partial}{\partial \varphi}\left(
\cos\varphi \int\limits_{-h}^{\eta} v' v' \, dz\right)
- \frac{\tan\varphi}{a} \int\limits_{-h}^{\eta} u' u' \, dz \\
& \qquad - \left(\frac{3 \bar{v}}{a} - \tilde{f}_{\lambda}\right) \int\limits_{-h}^{\eta} w' \, dz
- \frac{3}{a} \int\limits_{-h}^{\eta} v' w \, dz
- \frac{1}{\rho_0 \, a} \int\limits_{-h}^{\eta}
\frac{\partial p'}{\partial \varphi} \, dz .
\end{aligned}
\tag{15}
$$

The metric terms are collected as in (6) and (7), and
$\int u' \hat{w} \, dz + \int u' w' \, dz = \int u' w \, dz$ by $w = \hat{w} + w'$, likewise for
$v'$.

The integrals of products of baroclinic velocities, $\int u' u' \, dz$, $\int u' v' \, dz$ and
$\int v' v' \, dz$, are the dispersion terms of Blumberg and Mellor (1987). The density part
of $\int \partial p' / \partial \lambda \, dz$ and $\int \partial p' / \partial \varphi \, dz$
is the depth-integrated baroclinic pressure gradient.

## 4. Baroclinic equations

Inserting (1) and
$\partial u / \partial t = \partial \bar{u} / \partial t + \partial u' / \partial t$ into (6),
and likewise into (7), with $\partial \bar{u} / \partial t$ and $\partial \bar{v} / \partial t$
from (14) and (15) through
$\partial \left(d \bar{u}\right) / \partial t
= d \, \partial \bar{u} / \partial t + \bar{u} \, \partial \eta / \partial t$, the terms
containing only baroclinic quantities are on the left. On the right are the terms of (14) and
(15) divided by $d$, the terms containing only barotropic quantities, the mixed barotropic and
baroclinic terms, and $\mathcal{F}_{\lambda}$ and $\mathcal{F}_{\varphi}$, which act on the full
velocity. The Coriolis terms $f \bar{v}$ and $f \bar{u}$ cancel. The surface-slope and
atmospheric-pressure terms of (14), (15) and $\hat{p}$ combine into
$g \left(\rho_0 / \left(\rho \, r\right) - 1/a\right) \partial \eta$ and
$\left(1 / \left(\rho \, r\right) - 1 / \left(\rho_0 \, a\right)\right) \partial p_a$, which
vanish in the thin-shell limit of §3. $\bar{u}$ and $\bar{v}$ are independent of $z$ and are
taken outside the vertical derivatives,

$$
\begin{aligned}
& \frac{\partial u'}{\partial t}
+ \frac{1}{r\cos\varphi}\frac{\partial \left(u' u' + \bar{u} \bar{u}\right)}{\partial \lambda}
+ \frac{1}{r\cos^{2}\varphi}\frac{\partial \left[\left(u' v' + \bar{u} \bar{v}\right) \cos^{2}\varphi\right]}{\partial \varphi}
+ \frac{1}{r^{2}}\frac{\partial \left[r^{2} \left(u' w' + \bar{u} \hat{w}\right)\right]}{\partial z}
- f v'
+ \left(\frac{u'}{r} + \tilde{f}_{\varphi}\right) w'
+ \frac{1}{\rho \, r \cos\varphi}\frac{\partial p'}{\partial \lambda} \\
& \quad = \mathcal{F}_{\lambda}
- \frac{1}{d} \int\limits_{-h}^{\eta} \mathcal{F}_{\lambda} \, dz
+ \frac{1}{a d \cos\varphi}\frac{\partial}{\partial \lambda}\left(
d \bar{u} \bar{u} + \int\limits_{-h}^{\eta} u' u' \, dz\right)
+ \frac{1}{a d \cos^{2}\varphi}\frac{\partial}{\partial \varphi}\left[
\cos^{2}\varphi \left(d \bar{u} \bar{v} + \int\limits_{-h}^{\eta} u' v' \, dz\right)\right] \\
& \qquad + \frac{1}{d}\left(\frac{3 \bar{u}}{a} + \tilde{f}_{\varphi}\right)
\int\limits_{-h}^{\eta} w \, dz
+ \frac{3}{a d} \int\limits_{-h}^{\eta} u' w \, dz
+ \frac{1}{\rho_0 \, a d \cos\varphi} \int\limits_{-h}^{\eta}
\frac{\partial p'}{\partial \lambda} \, dz
+ \frac{\bar{u}}{d}\frac{\partial \eta}{\partial t} \\
& \qquad - \frac{g}{\cos\varphi}\left(\frac{\rho_0}{\rho \, r} - \frac{1}{a}\right)
\frac{\partial \eta}{\partial \lambda}
- \frac{1}{\cos\varphi}\left(\frac{1}{\rho \, r} - \frac{1}{\rho_0 \, a}\right)
\frac{\partial p_a}{\partial \lambda}
+ \frac{\rho_0}{\rho \, r \cos\varphi}\frac{\partial}{\partial \lambda}\left[
\left(\eta - z\right)\left(\tilde{f}_{\varphi} \bar{u} - \tilde{f}_{\lambda} \bar{v}\right)\right] \\
& \qquad - \frac{1}{a \cos\varphi}\left[
\left(\tilde{f}_{\varphi} \bar{u} - \tilde{f}_{\lambda} \bar{v}\right)
\frac{\partial \eta}{\partial \lambda}
+ \frac{d}{2}\frac{\partial}{\partial \lambda}\left(
\tilde{f}_{\varphi} \bar{u} - \tilde{f}_{\lambda} \bar{v}\right)\right] \\
& \qquad - \left(\frac{\bar{u}}{r} + \tilde{f}_{\varphi}\right) \hat{w} \\
& \qquad - \frac{1}{r\cos\varphi}\frac{\partial \left(2 \, \bar{u} u'\right)}{\partial \lambda}
- \frac{1}{r\cos^{2}\varphi}\frac{\partial}{\partial \varphi}\left[
\left(\bar{u} v' + u' \bar{v}\right) \cos^{2}\varphi\right]
- \frac{\bar{u}}{r^{2}}\frac{\partial \left(r^{2} w'\right)}{\partial z}
- \frac{1}{r^{2}}\frac{\partial \left(r^{2} u' \hat{w}\right)}{\partial z}
- \frac{\bar{u} w' + u' \hat{w}}{r},
\end{aligned}
\tag{16}
$$

$$
\begin{aligned}
& \frac{\partial v'}{\partial t}
+ \frac{1}{r\cos\varphi}\frac{\partial \left(u' v' + \bar{u} \bar{v}\right)}{\partial \lambda}
+ \frac{1}{r\cos\varphi}\frac{\partial \left[\left(v' v' + \bar{v} \bar{v}\right) \cos\varphi\right]}{\partial \varphi}
+ \frac{1}{r^{2}}\frac{\partial \left[r^{2} \left(v' w' + \bar{v} \hat{w}\right)\right]}{\partial z}
+ \left(f + \frac{u' \tan\varphi}{r}\right) u'
+ \left(\frac{v'}{r} - \tilde{f}_{\lambda}\right) w'
+ \frac{1}{\rho \, r}\frac{\partial p'}{\partial \varphi} \\
& \quad = \mathcal{F}_{\varphi}
- \frac{1}{d} \int\limits_{-h}^{\eta} \mathcal{F}_{\varphi} \, dz
+ \frac{1}{a d \cos\varphi}\frac{\partial}{\partial \lambda}\left(
d \bar{u} \bar{v} + \int\limits_{-h}^{\eta} u' v' \, dz\right)
+ \frac{1}{a d \cos\varphi}\frac{\partial}{\partial \varphi}\left[
\cos\varphi \left(d \bar{v} \bar{v} + \int\limits_{-h}^{\eta} v' v' \, dz\right)\right]
+ \frac{\tan\varphi}{a d}\left(
d \bar{u} \bar{u} + \int\limits_{-h}^{\eta} u' u' \, dz\right) \\
& \qquad + \frac{1}{d}\left(\frac{3 \bar{v}}{a} - \tilde{f}_{\lambda}\right)
\int\limits_{-h}^{\eta} w \, dz
+ \frac{3}{a d} \int\limits_{-h}^{\eta} v' w \, dz
+ \frac{1}{\rho_0 \, a d} \int\limits_{-h}^{\eta}
\frac{\partial p'}{\partial \varphi} \, dz
+ \frac{\bar{v}}{d}\frac{\partial \eta}{\partial t} \\
& \qquad - g \left(\frac{\rho_0}{\rho \, r} - \frac{1}{a}\right)
\frac{\partial \eta}{\partial \varphi}
- \left(\frac{1}{\rho \, r} - \frac{1}{\rho_0 \, a}\right)
\frac{\partial p_a}{\partial \varphi}
+ \frac{\rho_0}{\rho \, r}\frac{\partial}{\partial \varphi}\left[
\left(\eta - z\right)\left(\tilde{f}_{\varphi} \bar{u} - \tilde{f}_{\lambda} \bar{v}\right)\right] \\
& \qquad - \frac{1}{a}\left[
\left(\tilde{f}_{\varphi} \bar{u} - \tilde{f}_{\lambda} \bar{v}\right)
\frac{\partial \eta}{\partial \varphi}
+ \frac{d}{2}\frac{\partial}{\partial \varphi}\left(
\tilde{f}_{\varphi} \bar{u} - \tilde{f}_{\lambda} \bar{v}\right)\right] \\
& \qquad - \frac{\bar{u} \bar{u} \tan\varphi}{r}
- \left(\frac{\bar{v}}{r} - \tilde{f}_{\lambda}\right) \hat{w} \\
& \qquad - \frac{1}{r\cos\varphi}\frac{\partial}{\partial \lambda}\left(
\bar{u} v' + u' \bar{v}\right)
- \frac{1}{r\cos\varphi}\frac{\partial \left(2 \, \bar{v} v' \cos\varphi\right)}{\partial \varphi}
- \frac{\bar{v}}{r^{2}}\frac{\partial \left(r^{2} w'\right)}{\partial z}
- \frac{1}{r^{2}}\frac{\partial \left(r^{2} v' \hat{w}\right)}{\partial z}
- \frac{\bar{v} w' + v' \hat{w} + 2 \, \bar{u} u' \tan\varphi}{r} .
\end{aligned}
\tag{17}
$$

## 5. References

Blumberg, A. F., and G. L. Mellor, 1987: A description of a three-dimensional coastal ocean
circulation model. *Three-Dimensional Coastal Ocean Models*, N. S. Heaps, Ed., Coastal and
Estuarine Sciences 4, American Geophysical Union, 1–16.

Lan, R., L. Ju, Z. Wang, M. Gunzburger, and P. Jones, 2022: High-order multirate explicit
time-stepping schemes for the baroclinic-barotropic split dynamics in primitive equations.
*Journal of Computational Physics*, arXiv:2105.13484.

Shchepetkin, A. F., and J. C. McWilliams, 2005: The regional oceanic modeling system (ROMS): a
split-explicit, free-surface, topography-following-coordinate oceanic model. *Ocean
Modelling*, **9**, 347–404.

Vallis, G. K., 2006: *Atmospheric and Oceanic Fluid Dynamics*. Cambridge University Press.
