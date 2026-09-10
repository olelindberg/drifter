### 7.5 Mixing and surface fluxes

**Mixing.** The terms $\mathcal{F}_{\theta'}$, $\mathcal{F}_{\phi'}$ of §4 are the divergence
of the turbulent stress, split into a radial and a horizontal part because the eddy
viscosities differ by several orders of magnitude,

$$
\mathcal{F}_{\theta'} = \frac{1}{r^{2}}\frac{\partial}{\partial r}
\left(r^{2} \nu_v \frac{\partial v}{\partial r}\right)
+ \nabla_h \cdot \left(\nu_h \nabla_h v\right),
\qquad
\mathcal{F}_{\phi'} = \frac{1}{r^{2}}\frac{\partial}{\partial r}
\left(r^{2} \nu_v \frac{\partial u}{\partial r}\right)
+ \nabla_h \cdot \left(\nu_h \nabla_h u\right),
\tag{74}
$$

with $\nabla_h = \left(\frac{1}{r}\partial / \partial \theta', \;
\frac{1}{r \sin\theta'}\partial / \partial \phi'\right)$ the horizontal part of the gradient
(9) on the rotated triad. The metric coupling between the components that a full vector
Laplacian carries is absorbed into the eddy coefficients. The radial component
$\mathcal{F}_r$ leaves the system with the quasi-hydrostatic reduction (52). The tracer
equations carry the corresponding diffusion,

$$
\mathcal{D}_T = \frac{1}{r^{2}}\frac{\partial}{\partial r}
\left(r^{2} \kappa_v \frac{\partial T}{\partial r}\right)
+ \nabla_h \cdot \left(\kappa_h \nabla_h T\right),
\tag{75}
$$

and likewise for $\mathcal{D}_S$. The eddy coefficients $\nu_v$, $\kappa_v$ are supplied by a
turbulence closure and are not part of the system above.

**Surface and bottom fluxes.** The forcing enters as boundary conditions on the radial stress
rather than as body forces. At the free surface the stress is the applied wind stress,

$$
\rho_0 \nu_v \frac{\partial \left(u, v\right)}{\partial r}\bigg|_{r = r_s}
= \left(\tau_s^{\phi'}, \; \tau_s^{\theta'}\right),
\tag{76}
$$

and at the bed it is the bottom drag, usually quadratic in the near-bed velocity,

$$
\rho_0 \nu_v \frac{\partial \left(u, v\right)}{\partial r}\bigg|_{r = r_b}
= \left(\tau_b^{\phi'}, \; \tau_b^{\theta'}\right),
\qquad
\boldsymbol{\tau}_b = \rho_0 C_d \left|\mathbf{u}_b\right| \mathbf{u}_b .
\tag{77}
$$

The tracers take the surface heat and freshwater fluxes and the no-flux bed condition of
§5.4.

None of these terms changes the closure count: they are expressed entirely in the existing
unknowns, given the externally prescribed forcing $\tau_s$, the surface tracer fluxes, and the
drag and mixing coefficients.
