# Sea Water Equation of State

The closure relation $\rho = \rho\left(T, S, p\right)$ of
[governing_equations.md](governing_equations.md) §6 is measured rather than derived, and the
literature states it in several forms. This document collects them. Density is in kg m⁻³
throughout, $\alpha$ denotes the thermal expansion coefficient, and $p$ is the gauge sea
pressure, that is the pressure less the atmospheric pressure at the free surface, in bars
where a fit is stated in bars and in decibars otherwise.

## 1. Temperature and salinity scales

Temperature is reported on the International Practical Temperature Scale of 1968, written
$t_{68}$ in °C, in the older fits, and on the International Temperature Scale of 1990, written
$t_{90}$, in the newer ones. Salinity is reported either as the dimensionless Practical
Salinity $S_P$ of the 1978 practical salinity scale, or as the Absolute Salinity $S_A$ in
g kg⁻¹. The conversions are

$$
t_{68} = 1.00024 \; t_{90},
\qquad
S_R = \frac{35.16504}{35}\, S_P \;\text{g kg}^{-1},
\qquad
S_A = S_R + \delta S_A,
\tag{1}
$$

with $S_R$ the Reference Salinity and $\delta S_A$ the Absolute Salinity anomaly, a spatially
varying correction for departures of the composition of seawater from the reference
composition. A fit taking a potential temperature takes the $\theta$ of
[governing_equations.md](governing_equations.md) §5.3, written $T$ in that document and
$\theta$ here.

## 2. Linear and quadratic forms

Expanding the equation of state to first order about a reference state
$\left(T_0, S_0\right)$ gives the linear equation of state,

$$
\rho = \rho_0 \left[1 - \alpha \left(T - T_0\right) + \beta \left(S - S_0\right)\right],
\qquad
\alpha = -\frac{1}{\rho}\left(\frac{\partial \rho}{\partial T}\right)_{S,p},
\qquad
\beta = \frac{1}{\rho}\left(\frac{\partial \rho}{\partial S}\right)_{T,p},
\tag{2}
$$

with $\alpha$ the thermal expansion coefficient and $\beta$ the haline contraction
coefficient, both held constant, and $\rho_0 = \rho\left(T_0, S_0, 0\right)$. Representative
surface values are $\rho_0 = 1027$ kg m⁻³, $\alpha = 1.7 \times 10^{-4}$ K⁻¹ and
$\beta = 7.6 \times 10^{-4}$ per unit practical salinity.

Carried to second order about $\left(T_0, S_0, p_0\right)$ the expansion is

$$
\rho = \rho_0 \Big[
1 - \alpha \left(T - T_0\right) + \beta \left(S - S_0\right)
- \tfrac{1}{2}\alpha_T \left(T - T_0\right)^{2}
- \alpha_p \left(T - T_0\right)\left(p - p_0\right)
+ \frac{p - p_0}{\rho_0 c_s^{2}}
\Big],
\tag{3}
$$

where $c_s$ is the speed of sound, so that
$\left(\partial \rho / \partial p\right)_{T,S} = 1 / c_s^{2}$, and

$$
\alpha_T = -\frac{1}{\rho_0}\frac{\partial^{2} \rho}{\partial T^{2}},
\qquad
\alpha_p = -\frac{1}{\rho_0}\frac{\partial^{2} \rho}{\partial T \, \partial p}
$$

are the cabbeling coefficient, the curvature of the isopycnals in the $T$–$S$ plane, and the
thermobaric coefficient, the pressure dependence of $\alpha$. Both are named and quantified by
McDougall (1987).

## 3. The UNESCO 1983 equation of state

Also called EOS-80, or the International Equation of State of Seawater 1980. It gives the
in-situ density from the in-situ temperature $t_{68}$ in °C, the practical salinity $S_P$ and
the pressure in bars; the two former are written $t$ and $S$ below. The one-atmosphere part is
the fit of Millero and Poisson (1981),

$$
\rho\left(t, S, 0\right) = \rho_w(t) + A(t)\, S + B(t)\, S^{3/2} + C\, S^{2},
\tag{4}
$$

in which $\rho_w$ is the density of Standard Mean Ocean Water,

$$
\rho_w(t) = 999.842594 + 6.793952 \times 10^{-2}\, t - 9.095290 \times 10^{-3}\, t^{2}
+ 1.001685 \times 10^{-4}\, t^{3} - 1.120083 \times 10^{-6}\, t^{4}
+ 6.536332 \times 10^{-9}\, t^{5},
$$

and the salinity coefficients are

$$
A(t) = 8.24493 \times 10^{-1} - 4.0899 \times 10^{-3}\, t + 7.6438 \times 10^{-5}\, t^{2}
- 8.2467 \times 10^{-7}\, t^{3} + 5.3875 \times 10^{-9}\, t^{4},
\qquad
C = 4.8314 \times 10^{-4},
$$

$$
B(t) = -5.72466 \times 10^{-3} + 1.0227 \times 10^{-4}\, t - 1.6546 \times 10^{-6}\, t^{2} .
$$

Compression is carried by the secant bulk modulus $K$ of Millero et al. (1980),

$$
\rho\left(t, S, p\right) = \frac{\rho\left(t, S, 0\right)}{1 - p / K\left(t, S, p\right)},
\qquad
K\left(t, S, p\right) = K_0\left(t, S\right) + A_K\left(t, S\right) p
+ B_K\left(t, S\right) p^{2},
\tag{5}
$$

with

$$
K_0 = 19652.21 + 148.4206\, t - 2.327105\, t^{2} + 1.360477 \times 10^{-2}\, t^{3}
- 5.155288 \times 10^{-5}\, t^{4}
$$
$$
+ \left(54.6746 - 0.603459\, t + 1.09987 \times 10^{-2}\, t^{2}
- 6.1670 \times 10^{-5}\, t^{3}\right) S
+ \left(7.944 \times 10^{-2} + 1.6483 \times 10^{-2}\, t
- 5.3009 \times 10^{-4}\, t^{2}\right) S^{3/2},
$$

$$
A_K = 3.239908 + 1.43713 \times 10^{-3}\, t + 1.16092 \times 10^{-4}\, t^{2}
- 5.77905 \times 10^{-7}\, t^{3}
+ \left(2.2838 \times 10^{-3} - 1.0981 \times 10^{-5}\, t
- 1.6078 \times 10^{-6}\, t^{2}\right) S
+ 1.91075 \times 10^{-4}\, S^{3/2},
$$

$$
B_K = 8.50935 \times 10^{-5} - 6.12293 \times 10^{-6}\, t + 5.2787 \times 10^{-8}\, t^{2}
+ \left(-9.9348 \times 10^{-7} + 2.0816 \times 10^{-8}\, t
+ 9.1697 \times 10^{-10}\, t^{2}\right) S .
$$

The fit holds for $-2 \le t \le 40$ °C, $0 \le S \le 42$ and $0 \le p \le 1000$ bar. The
algorithms are collected in Fofonoff and Millard (1983).

## 4. Fits in potential temperature

The forms of §3 take the in-situ temperature, which is not a prognostic variable of the
governing equations.

**Jackett and McDougall (1995).** The structure of (4) and (5) is kept and every coefficient
refitted so that the argument is the potential temperature,

$$
\rho\left(\theta, S, p\right)
= \frac{\rho\left(\theta, S, 0\right)}{1 - p / K\left(\theta, S, p\right)},
$$

over $-2 \le \theta \le 40$ °C, $0 \le S \le 42$ and $0 \le p \le 1000$ bar, with $p$ in bars.

**McDougall et al. (2003).** The secant bulk modulus is replaced by a rational function of the
same three arguments,

$$
\rho\left(\theta, S, p\right)
= \frac{P_1\left(\theta, S, p\right)}{P_2\left(\theta, S, p\right)},
$$

with $P_1$ a polynomial of twelve terms and $P_2$ of thirteen, the latter carrying a term in
$S^{3/2}$, and $p$ in decibars. The fit is stated over a funnel-shaped region of
$\left(\theta, S, p\right)$ space enclosing the observed properties of the world ocean.

**Wright (1997).** The form of Eckart (1958) is refitted to the modern data,

$$
\rho\left(\theta, S, p\right)
= \frac{p + p_0\left(\theta, S\right)}
{\lambda\left(\theta, S\right)
+ \alpha_0\left(\theta, S\right)\left[p + p_0\left(\theta, S\right)\right]},
$$

in which $\alpha_0$ is linear in $\theta$ and $S$ while $p_0$ and $\lambda$ are cubic in
$\theta$ with a bilinear salinity term, fifteen coefficients in all. Within this equation
$\alpha_0$, $p_0$ and $\lambda$ are Wright's symbols and carry none of the meanings they have
elsewhere in this document; the redefinition ends with the paragraph.

**Brydon et al. (1999).** The potential density anomaly
$\sigma = \rho\left(\theta, S, p_r\right) - 1000$ at a fixed reference pressure $p_r$ is
fitted by a cubic,

$$
\sigma\left(\theta, S\right) = c_1 + c_2 \theta + c_3 S + c_4 \theta^{2} + c_5 S \theta
+ c_6 \theta^{3} + c_7 S \theta^{2},
$$

whose seven coefficients are refitted for each $p_r$ over the range of $\theta$ and $S$ found
at that pressure. The form is linear in $S$ at fixed $\theta$ and so inverts for salinity in
closed form.

## 5. TEOS-10

The Thermodynamic Equation of Seawater 2010 defines every thermodynamic property of seawater
as a derivative of one Gibbs function $g\left(S_A, t, p\right)$, with $S_A$ the Absolute
Salinity of (1), $t$ the in-situ temperature $t_{90}$ and $p$ the gauge sea pressure. The
function is the sum of a pure water part, the 2009 IAPWS release for ordinary water substance,
and a saline part, the 2008 IAPWS release for seawater given by Feistel (2008), the latter
carrying a term in $S_A \ln S_A$ required by the thermodynamics of a dilute solution. The
specific volume $v$, and with it the density, is

$$
v = \frac{1}{\rho} = \left(\frac{\partial g}{\partial p}\right)_{S_A, t} .
\tag{6}
$$

The temperature variable of the standard is the Conservative Temperature $\Theta$ of
McDougall (2003), proportional to the potential enthalpy $h^0$, that is the specific enthalpy
at zero sea pressure,

$$
\Theta\left(S_A, \theta\right) = \frac{h^0\left(S_A, \theta\right)}{c_p^0},
\qquad
c_p^0 = 3991.867\,957\,119\,63 \;\text{J kg}^{-1}\text{K}^{-1},
$$

the constant $c_p^0$ being exact by definition. In these variables the equation of state is
$\rho = \rho\left(S_A, \Theta, p\right)$, obtained from (6) with $t$ recovered from $\Theta$.
Algorithms for the standard are given by Jackett et al. (2006) and in the manual of IOC, SCOR
and IAPSO (2010).

## 6. Polynomial approximations to TEOS-10

**Roquet et al. (2015a).** Equation (6) is fitted directly in the variables
$\left(S_A, \Theta, Z\right)$, where $Z$ is the depth in metres for the Boussinesq form and the
sea pressure in decibars for the compressible form. The arguments are scaled,

$$
s = \sqrt{\frac{S_A + \delta S}{S_{A,u}}},
\qquad
\tau = \frac{\Theta}{\Theta_u},
\qquad
\zeta = \frac{Z}{Z_u},
$$

with $S_{A,u} = 40 \times 35.16504 / 35$ g kg⁻¹, $\Theta_u = 40$ °C, $Z_u = 10^{4}$, and the
offset $\delta S = 32$ g kg⁻¹ for the density fit and $24$ g kg⁻¹ for the specific volume fit.
The square root in $s$ reproduces the $S^{3/2}$ dependence of (4). The fit is the sum of a
vertical reference profile and an anomaly,

$$
\rho\left(S_A, \Theta, Z\right) = r_0(\zeta) + r'\left(s, \tau, \zeta\right),
\qquad
r_0(\zeta) = \sum_{k = 1}^{6} R_k \zeta^{k},
\qquad
r'\left(s, \tau, \zeta\right) = \sum_{k = 0}^{3} r_k\left(s, \tau\right) \zeta^{k},
$$

the reference profile being a sixth-order polynomial in $\zeta$ without constant term and the
anomaly a 52-term polynomial, cubic in $\zeta$, whose coefficients $r_k$ are polynomials in
$s$ and $\tau$. The same construction applied to the specific volume gives a 75-term
expression, which additionally reproduces the sound speed of the standard.

**Roquet et al. (2015b).** A simplified form in the same variables,

$$
\rho - \rho_0 =
- a_0 \left(1 + \tfrac{1}{2}\lambda_1 \Theta_a + \mu_1 Z\right) \Theta_a
+ b_0 \left(1 - \tfrac{1}{2}\lambda_2 S_a - \mu_2 Z\right) S_a
- \nu\, \Theta_a S_a,
$$

with $\Theta_a = \Theta - 10$ °C, $S_a = S_A - 35$ g kg⁻¹, $Z$ the depth and
$\rho_0 = \rho\left(35, 10, 0\right)$. The seven coefficients are tabulated in the reference.
Setting $\lambda_1 = \lambda_2 = \mu_1 = \mu_2 = \nu = 0$ recovers (2); $\lambda_1$ and $\mu_1$
carry the cabbeling and thermobaric terms of (3), $\lambda_2$ the corresponding curvature in
salinity, and $\nu$ the cross term.

## 7. References

- Brydon, D., S. Sun and R. Bleck, 1999: A new approximation of the equation of state for
  seawater, suitable for numerical ocean models. *J. Geophys. Res.*, **104** (C1), 1537–1540.
- Eckart, C., 1958: Properties of water, Part II. The equation of state of water and sea water
  at low temperatures and pressures. *Am. J. Sci.*, **256**, 225–240.
- Feistel, R., 2008: A Gibbs function for seawater thermodynamics for −6 to 80 °C and salinity
  up to 120 g kg⁻¹. *Deep-Sea Res. I*, **55**, 1639–1671.
- Fofonoff, N. P. and R. C. Millard, 1983: Algorithms for computation of fundamental
  properties of seawater. *UNESCO Technical Papers in Marine Science*, **44**, 53 pp.
- IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of seawater – 2010:
  calculation and use of thermodynamic properties. *Intergovernmental Oceanographic
  Commission, Manuals and Guides No. 56*, UNESCO, 196 pp.
- Jackett, D. R. and T. J. McDougall, 1995: Minimal adjustment of hydrographic profiles to
  achieve static stability. *J. Atmos. Oceanic Technol.*, **12**, 381–389.
- Jackett, D. R., T. J. McDougall, R. Feistel, D. G. Wright and S. M. Griffies, 2006:
  Algorithms for density, potential temperature, conservative temperature, and the freezing
  temperature of seawater. *J. Atmos. Oceanic Technol.*, **23**, 1709–1728.
- McDougall, T. J., 1987: Thermobaricity, cabbeling, and water-mass conversion.
  *J. Geophys. Res.*, **92** (C5), 5448–5464.
- McDougall, T. J., 2003: Potential enthalpy: a conservative oceanic variable for evaluating
  heat content and heat fluxes. *J. Phys. Oceanogr.*, **33**, 945–963.
- McDougall, T. J., D. R. Jackett, D. G. Wright and R. Feistel, 2003: Accurate and
  computationally efficient algorithms for potential temperature and density of seawater.
  *J. Atmos. Oceanic Technol.*, **20**, 730–741.
- Millero, F. J., C.-T. Chen, A. Bradshaw and K. Schleicher, 1980: A new high pressure
  equation of state for seawater. *Deep-Sea Res.*, **27A**, 255–264.
- Millero, F. J. and A. Poisson, 1981: International one-atmosphere equation of state of
  seawater. *Deep-Sea Res.*, **28A**, 625–629.
- Roquet, F., G. Madec, T. J. McDougall and P. M. Barker, 2015a: Accurate polynomial
  expressions for the density and specific volume of seawater using the TEOS-10 standard.
  *Ocean Modelling*, **90**, 29–43.
- Roquet, F., G. Madec, L. Brodeau and J. Nycander, 2015b: Defining a simplified yet
  "realistic" equation of state for seawater. *J. Phys. Oceanogr.*, **45**, 2564–2579.
- Wright, D. G., 1997: An equation of state for use in ocean models: Eckart's formula
  revisited. *J. Atmos. Oceanic Technol.*, **14**, 735–740.
