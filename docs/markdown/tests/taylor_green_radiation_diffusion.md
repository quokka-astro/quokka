# Taylor-Green radiation diffusion MMS

## Purpose

This note derives a manufactured Taylor--Green radiation-hydrodynamics
solution whose velocity is constant in the Eulerian frame:
\\(\mathbf{v}^\star(\mathbf{x},t)=\mathbf{v}\_{\mathrm{TG}}(\mathbf{x})\\).
The radiation transport is treated in the diffusion approximation, so the
radiation flux is not an independently manufactured degree of freedom.
It is the algebraic diffusion-limit flux needed to balance the radiation
momentum equation.

The construction has two parts:

1. an exact diffusion-limit MMS that balances the gas momentum equation,
   the diffusion-limit radiation momentum equation, and the radiation
   energy equation; and
2. a finite-parameter P1 relaxation family with no flux source, whose
   limit is the diffusion MMS.

## Taylor--Green velocity

Use a two-dimensional periodic domain, with all fields independent of
\\(z\\). Let

<script type="math/tex; mode=display">
\begin{equation}
  \mathbf{v}_{\mathrm{TG}}(x,y)
  =
  \begin{pmatrix}
    U\sin(kx)\cos(ky)\\
   -U\cos(kx)\sin(ky)\\
    0
  \end{pmatrix},
  \qquad
  \rho^\star=\rho_0.
  \label{eq:tgv}
\end{equation}
</script>

The velocity is solenoidal:

<script type="math/tex; mode=display">
\begin{equation}
  \nabla\!\cdot\mathbf{v}_{\mathrm{TG}}
  =
  Uk\cos(kx)\cos(ky)-Uk\cos(kx)\cos(ky)=0.
  \label{eq:tgv-divergence}
\end{equation}
</script>

Define the canonical Taylor--Green pressure potential

<script type="math/tex; mode=display">
\begin{equation}
  \Pi_{\mathrm{TG}}(x,y)
  =
  \frac{\rho_0U^2}{4}
  \left[\cos(2kx)+\cos(2ky)\right].
  \label{eq:tg-pressure}
\end{equation}
</script>

Then

<script type="math/tex; mode=display">
\begin{equation}
  \rho_0(\mathbf{v}_{\mathrm{TG}}\cdot\nabla)\mathbf{v}_{\mathrm{TG}}
  =
  \frac{\rho_0U^2k}{2}
  \begin{pmatrix}
    \sin(2kx)\\
    \sin(2ky)\\
    0
  \end{pmatrix}
  =
  -\nabla\Pi_{\mathrm{TG}}.
  \label{eq:tgv-advective-acceleration}
\end{equation}
</script>

The continuity equation is satisfied by \\(\rho^\star=\rho_0\\) because
\\(\nabla\!\cdot\mathbf{v}\_{\mathrm{TG}}=0\\).

## Diffusion-limit equations

The gas momentum equation is

<script type="math/tex; mode=display">
\begin{equation}
  \rho\left(\partial_t\mathbf{v}+\mathbf{v}\cdot\nabla\mathbf{v}\right)
  + \nabla p
  =
  \mathbf{G},
  \label{eq:gas-momentum}
\end{equation}
</script>

where \\(\mathbf{G}\\) is the radiation force on the gas. In a P1 moment
system, the leading static-frame radiation momentum equation is

<script type="math/tex; mode=display">
\begin{equation}
  \partial_t\mathbf{F}_{\mathrm r}
  + \frac{c^2}{3}\nabla E_{\mathrm r}
  =
  -c\rho\kappa_R\mathbf{F}_{\mathrm r}.
  \label{eq:p1-momentum}
\end{equation}
</script>

The diffusion approximation drops the flux time derivative in
Eq. \\(\ref{eq:p1-momentum}\\), giving

<script type="math/tex; mode=display">
\begin{equation}
  \mathbf{F}_{\mathrm D}
  =
  -\frac{c}{3\rho\kappa_R}\nabla E_{\mathrm r},
  \qquad
  \mathbf{G}_{\mathrm D}
  =
  \frac{\rho\kappa_R}{c}\mathbf{F}_{\mathrm D}
  =
  -\frac{1}{3}\nabla E_{\mathrm r}.
  \label{eq:diffusion-flux-force}
\end{equation}
</script>

The comoving radiation diffusion energy equation is

<script type="math/tex; mode=display">
\begin{equation}
  \partial_t E_{\mathrm r}
  + \mathbf{v}\cdot\nabla E_{\mathrm r}
  + \frac{4}{3}E_{\mathrm r}\nabla\!\cdot\mathbf{v}
  =
  -\nabla\!\cdot\mathbf{F}_{\mathrm D}+S_E.
  \label{eq:diffusion-energy}
\end{equation}
</script>

For the Taylor--Green velocity, the compression term vanishes.

## Exact diffusion MMS

Let \\(\Phi(\mathbf{x},t)\\) be any smooth periodic potential. Choose

<script type="math/tex; mode=display">
\begin{align}
  E_{\mathrm r}^{\mathrm D}
  &=
  E_{\mathrm r,0}-3\Phi,
  \label{eq:diffusion-erad}\\
  \mathbf{F}_{\mathrm D}
  &=
  \frac{c}{\rho_0\kappa_R}\nabla\Phi,
  \label{eq:diffusion-flux}\\
  \mathbf{G}_{\mathrm D}
  &=
  \nabla\Phi,
  \label{eq:diffusion-force}\\
  p^{\mathrm D}
  &=
  p_0+\Pi_{\mathrm{TG}}+\Phi.
  \label{eq:diffusion-pressure}
\end{align}
</script>

This exactly balances gas momentum. Since the velocity is stationary in
the Eulerian frame, Eq. \\(\ref{eq:tgv-advective-acceleration}\\) gives

<script type="math/tex; mode=display">
\begin{align}
  \rho_0(\mathbf{v}_{\mathrm{TG}}\cdot\nabla)\mathbf{v}_{\mathrm{TG}}
  + \nabla p^{\mathrm D}
  &=
  -\nabla\Pi_{\mathrm{TG}}
  + \nabla(\Pi_{\mathrm{TG}}+\Phi) \nonumber\\
  &=
  \nabla\Phi
  =
  \mathbf{G}_{\mathrm D}.
  \label{eq:gas-momentum-balance}
\end{align}
</script>

It also exactly balances the diffusion-limit radiation momentum equation:

<script type="math/tex; mode=display">
\begin{equation}
  \frac{c^2}{3}\nabla E_{\mathrm r}^{\mathrm D}
  + c\rho_0\kappa_R\mathbf{F}_{\mathrm D}
  =
  -c^2\nabla\Phi+c^2\nabla\Phi
  =
  \mathbf{0}.
  \label{eq:radiation-momentum-balance}
\end{equation}
</script>

The radiation energy source is the residual of
Eq. \\(\ref{eq:diffusion-energy}\\). Because
\\(\nabla\!\cdot\mathbf{v}\_{\mathrm{TG}}=0\\),

<script type="math/tex; mode=display">
\begin{equation}
  \boxed{
  S_E^{\mathrm D}
  =
  -3\left(\partial_t\Phi+\mathbf{v}_{\mathrm{TG}}\cdot\nabla\Phi\right)
  + \frac{c}{\rho_0\kappa_R}\nabla^2\Phi.
  }
  \label{eq:diffusion-energy-source}
\end{equation}
</script>

The first term keeps the radiation energy profile fixed in the chosen
Eulerian time dependence while the gas advects through it. The second
term balances diffusion.

## Explicit harmonic mode

For a useful one-parameter MMS, take

<script type="math/tex; mode=display">
\begin{equation}
  \Phi(x,y,t)
  =
  A\cos(\omega t)
  \left[\cos(2kx)+\cos(2ky)\right].
  \label{eq:harmonic-mode}
\end{equation}
</script>

Then

<script type="math/tex; mode=display">
\begin{align}
  E_{\mathrm r}^{\mathrm D}
  &=
  E_{\mathrm r,0}
  -3A\cos(\omega t)
  \left[\cos(2kx)+\cos(2ky)\right],
  \label{eq:explicit-erad}\\
  p^{\mathrm D}
  &=
  p_0+
  \left[\frac{\rho_0U^2}{4}+A\cos(\omega t)\right]
  \left[\cos(2kx)+\cos(2ky)\right],
  \label{eq:explicit-pressure}\\
  \mathbf{F}_{\mathrm D}
  &=
  -\frac{2ckA\cos(\omega t)}{\rho_0\kappa_R}
  \begin{pmatrix}
    \sin(2kx)\\
    \sin(2ky)\\
    0
  \end{pmatrix}.
  \label{eq:explicit-flux}
\end{align}
</script>

The radiation energy source is

<script type="math/tex; mode=display">
\begin{align}
  S_E^{\mathrm D}
  &=
  3A\omega\sin(\omega t)
  \left[\cos(2kx)+\cos(2ky)\right] \nonumber\\
  &\quad
  +6kAU\cos(\omega t)
  \left[
    \sin(kx)\cos(ky)\sin(2kx)
    -
    \cos(kx)\sin(ky)\sin(2ky)
  \right] \nonumber\\
  &\quad
  -\frac{4ck^2A\cos(\omega t)}{\rho_0\kappa_R}
  \left[\cos(2kx)+\cos(2ky)\right].
  \label{eq:explicit-source}
\end{align}
</script>

For a stationary radiation profile, set \\(\omega=0\\). The velocity is
still Eulerian steady; the source retains the advection and diffusion
terms.

## P1 relaxation family

To approach the diffusion MMS from a finite P1 system, keep
\\(E_{\mathrm r}=E_{\mathrm r,0}-3\Phi\\) and use the P1 radiation momentum
equation with no manufactured flux source:

<script type="math/tex; mode=display">
\begin{equation}
  \partial_t\mathbf{F}_{\mathrm r}
  + \frac{c^2}{3}\nabla E_{\mathrm r}
  =
  -c\rho_0\kappa_R\mathbf{F}_{\mathrm r}.
  \label{eq:p1-family-momentum}
\end{equation}
</script>

With

<script type="math/tex; mode=display">
\begin{equation}
  \tau_F\equiv\frac{1}{c\rho_0\kappa_R},
  \qquad
  \mathbf{F}_{\mathrm D}
  =
  \frac{c}{\rho_0\kappa_R}\nabla\Phi,
  \label{eq:relaxation-definitions}
\end{equation}
</script>

Eq. \\(\ref{eq:p1-family-momentum}\\) becomes

<script type="math/tex; mode=display">
\begin{equation}
  \tau_F\partial_t\mathbf{F}_{\mathrm r}
  + \mathbf{F}_{\mathrm r}
  =
  \mathbf{F}_{\mathrm D}.
  \label{eq:p1-relaxation}
\end{equation}
</script>

After the homogeneous transient is removed, the solution remains a
gradient field. Write

<script type="math/tex; mode=display">
\begin{equation}
  \mathbf{F}_{\mathrm r}^{\tau}
  =
  \frac{c}{\rho_0\kappa_R}\nabla\Psi^\tau,
  \qquad
  \tau_F\partial_t\Psi^\tau+\Psi^\tau=\Phi.
  \label{eq:relaxed-potential}
\end{equation}
</script>

Then the P1 radiation momentum equation is exactly balanced:

<script type="math/tex; mode=display">
\begin{equation}
  \partial_t\mathbf{F}_{\mathrm r}^{\tau}
  + \frac{c^2}{3}\nabla E_{\mathrm r}
  + c\rho_0\kappa_R\mathbf{F}_{\mathrm r}^{\tau}
  =
  c^2\nabla\left(\tau_F\partial_t\Psi^\tau+\Psi^\tau-\Phi\right)
  =
  \mathbf{0}.
  \label{eq:p1-momentum-balance}
\end{equation}
</script>

The gas radiation force is now

<script type="math/tex; mode=display">
\begin{equation}
  \mathbf{G}^{\tau}
  =
  \frac{\rho_0\kappa_R}{c}\mathbf{F}_{\mathrm r}^{\tau}
  =
  \nabla\Psi^\tau.
  \label{eq:p1-force}
\end{equation}
</script>

Thus exact finite-P1 gas momentum balance is obtained by using

<script type="math/tex; mode=display">
\begin{equation}
  p^\tau=p_0+\Pi_{\mathrm{TG}}+\Psi^\tau.
  \label{eq:p1-pressure}
\end{equation}
</script>

The radiation energy source for this finite-P1 member is

<script type="math/tex; mode=display">
\begin{equation}
  \boxed{
  S_E^\tau
  =
  -3\left(\partial_t\Phi+\mathbf{v}_{\mathrm{TG}}\cdot\nabla\Phi\right)
  + \frac{c}{\rho_0\kappa_R}\nabla^2\Psi^\tau.
  }
  \label{eq:p1-energy-source}
\end{equation}
</script>

As \\(\tau_F\to0\\), \\(\Psi^\tau\to\Phi\\), so
\\(p^\tau\to p^{\mathrm D}\\), \\(\mathbf{F}_{\mathrm r}^\tau\to\mathbf{F}_{\mathrm D}\\),
and Eq. \\(\ref{eq:p1-energy-source}\\) reduces to the diffusion source in
Eq. \\(\ref{eq:diffusion-energy-source}\\).

### Harmonic relaxation

For

<script type="math/tex; mode=display">
\begin{equation}
  \Phi(\mathbf{x},t)
  =
  \operatorname{Re}\left[\widehat{\Phi}(\mathbf{x})e^{i\omega t}\right],
  \label{eq:complex-harmonic}
\end{equation}
</script>

the post-transient relaxed potential is

<script type="math/tex; mode=display">
\begin{equation}
  \Psi^\tau(\mathbf{x},t)
  =
  \operatorname{Re}\left[
    \frac{\widehat{\Phi}(\mathbf{x})e^{i\omega t}}{1+i\omega\tau_F}
  \right].
  \label{eq:harmonic-relaxed-potential}
\end{equation}
</script>

The relative amplitude difference between \\(\Psi^\tau\\) and \\(\Phi\\) is

<script type="math/tex; mode=display">
\begin{equation}
  \frac{\left|\Psi^\tau-\Phi\right|}{|\Phi|}
  =
  \frac{\omega\tau_F}{\sqrt{1+(\omega\tau_F)^2}}
  =
  \omega\tau_F+O((\omega\tau_F)^2).
  \label{eq:relaxation-error}
\end{equation}
</script>

For the cosine mode in Eq. \\(\ref{eq:harmonic-mode}\\),
\\(\nabla^2\Psi^\tau=-4k^2\Psi^\tau\\). The finite-P1 source differs from
the diffusion source only through the flux-divergence term,

<script type="math/tex; mode=display">
\begin{equation}
  S_E^\tau-S_E^{\mathrm D}
  =
  \frac{c}{\rho_0\kappa_R}\nabla^2(\Psi^\tau-\Phi),
  \label{eq:source-difference}
\end{equation}
</script>

so, after spatial and temporal truncation errors are small, the P1
relaxation error scales linearly with \\(\omega\tau_F\\).

## Positivity bounds

For the cosine mode, \\(|\cos(2kx)+\cos(2ky)|\le2\\). The diffusion-limit
fields are positive if

<script type="math/tex; mode=display">
\begin{align}
  E_{\mathrm r,0} &> 6|A|,
  \label{eq:erad-positivity}\\
  p_0 &>
  2\max_t\left|\frac{\rho_0U^2}{4}+A\cos(\omega t)\right|.
  \label{eq:pressure-positivity}
\end{align}
</script>

For a finite-P1 member, replace \\(A\cos(\omega t)\\) in the pressure bound
by the maximum amplitude of \\(\Psi^\tau\\), namely
\\(|A|/\sqrt{1+(\omega\tau_F)^2}\\).

## Suggested test problem parameters

A practical first implementation should keep the gas nearly
incompressible, the radiation flux safely subluminal, and the P1
relaxation error large enough to measure above discretization error. The
following dimensionless choice is a useful starting point:

| Quantity | Recommended value |
| --- | --- |
| Domain | \\([0,1]^2\\), periodic |
| Taylor-Green wavenumber | \\(k=2\pi\\) |
| Source mode | \\(q=\cos(4\pi x)+\cos(4\pi y)\\) |
| Density | \\(\rho_0=1\\) |
| Gas adiabatic index | \\(\gamma=5/3\\) |
| Gas pressure offset | \\(p_0=0.1\\) |
| Radiation potential | \\(\Phi=Aq\cos(\omega t)\\), \\(A=5\times10^{-3}\\) |
| Radiation energy offset | \\(E_{\mathrm r,0}=1\\) |
| Physical light speed | \\(c=100\\) |
| Reduced light speed, if used | \\(\hat{c}=1\\) |
| Velocity amplitude | \\(U=10^{-2}\\) or smaller |
| Opacity | \\(\kappa_R=100\\) |
| Driving frequency | \\(\omega=5\\) |
| Resolution | \\(64^2\\) for development, \\(128^2\\) for convergence |
| Comparison time | \\(t_f=\pi/(2\omega)\\) |

The table assumes the finite-P1 relaxation is tested with a reduced light
speed \\(\hat{c}\\), so

<script type="math/tex; mode=display">
\begin{equation}
  \Omega \equiv \omega\tau_F
  =
  \frac{\omega}{\hat{c}\rho_0\kappa_R}
  =
  0.05.
  \label{eq:recommended-omega}
\end{equation}
</script>

For a full-speed calculation, replace \\(\hat{c}\\) by \\(c\\) in
\\(\tau_F\\). With \\(\Omega=0.05\\), the finite-P1 flux and gas radiation
force differ from the diffusion manufactured solution by about \\(5\%\\).
This is large enough to measure, but small enough that the solution is
visibly in the diffusion regime.

At \\(64^2\\), the cell optical depth is

<script type="math/tex; mode=display">
\begin{equation}
  \tau_{\mathrm{cell}}
  =
  \rho_0\kappa_R\Delta x
  =
  \frac{100}{64}
  \approx 1.56,
  \label{eq:recommended-cell-tau}
\end{equation}
</script>

and the optical depth across one source-mode wavelength is

<script type="math/tex; mode=display">
\begin{equation}
  \tau_\lambda
  =
  \rho_0\kappa_R\left(\frac{1}{2}\right)
  =
  50.
  \label{eq:recommended-wavelength-tau}
\end{equation}
</script>

The maximum reduced flux is approximately

<script type="math/tex; mode=display">
\begin{equation}
  f_{\max}
  =
  \frac{|\mathbf{F}_{\mathrm r}|}{cE_{\mathrm r}}
  \sim
  \frac{4\pi A\sqrt{2}}{\rho_0\kappa_R(E_{\mathrm r,0}-6A)}
  \approx 9\times10^{-4},
  \label{eq:recommended-reduced-flux}
\end{equation}
</script>

well inside the P1 admissible region.

For an asymptotic sweep, keep \\(\omega=5\\) and use

<script type="math/tex; mode=display">
\begin{equation}
  \kappa_R \in \{50,100,200,400\},
  \qquad
  \Omega \in \{0.1,0.05,0.025,0.0125\}.
  \label{eq:recommended-sweep}
\end{equation}
</script>

The finite-P1 relaxation error should decrease linearly with
\\(\Omega\\) once the mesh and timestep errors are smaller than this
modeling error. The quarter-period comparison time
\\(t_f=\pi/(2\omega)\\) is preferable to a full period because it exposes
the leading phase lag directly; at a full period the leading phase error
cancels in a final-time snapshot.

These parameters assume the problem initializes the exact relaxed flux
from Eq. \\(\ref{eq:relaxed-potential}\\), rather than initializing the
diffusion flux \\(\mathbf{F}_{\mathrm D}\\). That removes the homogeneous
flux-relaxation transient.
