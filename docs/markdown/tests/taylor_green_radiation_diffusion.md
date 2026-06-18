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

## Explicit cosine mode

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

The recommended test sets \\(\omega=0\\), so this potential is steady in
the Eulerian frame. Keeping \\(\omega\\ne0\\) is useful only if one wants to
measure finite-P1 phase and amplitude errors.

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

For gradient initial data, the solution remains a gradient field. Write

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

For \\(\omega\ne0\\), this gives a finite-P1 model error that scales
linearly with \\(\omega\tau_F\\). The recommended test avoids this effect by
setting \\(\omega=0\\).

### Steady flux transient

When \\(\omega=0\\), \\(\Phi\\) and \\(\mathbf{F}_{\mathrm D}\\) are time
independent. The P1 relaxation equation has the exact solution

<script type="math/tex; mode=display">
\begin{equation}
  \mathbf{F}_{\mathrm r}(t)
  =
  \mathbf{F}_{\mathrm D}
  +
  \left[\mathbf{F}_{\mathrm r}(0)-\mathbf{F}_{\mathrm D}\right]e^{-t/\tau_F}.
  \label{eq:steady-flux-transient}
\end{equation}
</script>

Equivalently,

<script type="math/tex; mode=display">
\begin{equation}
  \Psi^\tau(t)
  =
  \Phi
  +
  \left[\Psi^\tau(0)-\Phi\right]e^{-t/\tau_F}.
  \label{eq:steady-potential-transient}
\end{equation}
</script>

This is the more direct asymptotic-diffusion test. Initialize the flux
away from the diffusion value, for example
\\(\mathbf{F}_{\mathrm r}(0)=\mathbf{0}\\), and measure

<script type="math/tex; mode=display">
\begin{equation}
  \frac{\|\mathbf{F}_{\mathrm r}(t)-\mathbf{F}_{\mathrm D}\|}
       {\|\mathbf{F}_{\mathrm r}(0)-\mathbf{F}_{\mathrm D}\|}
  =
  e^{-t/\tau_F}.
  \label{eq:steady-transient-decay}
\end{equation}
</script>

Initializing the code to the exact diffusion MMS state with the
consistent flux \\(\mathbf{F}_{\mathrm r}(0)=\mathbf{F}_{\mathrm D}\\) is not
a good code test. It removes the stiff flux-relaxation dynamics, so a
time discretization with an incorrect factor of \\(c\\), \\(\hat{c}\\), or
\\(\rho_0\kappa_R\\) in the relaxation term could still appear to preserve
the already-balanced state.

During the transient, exact finite-P1 gas momentum balance uses
\\(p^\tau=p_0+\Pi_{\mathrm{TG}}+\Psi^\tau(t)\\), and the exact finite-P1
radiation energy source is Eq. \\(\ref{eq:p1-energy-source}\\) with
\\(\partial_t\Phi=0\\). As \\(t/\tau_F\\to\infty\\), the solution converges to
the diffusion MMS:
\\(\Psi^\tau\to\Phi\\),
\\(\mathbf{F}_{\mathrm r}\to\mathbf{F}_{\mathrm D}\\), and
\\(S_E^\tau\to S_E^{\mathrm D}\\).

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
incompressible, the radiation flux safely subluminal, and the P1 flux
relaxation time well resolved. The following dimensionless choice is a
useful starting point:

| Quantity | Recommended value |
| --- | --- |
| Domain | \\([0,1]^2\\), periodic |
| Taylor-Green wavenumber | \\(k=2\pi\\) |
| Source mode | \\(q=\cos(4\pi x)+\cos(4\pi y)\\) |
| Density | \\(\rho_0=1\\) |
| Gas adiabatic index | \\(\gamma=5/3\\) |
| Gas pressure offset | \\(p_0=0.1\\) |
| Radiation potential | \\(\Phi=Aq\\), \\(A=5\times10^{-3}\\) |
| Radiation energy offset | \\(E_{\mathrm r,0}=1\\) |
| Physical light speed | \\(c=100\\) |
| Reduced light speed, if used | \\(\hat{c}=1\\) |
| Velocity amplitude | \\(U=10^{-2}\\) or smaller |
| Opacity | \\(\kappa_R=100\\) |
| Driving frequency | \\(\omega=0\\) |
| Initial P1 flux | \\(\mathbf{F}_{\mathrm r}(0)=\mathbf{0}\\) |
| Resolution | \\(64^2\\) for development, \\(128^2\\) for convergence |
| Comparison time | \\(t_f=5\tau_F\\) for decay, later for steady MMS error |

The table assumes the finite-P1 relaxation is tested with a reduced light
speed \\(\hat{c}\\), so

<script type="math/tex; mode=display">
\begin{equation}
  \tau_F
  =
  \frac{1}{\hat{c}\rho_0\kappa_R}
  =
  0.01.
  \label{eq:recommended-tauf}
\end{equation}
</script>

For a full-speed calculation, replace \\(\hat{c}\\) by \\(c\\) in
\\(\tau_F\\). With the zero-flux initialization, the expected transient is

<script type="math/tex; mode=display">
\begin{equation}
  \mathbf{F}_{\mathrm r}(t)-\mathbf{F}_{\mathrm D}
  =
  -\mathbf{F}_{\mathrm D}e^{-t/\tau_F}.
  \label{eq:recommended-flux-decay}
\end{equation}
</script>

At \\(t_f=5\tau_F\\), the flux error should be
\\(e^{-5}\approx6.7\times10^{-3}\\) times its initial value. The late-time
solution should then reproduce the diffusion MMS, up to the usual spatial
and temporal discretization errors.

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

For a relaxation sweep, keep \\(\hat{c}=1\\) and use

<script type="math/tex; mode=display">
\begin{equation}
  \kappa_R \in \{50,100,200,400\},
  \qquad
  \tau_F \in \{0.02,0.01,0.005,0.0025\}.
  \label{eq:recommended-sweep}
\end{equation}
</script>

The fitted decay rate should be \\(1/\tau_F\\), and the late-time state
should converge to the same diffusion MMS for all opacities. If the
problem instead initializes \\(\mathbf{F}_{\mathrm r}(0)=\mathbf{F}_{\mathrm D}\\),
the homogeneous transient is removed and the test becomes a purely steady
MMS balance. That initialization is useful only as a narrow equilibrium
preservation check; it should not be the main code test because it does
not exercise the scaling of the stiff time-discretized relaxation term.
