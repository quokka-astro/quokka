# Taylor-Green Radiation Diffusion test

## Purpose

The MAPP document describes a Taylor--Green--Brunner
radiation-hydrodynamics test in which a Taylor--Green vortex velocity is
prescribed, the radiation source cancels the hydrodynamic pressure
force, the density remains constant, and the velocity remains unchanged.
This note derives the continuum manufactured sources needed to realize
that problem. The MAPP text places the test in a multirate setting where
radiation diffusion is one of the slow implicit physics packages, so the
main derivation below is for a grey radiation diffusion equation. A P1
two-moment derivation is retained as a comparison because it is the
closest form to Quokka's radiation transport module.

The essential point is that the gas momentum equation is affected by
radiation through the radiation four-force \\((G^0,\bm{G})\\). An external
source added only to the radiation energy equation cannot by itself
cancel a gas pressure force. In a diffusion code this coupling is
simpler: the radiation pressure is \\(p_{\mathrm r}=E_{\mathrm r}/3\\), so
the gas momentum equation contains the gradient of the total pressure
\\(p+E_{\mathrm r}/3\\).

## Quokka radiation-hydrodynamics equations

Ignoring gravity, magnetic fields, dust, chemistry, and optically thin
cooling, the relevant full-speed \\((\hat{c}=c)\\) equations are
<script type="math/tex; mode=display">
\begin{align}
  \partial_t \rho + \nabla\!\cdot(\rho \bm{v}) &= 0, \label{eq:mass}\\
  \partial_t(\rho \bm{v}) + \nabla\!\cdot(\rho \bm{v}\bm{v} + p\mathsf{I}) &= \bm{G}, \label{eq:mom}\\
  \partial_t E_{\mathrm r}+ \nabla\!\cdot\bm{F}_{\mathrm r}&= -cG^0+ S_E, \label{eq:rad-energy}\\
  \partial_t \bm{F}_{\mathrm r}+ \nabla\!\cdot(c^2\mathsf{P}_{\mathrm r}) &= -c^2\bm{G}+ \bm{S}_F. \label{eq:rad-flux}
\end{align}
</script> For a P1 closure, <script type="math/tex; mode=display">
\begin{equation}
  \mathsf{P}_{\mathrm r}= \frac{E_{\mathrm r}}{3}\mathsf{I},
  \qquad
  \nabla\!\cdot(c^2\mathsf{P}_{\mathrm r}) = \frac{c^2}{3}\nabla E_{\mathrm r}.
  \label{eq:p1}
\end{equation}
</script> Therefore, for any desired exact radiation fields
\\((E_{\mathrm r}^\star,\bm{F}_{\mathrm r}^\star)\\) and exact radiation
four-force \\((G^{0,\star},\bm{G}^\star)\\), the required radiation moment
sources are the residuals <script type="math/tex; mode=display">
\begin{align}
  S_E^\star
    &= \partial_t E_{\mathrm r}^\star + \nabla\!\cdot\bm{F}_{\mathrm r}^\star + cG^{0,\star},
    \label{eq:general-energy-source}\\
  \bm{S}_F^\star
    &= \partial_t \bm{F}_{\mathrm r}^\star + \frac{c^2}{3}\nabla E_{\mathrm r}^\star
       + c^2\bm{G}^\star.
    \label{eq:general-flux-source}
\end{align}
</script> These signs follow from moving the Quokka matter-radiation
exchange terms, \\(-cG^0\\) and \\(-c^2\bm{G}\\), to the left-hand side of the
radiation equations.

## Taylor--Green vortex fields

Use a two-dimensional periodic domain, with the solution independent of
\\(z\\). Let <script type="math/tex; mode=display">
\begin{equation}
  \bm{v}_{\mathrm{TG}}(x,y)
  =
  \begin{pmatrix}
    U\sin(kx)\cos(ky)\\
   -U\cos(kx)\sin(ky)\\
    0
  \end{pmatrix},
  \qquad
  \rho^\star = \rho_0.
  \label{eq:tgv}
\end{equation}
</script> This velocity is exactly solenoidal: <script type="math/tex; mode=display">
\begin{equation}
  \nabla\!\cdot\bm{v}_{\mathrm{TG}}
  = Uk\cos(kx)\cos(ky) - Uk\cos(kx)\cos(ky) = 0.
  \label{eq:tgv-divergence}
\end{equation}
</script>

For later use, define the canonical Taylor--Green pressure perturbation
<script type="math/tex; mode=display">
\begin{equation}
  \Pi_{\mathrm{TG}}(x,y)
  =
  \frac{\rho_0 U^2}{4}
  \left[\cos(2kx)+\cos(2ky)\right].
  \label{eq:tg-pressure}
\end{equation}
</script> The advective acceleration is <script type="math/tex; mode=display">
\begin{equation}
  \rho_0(\bm{v}_{\mathrm{TG}}\cdot\nabla)\bm{v}_{\mathrm{TG}}
  =
  \frac{\rho_0 U^2 k}{2}
  \begin{pmatrix}
    \sin(2kx)\\
    \sin(2ky)\\
    0
  \end{pmatrix}
  = -\nabla\Pi_{\mathrm{TG}}.
  \label{eq:tgv-advective-acceleration}
\end{equation}
</script>

## Radiation force required to keep the velocity unchanged

Equation \\(\\ref{eq:mom}\\) can be written in primitive form as
<script type="math/tex; mode=display">
\begin{equation}
  \rho\left(\partial_t\bm{v} + \bm{v}\cdot\nabla\bm{v}\right) + \nabla p
  = \bm{G}.
  \label{eq:primitive-momentum}
\end{equation}
</script> Thus the most general manufactured radiation force that
makes a prescribed velocity field \\(\bm{v}^\star\\) an exact solution is
<script type="math/tex; mode=display">
\begin{equation}
  \bm{G}^\star
  =
  \rho^\star
  \left(\partial_t\bm{v}^\star + \bm{v}^\star\cdot\nabla\bm{v}^\star\right)
  + \nabla p^\star.
  \label{eq:general-force}
\end{equation}
</script> For the steady Eulerian Taylor--Green field in
Eq. \\(\\ref{eq:tgv}\\),
\\(\partial_t\bm{v}^\star=0\\), so <script type="math/tex; mode=display">
\begin{equation}
  \bm{G}^\star
  =
  \nabla\left(p^\star - \Pi_{\mathrm{TG}}\right).
  \label{eq:eulerian-force}
\end{equation}
</script> If \\(p^\star=p_0+\Pi_{\mathrm{TG}}\\), then
\\(\bm{G}^\star=0\\); that is the usual unsourced incompressible
Taylor--Green balance. If the intent is instead the pressure-canceling
Lagrangian balance stated in the MAPP description, set
<script type="math/tex; mode=display">
\begin{equation}
  \bm{G}^\star = \nabla p^\star.
  \label{eq:pressure-canceling-force}
\end{equation}
</script> Then
Eq. \\(\\ref{eq:primitive-momentum}\\) gives <script type="math/tex; mode=display">
\begin{equation}
  \rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=0,
  \qquad
  \frac{\mathrm{d}}{\mathrm{d}t}\equiv\partial_t+\bm{v}\cdot\nabla,
  \label{eq:lagrangian-velocity-constant}
\end{equation}
</script> so the velocity carried by each Lagrangian element is
unchanged.

Both cases can be represented by writing the required force as a
gradient, <script type="math/tex; mode=display">
\begin{equation}
  \bm{G}^\star = \nabla\Phi.
  \label{eq:force-potential}
\end{equation}
</script> For a pressure-canceling Lagrangian test,
\\(\Phi=p^\star\\). For an Eulerian steady Taylor--Green field,
\\(\Phi=p^\star-\Pi_{\mathrm{TG}}\\).

## Radiation diffusion formulation

For a grey radiation diffusion code, the radiation flux is not an
independent evolved variable. It is closed by Fick's law,
<script type="math/tex; mode=display">
\begin{equation}
  \bm{F}_{\mathrm r}= -D_{\mathrm r}\nabla E_{\mathrm r},
  \qquad
  D_{\mathrm r}= \frac{c}{3\rho\kappa_R},
  \label{eq:diffusion-flux}
\end{equation}
</script> where \\(\kappa_R\\) is the Rosseland, or flux-mean,
opacity. The corresponding radiation force on the gas is
<script type="math/tex; mode=display">
\begin{equation}
  \bm{G}= \frac{\rho\kappa_R}{c}\bm{F}_{\mathrm r}
  = -\frac{1}{3}\nabla E_{\mathrm r}.
  \label{eq:diffusion-force}
\end{equation}
</script> The gas momentum equation is therefore equivalently
<script type="math/tex; mode=display">
\begin{equation}
  \rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t} + \nabla\left(p+\frac{E_{\mathrm r}}{3}\right)=0.
  \label{eq:diffusion-momentum}
\end{equation}
</script>

To obtain the manufactured force \\(\bm{G}^\star=\nabla\Phi\\), choose
<script type="math/tex; mode=display">
\begin{equation}
  \boxed{E_{\mathrm r}^\star = E_{\mathrm r,0}- 3\Phi.}
  \label{eq:diffusion-erad-choice}
\end{equation}
</script> Then <script type="math/tex; mode=display">
\begin{equation}
  \bm{G}^\star
  =
  -\frac{1}{3}\nabla E_{\mathrm r}^\star
  =
  \nabla\Phi.
  \label{eq:diffusion-force-choice}
\end{equation}
</script> This is the main simplification relative to a P1
two-moment code: there is no radiation flux source to manufacture. The
diffusion closure automatically sets <script type="math/tex; mode=display">
\begin{equation}
  \bm{F}_{\mathrm r}^\star
  =
  -D_{\mathrm r}\nabla E_{\mathrm r}^\star
  =
  3D_{\mathrm r}\nabla\Phi
  =
  \frac{c}{\rho_0\kappa_R}\nabla\Phi
  \label{eq:diffusion-flux-choice}
\end{equation}
</script> for constant \\(\rho_0\\) and \\(\kappa_R\\).

In a Lagrangian radiation diffusion update, the comoving radiation
energy equation is commonly written <script type="math/tex; mode=display">
\begin{equation}
  \frac{\mathrm{d}E_{\mathrm r}}{\mathrm{d}t}
  + \frac{4}{3}E_{\mathrm r}\,\nabla\!\cdot\bm{v}
  =
  \nabla\!\cdot(D_{\mathrm r}\nabla E_{\mathrm r})
  - cG^0
  + S_E.
  \label{eq:diffusion-energy}
\end{equation}
</script> The manufactured scalar radiation source is therefore
<script type="math/tex; mode=display">
\begin{equation}
  S_E^\star
  =
  \frac{\mathrm{d}E_{\mathrm r}^\star}{\mathrm{d}t}
  + \frac{4}{3}E_{\mathrm r}^\star\nabla\!\cdot\bm{v}^\star
  - \nabla\!\cdot(D_{\mathrm r}\nabla E_{\mathrm r}^\star)
  + cG^{0,\star}.
  \label{eq:diffusion-source-general}
\end{equation}
</script> For the MAPP-style pure-Lagrange Taylor--Green setup,
the desired radiation profile is steady in the material frame, the
velocity is solenoidal, and we can choose \\(G^{0,\star}=0\\). Hence
<script type="math/tex; mode=display">
\begin{equation}
  \boxed{
  S_E^\star
  =
  -\nabla\!\cdot(D_{\mathrm r}\nabla E_{\mathrm r}^\star)
  =
  3\nabla\!\cdot(D_{\mathrm r}\nabla\Phi).
  }
  \label{eq:diffusion-source-potential}
\end{equation}
</script> If \\(D_{\mathrm r}\\) is constant, this reduces to
<script type="math/tex; mode=display">
\begin{equation}
  \boxed{
  S_E^\star = 3D_{\mathrm r}\nabla^2\Phi
  = \frac{c}{\rho_0\kappa_R}\nabla^2\Phi.
  }
  \label{eq:diffusion-source-constant-d}
\end{equation}
</script> Thus a diffusion implementation needs only this scalar
radiation-energy source. There is no independent vector source
\\(\bm{S}_F\\), because the flux is algebraically determined by
Eq. \\(\\ref{eq:diffusion-flux}\\).

## P1 radiation fields and sources for comparison

Choose a positive constant \\(E_{\mathrm r,0}\\) and define
<script type="math/tex; mode=display">
\begin{equation}
  E_{\mathrm r}^\star = E_{\mathrm r,0}- 3\Phi.
  \label{eq:erad-choice}
\end{equation}
</script> The constant \\(E_{\mathrm r,0}\\) must be large enough
that \\(E_{\mathrm r}^\star>0\\) everywhere. Then the P1 radiation pressure
term is <script type="math/tex; mode=display">
\begin{equation}
  \frac{c^2}{3}\nabla E_{\mathrm r}^\star
  =
  -c^2\nabla\Phi
  =
  -c^2\bm{G}^\star.
  \label{eq:p1-cancellation}
\end{equation}
</script> Substitution into
Eq. \\(\\ref{eq:general-flux-source}\\) gives <script type="math/tex; mode=display">
\begin{equation}
  \bm{S}_F^\star = \partial_t\bm{F}_{\mathrm r}^\star.
  \label{eq:sflux-reduced}
\end{equation}
</script> For a steady manufactured radiation flux, no external
radiation-flux source is needed: <script type="math/tex; mode=display">
\begin{equation}
  \boxed{\bm{S}_F^\star=0.}
  \label{eq:sflux-zero}
\end{equation}
</script>

### Direct four-force construction

If the problem implementation directly imposes the manufactured
four-force \\((G^{0,\star},\bm{G}^\star)\\), one may take <script type="math/tex; mode=display">
\begin{equation}
  G^{0,\star}=0,\qquad
  \bm{F}_{\mathrm r}^\star=\bm{0}.
  \label{eq:direct-force-rad-fields}
\end{equation}
</script>
Equations \\(\\ref{eq:general-energy-source}\\) and
\\(\\ref{eq:sflux-zero}\\) then give <script type="math/tex; mode=display">
\begin{equation}
  \boxed{S_E^\star=0,\qquad \bm{S}_F^\star=\bm{0}.}
  \label{eq:direct-force-sources}
\end{equation}
</script> In this construction the "source" that balances the gas
is the imposed radiation four-force itself. The radiation moment
equations remain exactly balanced because the choice
\\(E_{\mathrm r}^\star=E_{\mathrm r,0}-3\Phi\\) cancels the
\\(-c^2\bm{G}^\star\\) term in the radiation flux equation.

### Opacity-coupled construction using a radiation energy source

Quokka's existing single-group source update can generate a gas
radiation force through flux-mean opacity. In the leading-order,
static-frame limit \\((\beta_{\mathrm{order}}=0)\\), <script type="math/tex; mode=display">
\begin{equation}
  \bm{G}= \frac{\rho\kappa_F}{c}\bm{F}_{\mathrm r}.
  \label{eq:opacity-force}
\end{equation}
</script> Set \\(\kappa_P=\kappa_E=0\\), choose a constant
\\(\kappa_F>0\\), and prescribe <script type="math/tex; mode=display">
\begin{equation}
  \bm{F}_{\mathrm r}^\star
  =
  \frac{c}{\rho_0\kappa_F}\nabla\Phi.
  \label{eq:flux-choice}
\end{equation}
</script> Then
Eq. \\(\\ref{eq:opacity-force}\\) gives \\(\bm{G}^\star=\nabla\Phi\\). With
\\(G^{0,\star}=0\\) and steady fields,
Eq. \\(\\ref{eq:general-energy-source}\\) gives the only nonzero
manufactured radiation source: <script type="math/tex; mode=display">
\begin{equation}
  \boxed{
  S_E^\star
  =
  \nabla\!\cdot\bm{F}_{\mathrm r}^\star
  =
  \frac{c}{\rho_0\kappa_F}\nabla^2\Phi,
  \qquad
  \bm{S}_F^\star=\bm{0}.
  }
  \label{eq:energy-source-potential}
\end{equation}
</script> This is the form most closely aligned with Quokka's
current `radEnergySource` interface: the pressure-balancing force is
produced by \\(\bm{F}_{\mathrm r}^\star\\) and \\(\kappa_F\\), while the
explicit radiation energy source keeps the prescribed
\\(E_{\mathrm r}^\star\\) field stationary.

## P1 relaxation family with a diffusion limit

Quokka has a problem-level hook for \\(S_E\\) through `SetRadEnergySource`,
but the standard update does not provide an analogous manufactured
\\(\bm{S}_F\\) hook. For a real Quokka test problem it is therefore useful
to construct a finite-parameter P1 solution using the natural flux
relaxation of the P1 equations, with \\(\bm{S}_F=0\\).

For the static-frame source terms \\((\beta_{\mathrm{order}}=0)\\), constant
\\(\rho_0\\), and constant \\(\kappa_R\\), Quokka's reduced-speed P1 equations
are <script type="math/tex; mode=display">
\begin{align}
  \partial_tE_{\mathrm r}+ \nabla\!\cdot\left(\frac{\hat{c}}{c}\bm{F}_{\mathrm r}\right)
  &= S_{E,\hat{c}}, \label{eq:q-rsla-energy}\\
  \partial_t\bm{F}_{\mathrm r}+ \frac{\hat{c}c}{3}\nabla E_{\mathrm r}
  &= -\hat{c}\rho_0\kappa_R\bm{F}_{\mathrm r}. \label{eq:q-rsla-flux}
\end{align}
</script> The gas radiation force is still <script type="math/tex; mode=display">
\begin{equation}
  \bm{G}= \frac{\rho_0\kappa_R}{c}\bm{F}_{\mathrm r}.
  \label{eq:q-rsla-force}
\end{equation}
</script> Let <script type="math/tex; mode=display">
\begin{equation}
  E_{\mathrm r}^\star = E_{\mathrm r,0}- 3\Phi(\bm{x},t),
  \qquad
  \bm{F}_{\mathrm D}= \frac{c}{\rho_0\kappa_R}\nabla\Phi.
  \label{eq:p1-diffusion-target}
\end{equation}
</script> Then
Eq. \\(\\ref{eq:q-rsla-flux}\\) can be written as the relaxation equation
<script type="math/tex; mode=display">
\begin{equation}
  \tau_F \partial_t\bm{F}_{\mathrm r}+ \bm{F}_{\mathrm r}= \bm{F}_{\mathrm D},
  \qquad
  \tau_F \equiv \frac{1}{\hat{c}\rho_0\kappa_R}.
  \label{eq:p1-relaxation}
\end{equation}
</script> Thus the P1 asymptotic parameter is the flux relaxation
time divided by the time scale on which the diffusion flux changes. If
\\(\tau_F \partial_t \bm{F}_{\mathrm D}\\) is small, the P1 flux is
<script type="math/tex; mode=display">
\begin{equation}
  \bm{F}_{\mathrm r}= \bm{F}_{\mathrm D}- \tau_F\partial_t\bm{F}_{\mathrm D}
  + O(\tau_F^2).
  \label{eq:p1-relaxation-expansion}
\end{equation}
</script> The gas force is then <script type="math/tex; mode=display">
\begin{equation}
  \bm{G}
  =
  \nabla\Phi
  - \frac{1}{c\hat{c}}\partial_t\bm{F}_{\mathrm D}
  + O(\tau_F^2\rho_0\kappa_R/c),
  \label{eq:p1-force-error}
\end{equation}
</script> so the pressure-canceling diffusion force is recovered
as \\(\tau_F/T\to0\\). In the full-speed case this force error is
\\(O(c^{-2}\partial_t\bm{F}_{\mathrm D})\\).

### Harmonic finite-parameter solution

For a clean analytic test, choose a harmonic potential
<script type="math/tex; mode=display">
\begin{equation}
  \Phi(\bm{x},t) = \operatorname{Re}\left[\widehat{\Phi}(\bm{x})e^{i\omega t}\right].
  \label{eq:harmonic-potential}
\end{equation}
</script> After the homogeneous relaxation transient has decayed,
the exact P1 flux with \\(\bm{S}_F=0\\) is <script type="math/tex; mode=display">
\begin{equation}
  \bm{F}_{\mathrm r}^\tau
  =
  \frac{c}{\rho_0\kappa_R}
  \nabla
  \operatorname{Re}\left[
    \frac{\widehat{\Phi}(\bm{x})e^{i\omega t}}{1+i\omega\tau_F}
  \right].
  \label{eq:harmonic-flux}
\end{equation}
</script> Consequently <script type="math/tex; mode=display">
\begin{equation}
  \bm{G}^\tau
  =
  \nabla\Psi^\tau,
  \qquad
  \Psi^\tau
  \equiv
  \operatorname{Re}\left[
    \frac{\widehat{\Phi}(\bm{x})e^{i\omega t}}{1+i\omega\tau_F}
  \right].
  \label{eq:harmonic-force-potential}
\end{equation}
</script> The finite-\\(\tau_F\\) gas pressure that is exactly
balanced by the P1 radiation force is therefore \\(p^\tau=p_0+\Psi^\tau\\)
for the Lagrangian pressure-canceling version, or
\\(p^\tau=p_0+\Pi_{\mathrm{TG}}+\Psi^\tau\\) for the Eulerian steady
Taylor--Green velocity field. The constant \\(p_0\\) does not affect the
force; it is included to keep the gas pressure positive. The diffusion
pressure is recovered when \\(\tau_F\to0\\).

The relative amplitude error in the force potential is
<script type="math/tex; mode=display">
\begin{equation}
  \frac{\left|\Psi^\tau-\Phi\right|}{|\Phi|}
  =
  \frac{\omega\tau_F}{\sqrt{1+(\omega\tau_F)^2}}
  =
  \omega\tau_F + O((\omega\tau_F)^2).
  \label{eq:potential-error}
\end{equation}
</script> The same relative error applies to the radiation flux
and radiation force for each Fourier mode. This gives a direct analytic
estimate for the modeling error caused by a finite asymptotic parameter.

The exact stored radiation-energy source for Quokka's reduced-speed
equations is <script type="math/tex; mode=display">
\begin{equation}
  S_{E,\hat{c}}^\tau
  =
  \partial_tE_{\mathrm r}^\star
  + \nabla\!\cdot\left(\frac{\hat{c}}{c}\bm{F}_{\mathrm r}^\tau\right)
  =
  -3\partial_t\Phi
  + \frac{\hat{c}}{\rho_0\kappa_R}\nabla^2\Psi^\tau.
  \label{eq:finite-tau-stored-source}
\end{equation}
</script> For thermal radiation groups, Quokka internally
multiplies the user-provided `radEnergySource` by \\(\hat{c}/c\\). Therefore
the source passed through `SetRadEnergySource` should be
<script type="math/tex; mode=display">
\begin{equation}
  S_{\mathrm{input}}^\tau
  =
  \frac{c}{\hat{c}}S_{E,\hat{c}}^\tau
  =
  -3\frac{c}{\hat{c}}\partial_t\Phi
  + \frac{c}{\rho_0\kappa_R}\nabla^2\Psi^\tau.
  \label{eq:finite-tau-input-source}
\end{equation}
</script> In the diffusion limit, \\(\Psi^\tau\to\Phi\\), and this
reduces to the appropriate reduced-speed diffusion source for the same
target \\(E_{\mathrm r}^\star=E_{\mathrm r,0}-3\Phi\\).

### Implications for a Quokka test

A stationary manufactured profile has \\(\omega=0\\), and then
\\(\Psi^\tau=\Phi\\): the P1 construction is exactly balanced for any
positive \\(\kappa_R\\). Such a test is useful for checking the source
implementation, but it cannot measure the asymptotic P1-to-diffusion
error.

To test the finite asymptotic parameter, use a time-dependent potential
such as <script type="math/tex; mode=display">
\begin{equation}
  \Phi(\bm{x},t)
  =
  A_p \cos(\omega t)
  \left[\cos(2kx)+\cos(2ky)\right].
  \label{eq:test-harmonic-potential}
\end{equation}
</script> For this mode,
Eq. \\(\\ref{eq:potential-error}\\) predicts the finite-\\(\tau_F\\) pressure,
flux, and radiation-force error. Since
\\(\nabla^2[\cos(2kx)+\cos(2ky)]=-4k^2[\cos(2kx)+\cos(2ky)]\\), the
amplitude of the finite-\\(\tau_F\\) correction to the diffusion energy
source is <script type="math/tex; mode=display">
\begin{equation}
  12D_{\hat{c}}k^2A_p
  \frac{\omega\tau_F}{\sqrt{1+(\omega\tau_F)^2}},
  \qquad
  D_{\hat{c}}\equiv\frac{\hat{c}}{3\rho_0\kappa_R}.
  \label{eq:source-error-amplitude}
\end{equation}
</script> This provides a closed-form tolerance target: after
spatial and temporal discretization errors are made small, the remaining
difference between a finite-\\(\tau_F\\) P1 calculation and the diffusion
manufactured solution should scale linearly with \\(\omega\tau_F\\).

### Practical parameter choices

A practical first implementation should keep the gas nearly
incompressible, the radiation flux safely subluminal, and the flux
relaxation error large enough to measure above discretization error. The
following dimensionless choice is a useful starting point:

| Quantity | Recommended value |
| --- | --- |
| Domain | \\([0,1]^2\\), periodic |
| Taylor-Green wavenumber | \\(k=2\pi\\) |
| Pressure/source mode | \\(q=\cos(4\pi x)+\cos(4\pi y)\\) |
| Density | \\(\rho_0=1\\) |
| Gas adiabatic index | \\(\gamma=5/3\\) |
| Gas pressure floor/base | \\(p_0=0.1\\) |
| Pressure perturbation | \\(\Phi=A_p q\cos(\omega t)\\), \\(A_p=5\times10^{-3}\\) |
| Radiation energy offset | \\(E_{\mathrm r,0}=1\\) |
| Physical light speed | \\(c=100\\) |
| Reduced light speed | \\(\hat{c}=1\\) |
| Velocity amplitude | \\(U=10^{-2}\\) or smaller |
| Radiation source terms | \\(\kappa_P=\kappa_E=0\\), \\(\kappa_F=\kappa_R\\) |
| Four-force order | \\(\beta_{\mathrm{order}}=0\\) |
| Driving frequency | \\(\omega=5\\) |
| Default opacity | \\(\kappa_R=100\\) |
| Resolution | \\(64^2\\) for development, \\(128^2\\) for convergence |
| Comparison time | \\(t_f=\pi/(2\omega)\\) |



With these values, <script type="math/tex; mode=display">
\begin{equation}
  \Omega \equiv \omega\tau_F
  = \frac{\omega}{\hat{c}\rho_0\kappa_R}
  = 0.05,
  \label{eq:recommended-omega}
\end{equation}
</script> so the finite-P1 force and flux differ from the
diffusion manufactured solution by approximately \\(5\%\\). This is large
enough to measure, but still small enough that the solution is visibly
in the diffusion regime. At \\(64^2\\), the cell optical depth is
<script type="math/tex; mode=display">
\begin{equation}
  \tau_{\mathrm{cell}}=\rho_0\kappa_R\Delta x = \frac{100}{64}\approx1.56,
  \label{eq:recommended-cell-tau}
\end{equation}
</script> and the optical depth across one pressure-mode
wavelength is <script type="math/tex; mode=display">
\begin{equation}
  \tau_\lambda=\rho_0\kappa_R\left(\frac{1}{2}\right)=50.
  \label{eq:recommended-wavelength-tau}
\end{equation}
</script> The maximum reduced flux is approximately
<script type="math/tex; mode=display">
\begin{equation}
  f_{\max}
  =
  \frac{|\bm{F}_{\mathrm r}|}{cE_{\mathrm r}}
  \sim
  \frac{4\pi A_p\sqrt{2}}{\rho_0\kappa_R(E_{\mathrm r,0}-6A_p)}
  \approx 9\times10^{-4},
  \label{eq:recommended-reduced-flux}
\end{equation}
</script> well inside the P1/M1 admissible region.

For an asymptotic sweep, keep \\(\omega=5\\) and use <script type="math/tex; mode=display">
\begin{equation}
  \kappa_R \in \{50,100,200,400\},
  \qquad
  \Omega \in \{0.1,0.05,0.025,0.0125\}.
  \label{eq:recommended-sweep}
\end{equation}
</script> The finite-asymptotic error should decrease linearly
with \\(\Omega\\) once the mesh and timestep errors are smaller than this
modeling error. The quarter period comparison time \\(t_f=\pi/(2\omega)\\)
is preferable to a full period because it exposes the \\(O(\Omega)\\) phase
lag directly; at a full period the leading phase error cancels in a
final-time snapshot.

These parameters assume the problem initializes the exact
finite-\\(\tau_F\\) flux in
Eq. \\(\\ref{eq:harmonic-flux}\\), rather than the diffusion flux
\\(\bm{F}_{\mathrm D}\\). That removes the homogeneous flux-relaxation
transient. If the gas pressure is made explicitly time dependent to
match \\(\Psi^\tau\\), an ideal-gas implementation also needs a matching gas
internal energy source, <script type="math/tex; mode=display">
\begin{equation}
  S_{\mathrm{gas},e}
  =
  \frac{1}{\gamma-1}
  \left(\partial_t p^\tau+\bm{v}\cdot\nabla p^\tau\right),
  \label{eq:recommended-gas-energy-source}
\end{equation}
</script> or else the test should first be run with \\(U=0\\) to
isolate the radiation asymptotics.

#### Gas and radiation positivity

Pressure positivity should be enforced analytically by choosing \\(p_0\\)
larger than the largest possible negative pressure perturbation. In the
Lagrangian pressure-canceling version, <script type="math/tex; mode=display">
\begin{equation}
  p^\tau = p_0 + \Psi^\tau,
  \qquad
  |\Psi^\tau|
  \le
  \frac{2A_p}{\sqrt{1+(\omega\tau_F)^2}},
  \label{eq:lagrangian-pressure-bound}
\end{equation}
</script> because \\(|\cos(4\pi x)+\cos(4\pi y)|\le2\\). Hence a
sufficient condition is <script type="math/tex; mode=display">
\begin{equation}
  \boxed{
  p_0 >
  \frac{2A_p}{\sqrt{1+(\omega\tau_F)^2}}.
  }
  \label{eq:lagrangian-pressure-positivity}
\end{equation}
</script> For the Eulerian steady Taylor--Green velocity field,
the gas pressure also contains \\(\Pi_{\mathrm{TG}}\\). With \\(k=2\pi\\),
<script type="math/tex; mode=display">
\begin{equation}
  \Pi_{\mathrm{TG}}
  =
  \frac{\rho_0U^2}{4}
  \left[\cos(4\pi x)+\cos(4\pi y)\right],
  \label{eq:practical-pitg}
\end{equation}
</script> so a sufficient pressure-positivity condition is
<script type="math/tex; mode=display">
\begin{equation}
  \boxed{
  p_0 >
  2\left[
    \frac{\rho_0U^2}{4}
    +
    \frac{A_p}{\sqrt{1+(\omega\tau_F)^2}}
  \right].
  }
  \label{eq:eulerian-pressure-positivity}
\end{equation}
</script> For the recommended values \\(p_0=0.1\\),
\\(A_p=5\times10^{-3}\\), \\(\rho_0=1\\), \\(U=10^{-2}\\), and \\(\omega\tau_F=0.05\\),
this gives <script type="math/tex; mode=display">
\begin{equation}
  p_{\min}
  \gtrsim
  0.1 - 2\left(2.5\times10^{-5}+4.99\times10^{-3}\right)
  \approx 8.995\times10^{-2},
  \label{eq:recommended-pressure-min}
\end{equation}
</script> so the gas pressure remains comfortably positive.

Radiation energy positivity is separate. With the practical convention
that \\(\Phi\\) is the non-constant pressure potential and \\(p_0\\) is a
gas-pressure offset, <script type="math/tex; mode=display">
\begin{equation}
  E_{\mathrm r}^\star=E_{\mathrm r,0}-3\Phi,
  \qquad
  E_{\mathrm r,\min}^\star
  \ge
  E_{\mathrm r,0}-6A_p.
  \label{eq:radiation-energy-bound}
\end{equation}
</script> Thus \\(E_{\mathrm r,0}>6A_p\\) is sufficient for the
Lagrangian pressure-canceling version. The recommended
\\(E_{\mathrm r,0}=1\\) and \\(A_p=5\times10^{-3}\\) give
\\(E_{\mathrm r,\min}^\star\ge0.97\\).

## Explicit pressure-canceling Taylor--Green source

Let the pressure to be canceled be <script type="math/tex; mode=display">
\begin{equation}
  p^\star(x,y)
  =
  p_0 + A_p\left[\cos(2kx)+\cos(2ky)\right].
  \label{eq:generic-pressure}
\end{equation}
</script> For the MAPP-style pressure-canceling Lagrangian
interpretation, \\(\Phi=p^\star\\). Then <script type="math/tex; mode=display">
\begin{align}
  \nabla p^\star
  &=
  -2kA_p
  \begin{pmatrix}
    \sin(2kx)\\
    \sin(2ky)\\
    0
  \end{pmatrix},
  \label{eq:pressure-gradient}\\
  \nabla^2 p^\star
  &=
  -4k^2 A_p\left[\cos(2kx)+\cos(2ky)\right].
  \label{eq:pressure-laplacian}
\end{align}
</script> The diffusion manufactured radiation field, algebraic
flux, and scalar source are <script type="math/tex; mode=display">
\begin{align}
  E_{\mathrm r}^\star
  &=
  E_{\mathrm r,0}- 3p^\star,
  \label{eq:pressure-erad}\\
  \bm{F}_{\mathrm r}^\star
  &=
  -\frac{2ckA_p}{\rho_0\kappa_R}
  \begin{pmatrix}
    \sin(2kx)\\
    \sin(2ky)\\
    0
  \end{pmatrix},
  \label{eq:pressure-frad}\\
  S_E^\star
  &=
  -\frac{4ck^2A_p}{\rho_0\kappa_R}
  \left[\cos(2kx)+\cos(2ky)\right].
  \label{eq:pressure-source}
\end{align}
</script> There is no separate \\(\bm{S}_F^\star\\) in a diffusion
formulation because \\(\bm{F}_{\mathrm r}^\star\\) is defined algebraically
by
Eq. \\(\\ref{eq:diffusion-flux}\\). If \\(A_p=\rho_0U^2/4\\), this becomes
<script type="math/tex; mode=display">
\begin{align}
  \bm{F}_{\mathrm r}^\star
  &=
  -\frac{cU^2k}{2\kappa_R}
  \begin{pmatrix}
    \sin(2kx)\\
    \sin(2ky)\\
    0
  \end{pmatrix},
  \label{eq:canonical-frad}\\
  S_E^\star
  &=
  -\frac{cU^2k^2}{\kappa_R}
  \left[\cos(2kx)+\cos(2ky)\right].
  \label{eq:canonical-source}
\end{align}
</script> The leading-order opacity-coupled P1 construction gives
the same scalar source after replacing \\(\kappa_R\\) by \\(\kappa_F\\), but P1
has an independent radiation flux variable and therefore requires the
separate balance discussed above.

For a strictly Eulerian steady Taylor--Green field, use instead
\\(\Phi=p^\star-\Pi_{\mathrm{TG}}\\). With the pressure in
Eq. \\(\\ref{eq:generic-pressure}\\), <script type="math/tex; mode=display">
\begin{equation}
  \Phi
  =
  p_0 +
  \left(A_p-\frac{\rho_0U^2}{4}\right)
  \left[\cos(2kx)+\cos(2ky)\right],
  \label{eq:eulerian-potential-explicit}
\end{equation}
</script> and
Eq. \\(\\ref{eq:diffusion-source-constant-d}\\) applies with \\(A_p\\) replaced
by \\(A_p-\rho_0U^2/4\\). In the canonical Taylor--Green case
\\(A_p=\rho_0U^2/4\\), \\(\Phi\\) is constant and the radiation force and
manufactured radiation energy source both vanish.

## Why the density remains constant

The mass equation can be written in material form as <script type="math/tex; mode=display">
\begin{equation}
  \frac{\mathrm{d}\rho}{\mathrm{d}t} = -\rho\,\nabla\!\cdot\bm{v}.
  \label{eq:material-mass}
\end{equation}
</script> For the prescribed Taylor--Green velocity,
\\(\nabla\!\cdot\bm{v}_{\mathrm{TG}}=0\\) by
Eq. \\(\\ref{eq:tgv-divergence}\\). Therefore <script type="math/tex; mode=display">
\begin{equation}
  \frac{\mathrm{d}\rho}{\mathrm{d}t}=0.
  \label{eq:density-material-constant}
\end{equation}
</script> If \\(\rho=\rho_0\\) initially, then \\(\rho=\rho_0\\) for all
time. Equivalently, in Eulerian form, <script type="math/tex; mode=display">
\begin{equation}
  \partial_t\rho^\star + \nabla\!\cdot(\rho^\star\bm{v}_{\mathrm{TG}})
  =
  0+\rho_0\nabla\!\cdot\bm{v}_{\mathrm{TG}}
  =
  0.
  \label{eq:density-eulerian-constant}
\end{equation}
</script> This proves the constant-density part of the
manufactured solution.

## Why the velocity remains unchanged

In the diffusion formulation, the pressure-canceling construction is
immediate. With \\(\Phi=p^\star\\), <script type="math/tex; mode=display">
\begin{equation}
  p^\star + \frac{E_{\mathrm r}^\star}{3}
  =
  p^\star + \frac{E_{\mathrm r,0}-3p^\star}{3}
  =
  \frac{E_{\mathrm r,0}}{3},
  \label{eq:total-pressure-constant}
\end{equation}
</script> which is spatially constant. The diffusion momentum
equation
\\(\\ref{eq:diffusion-momentum}\\) therefore gives <script type="math/tex; mode=display">
\begin{equation}
  \rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}
  =
  -\nabla\left(p^\star+\frac{E_{\mathrm r}^\star}{3}\right)
  =
  \bm{0}.
  \label{eq:velocity-diffusion-proof}
\end{equation}
</script> Thus the velocity attached to each Lagrangian fluid
element remains unchanged, as specified in the MAPP description.

For the pressure-canceling Lagrangian construction, the manufactured
radiation force is \\(\bm{G}^\star=\nabla p^\star\\). Substitution into the
primitive momentum equation gives <script type="math/tex; mode=display">
\begin{equation}
  \rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}+\nabla p^\star=\nabla p^\star,
  \qquad
  \frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\bm{0}.
  \label{eq:velocity-lagrangian-proof}
\end{equation}
</script> This is the same statement written in Quokka's
four-force sign convention.

For an Eulerian code test in which the exact velocity field is required
to be the same function of position at all times,
\\(\bm{v}^\star(x,y,t)=\bm{v}_{\mathrm{TG}}(x,y)\\), use the more general
force in
Eq. \\(\\ref{eq:eulerian-force}\\). Then <script type="math/tex; mode=display">
\begin{align}
  \rho_0\partial_t\bm{v}^\star
  &=
  \bm{G}^\star
  - \rho_0(\bm{v}^\star\cdot\nabla)\bm{v}^\star
  - \nabla p^\star \nonumber\\
  &=
  \nabla(p^\star-\Pi_{\mathrm{TG}})
  - \left[-\nabla\Pi_{\mathrm{TG}}\right]
  - \nabla p^\star \nonumber\\
  &= \bm{0}.
  \label{eq:velocity-eulerian-proof}
\end{align}
</script> This proves that the Eulerian velocity field is stationary
when the manufactured force includes both the hydrodynamic pressure
residual and the Taylor--Green advective acceleration.

## Reduced-speed-of-light form used by Quokka

When Quokka uses a reduced speed of light \\(\hat{c}\ne c\\), the radiation
transport terms in the stored radiation variables are scaled as
<script type="math/tex; mode=display">
\begin{align}
  \partial_tE_{\mathrm r}+ \nabla\!\cdot\left(\frac{\hat{c}}{c}\bm{F}_{\mathrm r}\right)
  &= -\hat{c}G^0+ S_{E,\hat{c}},
  \label{eq:rsla-energy}\\
  \partial_t\bm{F}_{\mathrm r}+ \nabla\!\cdot(\hat{c}c\mathsf{P}_{\mathrm r})
  &= -\hat{c}c\bm{G}+ \bm{S}_{F,\hat{c}}.
  \label{eq:rsla-flux}
\end{align}
</script> Under P1, <script type="math/tex; mode=display">
\begin{align}
  S_{E,\hat{c}}^\star
  &=
  \partial_tE_{\mathrm r}^\star
  + \nabla\!\cdot\left(\frac{\hat{c}}{c}\bm{F}_{\mathrm r}^\star\right)
  + \hat{c}G^{0,\star},
  \label{eq:rsla-energy-source}\\
  \bm{S}_{F,\hat{c}}^\star
  &=
  \partial_t\bm{F}_{\mathrm r}^\star
  + \frac{\hat{c}c}{3}\nabla E_{\mathrm r}^\star
  + \hat{c}c\bm{G}^\star.
  \label{eq:rsla-flux-source}
\end{align}
</script> The same choice
\\(E_{\mathrm r}^\star=E_{\mathrm r,0}-3\Phi\\) makes
\\(\bm{S}_{F,\hat{c}}^\star=0\\) for steady fields. For the opacity-coupled
construction, \\(\bm{F}_{\mathrm r}^\star=c\nabla\Phi/(\rho_0\kappa_F)\\)
still gives the gas force
\\(\bm{G}^\star=\rho_0\kappa_F\bm{F}_{\mathrm r}^\star/c=\nabla\Phi\\), and
the stored energy source becomes <script type="math/tex; mode=display">
\begin{equation}
  S_{E,\hat{c}}^\star
  =
  \frac{\hat{c}}{\rho_0\kappa_F}\nabla^2\Phi.
  \label{eq:rsla-energy-source-final}
\end{equation}
</script> Quokka's thermal-group `radEnergySource` input is a
physical luminosity density and is internally multiplied by \\(\hat{c}/c\\),
so the value passed through that interface should be the full-speed
expression
\\(\nabla\!\cdot\bm{F}_{\mathrm r}^\star=c\nabla^2\Phi/(\rho_0\kappa_F)\\).
