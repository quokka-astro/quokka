# 1D D-type ionization front test

This problem is a one-dimensional, planar version of the D-type ionization front test (a "Marshak problem with ionizing photons"). A constant ionizing photon flux \\(F\\) (photons \\(\text{cm}^{-2}\,\text{s}^{-1}\\)) is injected at the \\(x = 0\\) boundary into a uniform, initially neutral hydrogen slab. With photochemistry and photoheating enabled, the gas ionizes, heats to \\(\sim 10^4\\) K, becomes over-pressured relative to the cold ambient medium, and drives a planar D-type ionization front into the neutral gas. The numerical front position is compared against an analytic planar D-type solution, the planar analog of the classical spherical Spitzer solution.

## Initial conditions

The domain is \\(x \in [0, L]\\) with \\(L = 2 \times 10^{19}\\) cm, resolved by 128 uniform cells. The gas is uniform, neutral atomic hydrogen with number density \\(n_0 = 100 \, \text{cm}^{-3}\\) at a cold temperature \\(T_0 = 100\\) K (species: \\(n_{\text{HI}} = n_0\\), with trace \\(n_e = n_{\text{HII}} = 10^{-10}\,\text{cm}^{-3}\\)). The radiation field is initialized to the (negligible) energy-density floor \\(E_{\text{rad,floor}}\\) in every cell.

## Boundary conditions

Reflecting boundaries are applied at \\(x = 0\\) and \\(x = L\\). The ionizing source is imposed as a radiation-energy source term in the first cell adjacent to \\(x = 0\\): since the internal source array carries a luminosity volume density (\\(\text{erg}\,\text{s}^{-1}\,\text{cm}^{-3}\\)), the injected value is

<script type="math/tex; mode=display">
S = \frac{F \, E_\gamma}{\Delta x} \, ,
</script>

where \\(E_\gamma = \tfrac{1}{2}(\nu_{\text{lo}} + \nu_{\text{hi}}) h\\) is the mean photon energy of the single ionizing band and \\(\Delta x = L/128\\) is the cell width. The run uses \\(F = 10^{10}\,\text{cm}^{-2}\,\text{s}^{-1}\\) and a reduced speed of light \\(\hat c = c/1000\\).

## Analytic solution

Let \\(x_i(t)\\) be the length of the ionized column, \\(n_i(t)\\) its (uniform) number density, \\(\alpha_B\\) the case-B recombination coefficient, and \\(c_i\\) the isothermal sound speed of the ionized gas.

**Photon conservation.** In balance, the incident flux equals the total recombinations in the ionized column,

<script type="math/tex; mode=display">
F = \alpha_B \, n_i^2 \, x_i \, ,
</script>

which defines the initial (undisturbed-density) Strömgren length and the density during expansion,

<script type="math/tex; mode=display">
x_S = \frac{F}{\alpha_B \, n_0^2} \, , \qquad n_i = n_0 \sqrt{x_S / x_i} \, .
</script>

**Momentum balance.** The ionized gas (pressure \\(\rho_i c_i^2\\)) drives a shock that sweeps the ambient gas into a thin shell of column \\(\rho_0 x_i\\). Momentum conservation for the shell reads

<script type="math/tex; mode=display">
\frac{d}{dt}\!\left( \rho_0 \, x_i \, \frac{dx_i}{dt} \right) = \rho_i \, c_i^2 = \rho_0 \, c_i^2 \sqrt{x_S / x_i} \, .
</script>

Seeking the self-similar solution \\(x_i \propto t^{4/5}\\) reduces this to

<script type="math/tex; mode=display">
\frac{dx_i}{dt} = K \, c_i \left( \frac{x_S}{x_i} \right)^{1/4} , \qquad K = \sqrt{4/3} \, ,
</script>

where the factor \\(K = \sqrt{4/3}\\) is the momentum-conserving amplitude (the planar Hosokawa–Inutsuka correction; the cruder "ram-pressure = thermal-pressure" closure gives \\(K = 1\\)). Integrating with \\(x_i(0) = x_S\\) gives the analytic front position

<script type="math/tex; mode=display">
\boxed{\; x_i(t) = x_S \left[ 1 + \frac{5}{4} \, K \, \frac{c_i \, t}{x_S} \right]^{4/5} \;}
</script>

This is the planar analog of the spherical Spitzer law \\(r_i = r_S \left[1 + \tfrac{7}{4} c_i t / r_S\right]^{4/7}\\); the exponent changes from \\(4/7\\) to \\(4/5\\) because \\(n_i \propto x^{-1/2}\\) in planar geometry versus \\(n_i \propto r^{-3/2}\\) in spherical geometry.

The temperature-dependent quantities are evaluated at the ionized-region temperature \\(T_i\\) **measured from the simulation** (the median temperature of ionized cells, \\(x_{\text{HII}} > 0.9\\)), so that the reference solution is not derived from an assumed heating/cooling equilibrium:

<script type="math/tex; mode=display">
\alpha_B(T_i) = 2.6 \times 10^{-13} \left( \frac{T_i}{10^4 \, \text{K}} \right)^{-0.7} \text{cm}^3\,\text{s}^{-1} , \qquad c_i = \sqrt{\frac{2 k_B T_i}{m_H}} \, .
</script>

## Answer check

The numerical front position is the effective ionized length, the integral of the ionized fraction over the domain,

<script type="math/tex; mode=display">
x_{\text{eff}} = \int_0^L \left(1 - x_{\text{HI}}\right) dx = \sum_{\text{cells}} \left(1 - x_{\text{HI}}\right) \Delta x , \qquad x_{\text{HI}} = \frac{n_{\text{HI}}}{n_{\text{HI}} + n_{\text{HII}}} \, .
</script>

Because the simulation starts fully neutral, the R-type front needs a formation time to build the Strömgren column; the analytic curve is therefore shifted by \\(t_{\text{form}}\\), the first time \\(x_{\text{eff}}\\) reaches \\(x_S\\). The run integrates to \\(t_{\text{end}} = 6 \times 10^{12}\\) s (\\(\sim 2\\) expansion times), keeping the front well inside the domain. The test passes if

1. the effective ionized length matches the analytic solution, \\(\left| x_{\text{eff}} - x_i(t_{\text{end}} - t_{\text{form}}) \right| / x_i < 15\%\\); and
2. the measured ionized-cavity temperature lies in the physical range \\(5000\,\text{K} < T_i < 20000\,\text{K}\\).

At the reference resolution the front expands to \\(\sim 2.7\, x_S\\) (confirming genuine D-type expansion rather than R-type saturation) and \\(x_{\text{eff}}\\) tracks \\(x_i(t)\\) to \\(\sim 5\%\\) throughout the expansion, with \\(T_i \approx 1.05 \times 10^4\\) K.
