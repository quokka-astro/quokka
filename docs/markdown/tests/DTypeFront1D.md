# 1D dust reprocessing test

This problem is a one-dimensional, planar test of dust absorbing a beamed optical source and re-emitting it in the infrared. A constant optical photon flux is injected at the \\(x = 0\\) boundary into a uniform, cold, dusty hydrogen slab. Dust absorbs the beam, heats until emission balances absorption, and radiates the energy back out in a band where the dust is nearly transparent, so the reprocessed light escapes. The test exercises a mixed thermal + chemical multigroup configuration, the directed (beamed) radiation source, and the separate dust-temperature solver.

## Radiation groups

Three groups are used, with frequency boundaries \\(\{10^{8},\, 10^{14},\, 3.29\times10^{15},\, 10^{19}\}\\) Hz:

| group | band | role |
|---|---|---|
| 0 | IR, below \\(10^{14}\\) Hz | unsourced; filled only by dust re-emission |
| 1 | optical, \\(10^{14}\\) Hz to the Lyman edge | carries the injected beam |
| 2 | ionizing, above the Lyman edge | chemistry band; sourced, transparent to dust, absorbed by photoionization |

Chemistry bands must always be the last groups. The two outermost boundaries should be read as \\(0\\) and \\(\infty\\) rather than as physical band edges: `ComputePlanckEnergyFractions` accumulates the Planck integral from zero, so group 0 receives the whole blackbody below \\(10^{14}\\) Hz whatever the first entry says, and emission above the first chemistry-band boundary is dropped rather than assigned to group 2, so the last entry never enters the emission budget.

## Initial conditions

The domain is \\(x \in [0, L]\\) with \\(L = 2 \times 10^{19}\\) cm, resolved by 128 uniform cells. The gas is uniform, neutral atomic hydrogen with number density \\(n_0 = 100 \, \text{cm}^{-3}\\) at \\(T_0 = 100\\) K. Every radiation group is initialized to the negligible energy-density floor \\(E_{\text{rad,floor}}\\). The reduced speed of light is \\(\hat c = c/1000\\) and the run integrates to \\(t_{\text{end}} = 4 \times 10^{11}\\) s, which keeps the light front at about 60% of the domain.

## Boundary conditions and source

Reflecting boundaries are applied at \\(x = 0\\) and \\(x = L\\). The source is imposed in the first cell adjacent to \\(x = 0\\). Because the internal source array carries a luminosity volume density (\\(\text{erg}\,\text{s}^{-1}\,\text{cm}^{-3}\\)), the injected value for a photon flux \\(F\\) is

<script type="math/tex; mode=display">
S = \frac{F \, E_\gamma}{\Delta x} \, ,
</script>

where \\(E_\gamma\\) is the mean photon energy and \\(\Delta x = L/128\\). The companion radiation *flux* source is set to \\(c\,S\\), so the injected radiation has reduced flux \\(f = F_{\text{rad}}/(c E_{\text{rad}}) = 1\\) and free-streams along \\(+x\\) instead of spreading isotropically.

A thermal group's source is scaled internally by \\(\hat c / c\\) and a chemistry band's is not, so the two shipped fluxes differ by exactly that factor and deliver equal energy: \\(F_{\text{opt}} = 10^{13}\\) and \\(F_{\text{ion}} = 10^{10}\,\text{cm}^{-2}\,\text{s}^{-1}\\). Photochemistry is enabled, so the ionizing band is not merely transported: it is absorbed by photoionization, and the resulting front stalls at its Strömgren column well inside \\(\hat c\, t\\).

## Dust model

Each thermal group carries a constant gray dust opacity, \\(\kappa_{\text{IR}} = 10\\) and \\(\kappa_{\text{opt}} = 10^{3}\,\text{cm}^{2}\,\text{g}^{-1}\\), giving domain optical depths of \\(0.033\\) and \\(3.3\\). Dust is therefore optically thick to the incoming light and thin to what it re-emits, which is what makes the reprocessing one-way. Opacity in Quokka is pure absorption, so an opaque group also emits its share of the local blackbody, and that is what supplies the re-emission.

A separate dust temperature is solved for, with the gas–dust collisional coupling switched off (`radiation.dust_gas_interaction_coeff = 0`). The dust is then fixed purely by radiative equilibrium with the local radiation field,

<script type="math/tex; mode=display">
a \, T_d^4 = E_{\text{IR}} + \frac{\kappa_{\text{opt}}}{\kappa_{\text{IR}}} \, E_{\text{opt}} \, ,
</script>

which gives \\(T_d \approx 130\\) K for these parameters. At that temperature \\(h\nu/(k T_d) = 37\\) at the IR/optical boundary, so the Planck function has nothing left above the boundary: essentially all re-emission lands in the IR group and none returns to the optical one. Note that this decoupling is thermal only — radiation momentum is a separate channel and is still deposited, so the beam accelerates the gas to \\(\sim 8 \times 10^{6}\,\text{cm}\,\text{s}^{-1}\\) and evacuates the cells nearest the source.

## Analytic solution

Because the dust returns nothing to the optical band, behind the light front the optical group is in pure attenuation,

<script type="math/tex; mode=display">
E_{\text{opt}}(x) = \frac{F_{\text{opt}} E_\gamma}{c} \, e^{-\kappa_{\text{opt}} \rho x} \quad (x < \hat c \, t) \, ,
</script>

so the surviving unprocessed energy has the closed form

<script type="math/tex; mode=display">
\int_0^{\hat c t} E_{\text{opt}} \, dx = \frac{F_{\text{opt}} E_\gamma}{c} \, \frac{1 - e^{-\tau_f}}{\kappa_{\text{opt}} \rho} \, , \qquad \tau_f = \kappa_{\text{opt}} \, \rho \, \hat c \, t \, .
</script>

Everything the optical band loses reappears in the IR, so the two thermal bands together must retain the energy injected into the optical one. The ionizing band is budgeted separately, since photoionization drains it: photons injected equal photons still in the field, plus the ionized column, plus recombinations.

## Answer check

The test passes if all of the following hold. Measured values at the reference resolution are given in the last column.

| # | check | tolerance | measured |
|---|---|---|---|
| 1 | IR + optical equals the energy injected into the optical band | 1% | 1.0000000 |
| 2 | optical light front sits at \\(\hat c \, t\\) | 10% | 0.33% |
| 3 | reduced flux at the injection cell, \\(f \in [0.9,\, 1+10^{-6}]\\) | — | \\(1 - f = 3\times10^{-15}\\) |
| 4a | fraction reprocessed into the IR | \\(> 0.25\\) | 0.552 |
| 4b | surviving optical energy against the \\(e^{-\tau}\\) integral above | 10% | 1.039 |
| 5a | ionizing band is depleted, surviving fraction in \\([0.02, 1)\\) | — | 0.198 |
| 5b | photons absorbed from the ionizing band cover the ionized column | — | \\(3.2\times10^{21} \ge 3.0\times10^{20}\\) |

Check 3 is the one that exercises the directed source: deleting the flux source gives \\(f = 0.09\\), and scaling it up by \\(c/\hat c\\) gives \\(f = 1000\\). It is needed because check 2 barely discriminates on its own — with the flux source removed the front still reaches 90% of \\(\hat c\, t\\), since the M1 closure lets the leading edge of an isotropic pulse free-stream anyway.

Check 5 pins the chemistry-band source scaling, which is not multiplied by \\(\hat c / c\\) the way a thermal band's is. The budget inequality 5b alone does not catch a mis-scaling, because fewer photons produce a proportionally smaller ionized column and the inequality still holds; the lower bound in 5a is what catches it, since the ionization front stalls at its Strömgren column well inside \\(\hat c\, t\\) and always leaves a stable fraction of the photons in flight. Mis-scaling the source by \\(c/\hat c\\) collapses the surviving fraction from 0.198 to \\(8\times10^{-5}\\).

The front position is located on the optical band using a threshold of 5% of the unattenuated plateau, rather than the more natural 50%, because dust absorption thins the beam by \\(e^{-\tau} \approx 0.14\\) before it reaches the front; a 50% threshold would report the absorption depth instead.
