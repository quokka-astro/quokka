# TallBoxRadiation

`TallBoxRadiation` is `TallBoxSf` — a 1x1x4 kpc galactic patch with self-consistent star formation, SN feedback, self-gravity, turbulent driving and tabulated cooling — with three-group radiation transport added on top. The initial vertical density profile, its solver (`solve_density_profile.py`) and the IC table are shared with `TallBoxSf`; see `src/problems/TallBoxSf/README.md` for that part.

## Radiation groups

The three groups are bounded by $10^{-6}$, 6, 11.2 and 13.6 eV:

| group | name | band | dust opacity |
|---|---|---|---|
| 0 | IR | $10^{-6}$ - 6 eV | 10 cm$^2$ g$^{-1}$ |
| 1 | FUV | 6 - 11.2 eV | $2 \times 10^4$ cm$^2$ g$^{-1}$ |
| 2 | LW | 11.2 - 13.6 eV | $4 \times 10^4$ cm$^2$ g$^{-1}$ |

Group 0 collects everything from the radio up to the Balmer edge, which is where a star puts most of its bolometric luminosity and where dust re-emits. Group 1 is the band that drives photoelectric heating of neutral gas, and group 2 is the Lyman-Werner band that dissociates H2. The Lyman continuum above 13.6 eV is deliberately left out, because this problem has no photochemistry and ionizing photons would have nothing to do except heat the dust.

Opacity is gray within each group (`OpacityModel::piecewise_constant_opacity`). The IR value is the usual Rosseland-mean dust opacity of the diffuse ISM; the two UV values are about three orders of magnitude larger, so FUV and LW light is absorbed within a few pc of a young star while the reprocessed IR escapes.

## Dust temperature

`ISM_Traits<TheProblem>::enable_dust_gas_thermal_coupling_model = true` turns on the separate dust-temperature solver, and `radiation.dust_gas_interaction_coeff = 0.0` in the input file switches the gas-dust collisional term off. The solver then takes its decoupled branch: the dust temperature is set purely by radiative equilibrium with the local radiation field and exchanges no energy with the gas. This mirrors `DTypeFront1D`. Note that this is thermal decoupling only — radiation momentum is still deposited, so absorbed UV light still pushes on the gas.

## Stellar luminosity table

The `StochasticStellarPop` particles radiate according to a slug2 isochrone table generated with the script added in PR #2204:

```
export slug2_path=/path/to/slug2
python3 scripts/python/slug_luminosity_table_for_quokka.py inputs/slug_IR_FUV_LW.csv \
    --eV 1e-6 6 --eV 6 11.2 --eV 11.2 13.6 \
    --m0 2.1 --m1 120 --nm 100 --t0 1e5 --t1 2e8 --nt 100
```

The table spans stellar ages $10^5$ - $2 \times 10^8$ yr and initial masses 2.1 - 120 $M_\odot$ on the `mist_2016_vvcrit_40` track set, log-spaced on both axes with 100 points each. The band edges given to the script are exactly the group boundaries in `RadSystem_Traits<TheProblem>::radBoundaries`, and the script refuses band sequences that are not contiguous and increasing in photon energy.

Luminosity is looked up per particle as a function of (age, current mass) with `OutOfBounds::clamp`, so particles outside the table's mass range take the value at the nearest edge. The high-mass stars that this problem samples from the IMF lie in 9 - 120 $M_\odot$ and so are fully covered; low-mass composite particles are heavier than 120 $M_\odot$ and are clamped to the top of the mass axis — see the PR description for the consequences.

## Reduced speed of light

`c_hat = 1000` km/s. At the base resolution ($\Delta x = 94$ pc) a radiation substep is $\mathrm{cfl} \, \Delta x / \hat{c} = 2.8 \times 10^{11}$ s against a hydro step of $\sim 1.9 \times 10^{12}$ s set by the sound speed of the $10^6$ K phase, so about 7 radiation substeps per hydro step — under `maxSubsteps_ = 10`, which means the hydro step is not throttled. $\hat{c}$ also crosses the 4 kpc box in 3.8 Myr, much less than the 64 Myr run time.
