# Cooling tables

This directory contains cooling/heating tables used by Quokka.  The tables are
not interchangeable: each one encodes a particular radiation field, shielding
prescription, abundance pattern, molecule treatment, and dust-grain model.  Users
should check that those assumptions are appropriate before using a table in a
simulation.

## `resample_ISRF_shielding_wide_density.h5`

This table is a contributed Cloudy table intended for tests that need an ISM
radiation field and densities above the range covered by the standard Grackle
Cloudy table.  It should not be treated as a general-purpose replacement for the
standard tables.  In particular, the high-density behavior depends on the
shielding and dust assumptions listed below.

### Cloudy version

The Cloudy input grid was run with a local Cloudy build from
`/public/home/yaoguangpei/cloudy`.  The source tree identifies itself as Cloudy
version `25.00` in `source/version.cpp` (`CLD_MAJOR = 25`, `CLD_MINOR = 0`).

### Input grid and generation script

The table was generated from a grid of Cloudy input files.
Each model has a name of the form `n_<density>_T_<temperature>.in`; for example:

```text
title n_1p000e00_T_1p000e04
hden -0.00000000
constant temperature linear 10000.00000000
stop temperature off
CMB
table ISM
abundances GASS10
metals 1 linear
grains ISM 1 linear
no CO molecules
cosmic ray background
iterate to convergence
stop thickness 2.00000000 parsecs
set save prefix "n_1p000e00_T_1p000e04"
save cooling last ".cool"
save heating last ".heat"
save molecules last ".mol"
```

The input files were generated with
`quokka/extern/cooling/generate_input_with_new.py`.  
The table used the following grid and physics settings:

- Hydrogen number density: `1e-6 <= n_H / cm^-3 <= 1e9`, sampled every `0.1` dex.
- Temperature: `10 <= T / K <= 1e9`, sampled every `0.05` dex.
- Radiation field: Cloudy `table ISM` plus the CMB.
- Abundances: Cloudy `abundances GASS10`; gas metallicity scale factor `1`.
- Cosmic rays: Cloudy `cosmic ray background`.
- Iteration: `iterate to convergence`.
- Stop temperature: `stop temperature off`, because each Cloudy run is a fixed-temperature model.
- Dust grains: Cloudy `grains ISM 1 linear` for `T <= 1e4 K`; grains are not included above `1e4 K`.
- Gas-dust collisional energy exchange: left at the Cloudy default.
- Molecules: CO is disabled by default with `no CO molecules`; CO is enabled only in the window `2.1 <= log10(n_H / cm^-3) <= 6.0` and `1.0 <= log10(T / K) <= 2.6`.
- H2 and grain-surface molecule physics are otherwise left at the Cloudy defaults.
- Species line cooling output was not saved for this table.

A representative command to regenerate the input grid is:

```bash
python3 generate_input_with_new.py \
  --logn-min -6 --logn-max 9 --logn-step 0.1 \
  --logT-min 1 --logT-max 9 --logT-step 0.05 \
  --target-NH 2.0e22 \
  --max-depth-pc 100 --min-depth-pc 1.0e-3 \
  --use-ism-field \
  --use-grains --grains-max-temp 1.0e4 \
  --molecule-mode no_co \
  --co-include-logn-min 2.1 --co-include-logn-max 6.0 \
  --co-include-logT-min 1.0 --co-include-logT-max 2.6 \
  --stop-temperature-off \
  --outdir ISM_with_co_more_wide_10av_target 
  
```

### Shielding and cloud-depth prescription

For each `(n_H, T)` grid point, the script computes a model depth from three
limits:

```text
L_Jeans  = sqrt(pi c_s^2 / (G rho))
L_shield = N_H,target / n_H
L_raw    = min(L_Jeans, L_shield, max_depth)
L_eff    = max(L_raw, min_depth)
```

For this table, `N_H,target = 2e22 cm^-2`, corresponding roughly to a maximum
shielding column of `A_V ~ 10` for a Galactic dust-to-gas ratio.  The depth is
also capped to the range `1e-3 pc <= L_eff <= 100 pc`.  The resulting Cloudy
input writes `stop thickness log10(L_eff / pc) parsecs`.

These shielding assumptions are problem-dependent.  They are reasonable for some
ISM applications, but they are not a universal model.

### Conversion to the Quokka HDF5 table

The raw Cloudy outputs were converted to a Quokka cooling table and then
resampled to the standard Quokka HDF5 layout.  The final file in this directory is
`resample_ISRF_shielding_wide_density.h5`, with datasets:

```text
grids/rho                  (150,)
grids/eint                 (150,)
data/cooling_rates         (150, 150)
data/temperatures          (150, 150)
data/pressures             (150, 150)
data/entropies             (150, 150)
data/sound_speeds          (150, 150)
```

The resampled density and internal-energy grid is the grid used by Quokka at run
time; it is not identical to the original Cloudy `(n_H, T)` sampling.

### Caveats

This table extends to high density by making explicit assumptions about dust,
shielding, molecules, and the radiation field.  Those assumptions can change the
cooling and heating rates substantially.  Before using this table, check whether
`table ISM`, the `A_V ~ 10` column cap, the `T <= 1e4 K` grain cutoff, the CO
window, solar GASS10 abundances, and the default Cloudy cosmic-ray background are
appropriate for the intended simulation.
More detailed information can be found in PR#1990:
https://github.com/quokka-astro/quokka/pull/1990

## Grackle cooling (resampled)

The units of the cooling rates in the Grackle tables are confusing. Document here what they actually are.

## Photoelectric heating

The file `photoelectric_heating_from_sfh.csv` is a table of the photoelectric heating rate in units of `erg / s / H / (Msun/kpc^2)` as a function of age (years), generated by the SLUG code. The weight function has been normalised so that if you define `dM` = change in stellar mass per unit area in units of `Msun/kpc^2`, then the photoelectric heating rate produced by those stars `dM` of age `t` is

```
d_Gamma_PE = w(t) * d_M
```

That is, the instantaneous photoelectric heating rate is given by

$$
\Gamma_{\rm PE} (t) = \int_{0}^{t} w(t - t') \Sigma_{\rm SF}(t') dt'.
$$

Here is how one could compute the PE heating rate in practice. If your box cross section is `area_kpc2`, you can just record in your table the time `t` (in years) and stellar mass `Mstar` (in solar mass) at each time, and then compute the instantaneous photoelectric heating rate as

```
Gamma_PE = sum_i * (Mstar_{i+1} - Mstar_i) / area_kpc2 * w(t_current - t_{i})
```

and get the photoelectric heating rate in units of `erg / s / H`. The normalisation has been chosen so that the solar neighbourhood SFR per unit area (`2.5 x 10^-3 Msun / kpc^2 / yr`) corresponds to `Gamma_PE = 2e-26 erg/s/H`.

The structure of the csv file follows Quokka's DataTable format (https://github.com/quokka-astro/quokka/pull/1373).
