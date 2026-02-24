## Add runtime parameter `particles.SN_p_terminal` for SN feedback

### Summary

- Replaces the hard-coded SN terminal momentum `p_snr_0 = 2.8e5 M_sun km/s` with a runtime parameter `particles.SN_p_terminal` (default: canonical value).
- Scales the shell-formation mass `M_sf` as `M_sf_scaled = M_sf_canonical * (p_snr_0 / p_snr_0_canonical)^2` so that the kinetic energy of the expanding shell, `p_snr^2 / (2 M_sf)`, is invariant when `p_snr_0` is changed.

### Changed files

- `src/particles/particle_types.hpp`: added `SN_p_terminal_canonical` (constexpr) and `SN_p_terminal` (inline runtime variable), and `pp.query("SN_p_terminal", SN_p_terminal)` in `particleParmParse()`.
- `src/particles/particle_deposition.hpp`:
  - In the outer SN deposition function: replaced `constexpr double p_snr_0 = ...` with `const double p_snr_0 = quokka::SN_p_terminal`.
  - In `depositThermalKineticMomentumSNR`: replaced fixed `M_sf` with `M_sf_canonical * n_H^{-0.26} * (p_snr_0 / p_snr_0_canonical)^2`.

### Usage

Set in the input file:
```
particles.SN_p_terminal = <value_in_cgs>
```
If omitted, defaults to the canonical value `2.8e5 M_sun km/s ≈ 5.57e43 g cm/s`.
