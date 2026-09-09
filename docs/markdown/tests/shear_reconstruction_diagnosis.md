# Diagnosis of the 32 × 32 xPPM shear failure

The [source-free shear reproducer](shear_reconstruction.md), with pressure `2e-7`, density contrast 19999:1, CFL 0.3, artificial viscosity off, RK2, and **dual energy enabled**, exposes a face-density positivity failure in xPPM. Its final median can undo the preceding monotonization. HLLC then receives a negative density and generates a large finite energy flux. The resulting heat makes the final substep fail its post-update CFL check.

These measurements use the native double-precision AppleClang Release CPU executable, one rank, base commit `c8e0bcce7178c63768452eaf893a495ade8b1fb7` plus the problem driver. Temporary instrumentation dumped native states, reconstructed faces before and after flattening, stage fluxes, and cell signal speeds. All instrumentation and diagnostic interventions were removed after the experiments. This is a diagnosis, not an adopted solver fix.

## Rejected cells

Indices are zero-based; cell centers are `((i+0.5)/32, (j+0.5)/32)`. Attempt numbers include accepted substeps. The table lists a maximum-signal cell for each rejected attempt; symmetry-related cells can tie within roundoff.

| Attempt | Start time | dt | Cell (i,j) | Density after update | Velocity magnitude | Sound speed | Total signal speed |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 0 | 0.008546458 | (25,22) | 9.74209e-5 | 10.58650 | 0.05225 | 10.63875 |
| 2 | 0 | 0.004273229 | (25,22) | 9.85983e-5 | 3.30469 | 0.05272 | 3.35741 |
| 4 | 0.002136614 | 0.002136614 | (25,22) | 8.19999e-5 | 9.47383 | 0.06066 | 9.53449 |
| 6 | 0.003204922 | 0.001068307 | (30,31) | 7.77816e-5 | 9.89242 | 0.04731 | 9.93973 |
| 11 | 0.005341536 | 0.000534154 | (25,22) | 3.92526e-5 | 24.70565 | 0.04687 | 24.75251 |
| 16 | 0.006409843 | 0.000267077 | (9,6) | 2.01994e-5 | 40.93544 | 0.11154 | 41.04698 |
| 30 | 0.008145843 | 0.000133538 | (18,3) | 6.24834e-5 | 36.26256 | 64.18409 | 100.44665 |

The terminal attempt starts at `0.008145842644206723`, with dt `0.0001335384040033889` (original dt/64). The allowed signal speed is `1.1*CFL*dx/dt = 77.22497567`. Cells **(18,3)** and **(2,19)** have the symmetry-related terminal failure. Their centers are (0.578125, 0.109375) and (0.078125, 0.609375).

Early rejections are predominantly velocity growth in tenuous cells, and occur with classic PPM too: its first rejected maximum signal speed is 10.45855. They are not by themselves evidence of the xPPM-specific defect. The distinguishing terminal event is the large sound speed produced by the energy flux below. First-order flux correction activates elsewhere but does not repair this positive-density cell's corrupt energy update.

## Exact reconstruction at the responsible face

In stage 1 of attempt 30, the lower y-face of cell (18,3) receives its left state from the upper edge of donor cell **(18,2)**. That donor's five density averages, ordered along y, are all positive:

```text
q[-2] = 0.02234486969211808
q[-1] = 0.0013623592269931292
q[ 0] = 0.00014408067275252544
q[+1] = 0.00007585230405546217
q[+2] = 1.2840798703969565
```

The native face value agrees with an independent replay of the formulas in `src/hyperbolic_system.hpp`. This case takes the **steep-gradient branch** of xPPM:

| Operation | Donor's upper density edge |
|---|---:|
| Unlimited fifth-order reconstruction | -0.06360734563214464 |
| Initial `MonotonizeEdges` | +0.00007585230405546217 |
| WENO reconstruction | -0.00007417040733369239 |
| `ComputeSteepPPM`, before median | +0.0000644809092726183 |
| Median of WENO, steep PPM, unlimited | -0.00007417040733369239 |
| Second `MonotonizeEdges` | +0.00007585230405546217 |
| Final median of monotonized, WENO, unlimited | **-0.00007417040733369239** |

In particular, the final assignment (line 704 in the measured source) is:

```cpp
new_a_plus = median(a_plus_mppm, a_plus_weno, a_plus);
```

Two of its three candidates are negative, so it discards the positive monotonized edge. Shock flattening leaves this face density unchanged.

The WENO weights are approximately `(0.00123418, 0.99876526, 5.59321e-7)`: almost all weight goes to the central quadratic. Its moments are `q_x = -0.0006321256499144941` and `q_xx = 0.0005868704692261751`. The edge `q + q_x/2 + q_xx/6` is negative. Smoothness weighting therefore does not ensure a positive edge even though the input averages are positive. Classic PPM applied to this **identical** stencil gives the positive upper edge `7.585230405546217e-5`.

This is not the first invalid face. Attempt 4, stage 2, first produces negative left and right densities at x-face (26,0), and its periodic symmetry partner (10,16). The left donor uses the steep branch and the right donor uses the extremum branch. Thus both final median branches can undo the positivity supplied by monotonization. Attempt 5 also contains negative face densities and is accepted: positive updated cell densities do not certify admissible Riemann inputs.

## HLLC and the energy update

At the offending y-face, after flattening, the native inputs are:

| Quantity | Left | Right |
|---|---:|---:|
| Density | -7.41704e-5 | +7.58523e-5 |
| Normal velocity | -3.87159 | -42.75991 |
| Transverse velocity | +0.72603 | +0.43992 |
| Specific internal energy | 31.59131 | 20.51280 |
| EOS pressure returned to HLLC | 1.26365e-99 | 0.0006223773 |

The default reconstruction variable is specific internal energy. The EOS clamps its local density copy, but the density subsequently used by HLLC remains negative. Consequently the left reconstructed total energy density is negative and the Roe square root of density is NaN. On this CPU build, the endpoint-first `std::min/max` operations still select finite endpoint wave estimates: `S_L = -31.41067`, `S_R = -16.03765`. The computed contact speed is `S_star = 5076.15908`, with star pressure `10.37639`, outside the admissible wave configuration. HLLC selects its left star flux and returns an energy flux of **+323.84511**.

The relevant operations are in `src/hydro/HLLC.hpp`: Roe density square roots, contact speed, star pressure, and the left-star energy flux. This flux is generated outside the solver's physical input domain; it is not an ordinary high-Mach negative pressure inferred from a valid cell's total energy.

For receiving cell (18,3), the measured energy densities are:

| State | Total energy | Kinetic energy | Thermal energy from their difference |
|---|---:|---:|---:|
| Before attempt | 0.07090785 | 0.06935190 | 0.00155594 |
| Stage 1, before floors/dual synchronization | 1.45476529 | 0.06371093 | 1.39105436 |
| Final RK2 state | 0.50073655 | 0.04108202 | 0.45965454 |

The signed stage-1 sum of energy fluxes into the cell is `323.84350688`; multiplying by `dt/dx = 0.004273228928` gives the observed energy increase `1.38385744`. The final RK2 averaged fluxes give the net increase `0.42982870`. The spurious heat therefore exists **before** dual-energy synchronization. Synchronization then accepts the positive thermal energy, as designed. At the final density `6.24834e-5`, pressure `0.18386181` produces sound speed `64.18409`. Adding velocity magnitude `36.26256` exceeds the allowed signal speed.

## Causal interventions

Two temporary interventions test the proposed chain without changing RK2, CFL, artificial viscosity, or dual energy:

1. Replay the exact native pre-attempt-30 conserved snapshot with its exact time and dt. The original reconstruction fails again, with maximum signal speed 100.44665. Replace only the left density at the two symmetry-related stage-1 y-faces (18,3) and (2,19) with their donor cell densities. **That same substep passes**, with maximum signal speed 54.72132, below 77.22498. Repairing only one face leaves its symmetry partner failing.
2. In a fresh full run, replace only nonpositive final xPPM density edges with the corresponding `MonotonizeEdges` values. Leave all positive edges and other reconstructed variables unchanged. **The run completes to t = 3**, in 483 coarse steps.

These interventions establish the role of invalid face density in this reproducer. They are not a proof of general positivity, entropy stability, or robustness for other flows, and the temporary guard is not a proposed finished fix. In particular, this diagnosis does not explain the separate 64 × 64 PPM failure.

Local evidence is retained under `build/2d/shear-diagnosis/`: native traces in `xppm/` and `ppm/`, `limiter_cases.json`, `limiter_audit.py`, `read_trace.py`, `inspect_cell.py`, archived instrumented sources and patches, `HydroShearRepro.instrumented`, `replay.py`, replay logs, and the full diagnostic-guard run in `density-repaired/`. Run `python3 build/2d/shear-diagnosis/replay.py` to repeat the original/repaired snapshot comparison. These build artifacts are local and are not versioned regression fixtures.

## Comparison with Rider, Greenough & Kamm (2007)

The supplied `rider2007.pdf` resolves the attribution question. Algorithm 2.1.2, step 4(c)(iv), printed page 1833 (PDF page 7), explicitly takes the final median of the monotonized steepened value, the WENO value, and the original high-order value. Quokka's final median follows that prescription. Step 4(b)(iii) uses the analogous median for the extremum branch. Restoring a nonmonotone value at this point is intentional in the published reconstruction; that operation is not a transcription error.

The published scalar reconstruction itself can return a negative value from positive averages. Evaluating Algorithm 2.2.4 exactly as printed on page 1834 (PDF page 8), using the captured donor density stencil, gives:

| Quantity | Left-biased candidate | Centered candidate | Right-biased candidate |
|---|---:|---:|---:|
| Third-order upper edge | +0.006123018699 | -0.000081708543 | -0.213902074589 |
| Final mapped weight | 0.006432488865 | 0.993565700225 | 0.000001810910 |

The paper uses optimal weights `(0.1, 0.6, 0.3)`, inverse smoothness indicators with delta `1e-40`, followed by the mapping and renormalization in step 4. Its resulting upper WENO edge is **-4.2183913170449514e-5**. Quokka's symmetric WENO-Z value is **-7.417040733369239e-5**. Changing to the paper's WENO therefore does not eliminate the negative candidate on this stencil.

The scalar audit also applies the paper's full limiter with its own initial edge options, not only with Quokka's fifth-order initial edges:

| Initial edges | Limiter branch | Final upper edge with paper's WENO and limiter |
|---|---|---:|
| Quokka fifth-order | Steep | -4.218391317e-5 |
| Paper Algorithm 2.2.1, fourth-order | Extremum | -4.218391317e-5 |
| Paper Algorithm 2.2.2, seventh-order | Steep | -4.218391317e-5 |

For the seventh-order evaluation, the two additional positive density averages are `q[-3] = 0.29559682942339255` and `q[+3] = 1.850405303359926`. In that case, the original upper edge is `-0.09951663514729106` and the monotonized upper edge is `+7.585230405546217e-5`. The final median selects the negative WENO candidate. Thus a lack of scalar reconstruction positivity is demonstrable using the paper's own choices.

Other implementation differences exist: Quokka uses fifth-order initial edges and symmetric WENO-Z instead of the paper's mapped WENO; its steepened slope cap uses `2*MC` instead of the printed `2*median(0, forward_difference, backward_difference)`. In the extremum branch, the paper's second bound in 4(b)(ii) prints the pre-monotonized WENO edge as its third argument, whereas Quokka's helper uses the neighbor-bounded intermediate edge. The audit follows the printed expression. These differences do not change the negative-edge conclusion for the captured stencil.

This is a counterexample to positivity of the **scalar reconstruction**, not a reproduction of the original paper's entire Euler solver. The paper describes reconstruction in characteristic variables and characteristic time centering; Quokka's primitive-variable reconstruction with RK2 and HLLC is a different complete scheme. The results therefore establish an inherited positivity limitation in the reconstruction and a missing admissibility safeguard in Quokka's use of it, without establishing that the authors' complete code would crash on this test.

The formula audit is saved as `build/2d/shear-diagnosis/compare_rider.py`, with numerical output in `rider_comparison.json`. No solver code was changed for this comparison.
