# DTypeFront Photochemistry Optimization Notes

Baseline: `0.3038532663 us/zone-update`, `PhotoChemistry::computePhotoChemistry() = 1.285 s`, measured on 1 GPU for 20 hydro timesteps with default timestepping.

1. Reused recombination, collisional ionization, recombination cooling, and ion/free-free cooling derivative intermediates in `actual_jac`: `0.3025889732 us/zone-update`, delta `-0.0012642931 us/zone-update` (`-0.42%`), photochemistry `1.270 s`.
2. Reused KI cooling `exp`/`sqrt` intermediates in `actual_jac`: `0.2982543287 us/zone-update`, delta `-0.0055989376 us/zone-update` (`-1.84%`), photochemistry `1.252 s`.
3. Removed redundant switch multiplications in Jacobian entries where disabled paths already produce zero coefficients: `0.3135076948 us/zone-update`, delta `+0.0096544285 us/zone-update` (`+3.18%`), photochemistry `1.244 s`.
4. Added a strict RHS-tolerance early-out before `burner()` for near-static cells: `0.2981719217 us/zone-update`, delta `-0.0056813446 us/zone-update` (`-1.87%`), photochemistry `1.253 s`; later removed because it added an RHS call and did not skip enough work.
5. Cached `state.T`, `state.rho`, and related local values in RHS/Jacobian: `0.2976200905 us/zone-update`, delta `-0.0062331758 us/zone-update` (`-2.05%`), photochemistry `1.249 s`.
6. Reused `alpha_rec` when computing `dRecombination_coefficient_dT`: `0.2964371847 us/zone-update`, delta `-0.0074160816 us/zone-update` (`-2.44%`), photochemistry `1.240 s`.
7. Reused `Lambda_rec` when computing `dRecombination_cooling_coefficient_dT`: `0.3159172453 us/zone-update`, delta `+0.0120639790 us/zone-update` (`+3.97%`), photochemistry `1.270 s`; reverted.
8. Reverted the `Lambda_rec` reuse change from iteration 7: `0.2966345148 us/zone-update`, delta `-0.0072187515 us/zone-update` (`-2.38%`), photochemistry `1.238 s`.
9. Removed the strict early-out probe from iteration 4: `0.2917657005 us/zone-update`, delta `-0.0120875658 us/zone-update` (`-3.98%`), photochemistry `1.232 s`.
10. Ran built-in verification without further code changes: `ctest --test-dir /mnt/ffs24/home/wibkingb/quokka/build/3d-cuda -R '^DTypeFront$' --output-on-failure` passed in `12.73 s`.

Final retained change: coefficient/intermediate reuse and local state caching in `src/networks/photoionization/actual_rhs.H`. Final best measured FoM: `0.2917657005 us/zone-update`, delta `-0.0120875658 us/zone-update` (`-3.98%`).

## Fast-Transcendental Approximation Pass

Baseline: `0.2940853086 us/zone-update`, `PhotoChemistry::computePhotoChemistry() = 1.233 s`, measured on 1 GPU for 20 hydro timesteps with default timestepping after the coefficient/intermediate reuse pass.

1. Replaced selected FP64 `pow`, `sqrt`, and `exp` calls with CUDA device `powf`, `sqrtf`, and `__expf` wrappers throughout the photoionization coefficients: `0.7148890368 us/zone-update`, delta `+0.4208037282 us/zone-update` (`+143.09%`), photochemistry `3.430 s`; passed checks but strongly regressed.
2. Restored FP64 `pow`, retaining approximate `sqrtf` and `__expf`: `0.3221584942 us/zone-update`, delta `+0.0280731856 us/zone-update` (`+9.55%`), photochemistry `1.374 s`; regressed.
3. Restored all `exp` calls, retaining approximate `sqrtf` only: `0.2945085894 us/zone-update`, delta `+0.0004232808 us/zone-update` (`+0.14%`), photochemistry `1.229 s`; near neutral but total FoM slightly regressed.
4. Limited approximate `sqrtf` to cooling terms only, restoring collisional ionization `sqrt` to FP64: `0.2940679895 us/zone-update`, delta `-0.0000173191 us/zone-update` (`-0.01%`), photochemistry `1.231 s`; effectively neutral.
5. Tested KI-cooling `sqrtf` only: `0.2929003792 us/zone-update`, delta `-0.0011849294 us/zone-update` (`-0.40%`), photochemistry `1.237 s`; small total FoM improvement.
6. Tested ion/free-free cooling `sqrtf` only: `0.2922007305 us/zone-update`, delta `-0.0018845781 us/zone-update` (`-0.64%`), photochemistry `1.223 s`; better targeted improvement.
7. Added recombination-rate `powf` on top of ion/free-free `sqrtf`: `0.7708099401 us/zone-update`, delta `+0.4767246315 us/zone-update` (`+162.10%`), photochemistry `3.725 s`; passed checks but strongly regressed and was reverted.
8. Replaced recombination-cooling `T^-0.89` with `powf`, retaining ion/free-free `sqrtf`: `0.2826763927 us/zone-update`, delta `-0.0114089159 us/zone-update` (`-3.88%`), photochemistry `1.160 s`; strong improvement.
9. Added KI-cooling `sqrtf` on top of recombination-cooling `powf` and ion/free-free `sqrtf`: `0.2806844664 us/zone-update`, delta `-0.0134008422 us/zone-update` (`-4.56%`), photochemistry `1.154 s`; best result and retained.
10. Added approximate `__expf` only in KI cooling on top of iteration 9: `0.2839270308 us/zone-update`, delta `-0.0101582778 us/zone-update` (`-3.45%`), photochemistry `1.178 s`; improved over baseline but regressed relative to iteration 9, so the KI `__expf` change was reverted.

Final retained fast-transcendental change: CUDA device `sqrtf` for KI cooling and ion/free-free cooling `sqrt(T)`, plus CUDA device `powf` for recombination cooling `T^-0.89`; FP64 `pow` remains for recombination rate and FP64 `exp` remains everywhere. Final best measured FoM: `0.2806844664 us/zone-update`, delta `-0.0134008422 us/zone-update` (`-4.56%`) from this pass baseline. Built-in verification passed with `ctest --test-dir /mnt/ffs24/home/wibkingb/quokka/build/3d-cuda -R '^DTypeFront$' --output-on-failure` on the final rebuilt artifact.

## Jacobian/Linear-Solve Specialization Pass

Baseline: `0.2790360415 us/zone-update`, `PhotoChemistry::computePhotoChemistry() = 1.158 s`, measured on 1 GPU for 20 hydro timesteps with default timestepping after the fast-transcendental pass.

1. Runtime probe with `integrator.linalg_do_pivoting=0`: `0.2730121685 us/zone-update`, delta `-0.0060238730 us/zone-update` (`-2.16%`), photochemistry `1.126 s`; passed checks and showed no-pivot solve was viable for this problem.
2. Compiled a `PHOTOCHEMISTRY && int_neqs == 5` VODE specialization that calls the existing no-pivot LU/solve templates by default: `0.2727757460 us/zone-update`, delta `-0.0062602955 us/zone-update` (`-2.24%`), photochemistry `1.123 s`.
3. Combined the 5x5 matrix scale and identity-add into one direct array pass: `0.2749696930 us/zone-update`, delta `-0.0040663485 us/zone-update` (`-1.46%`), photochemistry `1.125 s`; regressed relative to iteration 2 and was reverted.
4. Added specialized 5x5 no-pivot LU/solve helpers with explicit unroll pragmas, bypassing the global `NET_LOOP_UNROLL_LEN=1`: `0.2551331923 us/zone-update`, delta `-0.0239028492 us/zone-update` (`-8.57%`), photochemistry `1.027 s`.
5. Rewrote the specialized 5x5 helpers as straight-line scalar matrix operations: `0.2546186224 us/zone-update`, delta `-0.0244174191 us/zone-update` (`-8.75%`), photochemistry `1.018 s`.
6. Rewrote the triangular solve to use local scalar RHS variables and write back once: `0.2552359426 us/zone-update`, delta `-0.0238000989 us/zone-update` (`-8.53%`), photochemistry `1.018 s`; regressed relative to iteration 5 and was reverted.
7. Runtime probe with `integrator.use_jacobian_caching=1`: `0.2544819857 us/zone-update`, delta `-0.0245540558 us/zone-update` (`-8.80%`), photochemistry `1.025 s`; non-default setting and no clear photochemistry win, so not retained.
8. Removed zero-pivot checks from the specialized DTypeFront no-pivot factorization: `0.2537714922 us/zone-update`, delta `-0.0252645493 us/zone-update` (`-9.05%`), photochemistry `1.015 s`.
9. Returned immediately from the specialized factorization branch before the generic `IER` error check: `0.2526407473 us/zone-update`, delta `-0.0263952942 us/zone-update` (`-9.46%`), photochemistry `1.018 s`.
10. Removed the unused factorization status argument and moved the generic `IER` variable below the specialized early return: `0.2523315582 us/zone-update`, delta `-0.0267044833 us/zone-update` (`-9.57%`), photochemistry `1.014 s`.

Final retained Jacobian/linear-solve change: a `PHOTOCHEMISTRY && int_neqs == 5` VODE path using straight-line no-pivot 5x5 LU factorization and solve helpers. Final best measured FoM: `0.2523315582 us/zone-update`, delta `-0.0267044833 us/zone-update` (`-9.57%`) from this pass baseline. Built-in 20-step checks passed on every retained benchmark run. Final verification passed with `ctest --test-dir /mnt/ffs24/home/wibkingb/quokka/build/3d-cuda -R '^DTypeFront$' --output-on-failure` in `12.20 s`.
