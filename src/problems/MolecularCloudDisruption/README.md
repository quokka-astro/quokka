# Molecular-cloud disruption benchmark

This problem places a nominal `1e5 Msun`, `20 pc`, `100 K` cloud in pressure equilibrium with an ambient medium 100 times less dense. A deterministic analytic low-mode velocity field is normalized to a total virial parameter of two after accounting for thermal support and a coeval `2000 Msun` stellar population.

The population contains one `1600 Msun` low-mass composite plus twenty `20 Msun` SN progenitors. The first SN occurs at 4 Myr, after the planned 3.3 Myr early-feedback interval. On-the-fly star formation is disabled so every feedback variant starts from the same stellar population.

The two MassScalars are cloud-origin and ambient-origin partial densities. SN ejecta are assigned to the cloud-origin component by setting `particles.scalar_yield_per_SN` equal to the fixed `10 Msun` ejecta mass.

Build and run the 64-cell-per-side SN-only case with:

```sh
quokka build -d 3d MolecularCloudDisruption
quokka run -d 3d MolecularCloudDisruption --input inputs/MolecularCloudDisruption.toml
```

For a resolution sweep, invoke the executable directly from `tests/` and override `amr.n_cell`, `amr.max_grid_size`, and the history filename. The current three-cell feedback support radius is 7.5, 3.75, and 1.875 pc at 64, 128, and 256 cells per side, respectively.

```sh
../build/3d/src/problems/MolecularCloudDisruption/MolecularCloudDisruption ../inputs/MolecularCloudDisruption.toml "amr.n_cell=64 64 64" amr.max_grid_size=64 statistics_file=cloud_SN_N64.txt
../build/3d/src/problems/MolecularCloudDisruption/MolecularCloudDisruption ../inputs/MolecularCloudDisruption.toml "amr.n_cell=128 128 128" amr.max_grid_size=128 statistics_file=cloud_SN_N128.txt
../build/3d/src/problems/MolecularCloudDisruption/MolecularCloudDisruption ../inputs/MolecularCloudDisruption.toml "amr.n_cell=256 256 256" amr.max_grid_size=128 statistics_file=cloud_SN_N256.txt
```

The no-feedback control sets `particles.disable_SN_feedback=1`. The early-feedback-plus-SN branch adds `particles.EMF_enabled=1`; the problem uses the general `StochasticStellarPop` early-feedback source term rather than a private duplicate.

Compare `cloud_dense_fraction`, `cloud_cold_dense_fraction`, and `cloud_mass_Msun` in the history files. A useful disruption time is the first time the dense or cold-dense fraction falls below 0.5 of its EMF-disabled initial value. `feedback_coupling_radius_pc` records the physical coupling scale in every history row.

## Matched EMF A/B run

Run the SN-only (`EMF_enabled=0`) and EMF-plus-SN (`EMF_enabled=1`) cases with:

```sh
python3 scripts/python/run_molecular_cloud_emf_ab.py --resolution 64 --output-dir sims/cloud-emf-ab-n64
```

The script builds the problem once, launches both arms from separate directories with identical inputs and runtime overrides, and writes `comparison.json` and `comparison.csv` next to the two run directories. The comparison reports each disruption metric at the common final time and linearly interpolated `t50` crossing times relative to the `EMF_enabled=0` initial value. Extra matched overrides may be supplied repeatedly, for example:

```sh
python3 scripts/python/run_molecular_cloud_emf_ab.py \
  --resolution 128 \
  --output-dir sims/cloud-emf-ab-n128 \
  --override 'stop_time=6.0 * Myr' \
  --override statistics_interval=5
```

The runner refuses to start if `particles.EMF_enabled` is absent from the C++ sources, because AMReX otherwise permits an unused command-line parameter and the two arms could silently be identical.
