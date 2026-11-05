# Quokka chemical yield datatables

`quokka_yield_tables.tar.gz` contains the preprocessed Quokka datatables used by the table-driven chemical feedback tests and simulations. The archive is extracted only by targets/tests that need these tables.

The archive contains:

- `SNII_yield_table.csv`: a one-dimensional Quokka `DataTable` generated from the solar-metallicity Sukhbold et al. (2016) massive-star yield tables, using the explosive ejecta column.
- `WR_yield_table.csv`: a one-dimensional Quokka `DataTable` generated from the solar-metallicity Sukhbold et al. (2016) massive-star yield tables, using the wind column.
- `AGB_yield_table.csv`: a one-dimensional Quokka `DataTable` generated from the Karakas--Lugaro AGB yields together with the Doherty et al. super-AGB yields.
- `WR_mass_loss_distribution_table.csv`: a two-dimensional Quokka `DataTable` generated from solar-metallicity MIST tracks. It stores the cumulative WR mass-loss fraction as a function of stellar age and birth mass, using `surface_h1 < 0.4` to identify the WR phase. This table controls the time distribution of WR ejecta; the total WR isotope budget still comes from `WR_yield_table.csv`.

The raw stellar yield tables and MIST tracks are not committed to this repository. To regenerate the archive, obtain the raw source data locally and run:

```bash
python3 scripts/generate_chemical_yield_tables.py --yield-root extern/yields --mist-root extern/mist_tracks
COPYFILE_DISABLE=1 tar -C extern/yields -czf extern/yields/quokka_yield_tables.tar.gz \
  AGB_yield_table.csv SNII_yield_table.csv WR_yield_table.csv WR_mass_loss_distribution_table.csv quokka_yield_tables_manifest.txt
```

## Particle storage and checkpoint compatibility

Enable chemical feedback storage at compile time by setting
`static constexpr bool enable_chemical_feedback = true;` in the problem's
`Particle_Traits` specialization, then enable it at runtime with
`particles.enable_chemical_feedback = true`. The default trait is `false`, so
problems that do not opt in retain their original particle layout, including
when they use passive scalars. Enabling the runtime switch without the trait
is an error.

Opted-in problems reserve five chemistry-history blocks (total, SNII, WR, AGB,
SNIa) of
`Physics_Traits<problem_t>::numPassiveScalars` real components. Changing the
compile-time trait changes the checkpoint layout; do not restart a checkpoint
with a different trait setting. Checkpoints from earlier versions of this PR
that unconditionally included chemistry storage also require that layout.

Yield validation reports simulated mass, expected mass, and absolute error.
It tests `abs(simulated - expected) <= tolerance * abs(expected)` without
division; an expected zero must remain exactly zero with FPE traps enabled.

## SNIa prototype migration (not yet validated)

`SNIa_yield_table.csv` is tracked separately from the existing archive so this
migration does not replace the already-merged SNII, WR, or AGB data. It is copied
from the local SNIa prototype. Its manifest identifies `snia_ivo13_stable_z.txt`,
the `Z=0.02` column, and normalization by 1.4 solar masses of ejecta. The raw data,
conversion procedure, and provenance still require review; the existing table
generation command above does not regenerate this SNIa file.

SNIa requires explicit `particles.enable_SNIa_metal = 1` and inclusion of `SNIa`
in `particles.chemical_tracked_channels`. The default channel selection remains
SNII/WR/AGB. Loading a SNIa table does not generate events automatically.

The extra history block changes the layout of all chemistry-enabled stochastic
stellar particles, even when SNIa is disabled at runtime. Existing four-block
checkpoints are not compatible with this draft layout. The two yield-test ASCII
particle inputs have an additional zero-filled block; external particle inputs
need the same adaptation. Problems without chemistry storage are unchanged.
