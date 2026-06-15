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
tar -C extern/yields -czf extern/yields/quokka_yield_tables.tar.gz \
  AGB_yield_table.csv SNII_yield_table.csv WR_yield_table.csv WR_mass_loss_distribution_table.csv quokka_yield_tables_manifest.txt
```
