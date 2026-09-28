# TallBoxSf density-profile scripts

Python helpers that solve the hydrostatic vertical density profile used as the initial condition for the TallBoxSf problem. The C++ driver (`testTallBoxSf.cpp`) reads the DataTable CSV produced here via `problem.IC_file`.

## Equations

We solve for the gas density $\rho_1$ and the gravitational acceleration due to gas $g_1$:

$$
\sigma_1^2 \frac{d \rho_1}{d z}=\rho_1\left(g_1+g_{\mathrm{ext}}\right), \qquad
\frac{d g_1}{d z}=4 \pi G \rho_1,
$$

with $\sigma_1 = 7$ km/s (default). Boundary conditions: $g_1(z=0)=0$ and $\Sigma_{\mathrm{gas}} = 2 \int_0^{\infty} \rho_1\, dz$. The second constraint is enforced by iterating on the midplane density $\rho_{1,0} \equiv \rho_1(z=0)$ until the integrated surface density matches the target $\Sigma_{\mathrm{gas}}$.

The external potential (stars + dark matter; Kim & Ostriker 2017, after Kuijken & Gilmore 1989) is

$$
\Phi_{\mathrm{ext}}
= 2\pi G \Sigma_* z_* \Bigl[\bigl(1 + z^2/z_*^2\bigr)^{1/2} - 1\Bigr]
+ 2\pi G \rho_{\mathrm{dm}} R_0^2 \ln\bigl(1 + z^2/R_0^2\bigr),
$$

and $g_{\mathrm{ext}} = -d\Phi_{\mathrm{ext}}/dz$. Default solar-neighborhood values: $\Sigma_* = 42\,\mathrm{M}_\odot\,\mathrm{pc}^{-2}$, $z_* = 245$ pc, $\rho_{\mathrm{dm}} = 6.4\times 10^{-3}\,\mathrm{M}_\odot\,\mathrm{pc}^{-3}$, $R_0 = 8$ kpc.

The tabulated solution is written in dimensionless form $\xi = \xi(\theta)$ with $\xi \equiv \rho / \rho_{1,0}$ and $\theta \equiv z/z_*$ on $\theta \in [0, 20]$, plus physical columns (density, $g$, $\Phi$).

## Scripts

| File | Role |
| --- | --- |
| `solve_density_profile.py` | ODE solver, root find on $\rho_{1,0}$, CSV + plot output |
| `datatable.py` | Writer for the Quokka `DataTable` CSV format (imported, not run directly) |
| `run.sh` | Default solar-neighborhood case (`Sigma13-Z1`) |

## How to run

Install:

```bash
pip install numpy scipy astropy pandas matplotlib
```

Then from this directory:

```bash
cd src/problems/TallBoxSf
```

### Default case (solar neighborhood)

$\Sigma_{\mathrm{gas}} = 13\,\mathrm{M}_\odot\,\mathrm{pc}^{-2}$, $\Sigma_* = 42\,\mathrm{M}_\odot\,\mathrm{pc}^{-2}$, $\sigma_1 = 7$ km/s, $\rho_{\mathrm{dm}} = 6.4\times 10^{-3}\,\mathrm{M}_\odot\,\mathrm{pc}^{-3}$, $R_0 = 8000$ pc:

```bash
./run.sh
```

Equivalent direct call:

```bash
python solve_density_profile.py \
  --Sigma_gas 13.0 \
  --Sigma_star 42.0 \
  --sigma_1 7.0 \
  --rho_dm 6.4e-3 \
  --R0 8000.0 \
  --output_suffix "Sigma13-Z1"
```

Stdout from `run.sh` is written to `output/log_Sigma13-Z1.txt`.

### Custom parameters

All physical parameters are CLI flags (defaults match the solar neighborhood):

```bash
python solve_density_profile.py --help
```

| Flag | Units | Default |
| --- | --- | --- |
| `--Sigma_gas` | $\mathrm{M}_\odot\,\mathrm{pc}^{-2}$ | 13 |
| `--Sigma_star` | $\mathrm{M}_\odot\,\mathrm{pc}^{-2}$ | 42 |
| `--sigma_1` | km/s | 7 |
| `--rho_dm` | $\mathrm{M}_\odot\,\mathrm{pc}^{-3}$ | $6.4\times 10^{-3}$ |
| `--R0` | pc | 8000 |
| `--z_star` | pc | 245 |
| `--output_suffix` | string | (empty) |

Example with a different gas surface density:

```bash
python solve_density_profile.py --Sigma_gas 20.0 --output_suffix "Sigma20-Z1"
```

## Output

Files land in `output/` (created automatically by the Python script):

| File | Contents |
| --- | --- |
| `disk_solution_all_vars_<suffix>.csv` | Full solution: $\theta$, $\xi$, $z$, $\rho$, $g_1$, $g_{\mathrm{ext}}$, $g_{\mathrm{tot}}$, $\Phi$ |
| `disk_solution_datatable_<suffix>.csv` | Quokka DataTable: $z \rightarrow (g_1, g_{\mathrm{ext}}, \Phi_{\mathrm{tot}})$ |
| `profile_dimensionless_<suffix>.png` | $\xi(\theta)$ |
| `profile_physical_<suffix>.png` | $n_{\mathrm{H}}(z)$ |

The DataTable file is what the simulation consumes (`problem.IC_file` in `inputs/TallBoxSf.toml`).
