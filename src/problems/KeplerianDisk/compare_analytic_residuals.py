#!/usr/bin/env python3
"""Compare xPPM, exact-face midpoint, and Gauss-integrated equilibrium residuals.

Run from the repository root. The independent NumPy calculation verifies the
C++ controls and supplies an eight-point quadrature reference. No evolution
algorithm is changed. Requires NumPy and Matplotlib.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from measure_residuals import fields, profile, self_check, summarize


def equilibrium(x, y):
    r2 = x*x + y*y
    rho = 0.01 + np.exp(-((r2 - 1) / 0.3)**2)
    omega2 = np.maximum(r2, 0.25**2)**(-1.5)
    omega = np.sqrt(omega2)
    return rho, -omega*y, omega*x, omega2


def reference(n, nq):
    """Physical fluxes and gravitational source, independently integrated."""
    h = 4 / n
    centers = -2 + (np.arange(n) + 0.5)*h
    faces = -2 + np.arange(n + 1)*h
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    nodes, weights = nodes / 2, weights / 2
    x, y = np.meshgrid(centers, centers)
    xf, yf = np.meshgrid(faces, centers)
    xg, yg = np.meshgrid(centers, faces)
    fx = np.zeros((n, n + 1, 3))
    fy = np.zeros((n + 1, n, 3))
    source = np.zeros((n, n, 3))
    for a, w in zip(nodes, weights):
        rho, vx, vy, _ = equilibrium(xf, yf + a*h)
        fx += w*np.stack([rho*vx, rho*vx*vx + 0.001, rho*vx*vy], axis=-1)
        rho, vx, vy, _ = equilibrium(xg + a*h, yg)
        fy += w*np.stack([rho*vy, rho*vy*vx, rho*vy*vy + 0.001], axis=-1)
        for b, v in zip(nodes, weights):
            rho, _, _, omega2 = equilibrium(x + a*h, y + b*h)
            source[..., 1] -= w*v*rho*omega2*(x + a*h)
            source[..., 2] -= w*v*rho*omega2*(y + b*h)
    rhs = -(fx[:, 1:] - fx[:, :-1])/h - (fy[1:] - fy[:-1])/h + source
    rho, _, _, _ = equilibrium(x, y)
    return np.column_stack([x.ravel(), y.ravel(), rho.ravel(), rhs.reshape(-1, 3), np.zeros(n*n)])


def metrics(data):
    result = summarize(data)
    # These controls do not export a face-moment torque-defect budget.
    for key in ('torque_defect_rms', 'torque_defect_r_lt_1', 'outward_face_Lz_flux_r_lt_1'):
        result.pop(key)
    for key in ('rotation90_max_error', 'peak_annulus'):
        result[key].pop('torque_defect')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--input', type=Path, default=Path('inputs/KeplerianDisk.toml'))
    parser.add_argument('--output', type=Path, default=Path('build/2d/keplerian-analytic-residuals'))
    args = parser.parse_args()
    self_check()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for n in (128, 256):
        directory = output / f'n{n}'
        directory.mkdir(exist_ok=True)
        if args.executable:
            cmd = [str(args.executable.resolve()), str(args.input.resolve()),
                   f'amr.n_cell={n} {n}', 'max_timesteps=1', 'constant_dt=0.00025',
                   'stop_time=0.00025', 'hydro.reconstruction_order=5', 'hydro.use_dual_energy=1',
                   'hydro.artificial_viscosity_coefficient=0.1', 'disk.residual_prefix=residual',
                   'plotfile_interval=-1', 'checkpoint_interval=-1', 'suppress_output=1']
            (directory / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
            with (directory / 'run.log').open('w') as log:
                subprocess.run(cmd, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = {'n': n}
        for kind in ('spatial', 'analytic-q1', 'analytic-q2', 'analytic-q4'):
            data = np.loadtxt(directory / f'residual-{kind}.txt')
            data = data[np.lexsort((data[:, 0], data[:, 1]))]
            assert np.isfinite(data).all()
            if kind != 'spatial':
                independent = reference(n, int(kind[-1]))
                np.testing.assert_allclose(data[:, :3], independent[:, :3], rtol=1.e-14, atol=1.e-15)
                np.testing.assert_allclose(data[:, 3:6], independent[:, 3:6], rtol=0, atol=5.e-13)
            record[kind] = metrics(data)
            (directory / f'{kind}-profile.json').write_text(json.dumps(profile(data), indent=2) + '\n')
        high = reference(n, 8)
        record['analytic-q8-python'] = metrics(high)
        # The smooth annulus is away from the core's nonsmooth potential transition.
        r, _, vals = fields(high)
        mask = (r > 0.7) & (r < 1.3)
        for key in ('density_rate', 'radial_acceleration', 'normalized_Lz_rate'):
            assert np.max(np.abs(vals[key][mask])) < 1.e-11
        np.savetxt(directory / 'residual-analytic-q8-python.txt', high,
                   header='x y rho0 density_rate mx_rate my_rate unused_zero')
        results.append(record)
    (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'mpl-cache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(output / 'xdg-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    names = ('density_rate_rms', 'radial_acceleration_rms', 'normalized_Lz_rate_rms')
    labels = ('Density rate', 'Radial acceleration', 'Angular momentum density rate / rho0')
    methods = (('spatial', 'Current xPPM'), ('analytic-q1', 'Exact face midpoint'),
               ('analytic-q2', '2-point Gauss'), ('analytic-q4', '4-point Gauss'))
    for ax, name, label in zip(axes, names, labels):
        for method, title in methods:
            ax.loglog([r['n'] for r in results], [r[method][name] for r in results], 'o-', label=title)
        ax.set_xticks([128, 256], ['128', '256'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel('Cells per direction')
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25, which='both')
    axes[0].set_ylabel('RMS in 0.7 < r < 1.3 (code units)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Kepler equilibrium: reconstruction versus integration error')
    fig.tight_layout()
    fig.savefig(output / 'analytic-comparison.png', dpi=170)
    fig.savefig(output / 'analytic-comparison.pdf')
    plt.close(fig)
    for rec in results:
        print('Grid', rec['n'])
        for method, _ in methods:
            print(method, *(f'{rec[method][name]:.8e}' for name in names))
        print('peak radial:', {m: rec[m]['peak_annulus']['radial_acceleration'] for m, _ in methods})


if __name__ == '__main__':
    main()
