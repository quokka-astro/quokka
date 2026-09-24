#!/usr/bin/env python3
"""Fit Joseph density slices to the Pringle profile and plot viscosity.

Requires numpy, scipy and matplotlib. The paper's Appendix D fits tau(t) over
~180 late orbits; the selected full-duration fits are early-time effective estimates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import minimize_scalar
from scipy.special import ive

TAU0 = 0.018
H = 0.005
BACKGROUND = 1.e-7
# Joseph et al. (2023), arXiv:2308.03881v1, Table 1, Cartesian rows.
# These are the Figure 8 measurements, in the same R0^2*Omega0 units as our nu.
PAPER_FIGURE8 = {
    'PLUTO': np.array([[256, 4.68e-6, .19e-6], [512, 1.48e-6, .01e-6],
                       [1024, 4.16e-7, .03e-7], [2048, 9.45e-8, .11e-8], [4096, 1.32e-8, .08e-8]]),
    'Athena++': np.array([[256, 7.31e-6, .22e-6], [512, 2.04e-6, .05e-6],
                          [1024, 4.26e-7, .11e-7], [2048, 1.11e-7, .02e-7]]),
}


def kernel(radius, tau):
    return np.exp(-(radius-1.)**2/tau)*ive(.25, 2.*radius/tau)/(tau*radius**.25)


def fit_tau(radius, surface_density):
    normalization = kernel(1., TAU0)
    def residual(log_tau):
        model = kernel(radius, np.exp(log_tau))/normalization + BACKGROUND
        return np.sum((model-surface_density)**2)
    result = minimize_scalar(residual, bounds=(np.log(.001), np.log(1.)),
                             method='bounded', options={'xatol': 1.e-13})
    if not result.success:
        raise RuntimeError(result.message)
    tau = np.exp(result.x)
    if tau < .00101 or tau > .99:
        raise ValueError('Fit reached a search boundary')
    error = np.sqrt(result.fun/np.sum(surface_density**2))
    return tau, error


def measure(run, end_orbits=2., start_orbits=0.):
    if not (np.isfinite(start_orbits) and np.isfinite(end_orbits) and 0 <= start_orbits < end_orbits):
        raise ValueError('Require finite 0 <= start_orbits < end_orbits')
    meta = json.loads((run/'run.json').read_text())
    if meta.get('exit_code') != 0:
        raise ValueError(f'{run}: run has not completed successfully')
    cells = meta['cells']
    if cells not in (128, 256, 512):
        raise ValueError(f'{run}: requested resolutions are 128, 256 and 512')
    files = sorted(run.glob('joseph_profile_*.txt'))
    if len(files) < 3:
        raise ValueError(f'{run}: insufficient profiles for a time-slope fit')
    rows = []
    hashes = {}
    for path in files:
        text = path.read_text()
        orbits = float(re.search(r'orbits = (.*)', text.splitlines()[0]).group(1))
        data = np.loadtxt(path)
        if not np.isfinite(data).all() or np.any(data[:, 1] <= 0):
            raise ValueError(f'Invalid surface density: {path}')
        selected = (data[:, 0] >= .2) & (data[:, 0] <= 1.8)
        # The stored y=0 slice averages the two rows at y=+/-dx/2.
        # Sample the axisymmetric model through the identical measurement operator.
        radius = np.hypot(data[selected, 0], 2./cells)
        tau, error = fit_tau(radius, data[selected, 1])
        rows.append((orbits, tau, error))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    samples = np.array(rows)
    orbits, tau = samples[:, 0], samples[:, 1]
    if np.any(np.diff(orbits) <= 0) or abs(orbits[0]) > 1.e-12 or abs(orbits[-1]-end_orbits) > 1.e-9:
        raise ValueError(f'{run}: requires strictly increasing profiles spanning exactly 0--{end_orbits:g} orbits')
    initial_tau = float(tau[0])
    selected_times = (orbits >= start_orbits-1.e-10) & (orbits <= end_orbits+1.e-10)
    samples = samples[selected_times]
    hashes = {path.name: hashes[path.name] for path, keep in zip(files, selected_times) if keep}
    if len(samples) < 3:
        raise ValueError(f'{run}: insufficient profiles in the requested fit interval')
    orbits, tau = samples[:, 0], samples[:, 1]
    # tau = a + b * (t/P0); t_code = 2*pi*(t/P0), so nu=b/(24*pi).
    slope, intercept = np.polyfit(orbits, tau, 1)
    nu = slope/(24.*np.pi)
    line = intercept + slope*orbits
    midpoint = (start_orbits+end_orbits)/2.
    half = orbits >= midpoint-1.e-10
    late_nu = float(np.polyfit(orbits[half], tau[half], 1)[0]/(24.*np.pi)) if half.sum() >= 3 else None
    record = dict(run=str(run.resolve()), cells=cells, profiles=len(samples),
                  fit_orbits=[start_orbits, end_orbits], tau_intercept=float(intercept),
                  tau_initial_fitted=initial_tau, slope_per_orbit=float(slope),
                  nu=float(nu), alpha_at_R0=float(nu/H**2),
                  nu_second_half=late_nu, second_half_orbits=[midpoint, end_orbits],
                  tau_line_rms=float(np.sqrt(np.mean((tau-line)**2))),
                  max_profile_relative_L2=float(np.max(samples[:, 2])),
                  source_sha256=meta.get('source_sha256'),
                  executable_sha256=meta.get('executable_sha256'),
                  profile_sha256=hashes)
    return record, samples


def validate_fit():
    # Recover a known nonzero-intercept viscosity from independent, unscaled
    # Bessel values (safe argument range here); checks inversion and 2*pi units.
    from scipy.special import iv
    radius = np.linspace(.6, 1.4, 80)
    times = np.linspace(0., 2., 9)
    known_nu = 2.e-5
    fitted = []
    for t in times:
        tau = TAU0 + 24.*np.pi*known_nu*t
        density = np.exp(-(1.+radius**2)/tau)*iv(.25, 2.*radius/tau)/(tau*radius**.25)/kernel(1., TAU0)+BACKGROUND
        recovered, error = fit_tau(radius, density)
        np.testing.assert_allclose(recovered, tau, rtol=1.e-7)
        assert error < 1.e-7
        fitted.append(recovered)
    inferred = np.polyfit(times, fitted, 1)[0]/(24.*np.pi)
    np.testing.assert_allclose(inferred, known_nu, rtol=1.e-6)


def plot_paper_comparison(ax, records, power_law):
    """Overlay the tabulated Figure 8 points; refit their rounded values for lines."""
    n = np.array([r['cells'] for r in records])
    nu = np.array([r['nu'] for r in records])
    start, end = records[0]['fit_orbits']
    ax.plot(n, nu, 'o', color='#332288', ms=6, label=f"Quokka: {start:g}–{end:g} orbits", zorder=4)
    if power_law is not None:
        grid = np.geomspace(n.min(), n.max(), 200)
        ax.plot(grid, power_law['B']*(grid/256.)**power_law['p'], color='#332288', ls='--',
                label=rf"Quokka fit: $N^{{{power_law['p']:.3f}}}$")
    paper_fits = {}
    for code, color, marker, linestyle in [('PLUTO', '#0072B2', 'x', '--'),
                                          ('Athena++', '#D55E00', '+', ':')]:
        values = PAPER_FIGURE8[code]
        ax.errorbar(values[:, 0], values[:, 1], yerr=values[:, 2], fmt=marker,
                    color=color, ms=7, capsize=3, label=f'{code}: paper', zorder=3)
        exponent, intercept = np.polyfit(np.log(values[:, 0]), np.log(values[:, 1]), 1)
        grid = np.geomspace(values[0, 0], values[-1, 0], 200)
        ax.plot(grid, np.exp(intercept)*grid**exponent, color=color, ls=linestyle, lw=1,
                label=rf'{code} table fit: $N^{{{exponent:.3f}}}$')
        paper_fits[code] = dict(A=float(np.exp(intercept)), p=float(exponent),
                                method='Unweighted log-space refit to rounded Table 1 values')
    ax.set(xscale='log', yscale='log', xlim=(100, 5000),
           ylim=(min(8.e-9, nu.min()/1.6), max(3.e-4, nu.max()*1.6)),
           xlabel='Cells per side, N', ylabel=r'$\nu_{\rm eff}\;[R_0^2\Omega_0]$')
    ax.set_xticks([128, 256, 512, 1024, 2048, 4096], ['128', '256', '512', '1024', '2048', '4096'])
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False, loc='upper right', fontsize=8)
    return paper_fits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--end-orbits', type=float, default=2., help='Required completed duration and upper fit bound')
    p.add_argument('--start-orbits', type=float, default=0., help='Lower fit bound; use only profiles within the selected interval')
    p.add_argument('--power-law', action='store_true', help='Fit nu versus resolution using unweighted least squares in log space')
    p.add_argument('--paper-figure8', action='store_true', help='Overlay PLUTO and Athena++ values from the paper Table 1 / Figure 8')
    a = p.parse_args()
    if not np.isfinite(a.end_orbits) or a.end_orbits <= 0:
        p.error('--end-orbits must be positive and finite')
    if not np.isfinite(a.start_orbits) or not 0 <= a.start_orbits < a.end_orbits:
        p.error('--start-orbits must be finite and satisfy 0 <= start < end')
    validate_fit()
    measurements = sorted((measure(run, a.end_orbits, a.start_orbits) for run in a.runs), key=lambda item: item[0]['cells'])
    cells = [record['cells'] for record, _ in measurements]
    if len(set(cells)) != len(cells):
        raise ValueError('Select only one completed run per resolution')
    power_law = None
    if a.power_law:
        if len(cells) < 2:
            raise ValueError('The power-law comparison requires at least two resolutions')
        values = np.array([r['nu'] for r, _ in measurements])
        if np.any(values <= 0):
            raise ValueError('Log-space power-law fitting requires positive viscosities')
        x = np.log(np.array(cells)/256.)
        y = np.log(values)
        exponent, log_amplitude = np.polyfit(x, y, 1)
        residual = y - (log_amplitude + exponent*x)
        np.testing.assert_allclose([residual.sum(), residual@x], 0., atol=1.e-12)
        power_law = dict(model='nu = B*(N/256)^p', method='Unweighted least squares in natural-log space',
                         resolutions=cells, fit_orbits=[a.start_orbits, a.end_orbits],
                         residual_degrees_of_freedom=len(cells)-2,
                         B=float(np.exp(log_amplitude)), p=float(exponent),
                         A_for_nu_equals_A_N_to_p=float(np.exp(log_amplitude)/256.**exponent),
                         log_space_R_squared=float(1.-np.sum(residual**2)/np.sum((y-y.mean())**2)),
                         delta_x_exponent=float(-exponent),
                         fitted_nu=list(np.exp(log_amplitude+exponent*x)))
    a.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    colors = {128: '#0072B2', 256: '#D55E00', 512: '#009E73'}
    for record, samples in measurements:
        n = record['cells']; color = colors[n]
        t, tau = samples[:, 0], samples[:, 1]
        axes[0].plot(t, tau-TAU0, 'o', ms=3, color=color, label=f'{n}² profiles')
        axes[0].plot(t, record['tau_intercept']+record['slope_per_orbit']*t-TAU0,
                     color=color, label=f'{n}² linear fit')
        axes[1].plot(n, record['nu'], 'o', color=color, ms=7)
        axes[1].annotate(f"{record['nu']:.3g}", (n, record['nu']), xytext=(0, 10),
                         textcoords='offset points', ha='center', color=color)
        interval_nu = np.diff(tau)/(24.*np.pi*np.diff(t))
        np.savetxt(a.output/f'tau_{n}.csv', samples, delimiter=',',
                   header='orbits,tau,profile_relative_L2', comments='')
        np.savetxt(a.output/f'interval_viscosity_{n}.csv', np.column_stack(((t[1:]+t[:-1])/2., interval_nu)),
                   delimiter=',', header='midpoint_orbits,nu', comments='')
    axes[0].set(xlabel=r'$t/P_0$', ylabel=r'$\tau_{\rm fit}-0.018$', xlim=(a.start_orbits, a.end_orbits))
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(xlabel='Cells per side', ylabel=r'$\nu_{\rm eff}\;[R_0^2\Omega_0]$', xlim=(100, 650))
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks([128, 256, 512], ['128', '256', '512'])
    if power_law is not None:
        nfit = np.geomspace(min(cells), max(cells), 200)
        axes[1].plot(nfit, power_law['B']*(nfit/256.)**power_law['p'], '--', color='.35', zorder=0,
                     label=rf"Power law: $\nu\propto N^{{{power_law['p']:.3f}}}$")
        axes[1].legend(frameon=False, loc='upper right', fontsize=9)
    if all(record['nu'] > 0 for record, _ in measurements):
        axes[1].set_yscale('log')
        vals = [r['nu'] for r, _ in measurements]
        axes[1].set_ylim(min(vals)/1.6, max(vals)*1.6)
    else:
        axes[1].axhline(0, color='.7', lw=.8)
    for ax in axes:
        ax.tick_params(direction='in', top=True, right=True)
    paper_comparison = None
    if a.paper_figure8:
        axes[1].clear()
        paper_fits = plot_paper_comparison(axes[1], [r for r, _ in measurements], power_law)
        paper_comparison = dict(source='https://arxiv.org/html/2308.03881v1#S4.T1',
                                provenance='Table 1 Cartesian rows underlying Figure 8; quoted 1-sigma errors',
                                columns=['cells_per_side', 'nu', 'nu_one_sigma'],
                                data={code: values.tolist() for code, values in PAPER_FIGURE8.items()},
                                power_law_refits=paper_fits,
                                units='R0^2*Omega0 = sqrt(G*M_star*R0); no viscosity rescaling',
                                alpha_note='Figure 8 uses h=0.05 for its secondary alpha axis; this comparison uses nu directly.')
    fig.suptitle(f'Joseph resolved ring: effective viscosity fitted over {a.start_orbits:g}–{a.end_orbits:g} orbits', fontsize=12)
    caption = 'Early-time fits; the paper uses a much longer late-time interval.'
    if a.paper_figure8:
        caption += '\nPaper: Figure 8 / Table 1, with 1σ errors. Paper trend lines refitted to tabulated values.'
    missing = sorted(set([128, 256, 512])-set(cells))
    if missing:
        caption += '\nResolutions omitted: ' + ', '.join(f'{n}²' for n in missing) + '.'
    if power_law is not None and len(cells) == 2:
        caption += ' Two-point power law; no independent test of the scaling.'
    fig.text(.5, -.04, caption, ha='center', va='top', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(a.output/f'viscosity_fits.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    if a.paper_figure8:
        comparison_fig, comparison_ax = plt.subplots(figsize=(7.6, 5.6), layout='constrained')
        plot_paper_comparison(comparison_ax, [r for r, _ in measurements], power_law)
        comparison_ax.set_title('Joseph ring: comparison with Figure 8', fontsize=12)
        comparison_fig.text(.5, -.04, caption, ha='center', va='top', fontsize=8)
        for ext in ('png', 'pdf'):
            comparison_fig.savefig(a.output/f'figure8_comparison.{ext}', dpi=200, bbox_inches='tight')
        plt.close(comparison_fig)
        with (a.output/'paper_figure8.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['code', 'cells', 'nu', 'nu_one_sigma'])
            for code, values in PAPER_FIGURE8.items():
                writer.writerows([code, int(n), nu, error] for n, nu, error in values)
    records = [record for record, _ in measurements]
    keys = ['cells', 'nu', 'alpha_at_R0', 'nu_second_half', 'tau_initial_fitted', 'tau_line_rms', 'max_profile_relative_L2']
    with (a.output/'measurements.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        writer.writerows({key: r[key] for key in keys} for r in records)
    summary = dict(method='Appendix D: fit Pringle tau to each y=0 slice, then fit tau versus code time with a free intercept.',
                   paper='https://arxiv.org/html/2308.03881v1#A4',
                   radial_range=[.2, 1.8], spatial_weights='unweighted radial samples',
                   normalization='fixed initial ring normalization and background; no free amplitude',
                   time_window=f'all saved profiles from {a.start_orbits:g} to {a.end_orbits:g} orbits; not the paper late-time interval',
                   conversion='nu = (d tau / d orbits)/(24*pi); alpha(R0) = nu/(0.005**2)',
                   requested_resolutions=[128, 256, 512], available_resolutions=cells,
                   missing_resolutions=sorted(set([128, 256, 512])-set(cells)), measurements=records,
                   software=dict(numpy=np.__version__, scipy=scipy_version, matplotlib=matplotlib.__version__),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   power_law=power_law,
                   paper_comparison=paper_comparison,
                   validation='Known synthetic viscosity recovered from unscaled Bessel reference to relative error below 1e-6')
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps([{k: r[k] for k in keys} for r in records], indent=2))
    if power_law is not None:
        (a.output/'power_law.json').write_text(json.dumps(power_law, indent=2)+'\n')
        print(json.dumps(power_law, indent=2))


if __name__ == '__main__':
    main()
