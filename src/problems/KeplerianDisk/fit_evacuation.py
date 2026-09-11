#!/usr/bin/env python3
"""Fit resolved sub-cell evacuation radii and compare with Krumholz eq. (18)."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('build/2d/keplerian-powerlaw-figure8-subcell/power_law.csv'))
    parser.add_argument('--output', type=Path, default=Path('build/2d/keplerian-powerlaw-equation18'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = np.genfromtxt(args.input, delimiter=',', names=True)
    time = data['time_periods_at_4dx']
    radius = data['r_evac_cells']
    valid = np.isfinite(radius) & (radius > 0) & (time > 0)
    t, r = time[valid], radius[valid]
    slope, intercept = np.polyfit(np.log(t), np.log(r), 1)
    amplitude = np.exp(intercept)
    fitted = amplitude * t**slope
    paper = 6.1 * t**.23
    residual = np.log(r) - np.log(fitted)
    r_squared = 1 - np.sum(residual**2) / np.sum((np.log(r)-np.log(r).mean())**2)
    # Independent normal-equation verification of the log-space fit.
    x = np.log(t)
    y = np.log(r)
    independent_slope = np.sum((x-x.mean())*(y-y.mean())) / np.sum((x-x.mean())**2)
    np.testing.assert_allclose(slope, independent_slope, rtol=1e-13)
    summary = dict(input=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   model='r_evac/dx = A [t/P(4dx)]^p', method='Unweighted least squares in natural-log space; one weight per resolved plotfile',
                   samples=len(t), excluded_samples=int((~valid).sum()), first_period=float(t[0]), last_period=float(t[-1]),
                   normalization=float(amplitude), exponent=float(slope), log_space_R_squared=float(r_squared),
                   log_space_RMS_residual=float(np.sqrt(np.mean(residual**2))),
                   paper_normalization=6.1, paper_exponent=.23,
                   normalization_ratio=float(amplitude/6.1), exponent_difference=float(slope-.23),
                   final_measured_radius_cells=float(r[-1]), final_fit_radius_cells=float(fitted[-1]),
                   final_paper_radius_cells=float(paper[-1]), final_fit_to_paper_ratio=float(fitted[-1]/paper[-1]),
                   comparison_limit='Equation 18 fits a calculation with accretion; this run has no accretion. Retained dx=1.85 AU differs from printed paper cm value.')
    (args.output/'fit.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savetxt(args.output/'comparison.csv', np.column_stack((t,r,fitted,paper,r/paper)), delimiter=',', comments='',
               header='periods_at_4dx,measured_radius_cells,fitted_radius_cells,equation18_radius_cells,measured_to_equation18')
    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.loglog(time, radius*1.85, color='.4', lw=.8, label='Measured, no accretion')
    grid = np.geomspace(t[0],t[-1],300)
    ax.loglog(grid,1.85*amplitude*grid**slope,color='#0072B2',lw=2,
              label=rf'Fit: $r/\Delta x={amplitude:.3f}\,T^{{{slope:.3f}}}$')
    ax.loglog(grid,1.85*6.1*grid**.23,'--',color='#D55E00',lw=1.8,
              label=r'Eq. (18), with accretion: $6.1\,T^{0.23}$')
    ax.set_xlabel(r'Time $T=t/P(4\Delta x)$ (Periods)')
    ax.set_ylabel('Radius (AU)')
    sec=ax.secondary_yaxis('right',functions=(lambda au:au/1.85,lambda cells:cells*1.85))
    sec.set_ylabel('Radius (cells)')
    for axis, ticks in ((ax.xaxis,[4,5,10,20,50]),(ax.yaxis,[8,10,15,20,30]),(sec.yaxis,[4,5,6,8,10,15,20])):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='both',direction='in',top=True)
    sec.tick_params(which='both',direction='in')
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(args.output/f'equation18.{extension}',dpi=200)
    plt.close(fig)
    (args.output/'README.md').write_text(f'''# Evacuation power-law fit

Fit all {len(t)} resolved sub-cell measurements over {t[0]:.6f}–{t[-1]:.6f}
orbital periods at four cells. Unresolved measurements are excluded, not
assigned a cutoff radius. Fit is unweighted least squares of log(radius) on
log(time), one weight per saved plotfile. No temporal smoothing or fit-window
selection is applied. Samples are temporally correlated; no independent-sample
statistical confidence interval is claimed. The normalization at one period
is an extrapolation beyond the resolved fitting window.

r_evac/dx = {amplitude:.9f} [t/P(4dx)]^{slope:.9f}

Equation (18) is 6.1 [t/P(4dx)]^0.23, fitted to a calculation WITH accretion.
Our stationary fixed-potential disk has NO accretion. The grid uses the retained
1.85 AU cell length, differing from the paper's printed cm value. Thus this is
a comparison of empirical trends, not an identical numerical experiment.

`fit.json` records provenance, method, fit parameters and comparison values.
`comparison.csv` includes the measured and fitted radii and Equation (18).
The plot retains AU/cell radius axes and logarithmic time/radius scales.

Reproduce from the repository root:

```sh
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 src/problems/KeplerianDisk/fit_evacuation.py
```
''')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
