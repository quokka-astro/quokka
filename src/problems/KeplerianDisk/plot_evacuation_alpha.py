#!/usr/bin/env python3
"""Remake Fig. 10 using the disk-evacuation fit substituted into eq. (21)."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, default=Path('build/2d/keplerian-powerlaw-equation18/fit.json'))
    parser.add_argument('--sound-speed-source', type=Path,
                        default=Path('build/2d/keplerian-figure10-ppm-floor7/r20/keplerian_disk_alpha.txt'))
    parser.add_argument('--output', type=Path, default=Path('build/2d/keplerian-figure10-evacuation-alpha'))
    args = parser.parse_args()
    fit = json.loads(args.fit.read_text())
    header = args.sound_speed_source.read_text().splitlines()[1]
    metadata = dict(item.strip().split(' = ') for item in header.lstrip('# ').split(','))
    # Both disk and ring use the same compiled isothermal EOS, unit dx and GM=1.
    bondi_cells = 1. / float(metadata['cs_squared'])
    A, p = fit['normalization'], fit['exponent']
    coefficient = 3. / (32*np.pi) * A**(1/p)
    normalization = coefficient * bondi_cells
    exponent = .5 - 1/p
    x = np.array([20.,40.,60.])
    y = normalization * x**exponent
    # Independent direct substitution into eq. (21), with GM=dx=1.
    t_acc = 16*np.pi*(x/A)**(1/p)
    np.testing.assert_allclose(y,1.5*np.sqrt(x)*bondi_cells/t_acc,rtol=1e-13)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(fit_source=str(args.fit.resolve()), fit_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),
                   sound_speed_source=str(args.sound_speed_source.resolve()),
                   sound_speed_source_sha256=hashlib.sha256(args.sound_speed_source.read_bytes()).hexdigest(),
                   bondi_radius_cells=bondi_cells, normalization=normalization, exponent=exponent,
                   coefficient_multiplying_bondi_radius=coefficient,
                   fitted_radius_range_cells=[A*fit['first_period']**p,A*fit['last_period']**p],
                   extrapolated_values=[dict(radius_cells=float(r),alpha=float(a)) for r,a in zip(x,y)],
                   method='Evacuation fit r/dx=A[t/P(4dx)]^p substituted into eq.21; alpha=(3/(32pi))A^(1/p)(r_B/dx)(r/dx)^(1/2-1/p)',
                   paper_eq22='78 * 1.36e4 * (r/dx)^(-3.85)',paper_eq24='1100 * (r/dx)^(-2.37)',
                   caveat='Extrapolated estimates, not measured ring alphas. Disk run has no accretion; paper eq22 derives from an accreting run.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savetxt(args.output/'extrapolated_alpha.csv',np.column_stack((x,y)),delimiter=',',comments='',header='radius_cells,extrapolated_alpha')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    radii=np.geomspace(10,70,400)
    fig,ax=plt.subplots(figsize=(7.2,5.4),layout='constrained')
    ax.loglog(radii,78*1.36e4*radii**-3.85,color='.3',lw=1.8,label='Paper: disk evacuation (eq. 22)')
    ax.loglog(radii,1100*radii**-2.37,color='.55',ls='--',lw=1.8,label='Paper: ring fit (eq. 24)')
    ax.loglog(radii,normalization*radii**exponent,color='#0072B2',lw=2.2,
              label=rf'Quokka extrapolation: $\alpha={normalization/1e4:.3f}\times10^4(r/\Delta x)^{{{exponent:.3f}}}$')
    ax.loglog(x,y,'o',color='#0072B2',ms=7,zorder=5,label='Extrapolated values at 20, 40, 60 cells')
    for r,a,offset,align in zip(x,y,[(7,7),(-8,-18),(-8,12)],['left','right','right']):
        ax.annotate(f'{a:.4g}',(r,a),xytext=offset,ha=align,textcoords='offset points',color='#0072B2',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    ax.set(xlabel=r'Radius $r/\Delta x$',ylabel=r'Effective viscosity $\alpha$',
           title='Figure 10 comparison: evacuation-based alpha',xlim=(10,70),ylim=(.04,30))
    ax.set_xticks([10,20,30,40,50,60,70],labels=['10','20','30','40','50','60','70'])
    ax.grid(which='major',color='.9',lw=.7)
    ax.legend(loc='upper right',fontsize=8.5,frameon=False)
    fig.get_layout_engine().set(rect=(0,.095,1,.905))
    fig.text(.5,.025,'Extrapolated from evacuation radii ≈4.3–9.2 cells; not ring measurements\nPPM, fixed central mass, no accretion; density floor = 10⁻⁷',ha='center',fontsize=9,color='.35')
    for ext in ('png','pdf'):
        fig.savefig(args.output/f'figure10.{ext}',dpi=220)
    plt.close(fig)
    (args.output/'README.md').write_text('''# Figure 10: extrapolated evacuation alpha

The fitted sub-cell evacuation law is substituted into Equation (21), identifying
evacuation time with accretion time as in the paper. With x=r/dx and the time
unit P(4 dx), alpha = (3/(32 pi)) A^(1/p) (r_B/dx) x^(1/2-1/p).
The same sound speed used by the continuous disk is read from the saved ring
EOS metadata (GM=dx=1). Source paths, hashes and full precision coefficients
are recorded in summary.json. The independent Equation (21) substitution
is checked numerically at all three tabulated radii.

All blue points are extrapolated predictions, not new ring measurements.
The underlying evacuation fit spans approximately 4.3–9.2 cells. The entire
plotted range 10–70 cells is an extrapolation. Paper references are the published
Equation (22), with r_B/dx=1.36e4, and Equation (24), not digitized measurements.
The y-axis extends to 30 to show the extrapolated curve near 10 cells.
Our disk has no accretion; the paper's Equation (22) derives from an accreting
run. This is an approximate effective-viscosity inference, not a direct
measurement of viscous stress.

Reproduce from the repository root:

```sh
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 src/problems/KeplerianDisk/plot_evacuation_alpha.py
```
''')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
