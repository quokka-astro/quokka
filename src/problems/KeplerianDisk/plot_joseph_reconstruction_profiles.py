#!/usr/bin/env python3
"""Compare PLM/PPM/xPPM ring profiles, with slice and annular diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def sample(run, requested):
    paths = sorted(run.glob('joseph_profile_*.txt'))
    times = np.array([float(re.search(r'orbits = (.*)', p.read_text().splitlines()[0]).group(1)) for p in paths])
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f'{run}: missing or nonmonotonic profiles')
    if min(requested) < times[0]-1e-12 or max(requested) > times[-1]+1e-10:
        raise ValueError(f'{run}: requested times outside available profiles')
    initial = np.loadtxt(paths[0])
    measurements = []
    profiles = []
    for t in requested:
        hi = min(int(np.searchsorted(times,t)),len(times)-1)
        lo = hi if hi == 0 or abs(times[hi]-t)<1e-10 else hi-1
        weight = 0. if hi == lo else (t-times[lo])/(times[hi]-times[lo])
        low,high = np.loadtxt(paths[lo]),np.loadtxt(paths[hi])
        np.testing.assert_array_equal(low[:,0],initial[:,0])
        np.testing.assert_array_equal(high[:,0],initial[:,0])
        data = (1-weight)*low+weight*high
        assert np.isfinite(data).all() and (data[:,1:3]>0).all()
        profiles.append(data)
        measurements.append(dict(orbits=t,bracket_orbits=[float(times[lo]),float(times[hi])],
                                 interpolation_weight=float(weight),
                                 profile_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in {paths[lo],paths[hi]}}))
    return initial,profiles,measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs=3,help='128-square PLM, PPM and xPPM directories, in that order')
    parser.add_argument('--times',type=float,nargs='+',default=[0.,5.,10.])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not np.isfinite(args.times).all():
        parser.error('Times must be finite')
    args.output.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(2,len(args.times),figsize=(4*len(args.times),6.3),
                            sharex=True,sharey=True,squeeze=False,layout='constrained')
    provenance = []
    reference = None
    maximum = 1.
    for run,label,color in zip(args.runs,['PLM','PPM','xPPM'],['#0072B2','#D55E00','#009E73']):
        meta = json.loads((run/'run.json').read_text())
        assert meta['cells']==128 and meta['exit_code']==0
        initial,profiles,sampling = sample(run,args.times)
        if reference is None:
            reference = initial
        else:
            np.testing.assert_array_equal(initial,reference)
        for j,(t,data) in enumerate(zip(args.times,profiles)):
            for row,col in enumerate([1,2]):
                axes[row,j].plot(data[:,0],data[:,col],color=color,label=label,lw=1.5)
                selected = (data[:,0]>=.2)&(data[:,0]<=1.8)
                maximum = max(maximum,float(data[selected,col].max()))
            np.savetxt(args.output/f'{label.lower()}_{t:g}_orbits.csv',data,delimiter=',',
                       header='R,slice_y0,Sigma_annular,initial_Sigma',comments='')
        provenance.append(dict(scheme=label,run=str(run.resolve()),run_metadata=meta,sampling=sampling))
    for j,t in enumerate(args.times):
        axes[0,j].set_title(f'{t:g} orbits')
        for row,col in enumerate([1,2]):
            axes[row,j].plot(reference[:,0],reference[:,col],'--',color='.45',lw=1.,label='Initial',zorder=0)
            axes[row,j].set(xlim=(.2,1.8),ylim=(0,maximum*1.08))
            axes[row,j].tick_params(direction='in',top=True,right=True)
        axes[1,j].set_xlabel(r'$R/R_0$')
    axes[0,0].set_ylabel('Slice at y = 0\n'+r'$\Sigma/\Sigma_{\rm ref}$')
    axes[1,0].set_ylabel('Annular average\n'+r'$\Sigma/\Sigma_{\rm ref}$')
    axes[0,-1].legend(frameon=False,fontsize=9)
    fig.suptitle('Joseph ring, 128²: PLM, PPM and xPPM',fontsize=14)
    fig.text(.5,-.02,'Same initial profile for all schemes. Intermediate times interpolated between saved outputs.',
             ha='center',va='top',fontsize=9)
    for ext in ('png','pdf'):
        fig.savefig(args.output/f'ring_profiles.{ext}',dpi=200,bbox_inches='tight')
    (args.output/'summary.json').write_text(json.dumps(dict(requested_orbits=args.times,runs=provenance,
        slice='Positive-x y=0 slice from the two adjacent rows; same as viscosity fit',
        annular='Area-weighted one-cell radial bins',
        interpolation='Linear in time; no radial resampling or extrapolation'),indent=2)+'\n')
    print(f'Saved ring profiles to {args.output}')


if __name__ == '__main__':
    main()
