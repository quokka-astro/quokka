#!/usr/bin/env python3
"""Plot Joseph ring surface-density slices at requested orbital times.

Defaults to Figure 7's inviscid times. Refuses to extrapolate unavailable data.
"""
import argparse
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
p.add_argument('--times',type=float,nargs='+',default=[224.,748.])
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
files=sorted(a.run.glob('joseph_profile_*.txt'))
if not files:raise ValueError('No radial profiles found')
times=[];profiles=[];radii=None
for file in files:
 t=float(re.search(r'orbits = (.*)',file.read_text().splitlines()[0]).group(1))
 data=np.loadtxt(file)
 if radii is None:radii=data[:,0]
 np.testing.assert_array_equal(radii,data[:,0])
 if not np.isfinite(data).all() or np.any(data[:,1:3]<=0):raise ValueError(f'Invalid profile: {file}')
 times.append(t);profiles.append(data)
times=np.asarray(times);profiles=np.asarray(profiles)
if np.any(np.diff(times)<=0):raise ValueError('Nonmonotonic profile times')
if min(a.times)<times[0]-1e-12 or max(a.times)>times[-1]+1e-10:
 raise ValueError(f'Requested {a.times}; available interval is {times[0]:.8g}–{times[-1]:.8g} orbits')
a.output.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output/'profiles.npz',orbits=times,radius=radii,slice_y0=profiles[:,:,1],annular=profiles[:,:,2],initial=profiles[0,:,3])
fig,ax=plt.subplots(figsize=(6.4,4.6),layout='constrained')
measurements=[]
for j,t in enumerate(a.times):
 hi=int(np.searchsorted(times,t));hi=min(hi,len(times)-1)
 if hi==0 or abs(times[hi]-t)<1e-10:
  profile=profiles[hi];lo=hi;weight=0.
 else:
  lo=hi-1;weight=(t-times[lo])/(times[hi]-times[lo]);profile=(1-weight)*profiles[lo]+weight*profiles[hi]
 ax.plot(radii,profile[:,1],color='#0072B2',ls=['-','--',':','-.'][j%4],label=f'{t:g} orbit' + ('' if t == 1 else 's'))
 np.savetxt(a.output/f'profile_{t:g}_orbits.csv',profile,delimiter=',',header='R,slice_y0,Sigma_annular,initial_Sigma',comments='')
 measurements.append(dict(orbits=t,bracket_orbits=[float(times[lo]),float(times[hi])],interpolation_weight=float(weight)))
run_metadata=json.loads((a.run/'run.json').read_text())
ncells=run_metadata['cells']
ax.set(xlabel=r'$R/R_0$',ylabel=r'$\Sigma/\Sigma_{\rm ref}$',xlim=(.2,1.8),ylim=(0,1.1),title=f'Joseph inviscid ring: {ncells} × {ncells}')
ax.set_xticks([.2,.6,1.,1.4,1.8]);ax.legend(frameon=False)
ax.tick_params(which='both',direction='in',top=True,right=True)
for ext in ('png','pdf'):fig.savefig(a.output/f'profiles.{ext}',dpi=200)
plt.close(fig)
summary=dict(run=str(a.run.resolve()),samples=len(times),available_orbits=[float(times[0]),float(times[-1])],requested=measurements,
             slice='Linear interpolation to y=0 from the two adjacent cell rows; positive x side',annuli='Area-weighted one-cell radial bins; retained in NPZ and CSV',time_sampling='Linear interpolation between output profiles, no extrapolation')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
