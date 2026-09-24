#!/usr/bin/env python3
"""Run a reproducible Joseph inviscid-ring case; preserve command and hashes."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--cells',type=int,default=1024)
p.add_argument('--max-grid-size',type=int,help='Override AMReX grid size for GPU launch efficiency')
p.add_argument('--blocking-factor',type=int,help='Override AMReX blocking factor (use 1024 for the 2D H200 run)')
p.add_argument('--executable',type=Path,help='Override the local CPU binary, e.g. a GPU build')
p.add_argument('--orbits',type=float,default=748)
p.add_argument('--max-steps',type=int)
p.add_argument('--init-shrink',type=float,help='Override the input first-step reduction factor')
p.add_argument('--density-floor',type=float,help='Override the input density floor')
p.add_argument('--reconstruction-order',type=int,choices=[1,2,3,5],help='Hydro reconstruction: 2=PLM, 3=PPM, 5=xPPM')
p.add_argument('--max-walltime',help='AMReX wall-time budget in HH:MM:SS (reserves time for final output)')
p.add_argument('--verbose',action='store_true',help='Print timestep diagnostics')
p.add_argument('--profile-interval',type=float,default=1.)
p.add_argument('--ranks',type=int,default=1)
p.add_argument('--restart',type=Path,help='Resume a checkpoint into a fresh output directory')
p.add_argument('--checkpoint-interval',type=int,default=10000)
a=p.parse_args()
repo=Path(__file__).resolve().parents[3]
exe=a.executable.resolve() if a.executable else repo/'build/2d/src/problems/KeplerianDisk/KeplerianDiskJoseph'
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
source=repo/'inputs/KeplerianDiskJoseph.toml'
(a.output/'input.toml').write_bytes(source.read_bytes())
command=[str(exe),str(a.output/'input.toml'),f'amr.n_cell={a.cells} {a.cells}',f'disk.orbits={a.orbits}',f'disk.profile_interval_orbits={a.profile_interval}',f'checkpoint_interval={a.checkpoint_interval}']
if a.max_steps is not None:command.append(f'max_timesteps={a.max_steps}')
if a.max_grid_size is not None:command.append(f'amr.max_grid_size={a.max_grid_size}')
if a.blocking_factor is not None:command.append(f'amr.blocking_factor={a.blocking_factor}')
if a.init_shrink is not None:command.append(f'init_shrink={a.init_shrink}')
if a.density_floor is not None:command.append(f'density_floor={a.density_floor}')
if a.reconstruction_order is not None:command.append(f'hydro.reconstruction_order={a.reconstruction_order}')
if a.max_walltime is not None:command.append(f'max_walltime={a.max_walltime}')
if a.verbose:command.append('suppress_output=0')
if a.restart:command.append(f'restartfile={a.restart.resolve()}')
if a.ranks>1:command=['mpirun','--oversubscribe','-np',str(a.ranks)]+command
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
meta=dict(command=command,cwd=str(a.output),cells=a.cells,target_orbits=a.orbits,source_sha256=sha(Path(__file__).with_name('testKeplerianDiskJoseph.cpp')),executable_sha256=sha(exe),input_sha256=sha(source),OMP_NUM_THREADS='1',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
(a.output/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
start=time.monotonic()
with (a.output/'run.log').open('w') as log:
 result=subprocess.run(command,cwd=a.output,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'})
meta.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-start)
profiles=sorted(a.output.glob('joseph_profile_*.txt'))
if profiles:
 meta['achieved_orbits']=float(re.search(r'orbits = (.*)',profiles[-1].read_text().splitlines()[0]).group(1))
 meta['target_reached']=meta['achieved_orbits']>=a.orbits-1.e-10
(a.output/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2),flush=True)
raise SystemExit(result.returncode)
