#!/usr/bin/env python3
import glob
import os
import re
import time

import yt


yt.funcs.mylog.setLevel(30)

base = '/Users/meow/quokka/tests'
out_d = os.path.join(base, 'chuhan_current_density_plots')
out_m = os.path.join(base, 'chuhan_current_metallicity_plots')
os.makedirs(out_d, exist_ok=True)
os.makedirs(out_m, exist_ok=True)

# Clean only this run's outputs
for p in glob.glob(os.path.join(out_d, '*.png')):
    os.remove(p)
for p in glob.glob(os.path.join(out_m, '*.png')):
    os.remove(p)

plot_dirs = [p for p in glob.glob(os.path.join(base, 'plt[0-9]*')) if os.path.isdir(p) and '.old' not in p]
idx_to_path = {}
mtime_by_idx = {}
for p in plot_dirs:
    m = re.search(r'plt(\d+)$', os.path.basename(p))
    if m:
        idx = int(m.group(1))
        idx_to_path[idx] = p
        header = os.path.join(p, 'Header')
        mtime_by_idx[idx] = os.path.getmtime(header) if os.path.exists(header) else os.path.getmtime(p)

indices = sorted(idx_to_path.keys())
if not indices:
    raise SystemExit('No plotfiles found')

# Select files from the newest run only. We keep plotfiles whose mtime is close
# to the newest file, so stale files from older runs are not mixed in.
latest_mtime = max(mtime_by_idx.values())
window_sec = 180.0
sel = sorted([idx for idx, mt in mtime_by_idx.items() if (latest_mtime - mt) <= window_sec])
if not sel:
    raise SystemExit('No plotfiles selected for latest run window')

print(
    f"Using latest-run window ({window_sec:.0f}s): "
    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_mtime))}, "
    f"frames={len(sel)}, first=plt{sel[0]:07d}, last=plt{sel[-1]:07d}"
)

for i in sel:
    pf = idx_to_path[i]
    ds = yt.load(pf)
    ad = ds.all_data()

    rho = ad[("boxlib", "gasDensity")]
    rho_min = float(rho.min())
    rho_max = float(rho.max())
    if rho_min == rho_max:
        rho_min *= 0.999
        rho_max *= 1.001

    s1 = yt.SlicePlot(ds, 'z', 'gasDensity', width=(1e20, 1e20))
    s1.set_cmap('gasDensity', 'Blues')
    s1.set_log('gasDensity', True)
    s1.set_zlim('gasDensity', rho_min, rho_max)
    s1.save(os.path.join(out_d, f'density_ts{i:07d}_Slice_z_gasDensity'), mpl_kwargs={'dpi': 150})

    s0 = ad[("boxlib", "scalar_0")]
    s0_max = float(s0.max())
    s0_hi = s0_max if s0_max > 0.0 else 1.0

    s2 = yt.SlicePlot(ds, 'z', 'scalar_0', width=(1e20, 1e20))
    s2.set_cmap('scalar_0', 'hot')
    s2.set_log('scalar_0', False)
    s2.set_zlim('scalar_0', 0.0, s0_hi)
    s2.save(os.path.join(out_m, f'scalar_0_ts{i:07d}_Slice_z_scalar_0'), mpl_kwargs={'dpi': 150})

    print(f'Saved frame {i:07d} rho=[{rho_min:.3e},{rho_max:.3e}] s0_max={s0_max:.3e}')

print('DONE')
