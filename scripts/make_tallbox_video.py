#!/usr/bin/env python3
"""Generate 2D projection videos from TallBoxSf chemistry simulation plotfiles.

Fields: gas density, total C12, SNII C12, WR C12, AGB C12
"""
import argparse, glob, os, subprocess, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plotdir', default='tests/tallbox_chem')
    p.add_argument('--outdir', default='tests/tallbox_chem/videos')
    p.add_argument('--fps', type=int, default=4)
    args = p.parse_args()

    plts = sorted([d for d in glob.glob(os.path.join(args.plotdir, 'plt*'))
                   if os.path.isdir(d) and '.old.' not in os.path.basename(d)])
    if not plts:
        print('No plotfiles found')
        sys.exit(1)
    print(f'Found {len(plts)} plotfiles')

    os.makedirs(args.outdir, exist_ok=True)

    channels = [
        ('gasDensity', 'Gas Density', 'viridis'),
        ('scalar_1', 'Total C12', 'inferno'),
        ('scalar_6', 'SNII C12', 'plasma'),
        ('scalar_11', 'WR C12', 'cividis'),
        ('scalar_16', 'AGB C12', 'magma'),
    ]

    all_frames = {ch[0]: [] for ch in channels}
    combined_frames = []

    for i, pf in enumerate(plts):
        ds = yt.load(pf)
        t_myr = ds.current_time / 3.15576e13

        for field, title, cmap in channels:
            fld_tuple = ('boxlib', field)
            if fld_tuple not in ds.field_list:
                continue
            slc = yt.ProjectionPlot(ds, 'z', fld_tuple, weight_field=None)
            slc.set_cmap(fld_tuple, cmap)
            slc.set_axes_unit('kpc')
            slc.annotate_title(f'{title}  t={t_myr:.1f} Myr')
            frame_path = os.path.join(args.outdir, f'{field}_{i:04d}.png')
            slc.save(frame_path)
            plt.close('all')
            all_frames[field].append(frame_path)

        # Combined 5-panel
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f'TallBox Chemistry -- t = {t_myr:.1f} Myr', fontsize=14)
        for j, (field, title, cmap) in enumerate(channels):
            ax = axes[j // 3][j % 3]
            fld_tuple = ('boxlib', field)
            if fld_tuple not in ds.field_list:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                continue
            slc = yt.ProjectionPlot(ds, 'z', fld_tuple, weight_field=None)
            slc.set_cmap(fld_tuple, cmap)
            slc.set_axes_unit('kpc')
            slc.annotate_title(title)
            tmp = os.path.join(args.outdir, f'_tmp_{i}_{j}.png')
            slc.save(tmp)
            img = plt.imread(tmp)
            ax.imshow(img, origin='lower')
            ax.set_title(title)
            ax.axis('off')
            os.remove(tmp)
        axes[1][2].set_visible(False)
        plt.tight_layout()
        cframe = os.path.join(args.outdir, f'combined_{i:04d}.png')
        fig.savefig(cframe, dpi=120, bbox_inches='tight')
        plt.close(fig)
        combined_frames.append(cframe)

        if i % 3 == 0:
            print(f'  Frame {i+1}/{len(plts)}  t={t_myr:.1f} Myr')

    for field, title, cmap in channels:
        frames = all_frames[field]
        if not frames:
            continue
        vid = os.path.join(args.outdir, f'tallbox_{field}.mp4')
        flist = os.path.join(args.outdir, f'_flist_{field}.txt')
        with open(flist, 'w') as f:
            for fr in frames:
                f.write(f"file '{os.path.abspath(fr)}'\n")
                f.write(f"duration {1/args.fps}\n")
        with open(flist, 'a') as f:
            f.write(f"file '{os.path.abspath(frames[-1])}'\n")
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', flist,
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', vid], check=True)
        os.remove(flist)
        print(f'  {title}: {vid}')

    cvid = os.path.join(args.outdir, 'tallbox_combined.mp4')
    flist = os.path.join(args.outdir, '_flist_combined.txt')
    with open(flist, 'w') as f:
        for fr in combined_frames:
            f.write(f"file '{os.path.abspath(fr)}'\n")
            f.write(f"duration {1/args.fps}\n")
    with open(flist, 'a') as f:
        f.write(f"file '{os.path.abspath(combined_frames[-1])}'\n")
    subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', flist,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', cvid], check=True)
    os.remove(flist)
    print(f'  Combined: {cvid}')
    print('\nDone!')

if __name__ == '__main__':
    main()
