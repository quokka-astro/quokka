#!/usr/bin/env python3
"""
Generate density and metallicity visualizations from new plotfiles using yt
This will create PNG images of z-slices showing the SN event
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import os
import glob
from pathlib import Path

# Suppress some yt warnings
yt.funcs.mylog.setLevel(30)  # Set to WARNING level

def load_plotfile(plotfile_path):
    """Load a plotfile using yt"""
    try:
        ds = yt.load(plotfile_path)
        return ds
    except Exception as e:
        print(f"Error loading {plotfile_path}: {e}")
        return None

def create_density_slice(ds, output_dir, ts_num):
    """Create a density slice (z-direction) using yt"""
    try:
        # Create slice
        slc = yt.SlicePlot(ds, 'z', 'gasDensity', width=(1e20, 1e20))
        
        # Set colormap and normalization
        slc.set_cmap('gasDensity', 'Blues')
        slc.set_log('gasDensity', True)
        
        # Save
        filename = f'density_ts{ts_num:07d}_Slice_z_gasDensity'
        output_path = os.path.join(output_dir, filename)
        slc.save(output_path, mpl_kwargs={'dpi': 150})
        
        print(f"✓ Saved: {filename}.png")
        return True
    except Exception as e:
        print(f"✗ Error creating density slice: {e}")
        return False

def create_metallicity_slice(ds, output_dir, ts_num, scalar_idx=0):
    """Create a metallicity/scalar slice (z-direction) using yt"""
    try:
        scalar_name = f'scalar_{scalar_idx}'
        
        # Create slice
        slc = yt.SlicePlot(ds, 'z', scalar_name, width=(1e20, 1e20))
        
        # Set colormap - use hot for metallicity
        slc.set_cmap(scalar_name, 'hot')
        slc.set_log(scalar_name, False)
        
        # Save
        filename = f'scalar_{scalar_idx}_ts{ts_num:07d}_Slice_z_scalar_{scalar_idx}'
        output_path = os.path.join(output_dir, filename)
        slc.save(output_path, mpl_kwargs={'dpi': 150})
        
        print(f"✓ Saved: {filename}.png")
        return True
    except Exception as e:
        print(f"✗ Error creating metallicity slice for scalar_{scalar_idx}: {e}")
        return False

def main():
    """Main workflow"""
    plotfile_base = "/Users/meow/quokka/tests"
    output_density_dir = os.path.join(plotfile_base, "chuhan_density_sn_plots")
    output_metallicity_dir = os.path.join(plotfile_base, "chuhan_metallicity_plots")
    
    # Create output directories if needed
    os.makedirs(output_density_dir, exist_ok=True)
    os.makedirs(output_metallicity_dir, exist_ok=True)
    
    # Get all new plotfiles (skip .old versions)
    all_plotfiles = sorted(glob.glob(os.path.join(plotfile_base, "plt[0-9]*")))
    plotfiles = [pf for pf in all_plotfiles if os.path.isdir(pf) and '.old' not in pf]
    
    print(f"Found {len(plotfiles)} plotfiles")
    print(f"Generating visualizations for key frames (especially frame 12 with SN)...\n")
    
    # Generate for specific frames: 0, 5, 10, 12 (SN frame!), 15, 20, 25, 30
    key_frames = [0, 5, 10, 12, 15, 20, 25, 30]
    
    for frame_idx in key_frames:
        if frame_idx < len(plotfiles):
            plotfile = plotfiles[frame_idx]
            ts_num = frame_idx
            
            print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            if frame_idx == 12:
                print(f"🌟 FRAME #{frame_idx:2d} (SN EVENT!) - Loading {os.path.basename(plotfile)}")
            else:
                print(f"📊 FRAME #{frame_idx:2d} - Loading {os.path.basename(plotfile)}")
            
            # Load plotfile
            ds = load_plotfile(plotfile)
            if ds is None:
                print(f"✗ Failed to load plotfile {frame_idx}")
                continue
            
            # Get metadata
            try:
                time_value = float(ds.current_time.to('Myr'))
                print(f"  Time: {time_value:.2f} Myr")
            except:
                print(f"  Time: [unable to extract]")
            
            # Create density slice
            print(f"  Creating density slice...")
            create_density_slice(ds, output_density_dir, ts_num)
            
            # Create metallicity slices (scalars 0, 1, 2)
            print(f"  Creating metallicity slices...")
            for scalar_idx in range(3):
                create_metallicity_slice(ds, output_metallicity_dir, ts_num, scalar_idx)
            
            print(f"  ✓ Complete for frame {frame_idx}")

    print("\n" + "="*60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*60)
    print(f"""
✓ Density plots saved to:     {output_density_dir}/
✓ Metallicity plots saved to: {output_metallicity_dir}/

Key frames with visualizations:
  Frame #0  - Initial conditions
  Frame #5  - Normal evolution
  Frame #10 - Before SN
  Frame #12 - ⭐ SN EVENT! (Look here for bright spot + metal enrichment)
  Frame #15 - After SN explosion
  Frame #20 - Shock wave expansion
  Frame #25 - Late evolution
  Frame #30 - Final state

To view the SN:
  1. Open: density_ts0000012_Slice_z_gasDensity.png
     → Look for BRIGHT WHITE SPOT (SN center)
  
  2. Open: scalar_0_ts0000012_Slice_z_scalar_0.png (metal 1)
  3. Open: scalar_1_ts0000012_Slice_z_scalar_1.png (metal 2)
  4. Open: scalar_2_ts0000012_Slice_z_scalar_2.png (metal 3)
     → Look for YELLOW/RED/ORANGE REGIONS (metal enrichment)
""")

if __name__ == '__main__':
    main()
