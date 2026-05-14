#!/usr/bin/env python3
"""
Generate density visualizations with manual range adjustment.

This script checks whether the density field itself carries any visible
contrast in the selected frames.
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

def create_contrast_density_slice(plotfile_path, output_dir, ts_num, vmin_exp=-30, vmax_exp=-20):
    """Create density slice with enhanced contrast"""
    try:
        print(f"  Loading {os.path.basename(plotfile_path)}...")
        ds = yt.load(plotfile_path)
        
        # Extract z-slice data
        print(f"  Extracting density field...")
        ad = ds.all_data()
        
        # Create slice plot
        slc = yt.SlicePlot(ds, 'z', 'gasDensity', width=(1e20, 1e20))
        
        # Method 1: Log scale with manual range
        slc.set_cmap('gasDensity', 'viridis')
        slc.set_log('gasDensity', True)
        
        # Use the actual field range. These frames are effectively uniform, so
        # arbitrary log ranges can trigger normalization problems in yt.
        dens = ad[("boxlib", "gasDensity")]
        dens_min = float(dens.min().to_value())
        dens_max = float(dens.max().to_value())
        if dens_min == dens_max:
            dens_min *= 0.999
            dens_max *= 1.001

        print(f"  Setting density range: {dens_min:.3e} to {dens_max:.3e} g/cm^3")
        slc.set_zlim('gasDensity', dens_min, dens_max)
        
        # Save
        filename_base = f'density_ts{ts_num:07d}_Slice_z_gasDensity_HighContrast'
        output_path = os.path.join(output_dir, filename_base)
        slc.save(output_path, mpl_kwargs={'dpi': 200})
        
        print(f"✓ Saved: {filename_base}.png")
        
        # Also create a linear scale version
        slc2 = yt.SlicePlot(ds, 'z', 'gasDensity', width=(1e20, 1e20))
        slc2.set_cmap('gasDensity', 'hot')
        slc2.set_log('gasDensity', False)  # Linear scale
        slc2.set_zlim('gasDensity', dens_min, dens_max)
        
        filename_base2 = f'density_ts{ts_num:07d}_Slice_z_gasDensity_Linear'
        output_path2 = os.path.join(output_dir, filename_base2)
        slc2.save(output_path2, mpl_kwargs={'dpi': 200})
        
        print(f"✓ Saved: {filename_base2}.png")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Main workflow"""
    plotfile_base = "/Users/meow/quokka/tests"
    output_dir = os.path.join(plotfile_base, "chuhan_density_sn_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Focus on frames around the currently inspected region.
    # The density field is uniform in these frames, so the goal is to check
    # whether any tiny contrast exists at all.
    key_frames = [
        (0, "初始条件"),
        (5, "早期演化"),
        (10, "候选帧"),
        (12, "候选帧"),
        (15, "后续帧"),
    ]
    
    print("="*70)
    print("生成高对比度密度图以检测SN信号")
    print("="*70)
    
    for frame_idx, description in key_frames:
        plotfile = os.path.join(plotfile_base, f"plt{frame_idx:07d}")
        if os.path.exists(plotfile):
            print(f"\n🔍 Frame #{frame_idx} - {description}")
            
            create_contrast_density_slice(plotfile, output_dir, frame_idx)

    print("\n" + "="*70)
    print("✓ 高对比度可视化完成！")
    print("="*70)
    print("""
生成的文件：
  • density_ts*_*_HighContrast.png  (对数缩放，增强对比)
  • density_ts*_*_Linear.png        (线性缩放)

检查 Frame 10, 12, 15 的图像：
    • 如果仍然是单一色块，说明 density 场本身基本均匀
    • 这时应优先检查 metallicity 或 temperature，而不是继续调整色标
""")

if __name__ == '__main__':
    main()
