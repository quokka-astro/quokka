#!/usr/bin/env python3
"""
Plot metallicity (chemical scalars) for ChuhanProblem at all timesteps.

Usage:
    ./plot_chuhan_metallicity.py <output_dir> [-o <output_dir>] [-j <n_procs>]

Example:
    ./plot_chuhan_metallicity.py build-chuhan/src/problems/ChuhanProblem/chuhan_*/ -o figures/ -j 4
"""

import sys
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool

try:
    import yt
    yt.set_log_level(40)
except ImportError:
    print("ERROR: yt not installed. Install with:")
    print("  pip install yt")
    sys.exit(1)

# Chemical scalar field mapping
METALLICITY_FIELDS = {
    "scalar_0": {"name": "C12", "color": "viridis"},
    "scalar_1": {"name": "N14", "color": "plasma"},
    "scalar_2": {"name": "O16", "color": "cool"},
}

def plot_metallicity_field(args_tuple):
    """Plot a single metallicity field for a single timestep."""
    pltdir, field_key, field_info, output_dir = args_tuple
    
    try:
        # Load dataset
        ds = yt.load(pltdir)
        
        # Extract timestep number from path
        pltdir_str = str(pltdir)
        if "plt" in pltdir_str:
            ts_str = pltdir_str.split("plt")[-1].split("/")[0]
        else:
            ts_str = "unknown"
        
        # Create slice plot for this field
        slc = yt.SlicePlot(
            ds, "z", field_key,
            center="c",
            width=(1.0, "code_length")
        )
        
        # Customize appearance
        slc.set_cmap(field_key, field_info["color"])
        # Let yt auto-scale the color limits for each field
        slc.annotate_grids(linewidth=0.5, alpha=0.3)
        
        # Try to annotate particles if they exist
        try:
            slc.annotate_particles(
                1.0, 
                ptype="io",
                marker="*",
                markersize=5
            )
        except:
            pass  # Particles not available or not in snapshot
        
        # Save figure
        output_path = Path(output_dir) / f"{field_key}_ts{ts_str}"
        slc.save(str(output_path), mpl_kwargs={"dpi": 150, "bbox_inches": "tight"})
        
        print(f"✓ {pltdir}/{field_key} → {output_path}.png")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {pltdir}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate metallicity (chemical scalar) plots for ChuhanProblem"
    )
    parser.add_argument(
        "plotdirs",
        nargs="+",
        help="Plotfile directories (plt*/) or glob pattern"
    )
    parser.add_argument(
        "-o", "--output",
        default="figures",
        help="Output directory for PNG files (default: figures/)"
    )
    parser.add_argument(
        "-j", "--n_processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns in plotdirs
    all_pltdirs = []
    for pattern in args.plotdirs:
        matched = sorted(glob.glob(pattern))
        if matched:
            all_pltdirs.extend(matched)
        else:
            # Try as direct directory
            if os.path.isdir(pattern):
                all_pltdirs.append(pattern)
    
    if not all_pltdirs:
        print("ERROR: No plotfile directories found!")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(all_pltdirs)} plotfiles")
    field_names = ", ".join([v["name"] for v in METALLICITY_FIELDS.values()])
    print(f"Chemical fields: {field_names}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Build task list: (pltdir, field_key, field_info, output_dir)
    tasks = []
    for pltdir in all_pltdirs:
        for field_key, field_info in METALLICITY_FIELDS.items():
            tasks.append((pltdir, field_key, field_info, output_dir))
    
    # Process tasks
    if args.n_processes > 1:
        print(f"Processing {len(tasks)} plots with {args.n_processes} workers...\n")
        with Pool(processes=args.n_processes) as pool:
            results = pool.map(plot_metallicity_field, tasks)
    else:
        print(f"Processing {len(tasks)} plots (serial)...\n")
        results = [plot_metallicity_field(task) for task in tasks]
    
    # Summary
    success = sum(results)
    print()
    print(f"✓ Successfully generated {success}/{len(tasks)} plots")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
