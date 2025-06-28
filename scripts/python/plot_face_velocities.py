#!/usr/bin/env python3
"""
Plot face velocity FAB files with ghost cells highlighted
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def parse_fab_file(filename):
    """Parse a FAB file and return the data and metadata"""
    data = []
    box_info = None
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('# Box:'):
                # Parse box bounds like ((-1) (113) (1)) or ((lo_x lo_y lo_z) (hi_x hi_y hi_z))
                box_str = line.split(':', 1)[1].strip()
                # Extract numbers from the box string
                numbers = re.findall(r'-?\d+', box_str)
                numbers = [int(n) for n in numbers]
                
                # Parse box format which can be:
                # 1D: ((-1) (113) (1)) - single direction with placeholders
                # 2D: ((-1,-1) (17,16) (1,0)) - two directions with placeholders  
                # 3D: ((lo_x,lo_y,lo_z) (hi_x,hi_y,hi_z)) - full 3D
                
                # Determine dimensionality from the data format
                # Count commas to determine actual dimensions
                comma_count = box_str.count(',')
                
                if comma_count == 0:  # 1D: ((-1) (113) (1))
                    box_info = {
                        'lo': [numbers[0]],
                        'hi': [numbers[1]]
                    }
                elif comma_count == 3:  # 2D: ((-1,-1) (17,16) (1,0))
                    # For 2D, first 4 numbers are the actual bounds: lo_i, lo_j, hi_i, hi_j
                    # The last 2 numbers (1,0) are stride/type indicators
                    box_info = {
                        'lo': [numbers[0], numbers[1]], 
                        'hi': [numbers[2], numbers[3]]
                    }
                elif comma_count == 5:  # 3D: ((lo_x,lo_y,lo_z) (hi_x,hi_y,hi_z))
                    box_info = {
                        'lo': [numbers[0], numbers[1], numbers[2]],
                        'hi': [numbers[3], numbers[4], numbers[5]]
                    }
                else:
                    print(f"Warning: Could not parse box format: {box_str}")
                    box_info = None
            elif not line.startswith('#'):
                parts = line.strip().split()
                if parts:
                    data.append([float(x) for x in parts])
    
    return np.array(data), box_info


def plot_1d_face_velocities(dirname, timestep=0, level=0):
    """Plot 1D face velocities from FAB files"""
    fab_dir = Path(f"facevel_lev{level}_step{timestep}")
    if not fab_dir.exists():
        print(f"Directory {fab_dir} not found!")
        return
    
    # Find all x-direction FAB files
    fab_files = sorted(fab_dir.glob("facevel_x_box_*.fab"))
    
    if not fab_files:
        print(f"No FAB files found in {fab_dir}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fab_files)))
    
    for i, fab_file in enumerate(fab_files):
        data, box_info = parse_fab_file(fab_file)
        
        if data.size == 0:
            continue
            
        # For 1D data: columns are [i, value]
        indices = data[:, 0].astype(int)
        values = data[:, 1]
        
        # Determine valid region (exclude first and last ghost cells)
        lo_idx = box_info['lo'][0]
        hi_idx = box_info['hi'][0]
        
        # Create masks for ghost and valid cells
        is_ghost = np.logical_or(indices == lo_idx, indices == hi_idx)
        
        # Plot valid cells
        valid_mask = ~is_ghost
        ax.plot(indices[valid_mask], values[valid_mask], 'o-', 
                color=colors[i], label=f'Box {i} (valid)', markersize=3)
        
        # Plot ghost cells with different style
        if np.any(is_ghost):
            ax.plot(indices[is_ghost], values[is_ghost], 's', 
                    color=colors[i], markersize=4, markerfacecolor='none',
                    markeredgewidth=1.5, label=f'Box {i} (ghost)')
        
        # Add vertical lines at box boundaries
        if i > 0:  # Don't draw line before first box
            ax.axvline(x=lo_idx + 0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Face Index (i)')
    ax.set_ylabel('Face Velocity')
    ax.set_title(f'Face Velocities at Level {level}, Timestep {timestep}\n'
                 f'Ghost cells shown as squares')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'face_velocities_1d_lev{level}_step{timestep}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_2d_face_velocities(dirname, direction='x', timestep=0, level=0):
    """Plot 2D face velocities from FAB files"""
    fab_dir = Path(f"facevel_lev{level}_step{timestep}")
    if not fab_dir.exists():
        print(f"Directory {fab_dir} not found!")
        return
    
    # Find all FAB files for the specified direction
    fab_files = sorted(fab_dir.glob(f"facevel_{direction}_box_*.fab"))
    
    if not fab_files:
        print(f"No FAB files found for direction {direction} in {fab_dir}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Collect all data to determine global bounds
    all_data = []
    all_boxes = []
    
    for fab_file in fab_files:
        data, box_info = parse_fab_file(fab_file)
        if data.size > 0:
            all_data.append(data)
            all_boxes.append(box_info)
    
    if not all_data:
        print("No data found in FAB files")
        return
    
    # For 2D data: columns are [i, j, value]
    # Determine global bounds
    i_min = min(int(data[:, 0].min()) for data in all_data)
    i_max = max(int(data[:, 0].max()) for data in all_data)
    j_min = min(int(data[:, 1].min()) for data in all_data)
    j_max = max(int(data[:, 1].max()) for data in all_data)
    
    # Create a grid for plotting
    grid = np.full((j_max - j_min + 1, i_max - i_min + 1), np.nan)
    ghost_mask = np.zeros_like(grid, dtype=bool)
    
    # Fill the grid with data from each box
    for data, box_info in zip(all_data, all_boxes):
        indices_i = data[:, 0].astype(int)
        indices_j = data[:, 1].astype(int)
        values = data[:, 2]
        
        lo_i, lo_j = box_info['lo']
        hi_i, hi_j = box_info['hi']
        
        for idx in range(len(indices_i)):
            i = indices_i[idx]
            j = indices_j[idx]
            grid_i = i - i_min
            grid_j = j - j_min
            
            grid[grid_j, grid_i] = values[idx]
            
            # Mark ghost cells
            if i == lo_i or i == hi_i or j == lo_j or j == hi_j:
                ghost_mask[grid_j, grid_i] = True
    
    # Create the plot - use scatter plot approach for better control
    # Use a dictionary to avoid plotting the same face multiple times
    face_data = {}  # (i,j) -> (value, is_ghost)
    
    # Collect all data points, handling duplicates
    for data, box_info in zip(all_data, all_boxes):
        indices_i = data[:, 0].astype(int)
        indices_j = data[:, 1].astype(int)
        values = data[:, 2]
        
        lo_i, lo_j = box_info['lo']
        hi_i, hi_j = box_info['hi']
        
        for idx in range(len(indices_i)):
            i = indices_i[idx]
            j = indices_j[idx]
            key = (i, j)
            
            # Mark ghost cells (at boundaries of this box)
            is_ghost = (i == lo_i or i == hi_i or j == lo_j or j == hi_j)
            
            # If this face already exists, prefer non-ghost classification
            if key in face_data:
                existing_value, existing_is_ghost = face_data[key]
                # Keep existing if it's not a ghost, or update if current is not ghost
                if existing_is_ghost and not is_ghost:
                    face_data[key] = (values[idx], is_ghost)
            else:
                face_data[key] = (values[idx], is_ghost)
    
    # Convert to arrays
    all_i = np.array([pos[0] for pos in face_data.keys()])
    all_j = np.array([pos[1] for pos in face_data.keys()])
    all_values = np.array([data[0] for data in face_data.values()])
    all_is_ghost = np.array([data[1] for data in face_data.values()])
    
    # Plot valid cells
    valid_mask = ~all_is_ghost
    if np.any(valid_mask):
        scatter = ax.scatter(all_i[valid_mask], all_j[valid_mask], 
                           c=all_values[valid_mask], s=15, cmap='viridis', 
                           marker='o', edgecolors='black', linewidth=0.3)
    
    # Plot ghost cells with different style
    if np.any(all_is_ghost):
        ax.scatter(all_i[all_is_ghost], all_j[all_is_ghost], 
                  c=all_values[all_is_ghost], s=25, cmap='viridis',
                  marker='s', edgecolors='red', linewidth=1)
    
    # Draw box boundaries
    for box_info in all_boxes:
        lo_i, lo_j = box_info['lo']
        hi_i, hi_j = box_info['hi']
        
        # Draw rectangle for each box (excluding ghost cells)
        rect = plt.Rectangle((lo_i + 0.5, lo_j + 0.5), 
                           hi_i - lo_i - 1, hi_j - lo_j - 1,
                           fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    ax.set_xlabel(f'Face Index i ({direction}-faces)')
    ax.set_ylabel('Face Index j')
    ax.set_title(f'{direction.upper()}-Face Velocities at Level {level}, Timestep {timestep}\n'
                 f'Ghost cells shown as red squares, box boundaries in red')
    
    # Add colorbar if we have valid data
    if np.any(valid_mask):
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Face Velocity')
    
    plt.tight_layout()
    plt.savefig(f'face_velocities_2d_{direction}_lev{level}_step{timestep}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot face velocity FAB files')
    parser.add_argument('--dim', type=int, default=1, choices=[1, 2],
                        help='Dimensionality of the plot (1 or 2)')
    parser.add_argument('--timestep', type=int, default=0,
                        help='Timestep to plot')
    parser.add_argument('--level', type=int, default=0,
                        help='AMR level to plot')
    parser.add_argument('--direction', type=str, default='x', choices=['x', 'y', 'z'],
                        help='Direction for face velocities (for 2D plots)')
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing facevel_* subdirectories')
    
    args = parser.parse_args()
    
    # Change to specified directory
    if args.dir != '.':
        os.chdir(args.dir)
    
    if args.dim == 1:
        plot_1d_face_velocities(args.dir, args.timestep, args.level)
    elif args.dim == 2:
        plot_2d_face_velocities(args.dir, args.direction, args.timestep, args.level)


if __name__ == '__main__':
    main()