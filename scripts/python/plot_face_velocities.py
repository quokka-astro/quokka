#!/usr/bin/env python3
# ABOUTME: Plot face velocities from FAB files with ghost cell validation
# ABOUTME: Validates ghost cell consistency and outputs statistics for both correct and incorrect ghost cells
"""
Plot face velocity and reconstructed state FAB files with ghost cells highlighted.
Validates ghost cell consistency and outputs statistics for both correct and incorrect ghost cells.
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
            elif line.startswith('# Valid box:'):
                # Parse valid box bounds
                valid_str = line.split(':', 1)[1].strip()
                numbers = re.findall(r'-?\d+', valid_str)
                numbers = [int(n) for n in numbers]
                
                comma_count = valid_str.count(',')
                
                if box_info is None:
                    box_info = {}
                
                if comma_count == 0:  # 1D: ((0) (28) (1))
                    box_info['valid_lo'] = [numbers[0]]
                    box_info['valid_hi'] = [numbers[1]]
                elif comma_count == 3:  # 2D: ((0,0) (17,16) (1,0))
                    box_info['valid_lo'] = [numbers[0], numbers[1]]
                    box_info['valid_hi'] = [numbers[2], numbers[3]]
                elif comma_count == 5:  # 3D: ((lo_x,lo_y,lo_z) (hi_x,hi_y,hi_z))
                    box_info['valid_lo'] = [numbers[0], numbers[1], numbers[2]]
                    box_info['valid_hi'] = [numbers[3], numbers[4], numbers[5]]
            elif not line.startswith('#'):
                parts = line.strip().split()
                if parts:
                    data.append([float(x) for x in parts])
    
    return np.array(data), box_info


def parse_reconstructed_fab_file(filename):
    """Parse a reconstructed state FAB file and return the data and metadata"""
    data = []
    box_info = None
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('# Box:'):
                # Parse box bounds like ((-2) (30) (1))
                box_str = line.split(':', 1)[1].strip()
                # Extract numbers from the box string
                numbers = re.findall(r'-?\d+', box_str)
                numbers = [int(n) for n in numbers]
                
                # Parse box format which can be:
                # 1D: ((-2) (30) (1)) - single direction with placeholders
                # 2D: ((-2,-2) (18,17) (1,0)) - two directions with placeholders  
                
                # Determine dimensionality from the data format
                # Count commas to determine actual dimensions
                comma_count = box_str.count(',')
                
                if comma_count == 0:  # 1D: ((-2) (30) (1))
                    box_info = {
                        'lo': [numbers[0]],
                        'hi': [numbers[1]]
                    }
                elif comma_count == 3:  # 2D: ((-2,-2) (18,17) (1,0))
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
            elif line.startswith('# Valid box:'):
                # Parse valid box bounds
                valid_str = line.split(':', 1)[1].strip()
                numbers = re.findall(r'-?\d+', valid_str)
                numbers = [int(n) for n in numbers]
                
                comma_count = valid_str.count(',')
                
                if box_info is None:
                    box_info = {}
                
                if comma_count == 0:  # 1D: ((0) (28) (1))
                    box_info['valid_lo'] = [numbers[0]]
                    box_info['valid_hi'] = [numbers[1]]
                elif comma_count == 3:  # 2D: ((0,0) (17,16) (1,0))
                    box_info['valid_lo'] = [numbers[0], numbers[1]]
                    box_info['valid_hi'] = [numbers[2], numbers[3]]
                elif comma_count == 5:  # 3D: ((lo_x,lo_y,lo_z) (hi_x,hi_y,hi_z))
                    box_info['valid_lo'] = [numbers[0], numbers[1], numbers[2]]
                    box_info['valid_hi'] = [numbers[3], numbers[4], numbers[5]]
            elif not line.startswith('#'):
                parts = line.strip().split()
                if parts:
                    data.append([float(x) for x in parts])
    
    return np.array(data), box_info


def check_ghost_cell_consistency(all_data_list, all_boxes, tolerance=1e-10):
    """Check if ghost cells match valid cells from adjacent boxes and list all ghost cells"""
    mismatches = []
    correct_matches = []
    all_ghost_cells = []
    
    # Create a dictionary mapping indices to (box_id, value, is_ghost)
    index_map = {}
    
    for box_id, (data, box_info) in enumerate(zip(all_data_list, all_boxes)):
        if data.size == 0:
            continue
            
        indices = data[:, 0].astype(int)
        values = data[:, 1] if data.shape[1] == 2 else data[:, 1:]  # Handle multi-component data
        
        # Parse box bounds and valid box bounds from header
        box_lo = box_info['lo'][0]
        box_hi = box_info['hi'][0]
        valid_lo = box_info.get('valid_lo', [box_lo])[0] if 'valid_lo' in box_info else box_lo
        valid_hi = box_info.get('valid_hi', [box_hi])[0] if 'valid_hi' in box_info else box_hi
        
        # Count ghost cells properly: any cell outside the valid region
        for idx, val in zip(indices, values):
            is_ghost = (idx < valid_lo or idx > valid_hi)
            is_boundary = (idx == box_lo or idx == box_hi)
            
            # Collect ALL ghost cells (outside valid region)
            if is_ghost:
                if idx < valid_lo:
                    boundary_type = 'lo_ghost'
                    ghost_layer = valid_lo - idx  # how many cells from valid boundary
                else:
                    boundary_type = 'hi_ghost'
                    ghost_layer = idx - valid_hi  # how many cells from valid boundary
                
                all_ghost_cells.append({
                    'index': idx,
                    'box_id': box_id,
                    'value': val,
                    'boundary_type': boundary_type,
                    'ghost_layer': ghost_layer,
                    'is_boundary': is_boundary
                })
            
            if idx not in index_map:
                index_map[idx] = []
            
            index_map[idx].append({
                'box_id': box_id,
                'value': val,
                'is_ghost': is_boundary,  # Keep this for interface checking
                'lo': box_lo,
                'hi': box_hi
            })
    
    # Check for mismatches and correct matches (interface ghost cells only)
    for idx, entries in index_map.items():
        if len(entries) > 1:  # This index appears in multiple boxes
            # Find valid cell value(s)
            valid_entries = [e for e in entries if not e['is_ghost']]
            ghost_entries = [e for e in entries if e['is_ghost']]
            
            if valid_entries and ghost_entries:
                # There should be exactly one valid value
                if len(valid_entries) == 1:
                    valid_val = valid_entries[0]['value']
                    
                    # Check all ghost cells against this valid value
                    for ghost in ghost_entries:
                        if np.any(np.abs(ghost['value'] - valid_val) > tolerance):
                            mismatches.append({
                                'index': idx,
                                'valid_box': valid_entries[0]['box_id'],
                                'valid_value': valid_val,
                                'ghost_box': ghost['box_id'],
                                'ghost_value': ghost['value'],
                                'difference': ghost['value'] - valid_val
                            })
                        else:
                            correct_matches.append({
                                'index': idx,
                                'valid_box': valid_entries[0]['box_id'],
                                'valid_value': valid_val,
                                'ghost_box': ghost['box_id'],
                                'ghost_value': ghost['value']
                            })
                else:
                    # Multiple valid entries for same index - this shouldn't happen
                    print(f"Warning: Multiple valid cells at index {idx}")
    
    return mismatches, correct_matches, all_ghost_cells


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
    
    # Collect all data for ghost cell checking
    all_data = []
    all_boxes = []
    
    for fab_file in fab_files:
        data, box_info = parse_fab_file(fab_file)
        if data.size > 0:
            all_data.append(data)
            all_boxes.append(box_info)
    
    # Check ghost cell consistency
    mismatches, correct_matches, all_ghost_cells = check_ghost_cell_consistency(all_data, all_boxes)
    
    print(f"\n📊 All ghost cells in face velocity data:")
    print(f"  👻 Total ghost cells found: {len(all_ghost_cells)}")
    print(f"  ✅ Interface ghost cells (correctly matched): {len(correct_matches)}")
    print(f"  ⚠️  Interface ghost cells (disagreeing): {len(mismatches)}")
    
    if all_ghost_cells:
        print(f"\n👻 All {len(all_ghost_cells)} ghost cells:")
        for ghost in sorted(all_ghost_cells, key=lambda x: (x['box_id'], x['index'])):
            boundary_marker = " (boundary)" if ghost['is_boundary'] else ""
            print(f"  Box {ghost['box_id']}, Index {ghost['index']} ({ghost['boundary_type']}, layer {ghost['ghost_layer']}){boundary_marker}: {ghost['value']:.6e}")
    
    if correct_matches:
        print(f"\n✅ Details of {len(correct_matches)} correctly-filled interface ghost cells:")
        for m in correct_matches:
            print(f"  Index {m['index']}: Box {m['valid_box']} (valid) = {m['valid_value']:.6e}, "
                  f"Box {m['ghost_box']} (ghost) = {m['ghost_value']:.6e}")
    
    if mismatches:
        print(f"\n⚠️  Details of {len(mismatches)} interface ghost cell mismatches:")
        for m in mismatches:
            print(f"  Index {m['index']}: Box {m['valid_box']} (valid) = {m['valid_value']:.6e}, "
                  f"Box {m['ghost_box']} (ghost) = {m['ghost_value']:.6e}, "
                  f"diff = {m['difference']:.6e}")
    
    _, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fab_files)))
    
    for i, (data, box_info) in enumerate(zip(all_data, all_boxes)):
        if data.size == 0:
            continue
            
        # For 1D data: columns are [i, value]
        indices = data[:, 0].astype(int)
        values = data[:, 1]
        
        # Determine valid region from header information
        valid_lo = box_info.get('valid_lo', [box_info['lo'][0]])[0] if 'valid_lo' in box_info else box_info['lo'][0]
        valid_hi = box_info.get('valid_hi', [box_info['hi'][0]])[0] if 'valid_hi' in box_info else box_info['hi'][0]
        
        # Create masks for ghost and valid cells (ghost = outside valid region)
        is_ghost = np.logical_or(indices < valid_lo, indices > valid_hi)
        
        # Plot valid cells
        valid_mask = ~is_ghost
        ax.plot(indices[valid_mask], values[valid_mask], 'o-', 
                color=colors[i], label=f'Box {i} (valid)', markersize=3)
        
        # Plot ghost cells with different style
        if np.any(is_ghost):
            ax.plot(indices[is_ghost], values[is_ghost], 's', 
                    color=colors[i], markersize=4, markerfacecolor='none',
                    markeredgewidth=1.5, label=f'Box {i} (ghost)')
        
        # Add vertical lines at shared face locations between adjacent boxes
        if i > 0:  # Don't draw line before first box
            # The shared face is at the start of this box's valid region
            ax.axvline(x=valid_lo, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Face Index (i)')
    ax.set_ylabel('Face Velocity')
    ax.set_title(f'Face Velocities at Level {level}, Timestep {timestep}\n'
                 f'Ghost cells shown as squares')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'face_velocities_1d_lev{level}_step{timestep}.png', dpi=150, bbox_inches='tight')
    plt.show()




def check_reconstructed_ghost_cells(all_data_list, all_boxes, var_col=1, tolerance=1e-10):
    """Check if ghost cells match valid cells from adjacent boxes for reconstructed states"""
    mismatches = []
    correct_matches = []
    all_ghost_cells = []
    
    # Create a dictionary mapping indices to (box_id, value, is_ghost)
    index_map = {}
    
    for box_id, (data, box_info) in enumerate(zip(all_data_list, all_boxes)):
        if data.size == 0:
            continue
            
        indices = data[:, 0].astype(int)
        values = data[:, var_col]  # Get the specific variable column
        
        # Parse box bounds and valid box bounds from header
        box_lo = box_info['lo'][0]
        box_hi = box_info['hi'][0]
        valid_lo = box_info.get('valid_lo', [box_lo])[0] if 'valid_lo' in box_info else box_lo
        valid_hi = box_info.get('valid_hi', [box_hi])[0] if 'valid_hi' in box_info else box_hi
        
        for idx, val in zip(indices, values):
            is_ghost = (idx < valid_lo or idx > valid_hi)
            is_boundary = (idx == box_lo or idx == box_hi)
            
            # Collect ALL ghost cells (outside valid region)
            if is_ghost:
                if idx < valid_lo:
                    boundary_type = 'lo_ghost'
                    ghost_layer = valid_lo - idx  # how many cells from valid boundary
                else:
                    boundary_type = 'hi_ghost'
                    ghost_layer = idx - valid_hi  # how many cells from valid boundary
                
                all_ghost_cells.append({
                    'index': idx,
                    'box_id': box_id,
                    'value': val,
                    'boundary_type': boundary_type,
                    'ghost_layer': ghost_layer,
                    'is_boundary': is_boundary
                })
            
            if idx not in index_map:
                index_map[idx] = []
            
            index_map[idx].append({
                'box_id': box_id,
                'value': val,
                'is_ghost': is_boundary,  # Keep this for interface checking
                'lo': box_lo,
                'hi': box_hi
            })
    
    # Check for mismatches and correct matches (interface ghost cells only)
    for idx, entries in index_map.items():
        if len(entries) > 1:  # This index appears in multiple boxes
            # Find valid cell value(s)
            valid_entries = [e for e in entries if not e['is_ghost']]
            ghost_entries = [e for e in entries if e['is_ghost']]
            
            if valid_entries and ghost_entries:
                # There should be exactly one valid value
                if len(valid_entries) == 1:
                    valid_val = valid_entries[0]['value']
                    
                    # Check all ghost cells against this valid value
                    for ghost in ghost_entries:
                        if np.abs(ghost['value'] - valid_val) > tolerance:
                            mismatches.append({
                                'index': idx,
                                'valid_box': valid_entries[0]['box_id'],
                                'valid_value': valid_val,
                                'ghost_box': ghost['box_id'],
                                'ghost_value': ghost['value'],
                                'difference': ghost['value'] - valid_val
                            })
                        else:
                            correct_matches.append({
                                'index': idx,
                                'valid_box': valid_entries[0]['box_id'],
                                'valid_value': valid_val,
                                'ghost_box': ghost['box_id'],
                                'ghost_value': ghost['value']
                            })
                else:
                    # Multiple valid entries for same index - this shouldn't happen
                    print(f"Warning: Multiple valid cells at index {idx}")
    
    return mismatches, correct_matches, all_ghost_cells


def plot_1d_reconstructed_states(dirname, timestep=0, level=0, variable='density'):
    """Plot 1D reconstructed states from FAB files"""
    reconst_dir = Path(f"reconst_lev{level}_step{timestep}")
    if not reconst_dir.exists():
        print(f"Directory {reconst_dir} not found!")
        return
    
    # Find all left and right state FAB files for x-direction
    left_files = sorted(reconst_dir.glob("reconst_left_x_box_*.fab"))
    right_files = sorted(reconst_dir.glob("reconst_right_x_box_*.fab"))
    
    if not left_files or not right_files:
        print(f"No reconstructed state files found in {reconst_dir}")
        return
    
    # Variable mapping: variable name -> column index
    var_map = {
        'density': 1,
        'xmom': 2,
        'ymom': 3,
        'zmom': 4,
        'energy': 5,
        'intenergy': 6
    }
    
    if variable not in var_map:
        print(f"Unknown variable: {variable}. Available: {list(var_map.keys())}")
        return
    
    var_col = var_map[variable]
    
    # Collect all data for ghost cell checking
    all_left_data = []
    all_left_boxes = []
    all_right_data = []
    all_right_boxes = []
    
    for left_file in left_files:
        data, box_info = parse_reconstructed_fab_file(left_file)
        if data.size > 0:
            all_left_data.append(data)
            all_left_boxes.append(box_info)
    
    for right_file in right_files:
        data, box_info = parse_reconstructed_fab_file(right_file)
        if data.size > 0:
            all_right_data.append(data)
            all_right_boxes.append(box_info)
    
    # Check ghost cell consistency
    print(f"\nChecking ghost cell consistency for {variable}:")
    
    left_mismatches, left_correct, left_all_ghosts = check_reconstructed_ghost_cells(all_left_data, all_left_boxes, var_col)
    print(f"\n📊 LEFT state ghost cell statistics:")
    print(f"  👻 Total ghost cells found: {len(left_all_ghosts)}")
    print(f"  ✅ Interface ghost cells (correctly matched): {len(left_correct)}")
    print(f"  ⚠️  Interface ghost cells (disagreeing): {len(left_mismatches)}")
    
    if left_all_ghosts:
        print(f"\n👻 All {len(left_all_ghosts)} LEFT state ghost cells:")
        for ghost in sorted(left_all_ghosts, key=lambda x: (x['box_id'], x['index'])):
            boundary_marker = " (boundary)" if ghost['is_boundary'] else ""
            print(f"  Box {ghost['box_id']}, Index {ghost['index']} ({ghost['boundary_type']}, layer {ghost['ghost_layer']}){boundary_marker}: {ghost['value']:.6e}")
    
    if left_correct:
        print(f"\n✅ Details of {len(left_correct)} correctly-filled LEFT state interface ghost cells:")
        for m in left_correct:
            print(f"  Index {m['index']}: Box {m['valid_box']} (valid) = {m['valid_value']:.6e}, "
                  f"Box {m['ghost_box']} (ghost) = {m['ghost_value']:.6e}")
    
    if left_mismatches:
        print(f"\n⚠️  Details of {len(left_mismatches)} LEFT state interface ghost cell mismatches:")
        for m in left_mismatches:
            print(f"  Index {m['index']}: Box {m['valid_box']} (valid) = {m['valid_value']:.6e}, "
                  f"Box {m['ghost_box']} (ghost) = {m['ghost_value']:.6e}, "
                  f"diff = {m['difference']:.6e}")
    
    right_mismatches, right_correct, right_all_ghosts = check_reconstructed_ghost_cells(all_right_data, all_right_boxes, var_col)
    print(f"\n📊 RIGHT state ghost cell statistics:")
    print(f"  👻 Total ghost cells found: {len(right_all_ghosts)}")
    print(f"  ✅ Interface ghost cells (correctly matched): {len(right_correct)}")
    print(f"  ⚠️  Interface ghost cells (disagreeing): {len(right_mismatches)}")
    
    if right_all_ghosts:
        print(f"\n👻 All {len(right_all_ghosts)} RIGHT state ghost cells:")
        for ghost in sorted(right_all_ghosts, key=lambda x: (x['box_id'], x['index'])):
            boundary_marker = " (boundary)" if ghost['is_boundary'] else ""
            print(f"  Box {ghost['box_id']}, Index {ghost['index']} ({ghost['boundary_type']}, layer {ghost['ghost_layer']}){boundary_marker}: {ghost['value']:.6e}")
    
    if right_correct:
        print(f"\n✅ Details of {len(right_correct)} correctly-filled RIGHT state interface ghost cells:")
        for m in right_correct:
            print(f"  Index {m['index']}: Box {m['valid_box']} (valid) = {m['valid_value']:.6e}, "
                  f"Box {m['ghost_box']} (ghost) = {m['ghost_value']:.6e}")
    
    if right_mismatches:
        print(f"\n⚠️  Details of {len(right_mismatches)} RIGHT state interface ghost cell mismatches:")
        for m in right_mismatches:
            print(f"  Index {m['index']}: Box {m['valid_box']} (valid) = {m['valid_value']:.6e}, "
                  f"Box {m['ghost_box']} (ghost) = {m['ghost_value']:.6e}, "
                  f"diff = {m['difference']:.6e}")
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(left_files), len(right_files))))
    
    # Plot left states
    for i, left_file in enumerate(left_files):
        data, box_info = parse_reconstructed_fab_file(left_file)
        
        if data.size == 0:
            continue
            
        # For reconstructed state data: columns are [i, density, xmom, ymom, zmom, energy, intenergy]
        indices = data[:, 0].astype(int)
        values = data[:, var_col]
        
        # Determine valid region from header information
        valid_lo = box_info.get('valid_lo', [box_info['lo'][0]])[0] if 'valid_lo' in box_info else box_info['lo'][0]
        valid_hi = box_info.get('valid_hi', [box_info['hi'][0]])[0] if 'valid_hi' in box_info else box_info['hi'][0]
        
        # Create masks for ghost and valid cells (ghost = outside valid region)
        is_ghost = np.logical_or(indices < valid_lo, indices > valid_hi)
        
        # Plot valid cells
        valid_mask = ~is_ghost
        ax1.plot(indices[valid_mask], values[valid_mask], 'o-', 
                color=colors[i], label=f'Box {i} (valid)', markersize=3)
        
        # Plot ghost cells with different style
        if np.any(is_ghost):
            ax1.plot(indices[is_ghost], values[is_ghost], 's', 
                    color=colors[i], markersize=4, markerfacecolor='none',
                    markeredgewidth=1.5, label=f'Box {i} (ghost)')
        
        # Add vertical lines at shared face locations between adjacent boxes
        if i > 0:  # Don't draw line before first box
            ax1.axvline(x=valid_lo, color='gray', linestyle='--', alpha=0.5)
    
    # Plot right states
    for i, right_file in enumerate(right_files):
        data, box_info = parse_reconstructed_fab_file(right_file)
        
        if data.size == 0:
            continue
            
        # For reconstructed state data: columns are [i, density, xmom, ymom, zmom, energy, intenergy]
        indices = data[:, 0].astype(int)
        values = data[:, var_col]
        
        # Determine valid region from header information
        valid_lo = box_info.get('valid_lo', [box_info['lo'][0]])[0] if 'valid_lo' in box_info else box_info['lo'][0]
        valid_hi = box_info.get('valid_hi', [box_info['hi'][0]])[0] if 'valid_hi' in box_info else box_info['hi'][0]
        
        # Create masks for ghost and valid cells (ghost = outside valid region)
        is_ghost = np.logical_or(indices < valid_lo, indices > valid_hi)
        
        # Plot valid cells
        valid_mask = ~is_ghost
        ax2.plot(indices[valid_mask], values[valid_mask], 'o-', 
                color=colors[i], label=f'Box {i} (valid)', markersize=3)
        
        # Plot ghost cells with different style
        if np.any(is_ghost):
            ax2.plot(indices[is_ghost], values[is_ghost], 's', 
                    color=colors[i], markersize=4, markerfacecolor='none',
                    markeredgewidth=1.5, label=f'Box {i} (ghost)')
        
        # Add vertical lines at box boundaries
        if i > 0:  # Don't draw line before first box
            ax2.axvline(x=valid_lo, color='gray', linestyle='--', alpha=0.5)
    
    # Configure plots
    ax1.set_xlabel('Face Index (i)')
    ax1.set_ylabel(f'Left {variable}')
    ax1.set_title(f'Left Reconstructed States ({variable}) at Level {level}, Timestep {timestep}\n'
                  f'Ghost cells shown as squares')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_xlabel('Face Index (i)')
    ax2.set_ylabel(f'Right {variable}')
    ax2.set_title(f'Right Reconstructed States ({variable}) at Level {level}, Timestep {timestep}\n'
                  f'Ghost cells shown as squares')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'reconstructed_states_1d_{variable}_lev{level}_step{timestep}.png', dpi=150, bbox_inches='tight')
    plt.show()




def main():
    parser = argparse.ArgumentParser(description='Plot 1D face velocity and reconstructed state FAB files')
    parser.add_argument('--mode', type=str, default='facevel', choices=['facevel', 'reconst'],
                        help='Plotting mode: facevel (face velocities) or reconst (reconstructed states)')
    parser.add_argument('--timestep', type=int, default=0,
                        help='Timestep to plot')
    parser.add_argument('--level', type=int, default=0,
                        help='AMR level to plot')
    parser.add_argument('--variable', type=str, default='xmom', 
                        choices=['density', 'xmom', 'ymom', 'zmom', 'energy', 'intenergy'],
                        help='Variable to plot for reconstructed states')
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing facevel_* and reconst_* subdirectories')
    
    args = parser.parse_args()
    
    # Change to specified directory
    if args.dir != '.':
        os.chdir(args.dir)
    
    if args.mode == 'facevel':
        plot_1d_face_velocities(args.dir, args.timestep, args.level)
    elif args.mode == 'reconst':
        plot_1d_reconstructed_states(args.dir, args.timestep, args.level, args.variable)


if __name__ == '__main__':
    main()