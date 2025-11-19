#!/usr/bin/env python3
"""
Script to parse Quokka problem directories and generate a table of all problems.
Run it in the root of the Quokka repository to generate the table.

    python3 docs/markdown/tests/parse_problems.py

This will generate the table in `docs/markdown/tests/table_of_all_problems.md`.
"""

import re
from pathlib import Path
from typing import Dict, Optional, List


def parse_cmake_dim(cmake_path: Path) -> Optional[str]:
    """Parse CMakeLists.txt to determine dimensionality."""
    if not cmake_path.exists():
        return None

    with open(cmake_path, 'r') as f:
        content = f.read()

    # Check for AMReX_SPACEDIM conditions
    if 'AMReX_SPACEDIM EQUAL 3' in content:
        return '3'
    elif 'AMReX_SPACEDIM EQUAL 2' in content:
        return '2'
    elif 'AMReX_SPACEDIM EQUAL 1' in content:
        return '1'
    elif 'AMReX_SPACEDIM GREATER_EQUAL 3' in content:
        return '3'
    elif 'AMReX_SPACEDIM GREATER_EQUAL 2' in content:
        return '2'
    elif 'AMReX_SPACEDIM GREATER_EQUAL 1' in content:
        return '1'
    else:
        # No AMReX_SPACEDIM means it works for any dimension (or is 1D)
        return '1'


def parse_cpp_flags(cpp_path: Path) -> Dict[str, any]:
    """Parse C++ file to extract problem flags."""
    if not cpp_path.exists():
        return {}

    with open(cpp_path, 'r') as f:
        content = f.read()

    # Remove commented lines (lines starting with //)
    lines = content.split('\n')
    uncommented_lines = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith('//'):
            uncommented_lines.append(line)
    content = '\n'.join(uncommented_lines)

    flags = {}

    # Parse boolean flags (may be direct values or variable references)
    # Handle multiple occurrences - if any is true, use true
    bool_flags = [
        'is_hydro_enabled',
        'is_mhd_enabled',
        'is_radiation_enabled',
        'is_self_gravity_enabled',
        'enable_dust_gas_thermal_coupling_model',
        'enable_photoelectric_heating',
    ]

    for key in bool_flags:
        # Find all matches for this flag
        matches = re.findall(rf'{key}\s*=\s*(\w+)', content)
        bool_values = []

        for value_or_var in matches:
            if value_or_var in ('true', 'false'):
                # Direct boolean value
                bool_values.append(value_or_var == 'true')
            else:
                # Variable reference - search for its definition
                var_def_match = re.search(rf'constexpr\s+bool\s+{value_or_var}\s*=\s*(true|false)', content)
                if var_def_match:
                    bool_values.append(var_def_match.group(1) == 'true')

        # If any value is true, set flag to true (handles files with multiple trait structs)
        if bool_values:
            flags[key] = any(bool_values)

    # Parse integer flags
    int_patterns = {
        'numPassiveScalars': r'numPassiveScalars\s*=\s*(?:numMassScalars\s*\+\s*)?(\d+)',
        'numMassScalars': r'numMassScalars\s*=\s*(\d+)',
    }

    for key, pattern in int_patterns.items():
        match = re.search(pattern, content)
        if match:
            flags[key] = int(match.group(1))

    # Parse nGroups (special case - may be a variable reference or multiple definitions)
    # Find all nGroups assignments
    ngroups_matches = re.findall(r'nGroups\s*=\s*(\w+)', content)
    ngroups_values = []

    for value_or_var in ngroups_matches:
        # Check if it's a direct number
        if value_or_var.isdigit():
            ngroups_values.append(int(value_or_var))
        else:
            # It's a variable, search for its definition
            var_def_match = re.search(rf'constexpr\s+int\s+{value_or_var}\s*=\s*(\d+)', content)
            if var_def_match:
                ngroups_values.append(int(var_def_match.group(1)))

    # Take the maximum value if we found any (handles files with multiple problem types)
    if ngroups_values:
        flags['nGroups'] = max(ngroups_values)

    # Parse particle_switch (special case)
    particle_match = re.search(r'particle_switch\s*=\s*([^;]+);', content)
    if particle_match:
        particles_str = particle_match.group(1)
        # Extract particle types (e.g., "ParticleSwitch::CIC | ParticleSwitch::Rad | ParticleSwitch::CICRad")
        particles = re.findall(r'ParticleSwitch::(\w+)', particles_str)
        flags['particle_switch'] = particles

    return flags


def format_particles(particles: Optional[List[str]]) -> str:
    """Format particle list for table."""
    if not particles:
        return '❌'
    return ', '.join(particles)


def format_radiation(flags: Dict) -> str:
    """Format radiation column based on flags."""
    if not flags.get('is_radiation_enabled', False):
        return '❌'

    n_groups = flags.get('nGroups', 1)

    # Base: SG or MG
    if n_groups == 1:
        rad_str = 'SG'
    else:
        rad_str = 'MG'

    # Add modifiers
    modifiers = []
    if flags.get('enable_dust_gas_thermal_coupling_model', False):
        modifiers.append('ThermalDust')
    if flags.get('enable_photoelectric_heating', False):
        modifiers.append('PE')

    if modifiers:
        rad_str += '+' + '+'.join(modifiers)

    return rad_str


def format_passive_scalars(flags: Dict) -> str:
    """Format passive scalars column."""
    num_passive = flags.get('numPassiveScalars', 0)
    num_mass = flags.get('numMassScalars', 0)

    if num_passive == 0 and num_mass == 0:
        return '❌'

    # Try to provide meaningful info
    if num_passive > 0 or num_mass > 0:
        return str(max(num_passive, num_mass))

    return '❌'


def main():
    """Main function to parse all problems and generate table."""
    problems_dir = Path('src/problems')

    if not problems_dir.exists():
        print(f"Error: {problems_dir} not found")
        return

    # Get all problem directories
    problem_dirs = sorted([d for d in problems_dir.iterdir() if d.is_dir()])

    results = []

    for problem_dir in problem_dirs:
        problem_name = problem_dir.name

        # Parse CMakeLists.txt
        cmake_path = problem_dir / 'CMakeLists.txt'
        dim = parse_cmake_dim(cmake_path)

        # Find and parse cpp file
        cpp_files = list(problem_dir.glob('*.cpp'))
        if not cpp_files:
            print(f"Warning: No cpp file found for {problem_name}")
            continue

        cpp_path = cpp_files[0]  # Use first cpp file found
        flags = parse_cpp_flags(cpp_path)

        # Build row data
        row = {
            'Problem': problem_name,
            'DIM': dim or '',
            'Hydro': '✅' if flags.get('is_hydro_enabled', False) else '❌',
            'MHD': '✅' if flags.get('is_mhd_enabled', False) else '❌',
            'Rad': format_radiation(flags),
            'Gravity': '✅' if flags.get('is_self_gravity_enabled', False) else '❌',
            'Particles': format_particles(flags.get('particle_switch')),
            'PassiveScalars': format_passive_scalars(flags),
        }

        results.append(row)

    # Generate markdown table
    headers = ['Problem', 'DIM', 'Hydro', 'MHD', 'Rad', 'Gravity', 'Particles', 'PassiveScalars']

    # Calculate column widths
    widths = {h: len(h) for h in headers}
    for row in results:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ''))))

    # Print table
    print('# Quokka Test Problems\n')
    print('This table lists all test problems in the Quokka codebase.\n')

    # Header row
    header_row = '| ' + ' | '.join(h.ljust(widths[h]) for h in headers) + ' |'
    print(header_row)

    # Separator row
    sep_row = '|' + '|'.join('-' * (widths[h] + 2) for h in headers) + '|'
    print(sep_row)

    # Data rows
    for row in results:
        data_row = '| ' + ' | '.join(str(row.get(h, '')).ljust(widths[h]) for h in headers) + ' |'
        print(data_row)

    # Also write to file
    output_path = Path('docs/markdown/tests/table_of_all_problems.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('# Table of all test problems\n\n')
        f.write('This table lists all test problems in the Quokka codebase.\n\n')
        f.write(header_row + '\n')
        f.write(sep_row + '\n')
        for row in results:
            data_row = '| ' + ' | '.join(str(row.get(h, '')).ljust(widths[h]) for h in headers) + ' |'
            f.write(data_row + '\n')

    print(f'\n✅ Table written to {output_path}')


if __name__ == '__main__':
    main()
