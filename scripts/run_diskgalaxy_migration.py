#!/usr/bin/env python3
"""Run one paired 100-step DiskGalaxy sample in fresh directories.

This records execution and profiler evidence only. Successful execution does not
establish output equivalence or performance acceptance; compare outputs separately
and use repeated, alternating-order samples for timing conclusions.
"""
import argparse
import json
import math
import pathlib
import platform
import subprocess
import time

from benchmark_particle_migration import inclusive_times, sha256


def checkpoint_state(path):
    lines = (path / 'Header').read_text().splitlines()
    if lines[0] != 'Checkpoint file for QuokkaCode':
        raise ValueError(f'Unexpected checkpoint format: {path}')
    finest = int(lines[1])
    steps = list(map(int, lines[2].split()))[:finest + 1]
    times = list(map(float, lines[4].split()))[:finest + 1]
    if len(steps) != finest + 1 or any(step != 100 for step in steps):
        raise ValueError(f'Incomplete 100-step run: {path}, steps={steps}')
    if len(times) != finest + 1 or any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError(f'Invalid checkpoint times: {path}, times={times}')
    if any(t != times[0] for t in times):
        raise ValueError(f'Unsynchronized checkpoint levels: {times}')
    return {'finest_level': finest, 'steps': steps, 'times': times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=pathlib.Path,
                        help='Isolated directory containing aos-build, rtsoa-build, and rtsoa sources')
    parser.add_argument('--output', required=True, type=pathlib.Path, help='New directory for this paired sample')
    parser.add_argument('--backend', required=True, choices=['CUDA', 'HIP'])
    parser.add_argument('--order', choices=['aos-first', 'rtsoa-first'], default='aos-first')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = root / 'rtsoa'
    for name in ('inputs', 'extern'):
        (output / name).symlink_to(source / name, target_is_directory=True)
    inputs = source / 'inputs/DiskGalaxy.toml'
    data = {
        'CloudyData_UVB=HM2012_resampled.h5': source / 'extern/cooling/CloudyData_UVB=HM2012_resampled.h5',
        'AgoraGalaxy_particles_LOW.txt': source / 'extern/agora_data/AgoraGalaxy_particles_LOW.txt',
        'vcirc_with_constant_halo.csv': source / 'extern/agora_data/vcirc_with_constant_halo.csv',
    }
    parameters = [
        'max_timesteps=100', 'max_walltime=24:00:00',
        'plotfile_interval=100', 'checkpoint_interval=100',
        'tiny_profiler.enabled=1', 'tiny_profiler.print_threshold=0',
        'tiny_profiler.output_file=stdout', 'tiny_profiler.device_synchronize_around_region=1',
    ]
    report = {
        'host': platform.node(), 'backend': args.backend, 'processes': 1,
        'runner_sha256': sha256(pathlib.Path(__file__)),
        'input': {'path': str(inputs), 'sha256': sha256(inputs)},
        'data': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in data.items()},
        'parameters': parameters, 'samples': [],
        'timing_scope': 'Synchronized TinyProfiler regions; one paired sample, not a performance acceptance verdict',
    }
    def save():
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    order = ('aos', 'rtsoa') if args.order == 'aos-first' else ('rtsoa', 'aos')
    for variant in order:
        executable = root / f'{variant}-build/src/problems/DiskGalaxy/DiskGalaxy'
        run = output / variant
        run.mkdir()
        for name, path in data.items():
            (run / name).symlink_to(path)
        command = [str(executable), str(inputs)] + parameters
        sample = {'variant': variant, 'command': command, 'executable_sha256': sha256(executable), 'run': str(run),
                  'cmake_cache_sha256': sha256(root / f'{variant}-build/CMakeCache.txt')}
        report['samples'].append(sample)
        save()
        print(f'Starting {variant}', flush=True)
        started = time.monotonic()
        with (run / 'output.log').open('w') as log:
            result = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT)
        sample.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started)
        text = (run / 'output.log').read_text()
        sample['regions'] = inclusive_times(text)
        save()
        result.check_returncode()
        if f'Initializing {args.backend}...' not in text:
            raise RuntimeError(f'{variant} did not initialize {args.backend}')
        checkpoint = run / 'chk0000100'
        plotfile = run / 'plt0000100'
        sample['checkpoint'] = checkpoint_state(checkpoint)
        if not (plotfile / 'Header').is_file():
            raise RuntimeError(f'Missing step-100 plotfile: {plotfile}')
        sample['checkpoint_header_sha256'] = sha256(checkpoint / 'Header')
        sample['plotfile_header_sha256'] = sha256(plotfile / 'Header')
        if not sample['regions']:
            raise RuntimeError('Missing profiler evidence')
        sample['completed_100_steps'] = True
        save()
        print(f'Completed {variant}: 100 steps', flush=True)


if __name__ == '__main__':
    main()
