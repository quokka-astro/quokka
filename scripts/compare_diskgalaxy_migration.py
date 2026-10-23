#!/usr/bin/env python3
"""Compare a completed paired DiskGalaxy sample; retain evidence even on failure."""
import argparse
import json
import math
import pathlib
import subprocess
import sys

from compare_particle_records import compare
from run_diskgalaxy_migration import checkpoint_state


def plot_state(path):
    lines = iter((path / 'Header').read_text().splitlines())
    version = next(lines)
    names = [next(lines) for _ in range(int(next(lines)))]
    dimension, time, finest = int(next(lines)), float(next(lines)), int(next(lines))
    lo, hi = list(map(float, next(lines).split())), list(map(float, next(lines).split()))
    ratios, domains = next(lines), next(lines)
    steps = list(map(int, next(lines).split()))
    if dimension != 3 or len(steps) != finest + 1 or any(step != 100 for step in steps):
        raise ValueError('Plotfile is not a complete 3D 100-step output')
    if not math.isfinite(time) or time <= 0:
        raise ValueError('Invalid plotfile time')
    return {'version': version, 'names': names, 'dimension': dimension, 'finest_level': finest,
            'lo': lo, 'hi': hi, 'ratios': ratios, 'domains': domains, 'steps': steps}, time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', type=pathlib.Path, help='Paired sample containing aos/ and rtsoa/')
    parser.add_argument('--fcompare', required=True, type=pathlib.Path)
    parser.add_argument('--roundoff-factor', type=float, default=64)
    args = parser.parse_args()
    if not math.isfinite(args.roundoff_factor) or args.roundoff_factor < 0:
        parser.error('Roundoff factor must be finite and nonnegative')
    root = args.sample.resolve(strict=True)
    report = {'roundoff_factor': args.roundoff_factor, 'checks': {}}
    tolerance = args.roundoff_factor * sys.float_info.epsilon

    def check(name, function):
        try:
            value = function()
            report['checks'][name] = {'passed': value.get('passed', True), 'result': value}
        except (ValueError, AssertionError, OSError, StopIteration) as error:
            report['checks'][name] = {'passed': False, 'error': str(error)}

    def completion():
        states = [checkpoint_state(root / variant / 'chk0000100') for variant in ('aos', 'rtsoa')]
        if states[0]['finest_level'] != states[1]['finest_level'] or states[0]['steps'] != states[1]['steps']:
            raise ValueError('Checkpoint hierarchy or step mismatch')
        for a, b in zip(states[0]['times'], states[1]['times']):
            if abs(a - b) > tolerance * max(abs(a), abs(b)):
                raise ValueError(f'Checkpoint times differ beyond roundoff: {a}, {b}')
        plots = [plot_state(root / variant / 'plt0000100') for variant in ('aos', 'rtsoa')]
        if plots[0][0] != plots[1][0]:
            raise ValueError('Plotfile field order or hierarchy metadata mismatch')
        for state, (_, time) in zip(states, plots):
            if time != state['times'][0]:
                raise ValueError('Plotfile and checkpoint times disagree')
        return {'checkpoints': states, 'plotfiles': plots}

    check('completion_and_metadata', completion)
    for kind in ('plt', 'chk'):
        paths = [root / variant / f'{kind}0000100' for variant in ('aos', 'rtsoa')]
        populations = [{p.name for p in path.glob('*_particles') if p.is_dir()} for path in paths]
        expected = {'CIC_particles', 'StochasticStellarPop_particles'}
        report['checks'][f'{kind}_populations'] = {
            'passed': populations[0] == populations[1] == expected,
            'populations': [sorted(p) for p in populations],
        }
        for name in sorted(set.union(*populations) | expected):
            check(f'{kind}/{name}', lambda name=name: compare(paths[0] / name, paths[1] / name, args.roundoff_factor))
    command = [str(args.fcompare.resolve(strict=True)), '--abort_if_not_all_found',
               '--rel_tol', str(tolerance), '--abs_tol', '0',
               str(root / 'aos/plt0000100'), str(root / 'rtsoa/plt0000100')]
    with (root / 'mesh-comparison.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report['checks']['mesh_fields'] = {'passed': result.returncode == 0, 'returncode': result.returncode, 'command': command}
    report['passed'] = all(item['passed'] for item in report['checks'].values())
    (root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: item['passed'] for name, item in report['checks'].items()}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
