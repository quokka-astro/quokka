#!/usr/bin/env python3
"""Report matched DiskGalaxy particle timings using the agreed 5% acceptance rule.

Median changes smaller than 5% in magnitude count as equal for acceptance.
A slowdown of 5% or more fails; this is a user-defined acceptance margin,
not an estimate of measurement noise or statistical equivalence.
"""
import argparse
import json
import math
import pathlib
import statistics


def summarize(paths):
    roots = [p.resolve(strict=True) for p in paths]
    if len(roots) < 3 or len(set(roots)) != len(roots):
        raise ValueError('Use at least three distinct paired samples')
    reports = [json.loads((p / 'summary.json').read_text()) for p in roots]
    for key in ('host', 'backend', 'processes', 'runner_sha256', 'input', 'data', 'parameters'):
        if any(report[key] != reports[0][key] for report in reports):
            raise ValueError(f'Inconsistent provenance: {key}')
    samples = []
    for report in reports:
        pair = {sample['variant']: sample for sample in report['samples']}
        if len(report['samples']) != 2 or set(pair) != {'aos', 'rtsoa'}:
            raise ValueError('Missing or duplicate layout samples')
        for variant, sample in pair.items():
            if not sample.get('completed_100_steps') or sample['returncode'] != 0:
                raise ValueError('Incomplete sample')
            if samples:
                for key in ('executable_sha256', 'cmake_cache_sha256', 'command'):
                    if sample[key] != samples[0][variant][key]:
                        raise ValueError(f'Changed {variant} {key}')
        samples.append(pair)
    names = {name for name in samples[0]['aos']['regions'] if name.startswith('PhysicsParticleRegister::')}
    if not names:
        raise ValueError('Missing particle profiler regions')
    result = {'host': reports[0]['host'], 'backend': reports[0]['backend'], 'samples': list(map(str, roots)),
              'scope': __doc__, 'regions': {}}
    for pair in samples:
        for sample in pair.values():
            if {name for name in sample['regions'] if name.startswith('PhysicsParticleRegister::')} != names:
                raise ValueError('Particle region coverage differs')
    for name in sorted(names):
        calls = {pair[v]['regions'][name]['calls'] for pair in samples for v in ('aos', 'rtsoa')}
        if len(calls) != 1 or min(calls) <= 0:
            raise ValueError(f'Inconsistent call counts: {name}')
        entry = {'calls_per_run': min(calls)}
        for variant in ('aos', 'rtsoa'):
            values = [pair[variant]['regions'][name]['max_seconds'] for pair in samples]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f'Invalid timing: {name}')
            entry[variant] = {'seconds': values, 'median_seconds': statistics.median(values),
                              'min_seconds': min(values), 'max_seconds': max(values)}
        entry['median_change_percent'] = 100 * (entry['rtsoa']['median_seconds'] / entry['aos']['median_seconds'] - 1)
        entry['paired_changes_percent'] = [100 * (b / a - 1) for a, b in
                                           zip(entry['aos']['seconds'], entry['rtsoa']['seconds'])]
        result['regions'][name] = entry
    result['slower_medians'] = [name for name, entry in result['regions'].items() if entry['median_change_percent'] > 0]
    result['acceptance_margin_percent'] = 5.0
    for entry in result['regions'].values():
        delta = entry['median_change_percent']
        entry['acceptance'] = 'equal' if abs(delta) < 5.0 else ('better' if delta < 0 else 'slower')
    result['performance_failures'] = [name for name, entry in result['regions'].items() if entry['acceptance'] == 'slower']
    result['performance_passed'] = not result['performance_failures']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=pathlib.Path, nargs='+')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = summarize(args.samples)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    for name, entry in result['regions'].items():
        print(f'{entry["median_change_percent"]:+8.2f}% {name}')


if __name__ == '__main__':
    main()
