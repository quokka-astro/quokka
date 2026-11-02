#!/usr/bin/env python3
"""Assess DiskGalaxy numerical agreement against same-host AoS repeat variation.

This is a finite-sample numerical check, not a performance verdict or a proof
of a universal error bound. Every cross-layout pair is checked against the
largest observed AoS repeat difference for that field and AMR level, with a
roundoff floor. RTSoA repeat differences are reported separately.
"""
import argparse
import itertools
import json
import math
import pathlib
import subprocess
import sys

from compare_diskgalaxy_migration import plot_state
from compare_particle_records import compare, read
from run_diskgalaxy_migration import checkpoint_state

EPSILON_BOUND = 64 * sys.float_info.epsilon
POPULATIONS = ('CIC_particles', 'StochasticStellarPop_particles')


def mesh_errors(text):
    """Read fcompare's L-infinity errors, keeping levels distinct."""
    result = {}
    level = None
    for line in text.splitlines():
        words = line.split()
        if words[:2] == ['level', '=']:
            level = int(words[2])
        elif level is not None and len(words) == 3:
            try:
                absolute, relative = map(float, words[1:])
            except ValueError:
                continue
            if not all(math.isfinite(x) and x >= 0 for x in (absolute, relative)):
                raise ValueError(f'Invalid mesh comparison: {line}')
            key = f'{level}/{words[0]}'
            if key in result:
                raise ValueError(f'Duplicate mesh field: {key}')
            result[key] = {'absolute': absolute, 'relative': relative}
    return result


def counter_state(path):
    metadata, records = read(path)
    version = (path / 'Header').read_text().splitlines()[0]
    ids = []
    for first, second in records:
        if version == 'Version_Two_Dot_One_double':
            packed = ((first & 0xffffffff) << 32) | (second & 0xffffffff)
            value = (packed >> 24) & 0x7fffffffff
            if not packed >> 63:
                raise ValueError('Invalid stored particle ID')
        else:
            value = first
        if value <= 0:
            raise ValueError('Nonpositive stored particle ID')
        ids.append(value)
    maximum = max(ids, default=0)
    if metadata[3] <= maximum:
        raise ValueError(f'Next-ID counter does not exceed stored IDs: {path}')
    return {'next_id': metadata[3], 'maximum_id': maximum, 'count': len(records)}


def assess(pairs):
    """Compare each cross-layout error against independent baseline envelopes."""
    baseline = [p for p in pairs if p['kind'] == 'aos_repeat']
    cross = [p for p in pairs if p['kind'] == 'cross_layout']
    if not baseline or not cross:
        raise ValueError('Missing baseline repeats or cross-layout comparisons')
    keys = baseline[0]['errors'].keys()
    if any(p['errors'].keys() != keys for p in pairs):
        raise ValueError('Comparison field coverage differs')
    envelope = {key: max(p['errors'][key]['absolute'] for p in baseline) for key in keys}
    failures = []
    for pair in cross:
        for key, error in pair['errors'].items():
            if error['absolute'] > envelope[key] and error['relative'] > EPSILON_BOUND:
                failures.append({'pair': pair['label'], 'field': key, 'error': error,
                                 'baseline_absolute_envelope': envelope[key]})
    return {'passed': not failures, 'baseline_absolute_envelope': envelope, 'failures': failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', nargs='+', type=pathlib.Path)
    parser.add_argument('--fcompare', type=pathlib.Path, required=True)
    parser.add_argument('--checkpoint-compare', type=pathlib.Path,
                        help='CPU checkpoint utility; defaults to compare_checkpoint_mesh beside fcompare')
    parser.add_argument('--output', type=pathlib.Path, required=True, help='New evidence directory')
    args = parser.parse_args()
    if len(args.samples) < 3:
        parser.error('At least three completed paired samples are required; five are preferred')
    roots = [p.resolve(strict=True) for p in args.samples]
    checkpoint_tool = (args.checkpoint_compare or args.fcompare.with_name('compare_checkpoint_mesh')).resolve(strict=True)
    if len(set(roots)) != len(roots):
        parser.error('Repeated sample directories are not independent samples')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'rule': __doc__, 'roundoff_floor': EPSILON_BOUND, 'samples': list(map(str, roots)),
              'pairs': [], 'counters': {}, 'initial_checks': []}
    summaries = [json.loads((p / 'summary.json').read_text()) for p in roots]
    for key in ('host', 'backend', 'processes', 'runner_sha256', 'input', 'data', 'parameters'):
        if any(s[key] != summaries[0][key] for s in summaries):
            raise ValueError(f'Sample provenance differs: {key}')
    runs = []
    for index, (root, summary) in enumerate(zip(roots, summaries)):
        samples = {s['variant']: s for s in summary['samples']}
        if len(summary['samples']) != 2 or set(samples) != {'aos', 'rtsoa'}:
            raise ValueError('Missing or duplicate layout samples')
        for variant, sample in samples.items():
            if not sample.get('completed_100_steps') or sample['returncode'] != 0:
                raise ValueError('Incomplete sample')
            first = next(s for s in summaries[0]['samples'] if s['variant'] == variant)
            for key in ('command', 'executable_sha256', 'cmake_cache_sha256'):
                if sample[key] != first[key]:
                    raise ValueError(f'{variant} provenance differs: {key}')
            path = root / variant
            checkpoint = checkpoint_state(path / 'chk0000100')
            timestep_line = (path / 'chk0000100/Header').read_text().splitlines()[3]
            timesteps = list(map(float, timestep_line.split()))
            if len(timesteps) != checkpoint['finest_level'] + 1 or any(not math.isfinite(t) or t <= 0 for t in timesteps):
                raise ValueError('Invalid checkpoint timesteps')
            metadata, time = plot_state(path / 'plt0000100')
            if time != checkpoint['times'][0]:
                raise ValueError('Plotfile and checkpoint times disagree')
            run = {'label': f'{index}-{variant}', 'variant': variant, 'path': path,
                   'metadata': metadata, 'time': time, 'timesteps': timesteps}
            runs.append(run)
            for step in ('0000000', '0000100'):
                for kind in ('plt', 'chk'):
                    output = path / f'{kind}{step}'
                    found = {p.name for p in output.glob('*_particles') if p.is_dir()}
                    if found != set(POPULATIONS):
                        raise ValueError(f'Unexpected particle populations: {output}')
                    counters = {p: counter_state(output / p) for p in POPULATIONS}
                    if variant == 'rtsoa':
                        values = {v['next_id'] for v in counters.values()}
                        if len(values) != 1 or min(values) <= max(v['maximum_id'] for v in counters.values()):
                            raise ValueError('Invalid shared next-ID state')
                    report['counters'][f'{run["label"]}/{kind}{step}'] = counters
    reference = runs[0]
    expected = {f'{lev}/{name}' for lev in range(reference['metadata']['finest_level'] + 1)
                for name in reference['metadata']['names']}

    def mesh(left, right, step, label):
        command = [str(args.fcompare.resolve(strict=True)), '--abort_if_not_all_found',
                   '--rel_tol', str(EPSILON_BOUND), '--abs_tol', '0',
                   str(left['path'] / f'plt{step}'), str(right['path'] / f'plt{step}')]
        result = subprocess.run(command, capture_output=True, text=True)
        (args.output / f'{label}-mesh-{step}.log').write_text(result.stdout + result.stderr)
        errors = mesh_errors(result.stdout)
        if result.returncode not in (0, 1) or errors.keys() != expected:
            raise ValueError(f'Incomplete or failed mesh comparison: {label}')
        checkpoint_command = [str(checkpoint_tool), str(left['path'] / f'chk{step}'),
                              str(right['path'] / f'chk{step}'), str(reference['metadata']['finest_level'])]
        checkpoint_result = subprocess.run(checkpoint_command, capture_output=True, text=True)
        (args.output / f'{label}-checkpoint-{step}.log').write_text(checkpoint_result.stdout + checkpoint_result.stderr)
        checkpoint_result.check_returncode()
        checkpoint_errors = {}
        for line in checkpoint_result.stdout.splitlines():
            words = line.split()
            if words[:1] != ['ERROR']:
                continue
            if len(words) != 4:
                raise ValueError(f'Malformed checkpoint error: {line}')
            absolute, relative = map(float, words[2:])
            if not all(math.isfinite(x) and x >= 0 for x in (absolute, relative)):
                raise ValueError(f'Nonfinite checkpoint error: {line}')
            if words[1] in checkpoint_errors:
                raise ValueError('Duplicate checkpoint field')
            checkpoint_errors[words[1]] = {'absolute': absolute, 'relative': relative}
        prefixes = {f'Level_{lev}/{name}' for lev in range(reference['metadata']['finest_level'] + 1)
                    for name in ('Cell', 'Face_x', 'Face_y', 'Face_z')}
        if {k.rsplit('/', 1)[0] for k in checkpoint_errors} != prefixes:
            raise ValueError('Incomplete checkpoint mesh comparison')
        return {**{'plot/' + k: v for k, v in errors.items()},
                **{'checkpoint/' + k: v for k, v in checkpoint_errors.items()}}

    for run in runs:
        if run['metadata'] != reference['metadata']:
            raise ValueError('Plotfile schema or hierarchy differs')
        for step in ('0000000', '0000100'):
            for kind in ('plt', 'chk'):
                metadata_path = f'{kind}{step}/metadata.yaml'
                if (run['path'] / metadata_path).read_bytes() != (reference['path'] / metadata_path).read_bytes():
                    raise ValueError(f'Output metadata differs: {run["label"]}/{metadata_path}')
        if run is reference:
            continue
        errors = mesh(reference, run, '0000000', f'initial-{run["label"]}')
        passed = all(e['relative'] <= EPSILON_BOUND for e in errors.values())
        for kind in ('plt', 'chk'):
            for population in POPULATIONS:
                value = compare(reference['path'] / f'{kind}0000000' / population,
                                run['path'] / f'{kind}0000000' / population, 64,
                                require_equal_next_id=run['variant'] == reference['variant'])
                passed &= value['passed']
        report['initial_checks'].append({'run': run['label'], 'passed': passed, 'mesh': errors})
    for left, right in itertools.combinations(runs, 2):
        label = f'{left["label"]}--{right["label"]}'
        kind = f'{left["variant"]}_repeat' if left['variant'] == right['variant'] else 'cross_layout'
        errors = {'mesh/' + k: v for k, v in mesh(left, right, '0000100', label).items()}
        difference = abs(left['time'] - right['time'])
        errors['simulation_time'] = {'absolute': difference, 'relative': difference / max(left['time'], right['time'])}
        for lev, (a, b) in enumerate(zip(left['timesteps'], right['timesteps'])):
            errors[f'timestep/{lev}'] = {'absolute': abs(a - b), 'relative': abs(a - b) / max(a, b)}
        for output_kind in ('plt', 'chk'):
            for population in POPULATIONS:
                value = compare(left['path'] / f'{output_kind}0000100' / population,
                                right['path'] / f'{output_kind}0000100' / population, 64,
                                require_equal_next_id=kind != 'cross_layout')
                for name, field in value['fields'].items():
                    errors[f'{output_kind}/{population}/{name}'] = {
                        'absolute': field['max_absolute_difference'],
                        'relative': field['max_absolute_difference'] / field['scale'] if field['scale'] else 0}
        report['pairs'].append({'label': label, 'kind': kind, 'errors': errors})
        print(f'Compared {label}', flush=True)
    report['assessment'] = assess(report['pairs'])
    report['passed'] = report['assessment']['passed'] and all(c['passed'] for c in report['initial_checks'])
    (args.output / 'variability.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Numerical variability assessment passed: {report["passed"]}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
