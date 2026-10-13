"""Run real checkpoint output with scheduled and unscheduled sentinel requests."""
import pathlib
import subprocess
import sys
import tempfile


def main():
    executable = str(pathlib.Path(sys.argv[1]).resolve())
    inputs = str(pathlib.Path(sys.argv[2]).resolve())
    failures = 0
    cases = [
        ('step-scheduled', ['checkpoint_interval=1'], True),
        ('time-scheduled', ['checkpoint_interval=-1', 'checkpointtime_interval=1e-10'], True),
        ('unscheduled', ['checkpoint_interval=10'], True),
        ('no-request', ['checkpoint_interval=10'], False),
    ]
    for name, parameters, request in cases:
        with tempfile.TemporaryDirectory(prefix='checkpoint-sentinel-') as directory:
            root = pathlib.Path(directory)
            sentinel = root / 'checkpointNow'
            if request:
                sentinel.touch()
            result = subprocess.run(
                sys.argv[3:] + [executable, inputs, 'max_timesteps=2', 'amr.n_cell=8 8 8',
                 'plotfile_interval=-1', 'suppress_output=1'] + parameters,
                cwd=root, capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            writes = [line.strip() for line in output.splitlines() if line.startswith('Writing checkpoint ')]
            step_one = writes.count('Writing checkpoint chk0000001')
            passed = result.returncode == 0 and not sentinel.exists() and step_one == int(request)
            if request:
                passed = passed and (root / 'chk0000001' / 'Header').is_file()
            if not passed:
                failures += 1
                print(f'{name}: exit={result.returncode}, sentinel={sentinel.exists()}, writes={writes}\n{output}')
    print(f'{len(cases)} cases, {failures} failures')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
