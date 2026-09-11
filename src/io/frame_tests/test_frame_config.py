"""Check initialization diagnostics in subprocesses so AMReX aborts are bounded."""
import pathlib
import subprocess
import sys
import tempfile


def main():
    executable = str(pathlib.Path(sys.argv[1]).resolve())
    dimension = int(sys.argv[2])
    failures = 0
    with tempfile.TemporaryDirectory() as directory:
        for filtered in (False, True):
            result = subprocess.run([executable, 'filtered' if filtered else 'plain'],
                                    cwd=directory, capture_output=True, text=True, timeout=20)
            output = result.stdout + result.stderr
            if dimension == 3:
                passed = result.returncode == 0 and 'Preparation completed' in output
                passed = passed and (('Filters are not available' in output) == filtered)
            else:
                passed = result.returncode != 0 and 'DiagFramePlane requires a 3D simulation' in output
                passed = passed and 'Initialization completed' not in output
            if not passed:
                print(output)
                failures += 1
    print(f'{dimension}D: {failures} configuration failures')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
