"""Exercise real CSVReader parsing in bounded subprocesses, including fatal errors."""
import pathlib
import subprocess
import sys
import tempfile


def fixture(dim):
    header = [str(dim), ','.join(['2'] * dim), '2',
              ','.join(f'x{i}' for i in range(dim)), 'a,b',
              ','.join(['cm'] * dim), 'g,s', ','.join(['0'] * dim),
              ','.join(['1'] * dim), ','.join(['linear'] * dim)]
    data = [f'{out * 100 + i + 1},{out * 100 + i + 2}'
            for out in range(2) for i in range(0, 2**dim, 2)]
    return header + data


def main():
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / 'table.csv'

        def run(name, dim, lines, valid=False, expected="CSVReader:"):
            path.write_text('\n'.join(lines) + '\n')
            result = subprocess.run([sys.argv[1], str(path), str(dim)],
                                    cwd=directory, capture_output=True, text=True, timeout=20)
            if valid:
                passed = result.returncode == 0
            else:
                passed = result.returncode != 0 and expected in result.stdout + result.stderr
            if not passed:
                failures.append(name)
                print(f'{name}: exit {result.returncode}\n{result.stdout}\n{result.stderr}')

        for dim in range(1, 5):
            lines = fixture(dim)
            run(f'valid-{dim}d', dim, lines, True)
            run(f'truncated-{dim}d', dim, lines[:-1])
            bad = lines.copy()
            bad[-1] = '101,invalid'
            run(f'invalid-number-{dim}d', dim, bad, expected='invalid field 1 in output 1 index')
            bad[-1] = '101 102'
            run(f'missing-comma-{dim}d', dim, bad)
            bad[-1] = '101,102,103'
            run(f'extra-column-{dim}d', dim, bad)
            run(f'extra-data-{dim}d', dim, lines + ['999'])
            run(f'whitespace-{dim}d', dim, ['  ' + line + '  \r' for line in lines] + ['  '], True)
            bad[-1] = '101,102,'
            run(f'trailing-comma-{dim}d', dim, bad)
        for row in range(10):
            lines = fixture(2)
            lines[row] += ',extra'
            run(f'extra-header-field-{row}', 2, lines)
            lines = fixture(2)
            lines[row] = ''
            run(f'missing-header-{row}', 2, lines)
        lines = fixture(2)
        lines[1] = '2;2'
        run('wrong-header-delimiter', 2, lines)
    print(f'{len(failures)} CSV regression failures')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
