"""Test sanitizer orchestration with controlled process exits and tool output."""
import os
import pathlib
import subprocess
import sys
import tempfile

RACE_CLEAN = '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
MEM_CLEAN = '========= ERROR SUMMARY: 0 errors'


def main():
    cases = [('clean', RACE_CLEAN, 0, MEM_CLEAN, 0, True)]
    for tool in ('race', 'mem'):
        for name, output, code in [
            ('process-failure', '', 7),
            ('failure-after-clean-summary', RACE_CLEAN if tool == 'race' else MEM_CLEAN, 9),
            ('missing-summary', 'tool stopped before finishing', 0),
            ('malformed-summary', '========= RACECHECK SUMMARY: unknown' if tool == 'race' else '========= ERROR SUMMARY: unknown', 0),
            ('reported-error', '========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)' if tool == 'race' else '========= ERROR SUMMARY: 1 error', 0),
            ('mixed-summaries', RACE_CLEAN + '\n========= RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)' if tool == 'race' else MEM_CLEAN + '\n========= ERROR SUMMARY: 2 errors', 0),
        ]:
            cases.append((tool + '-' + name, output if tool == 'race' else RACE_CLEAN,
                          code if tool == 'race' else 0, output if tool == 'mem' else MEM_CLEAN,
                          code if tool == 'mem' else 0, False))
    failures = []
    script = str(pathlib.Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        bin_dir = root / 'bin'
        bin_dir.mkdir()
        sanitizer = bin_dir / 'compute-sanitizer'
        sanitizer.write_text('''#!/bin/bash
printf '%s\\n' "$2" >> "$MOCK_CALLS"
if [ "$2" = racecheck ]; then
 printf '%s\\n' "$MOCK_RACE_OUTPUT"
 exit "$MOCK_RACE_EXIT"
fi
printf '%s\\n' "$MOCK_MEM_OUTPUT"
exit "$MOCK_MEM_EXIT"
''')
        sanitizer.chmod(0o755)
        # Keep the script's intentionally preserved logs inside this test's
        # temporary root on both BSD and GNU systems.
        mktemp = bin_dir / 'mktemp'
        mktemp.write_text('''#!/bin/bash
mkdir "$MOCK_TEMP/run"
printf '%s\\n' "$MOCK_TEMP/run"
''')
        mktemp.chmod(0o755)
        compare_dir = root / 'extern/amrex/Tools/Plotfile'
        compare_dir.mkdir(parents=True)
        (compare_dir / 'fcompare.mock.ex').touch()
        binary = root / 'target'
        binary.touch()
        inputs = root / 'input.toml'
        inputs.touch()
        for name, race, race_code, mem, mem_code, success in cases:
            temp = root / name
            temp.mkdir()
            calls = temp / 'calls'
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                       TMPDIR=str(temp), MOCK_TEMP=str(temp), MOCK_CALLS=str(calls), MOCK_RACE_OUTPUT=race,
                       MOCK_RACE_EXIT=str(race_code), MOCK_MEM_OUTPUT=mem, MOCK_MEM_EXIT=str(mem_code))
            result = subprocess.run(['bash', script, '-b', str(binary), '-i', str(inputs), '-n', '1', '-c'],
                                    cwd=root, env=env, text=True, capture_output=True, timeout=15)
            called_both = calls.exists() and calls.read_text().splitlines() == ['racecheck', 'memcheck']
            passed = (result.returncode == 0) == success and called_both
            if not success and 'No issues detected by compute-sanitizer.' in result.stdout:
                passed = False
            if not passed:
                failures.append(name)
                print(f'{name}: exit {result.returncode}\n{result.stdout}\n{result.stderr}')
    print(f'{len(cases)} cases, {len(failures)} failures')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
