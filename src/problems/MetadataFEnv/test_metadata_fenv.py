"""Check real metadata reads on normal and YAML exception paths."""

from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable = str(Path(sys.argv[1]).resolve())
    inputs = str(Path(sys.argv[2]).resolve())
    cases = {
        "valid": "number: 1.25\nstring: hello\nsequence: [1, 2]\nmap: {a: 1}\n",
        "missing": None,
        "malformed": "value: [\n",
        "conversion": "? [a, b]\n: value\n",
    }
    failures = []
    with tempfile.TemporaryDirectory(prefix="quokka-metadata-fenv-") as root:
        for name, contents in cases.items():
            directory = Path(root) / name
            directory.mkdir()
            if contents is not None:
                (directory / "metadata.yaml").write_text(contents)
            result = subprocess.run(
                sys.argv[3:]
                + [executable, inputs, f"metadata_test.directory={directory}",
                   f"metadata_test.expect_exception={int(name != 'valid')}", "suppress_output=1"],
                cwd=directory, capture_output=True, text=True, timeout=60, check=False,
            )
            print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
            if result.returncode != 0:
                failures.append(result.stdout + result.stderr)
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    main()
