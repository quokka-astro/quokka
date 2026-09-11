"""Run hydro with debug output and inspect every active-axis flattening plotfile."""

from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable = str(Path(sys.argv[1]).resolve())
    inputs = str(Path(sys.argv[2]).resolve())
    dimensions = int(sys.argv[3])
    with tempfile.TemporaryDirectory(prefix="quokka-hydro-debug-") as directory:
        result = subprocess.run(
            sys.argv[4:]
            + [
                executable,
                inputs,
                "max_timesteps=1",
                "amr.n_cell=8 8 8",
                "hydro.low_level_debugging_output=1",
                "plotfile_interval=-1",
                "checkpoint_interval=-1",
                "suppress_output=1",
            ],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        for axis_index, axis in enumerate("xyz"):
            outputs = list(Path(directory).glob(f"debug_flattening_{axis}*"))
            if axis_index >= dimensions:
                assert not outputs, f"Unexpected inactive-axis output: {outputs}"
                continue
            assert outputs, f"Missing {axis}-axis flattening output"
            for output in outputs:
                header = (output / "Header").read_text().splitlines()
                assert header[1:4] == ["1", "chi", str(dimensions)], header[:4]
                assert (output / "Level_0" / "Cell_H").is_file(), output
                assert any((output / "Level_0").glob("Cell_D_*")), output
        print(f"Validated {dimensions}D hydro debug flattening plotfiles")


if __name__ == "__main__":
    main()
