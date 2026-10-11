"""Check splitting through real checkpoints, plus direct factor validation."""

from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable = str(Path(sys.argv[1]).resolve())
    inputs = str(Path(sys.argv[2]).resolve())
    failures = []
    with tempfile.TemporaryDirectory(prefix="quokka-particle-split-") as temporary:
        root = Path(temporary)

        def run(name, arguments, rejected=False):
            directory = root / name
            directory.mkdir()
            (directory / "cic.txt").write_text("1\n0.5 0.5 0.5 8 0 0 0\n")
            (directory / "cicrad.txt").write_text(
                "1\n0.5 0.5 0.5 8 0 0 0 2 10 8 16 24\n"
            )
            result = subprocess.run(
                sys.argv[3:] + [executable, inputs] + arguments,
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            output = result.stdout + result.stderr
            if rejected:
                passed = result.returncode != 0 and "splitParticles requires splitFactor > 0" in output
            else:
                passed = result.returncode == 0 and "Particle split checks passed" in output
                if name.startswith("restart_"):
                    passed = passed and "Restart from checkpoint" in output
                if name == "restart_refined":
                    passed = passed and "Splitting CICRad_particles using split_factor = 8" in output
            if not passed:
                failures.append(f"{name}:\n{output}")
            print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
            return directory

        seed = run("seed", [])
        checkpoint = seed / "chk0000000"
        assert (checkpoint / "Header").is_file(), "Missing seed checkpoint"
        restart = [f"restartfile={checkpoint}"]
        run("restart_unchanged", restart)
        run("restart_refined", restart + ["amr.n_cell=16 16 16", "split_test.expected_count=8"])
        run("restart_refined_no_split", restart + ["amr.n_cell=16 16 16", "particles.split_particles_on_restart_refine=0"])
        run("factor_one", ["split_test.factor=1"])
        run("factor_three", ["split_test.factor=3", "split_test.expected_count=3"])
        run("factor_zero", ["split_test.factor=0"], rejected=True)
        # Test negatives only once zero is rejected, avoiding an invalid huge allocation on the old implementation.
        if not any(failure.startswith("factor_zero:") for failure in failures):
            run("factor_negative", ["split_test.factor=-1"], rejected=True)
        assert not failures, "\n".join(failures)


if __name__ == "__main__":
    main()
