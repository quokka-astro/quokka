from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import canary


def repo_root(start: Path | None = None) -> Path:
    """Find the Quokka repository root from a file inside regression/canary."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "CMakeLists.txt").is_file() and (candidate / "src").is_dir() and (candidate / "inputs").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate Quokka repository root")


def build_dir() -> Path:
    env = os.environ.get("QUOKKA_CANARY_BUILD_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "build"


def benchmark_root(test_name: str) -> Path:
    return repo_root() / "regression" / "canary" / "benchmarks" / test_name


def benchmark_plotfile(test_name: str, prefix: str) -> Path:
    root = benchmark_root(test_name)
    matches = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"No benchmark plotfile matching '{prefix}*' in {root}")
    return matches[-1]


def latest_plotfile(prefix: str, cwd: Path | None = None) -> Path:
    root = cwd or Path.cwd()
    matches = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"No plotfile matching '{prefix}*' in {root}")
    return matches[-1]


def quokka_executable(problem_name: str) -> Path:
    exe = build_dir() / "src" / "problems" / problem_name / problem_name
    if not exe.is_file():
        raise FileNotFoundError(f"Expected Quokka executable at {exe}")
    return exe


def find_tool(tool_name: str, env_var: str) -> Path:
    env = os.environ.get(env_var)
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise FileNotFoundError(f"{env_var} points to non-executable path: {path}")

    candidates: list[Path] = []
    candidates.extend(path for path in build_dir().rglob(tool_name) if path.is_file() and os.access(path, os.X_OK))

    amrex_root = repo_root() / "extern" / "amrex"
    for parent in (
        amrex_root / "Tools" / "Plotfile",
        amrex_root / "Tools" / "Postprocessing" / "C_Src",
    ):
        if not parent.is_dir():
            continue
        candidates.extend(
            path
            for path in parent.glob(f"{tool_name}*")
            if path.is_file() and os.access(path, os.X_OK)
        )

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate '{tool_name}' under {build_dir()} or extern/amrex. Set {env_var} to override."
        )
    return sorted(candidates)[0]


def snapshot_palette() -> Path | None:
    env = os.environ.get("QUOKKA_CANARY_PALETTE")
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"QUOKKA_CANARY_PALETTE points to missing file: {path}")

    for candidate in (
        repo_root() / "regression" / "canary" / "assets" / "Palette",
        repo_root() / "regression" / "canary" / "Palette",
    ):
        if candidate.is_file():
            return candidate
    return None


def runtime_env(instance: canary.TestInstance | None = None) -> dict[str, str]:
    env = dict(os.environ)
    self = instance or canary.get_instance()
    if self is not None and self.gpu_ids and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
    return env


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def case_log_name(test_name: str, suffix: str) -> str:
    return f"{slugify(test_name)}_{suffix}"


def run_simulation(
    *,
    executable: Path,
    input_file: str,
    plotfile_prefix: str,
    checkpoint_prefix: str,
    runtime_args: list[str] | None = None,
    nprocs: int = 1,
    env: dict[str, str] | None = None,
    ignore_return_code: bool = False,
) -> int:
    launcher = canary.Executable(os.environ.get("QUOKKA_CANARY_MPIEXEC", "mpirun"))
    numproc_flag = os.environ.get("QUOKKA_CANARY_MPIEXEC_NUMPROC_FLAG", "-n")
    args = [numproc_flag, str(nprocs), str(executable), input_file]
    args.extend(
        [
            f"plotfile_prefix={plotfile_prefix}",
            f"checkpoint_prefix={checkpoint_prefix}",
            "amr.checkpoint_files_output=0",
        ]
    )
    if runtime_args:
        args.extend(runtime_args)

    print(f"Launching: {' '.join(args)}")
    result = launcher(*args, env=env or runtime_env(), fail_on_error=False)
    if result.returncode != 0:
        if ignore_return_code:
            print(f"Simulation exited with code {result.returncode}, continuing because ignore_return_code is set")
            return result.returncode
        raise canary.TestFailed(f"Simulation exited with code {result.returncode}")
    return result.returncode


def compare_plotfiles(
    *,
    benchmark: Path,
    output: Path,
    log_name: str,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> None:
    fcompare = canary.Executable(find_tool("fcompare", "QUOKKA_CANARY_FCOMPARE"))
    args = ["--abort_if_not_all_found", "-n", "0"]
    if rel_tol is not None:
        args.extend(["--rel_tol", str(rel_tol)])
    if abs_tol is not None:
        args.extend(["--abs_tol", str(abs_tol)])
    args.extend([str(benchmark), str(output)])

    result = fcompare(*args, stdout=str, stderr=str, fail_on_error=False)
    Path(log_name).write_text(
        f"$ {result.cmd}\n\nstdout:\n{result.out or ''}\n\nstderr:\n{result.err or ''}\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise canary.TestDiffed(f"Mesh comparison failed; see {log_name}")
    if result.out and "< NaN present >" in result.out:
        raise canary.TestDiffed(f"Mesh comparison reported NaNs; see {log_name}")


def compare_particles(
    *,
    benchmark: Path,
    output: Path,
    particle_type: str,
    log_name: str,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> None:
    particle_compare = canary.Executable(find_tool("particle_compare", "QUOKKA_CANARY_PARTICLE_COMPARE"))
    args: list[str] = []
    if rel_tol is not None:
        args.extend(["--rel_tol", str(rel_tol)])
    if abs_tol is not None:
        args.extend(["--abs_tol", str(abs_tol)])
    args.extend([str(benchmark), str(output), particle_type])

    result = particle_compare(*args, stdout=str, stderr=str, fail_on_error=False)
    Path(log_name).write_text(
        f"$ {result.cmd}\n\nstdout:\n{result.out or ''}\n\nstderr:\n{result.err or ''}\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise canary.TestDiffed(f"Particle comparison failed for {particle_type}; see {log_name}")


def maybe_render_snapshot(*, plotfile: Path, variable: str, output_name: str) -> Path | None:
    try:
        fsnapshot = canary.Executable(find_tool("fsnapshot", "QUOKKA_CANARY_FSNAPSHOT"))
    except FileNotFoundError:
        print("fsnapshot not found; skipping visualization")
        return None

    before = set(Path.cwd().glob("*.ppm"))
    args = ["--variable", variable, str(plotfile)]
    palette = snapshot_palette()
    if palette is not None:
        args = ["--palette", str(palette), *args]
    else:
        print("Palette not found; using fsnapshot default palette")
    result = fsnapshot(*args, stdout=str, stderr=str, fail_on_error=False)
    if result.returncode != 0:
        print("fsnapshot failed; skipping visualization")
        return None

    after = set(Path.cwd().glob("*.ppm"))
    new_files = sorted(after - before)
    if not new_files:
        return None

    target = Path(output_name)
    new_files[-1].replace(target)
    return target


def replace_directory(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def rebaseline_plotfile(*, test_name: str, output_plotfile: Path) -> Path:
    root = benchmark_root(test_name)
    for existing in root.glob("*"):
        if existing.is_dir():
            shutil.rmtree(existing)
    destination = root / output_plotfile.name
    replace_directory(output_plotfile, destination)
    return destination


def run_nightly_case(case: dict[str, object]) -> int:
    self = canary.get_instance()
    if self is None:
        raise RuntimeError("Canary test instance is not available")

    test_name = str(case["test_name"])
    plotfile_prefix = str(case.get("plotfile_prefix", f"{test_name}_plt"))
    checkpoint_prefix = str(case.get("checkpoint_prefix", f"{test_name}_chk"))

    env = runtime_env(self)

    run_simulation(
        executable=quokka_executable(str(case["problem_name"])),
        input_file=str(case["input_file"]),
        plotfile_prefix=plotfile_prefix,
        checkpoint_prefix=checkpoint_prefix,
        runtime_args=list(case.get("runtime_args", [])),
        nprocs=int(case.get("nprocs", self.parameters.cpus)),
        env=env,
        ignore_return_code=bool(case.get("ignore_return_code", False)),
    )

    output_plotfile = latest_plotfile(plotfile_prefix)

    try:
        benchmark = benchmark_plotfile(test_name, plotfile_prefix)
    except FileNotFoundError as exc:
        print(exc)
        raise canary.TestDiffed("Benchmark is missing; run canary rebaseline after review") from None

    compare_plotfiles(
        benchmark=benchmark,
        output=output_plotfile,
        log_name=case_log_name(test_name, "fcompare.log"),
        rel_tol=case.get("rel_tol"),
        abs_tol=case.get("abs_tol"),
    )

    particle_types = [ptype for ptype in str(case.get("particle_types", "")).split() if ptype]
    for particle_type in particle_types:
        compare_particles(
            benchmark=benchmark,
            output=output_plotfile,
            particle_type=particle_type,
            log_name=case_log_name(test_name, f"{slugify(particle_type)}_particle_compare.log"),
            rel_tol=case.get("particle_rel_tol"),
            abs_tol=case.get("particle_abs_tol"),
        )

    vis_var = str(case.get("vis_var", ""))
    if vis_var:
        maybe_render_snapshot(
            plotfile=output_plotfile,
            variable=vis_var,
            output_name=case_log_name(test_name, f"{slugify(vis_var)}.ppm"),
        )

    return 0


def rebaseline_nightly_case(case: dict[str, object]) -> int:
    plotfile_prefix = str(case.get("plotfile_prefix", f"{case['test_name']}_plt"))
    output_plotfile = latest_plotfile(plotfile_prefix)
    destination = rebaseline_plotfile(test_name=str(case["test_name"]), output_plotfile=output_plotfile)
    print(f"Updated benchmark: {destination}")
    return 0


def nightly_case_main(case: dict[str, object], argv: list[str] | None = None) -> int:
    parser = canary.make_argument_parser()
    parser.add_argument("--rebaseline-benchmark", action="store_true")
    args = parser.parse_args(argv)
    if args.rebaseline_benchmark:
        return rebaseline_nightly_case(case)
    return run_nightly_case(case)
