#!/usr/bin/env python3
"""Compile a fresh snapshot, then independently kernel-check every module."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--coqc", default="coqc")
parser.add_argument("--coqchk", default="coqchk")
parser.add_argument("--build-dir", default="build/verification/radiation_coupling")
args = parser.parse_args()
source = Path(__file__).resolve().parent
expected = json.loads((source / "source_sha256.json").read_text())
paths = {f"{p.parent.name}.{p.stem}": p for p in source.glob("*/*.v")}
actual = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
if actual != expected:
    raise SystemExit("Proof source manifest mismatch; review source changes before updating it.")

def uncomment(text):
    # Rocq comments nest; do not let a scan accidentally hide an admission.
    out, depth, i = [], 0, 0
    while i < len(text):
        if text[i:i+2] == "(*": depth += 1; i += 2
        elif depth and text[i:i+2] == "*)": depth -= 1; i += 2
        else:
            if not depth: out.append(text[i])
            i += 1
    if depth: raise ValueError("Unclosed comment")
    return "".join(out)

deps = {}
for name, path in paths.items():
    text = uncomment(path.read_text())
    if re.search(r"\b(Admitted|admit|Axiom|Parameter|Conjecture|Abort)\b", text):
        raise SystemExit(f"Unfinished proof or custom assumption declaration: {name}")
    deps[name] = {ns+"."+m for ns, names in re.findall(
        r"From\s+(BlackBox|MultiGroup)\s+Require\s+(?:Import|Export)\s+([^.]*)\.", text, re.S)
        for m in names.split()}
order, visiting = [], set()
def visit(name):
    if name in order: return
    if name in visiting: raise ValueError(f"Import cycle: {name}")
    if name not in deps: raise ValueError(f"Missing module: {name}")
    visiting.add(name)
    for dep in sorted(deps[name]): visit(dep)
    visiting.remove(name)
    order.append(name)
for name in sorted(paths): visit(name)
root = Path(args.build_dir).resolve()
root.mkdir(parents=True, exist_ok=True)
build = Path(tempfile.mkdtemp(prefix="check-", dir=root))
for name, path in paths.items():
    target = build / path.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
flags = ["-Q", str(build/"BlackBox"), "BlackBox", "-Q", str(build/"MultiGroup"), "MultiGroup"]
env = os.environ.copy()
compiler = shutil.which(args.coqc)
checker = shutil.which(args.coqchk)
if not compiler or not checker: raise SystemExit("Install Rocq with Coquelicot and Flocq; provide --coqc and --coqchk if needed.")
env["PATH"] = str(Path(compiler).parent) + os.pathsep + env["PATH"]
with (build/"compile.log").open("w") as log:
    for name in order:
        print("Compiling", name, flush=True)
        subprocess.run([compiler, "-q", *flags, str(build/paths[name].relative_to(source))],
                       cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
with (build/"coqchk.log").open("w") as log:
    print("Independent kernel check", flush=True)
    subprocess.run([checker, *flags, *order], cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
if any(hashlib.sha256(p.read_bytes()).hexdigest() != actual[str(p.relative_to(source))] for p in paths.values()):
    raise SystemExit("Proof source changed during verification")
(build/"summary.json").write_text(json.dumps({"all_passed": True, "modules": order, "source_sha256": actual}, indent=2)+"\n")
print(f"PASS: {len(order)} modules compiled and kernel-checked; logs: {build}")
