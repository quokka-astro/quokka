#!/usr/bin/env python3
"""Kernel metrics and the paired-timing verdict for Quokka kernel autoresearch.

usage:
    analyze.py metric PROF_DIR KERNEL_RE     one line: kernel_ms launches total_ms vgpr sgpr scratch
    analyze.py kernels PROF_DIR              the run's kernels by total GPU time
    analyze.py verdict TIMING_DIR BAR ALLOW_CONTENDED

PROF_DIR holds a `rocprofv3 --kernel-trace --output-format csv` run. A
kernel's time is End_Timestamp - Start_Timestamp (ns); the metric is the sum
over every launch whose demangled name matches KERNEL_RE (re.search).

TIMING_DIR/runs.txt has one line per timed run:
    IDX LABEL RC T0 T1 METRIC_FILE FOREIGN
Consecutive runs (1,2), (3,4), ... form one pair of `cand` and `best`. Each
pair gives speedup = best kernel ms / cand kernel ms, so drift between pairs
cancels. A pair is flagged when either run failed, saw another GPU process
(unless ALLOW_CONTENDED), or ran at a busy GFX clock below 97% of the median.

The candidate meets the bar when:
- the median clean speedup is at least BAR;
- every clean pair is faster than 1.0, except that one may disagree once
  there are four or more;
- the median speedup of total GPU kernel time is at least 0.99, so work
  moved out of the matched kernels into others does not count as a win.
launch_match=0 when the candidate launches the matched kernels a different
number of times than the best: the runs did different work, so their times
are not comparable.
"""
import csv
import re
import statistics
import sys
from pathlib import Path

TOTAL_FLOOR = 0.99


def trace_rows(prof_dir):
    rows = []
    for path in sorted(Path(prof_dir).rglob("*kernel_trace.csv")):
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def col(row, *names):
    """First present column among NAMES (rocprofv3 versions rename a few)."""
    for n in names:
        if n in row and row[n] not in ("", None):
            return row[n]
    return None


def duration_ms(row):
    return (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e6


def metric(prof_dir, pattern):
    rows = trace_rows(prof_dir)
    if not rows:
        print("kernel_ms=- launches=0 total_ms=- vgpr=- sgpr=- scratch=- error=no-kernel-trace")
        return
    rx = re.compile(pattern)
    hits = [r for r in rows if rx.search(r.get("Kernel_Name", ""))]
    total = sum(duration_ms(r) for r in rows)

    def peak(*names):
        vals = [int(v) for r in hits if (v := col(r, *names)) and v.isdigit()]
        return max(vals) if vals else "-"
    # On gfx90a rocprofv3 misreports VGPR_Count for kernels whose ACCUM_OFFSET
    # is above 128 (ROCm/rocm-systems#13686); treat vgpr as a hint only.
    print(f"kernel_ms={sum(duration_ms(r) for r in hits):.3f} launches={len(hits)} total_ms={total:.3f} "
          f"vgpr={peak('Arch_VGPR_Count', 'VGPR_Count')} sgpr={peak('SGPR_Count')} "
          f"scratch={peak('Scratch_Size', 'Private_Segment_Size')}")


def kernels(prof_dir):
    by_name = {}
    for r in trace_rows(prof_dir):
        n = r.get("Kernel_Name", "?")
        t, c = by_name.get(n, (0.0, 0))
        by_name[n] = (t + duration_ms(r), c + 1)
    total = sum(t for t, _ in by_name.values()) or 1.0
    print(f"{'total_ms':>10} {'share':>6} {'launches':>8}  kernel")
    for n, (t, c) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:40]:
        print(f"{t:10.3f} {100 * t / total:5.1f}% {c:8d}  {n[:400]}")


def telemetry(path):
    """[(epoch_s, gfx_clk_mhz, gfx_util_pct)] from `amd-smi monitor --csv`."""
    samples, header = [], None
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return samples
    for row in csv.reader(lines):
        if row and row[0] == "timestamp":
            header = row
            continue
        if not header or len(row) != len(header):
            continue
        try:
            fields = dict(zip(header, row))
            samples.append((float(fields["timestamp"]), float(fields["gfx_clk"]), float(fields["gfx"])))
        except (KeyError, ValueError):
            continue
    return samples


def parse_metric(path):
    try:
        text = Path(path).read_text()
    except OSError:
        return {}
    out = {}
    for word in text.split():
        k, _, v = word.partition("=")
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def verdict(out, bar, allow_contended):
    samples = telemetry(out / "telemetry.csv")
    runs = []
    for line in (out / "runs.txt").read_text().splitlines():
        idx, label, rc, t0, t1, mfile, foreign = line.split()
        m = parse_metric(mfile)
        busy = [clk for t, clk, util in samples if float(t0) <= t <= float(t1) and util >= 50]
        runs.append({
            "idx": int(idx), "label": label, "rc": int(rc), "foreign": int(foreign),
            "ms": m.get("kernel_ms"), "total": m.get("total_ms"), "launches": m.get("launches"),
            "clk": statistics.mean(busy) if busy else None,
        })
    clocks = [r["clk"] for r in runs if r["clk"]]
    median_clk = statistics.median(clocks) if clocks else None
    for r in runs:
        flags = []
        if r["rc"] != 0 or not r["ms"]:
            flags.append("failed")
        if r["foreign"]:
            flags.append("contended")
        if median_clk and r["clk"] and r["clk"] < 0.97 * median_clk:
            flags.append(f"clock={r['clk']:.0f}")
        r["flags"] = flags

    pairs = []
    for a, b in zip(runs[0::2], runs[1::2]):
        cand, best = (a, b) if a["label"] == "cand" else (b, a)
        flags = sorted(set(cand["flags"] + best["flags"]))
        blocking = [f for f in flags if not (f == "contended" and allow_contended)]
        speedup = total = None
        if cand["ms"] and best["ms"]:
            speedup = best["ms"] / cand["ms"]
        if cand["total"] and best["total"]:
            total = best["total"] / cand["total"]
        pairs.append({"cand": cand, "best": best, "flags": flags, "clean": not blocking and speedup,
                      "speedup": speedup, "total": total})

    for n, p in enumerate(pairs, 1):
        fmt = lambda v: f"{v:.4f}" if v else "-"
        print(f"PAIR {n} speedup={fmt(p['speedup'])} total_speedup={fmt(p['total'])} "
              f"cand_ms={p['cand']['ms']} best_ms={p['best']['ms']} "
              f"launches={p['cand']['launches']}/{p['best']['launches']} "
              f"flags={','.join(p['flags']) or 'none'}{'' if p['clean'] else ' (excluded)'}")
    clean = [p for p in pairs if p["clean"]]
    print(f"CLEAN_PAIRS {len(clean)}")
    # Ask for re-runs only for transient flags; contention does not improve
    # with another pair.
    retry = [p for p in pairs if not p["clean"] and "contended" not in p["flags"]
             and "failed" not in p["cand"]["flags"]]
    print(f"FLAGGED_PAIRS {len(retry)}")

    failed = sum("failed" in r["flags"] for r in runs if r["label"] == "cand")
    result = {"clean_pairs": len(clean), "failed_runs": failed}
    if clean:
        s = [p["speedup"] for p in clean]
        tot = [p["total"] for p in clean if p["total"]]
        cand_ms = statistics.median(p["cand"]["ms"] for p in clean)
        best_ms = statistics.median(p["best"]["ms"] for p in clean)
        launch_match = all(p["cand"]["launches"] == p["best"]["launches"] for p in clean)
        agree = sum(x > 1.0 for x in s)
        total_speedup = statistics.median(tot) if tot else 0.0
        result.update({
            "median_speedup": f"{statistics.median(s):.4f}", "min_speedup": f"{min(s):.4f}",
            "max_speedup": f"{max(s):.4f}", "total_speedup": f"{total_speedup:.4f}",
            "agree": f"{agree}/{len(s)}", "cand_ms": f"{cand_ms:.3f}", "best_ms": f"{best_ms:.3f}",
            "aa_spread": f"{max(abs(x - 1.0) for x in s):.4f}",
            "aa_median_dev": f"{statistics.median(abs(x - 1.0) for x in s):.4f}",
            "contended_pairs": sum("contended" in p["flags"] for p in clean),
            "launch_match": int(launch_match),
            "meets_bar": int(len(clean) >= 2 and launch_match and statistics.median(s) >= bar
                             and total_speedup >= TOTAL_FLOOR
                             and agree >= len(s) - (1 if len(s) >= 4 else 0)),
        })
    else:
        result["meets_bar"] = 0
    # clean_pairs comes first and is followed by a space; remote.sh matches it.
    print("RESULT " + " ".join(f"{k}={v}" for k, v in result.items()))


def main():
    mode = sys.argv[1]
    if mode == "metric":
        metric(sys.argv[2], sys.argv[3])
    elif mode == "kernels":
        kernels(sys.argv[2])
    elif mode == "verdict":
        verdict(Path(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
