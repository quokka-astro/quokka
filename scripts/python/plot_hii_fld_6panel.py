#!/usr/bin/env python3

import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CM_PER_PC = 3.08567758e18
Q = 1.0e49
ALPHAB = 2.6e-13
NH = 1.0e3
T0 = 500.0
TION = 1.0e4
RS_CM = ((3.0 * Q) / (4.0 * math.pi * ALPHAB * NH * NH)) ** (1.0 / 3.0)
RS_PC = RS_CM / CM_PER_PC


def parse_header(plotfile: Path):
	lines = plotfile.joinpath("Header").read_text().splitlines()
	ncomp = int(lines[1].strip())
	comp_names = [lines[2 + i].strip() for i in range(ncomp)]

	box_line = next(l for l in lines if l.strip().startswith("((") and ") (" in l)
	box_triplets = re.findall(r"\(([-\d]+),([-\d]+),([-\d]+)\)", box_line)
	if len(box_triplets) < 2:
		raise RuntimeError(f"Could not parse domain box from {plotfile / 'Header'}")
	lo = np.array([int(v) for v in box_triplets[0]], dtype=int)
	hi = np.array([int(v) for v in box_triplets[1]], dtype=int)
	ncell = hi - lo + 1

	float_token = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$")
	float3 = []
	for line in lines:
		toks = line.strip().split()
		if len(toks) == 3 and all(float_token.match(t) for t in toks):
			float3.append([float(t) for t in toks])
	if len(float3) < 2:
		raise RuntimeError(f"Could not parse prob_lo/prob_hi from {plotfile / 'Header'}")

	prob_lo = np.array(float3[0])
	prob_hi = np.array(float3[1])
	return comp_names, ncell, prob_lo, prob_hi


def read_single_fab_data(plotfile: Path, ncell, ncomp):
	cell_file = plotfile / "Level_0" / "Cell_D_00000"
	with cell_file.open("rb") as f:
		_ = f.readline()  # FAB header
		arr = np.fromfile(f, dtype="<f8")

	ntot = int(np.prod(ncell))
	expected = ntot * ncomp
	if arr.size != expected:
		raise RuntimeError(f"Unexpected data size in {cell_file}: got {arr.size}, expected {expected}")

	out = np.empty((ncomp, ncell[2], ncell[1], ncell[0]), dtype=np.float64)
	for comp in range(ncomp):
		slab = arr[comp * ntot:(comp + 1) * ntot]
		out[comp] = slab.reshape((ncell[2], ncell[1], ncell[0]), order="C")
	return out


def get_fields(plotdir: Path):
	comp_names, ncell, prob_lo, prob_hi = parse_header(plotdir)
	data = read_single_fab_data(plotdir, ncell, len(comp_names))

	try:
		idx_temperature = comp_names.index("temperature")
	except ValueError as exc:
		raise RuntimeError(f"Plotfile {plotdir} is missing derived field 'temperature'.") from exc
	temperature = data[idx_temperature]

	zmid = ncell[2] // 2
	extent_pc = [prob_lo[0] / CM_PER_PC, prob_hi[0] / CM_PER_PC, prob_lo[1] / CM_PER_PC, prob_hi[1] / CM_PER_PC]
	return ncell[0], temperature[zmid, :, :], extent_pc


def main():
	parser = argparse.ArgumentParser(description="Make 6-panel HII FLD plot (temperature + intensity proxy at 16/32/64).")
	parser.add_argument(
		"--plotfiles",
		nargs=3,
		default=[
			"tests/plt_hii_fld_resid160000004",
			"tests/plt_hii_fld_resid320000004",
			"tests/plt_hii_fld_resid640000004",
		],
		help="Three plotfile directories (16, 32, 64).",
	)
	parser.add_argument("--output", default="tests/hii_fld_6panel_16_32_64_resid.png", help="Output image path.")
	args = parser.parse_args()

	plotfiles = [Path(p) for p in args.plotfiles]
	for p in plotfiles:
		if not p.exists():
			raise FileNotFoundError(f"Plotfile not found: {p}")

	fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

	for col, plotdir in enumerate(plotfiles):
		n, temp2d, extent = get_fields(plotdir)

		ax_t = axes[col]
		im_t = ax_t.imshow(temp2d, origin="lower", extent=extent, cmap="inferno", vmin=T0, vmax=TION, interpolation="nearest")
		ax_t.add_patch(plt.Circle((0.0, 0.0), RS_PC, color="cyan", fill=False, linewidth=1.6))
		ax_t.set_title(f"Temperature (N={n}^3)")
		ax_t.set_xlabel("x [pc]")
		ax_t.set_ylabel("y [pc]")
		ax_t.set_aspect("equal")

	cbar_t = fig.colorbar(im_t, ax=axes, shrink=0.92, pad=0.02)
	cbar_t.set_label("Temperature [K]")

	out = Path(args.output)
	fig.savefig(out, dpi=220)
	print(f"Wrote {out}")


if __name__ == "__main__":
	main()
