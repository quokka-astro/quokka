#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np


def lambda_fld(R: np.ndarray) -> np.ndarray:
	return (2.0 + R) / (6.0 + 3.0 * R + R * R)


def x_eq(alphaB: float, nH: float, sigma_HI: float, c_light: float, phi: np.ndarray) -> np.ndarray:
	phi_pos = np.maximum(phi, 0.0)
	a = alphaB * nH
	b = c_light * sigma_HI * phi_pos
	disc = b * b + 4.0 * a * b
	x = (-b + np.sqrt(disc)) / (2.0 * a)
	return np.clip(x, 0.0, 1.0)


def source_top_hat(rf: np.ndarray, Q: float, r_src: float) -> np.ndarray:
	src0 = 3.0 * Q / (4.0 * math.pi * r_src**3)
	r_in = rf[:-1]
	r_out = rf[1:]
	num = np.maximum(np.minimum(r_out, r_src) ** 3 - np.minimum(r_in, r_src) ** 3, 0.0)
	den = np.maximum(r_out**3 - r_in**3, 1.0e-60)
	frac = num / den
	return src0 * frac


def compute_face_diffusion(phi: np.ndarray, rf: np.ndarray, rc: np.ndarray, c_light: float, kappa: float) -> np.ndarray:
	nr = phi.size
	Df = np.zeros(nr + 1, dtype=np.float64)
	Df[0] = c_light / (3.0 * kappa)

	phi_L = np.maximum(phi[:-1], 0.0)
	phi_R = np.maximum(phi[1:], 0.0)
	grad = (phi_R - phi_L) / np.maximum(rc[1:] - rc[:-1], 1.0e-60)
	phi_face = 0.5 * (phi_L + phi_R)
	R = np.abs(grad) / np.maximum(kappa * phi_face, 1.0e-60)
	Df[1:nr] = c_light * lambda_fld(R) / kappa

	dr_out = max(rf[-1] - rc[-1], 1.0e-60)
	grad_o = -max(phi[-1], 0.0) / dr_out
	phi_face_o = 0.5 * max(phi[-1], 0.0)
	Ro = abs(grad_o) / max(kappa * phi_face_o, 1.0e-60)
	Df[-1] = c_light * lambda_fld(np.array([Ro]))[0] / kappa
	return Df


def compute_divergence(phi: np.ndarray, rf: np.ndarray, rc: np.ndarray, vol: np.ndarray, Df: np.ndarray) -> np.ndarray:
	nr = phi.size
	F = np.zeros(nr + 1, dtype=np.float64)
	grad = (phi[1:] - phi[:-1]) / np.maximum(rc[1:] - rc[:-1], 1.0e-60)
	F[0] = 0.0
	F[1:nr] = -Df[1:nr] * grad
	grad_o = -max(phi[-1], 0.0) / max(rf[-1] - rc[-1], 1.0e-60)
	F[-1] = -Df[-1] * grad_o
	return (rf[1:] ** 2 * F[1:] - rf[:-1] ** 2 * F[:-1]) / np.maximum(vol, 1.0e-60)


def build_linear_system(rc: np.ndarray, rf: np.ndarray, vol: np.ndarray, Df: np.ndarray, k_reac: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	nr = vol.size
	lower = np.zeros(nr, dtype=np.float64)
	diag = np.zeros(nr, dtype=np.float64)
	upper = np.zeros(nr, dtype=np.float64)

	for i in range(nr):
		aW = 0.0
		if i > 0:
			dr_w = rc[i] - rc[i - 1]
			aW = rf[i] ** 2 * Df[i] / (vol[i] * max(dr_w, 1.0e-60))

		if i < nr - 1:
			dr_e = rc[i + 1] - rc[i]
			aE = rf[i + 1] ** 2 * Df[i + 1] / (vol[i] * max(dr_e, 1.0e-60))
		else:
			dr_e = rf[-1] - rc[-1]
			aE = rf[-1] ** 2 * Df[-1] / (vol[i] * max(dr_e, 1.0e-60))

		lower[i] = -aW if i > 0 else 0.0
		diag[i] = aW + aE + k_reac[i]
		upper[i] = -aE if i < nr - 1 else 0.0

	return lower, diag, upper


def solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
	n = diag.size
	cprime = np.zeros(n, dtype=np.float64)
	dprime = np.zeros(n, dtype=np.float64)
	x = np.zeros(n, dtype=np.float64)

	cprime[0] = upper[0] / diag[0] if n > 1 else 0.0
	dprime[0] = rhs[0] / diag[0]
	for i in range(1, n):
		den = diag[i] - lower[i] * cprime[i - 1]
		cprime[i] = upper[i] / den if i < n - 1 else 0.0
		dprime[i] = (rhs[i] - lower[i] * dprime[i - 1]) / den

	x[-1] = dprime[-1]
	for i in range(n - 2, -1, -1):
		x[i] = dprime[i] - cprime[i] * x[i + 1]
	return x


def compute_rhs_dt_limit(
	phi: np.ndarray,
	rf: np.ndarray,
	rc: np.ndarray,
	vol: np.ndarray,
	src: np.ndarray,
	alphaB: float,
	nH: float,
	sigma_HI: float,
	c_light: float,
	kappa: float,
) -> tuple[np.ndarray, float]:
	nr = phi.size

	xc = x_eq(alphaB, nH, sigma_HI, c_light, phi)
	nHI = nH * (1.0 - xc)
	k_reac = c_light * sigma_HI * nHI
	sink = k_reac * phi

	Df = compute_face_diffusion(phi, rf, rc, c_light, kappa)
	div = compute_divergence(phi, rf, rc, vol, Df)
	rhs = -div + src - sink

	dt_limit = math.inf
	for i in range(nr):
		aW = 0.0
		if i > 0:
			dr_w = rc[i] - rc[i - 1]
			aW = rf[i] ** 2 * Df[i] / (vol[i] * max(dr_w, 1.0e-60))
		aE = 0.0
		if i < nr - 1:
			dr_e = rc[i + 1] - rc[i]
			aE = rf[i + 1] ** 2 * Df[i + 1] / (vol[i] * max(dr_e, 1.0e-60))
		else:
			dr_e = rf[-1] - rc[-1]
			aE = rf[-1] ** 2 * Df[-1] / (vol[i] * max(dr_e, 1.0e-60))
		den = aW + aE + k_reac[i]
		dt_limit = min(dt_limit, 0.5 / max(den, 1.0e-60))

	return rhs, dt_limit


def compute_explicit_and_reaction(
	phi: np.ndarray,
	rf: np.ndarray,
	rc: np.ndarray,
	vol: np.ndarray,
	src: np.ndarray,
	alphaB: float,
	nH: float,
	sigma_HI: float,
	c_light: float,
	kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
	nr = phi.size
	xc = x_eq(alphaB, nH, sigma_HI, c_light, phi)
	nHI = nH * (1.0 - xc)
	k_reac = c_light * sigma_HI * nHI

	Df = compute_face_diffusion(phi, rf, rc, c_light, kappa)
	div = compute_divergence(phi, rf, rc, vol, Df)
	E = -div + src
	return E, k_reac


def light_crossing_timestep(rf: np.ndarray, c_light: float) -> float:
	dr = rf[1:] - rf[:-1]
	return float(np.min(dr) / max(c_light, 1.0e-60))


def solve_relaxation(
	Q: float,
	alphaB: float,
	nH: float,
	sigma_HI: float,
	c_light: float,
	rmax: float,
	r_src: float,
	box_size: float,
	nr: int,
	max_iters: int,
	tol: float,
	method: str,
	cfl: float,
	log_every: int,
	osc_min_iters: int,
	osc_window: int,
	osc_rel_band: float,
	osc_net_decay: float,
):
	rs = ((3.0 * Q) / (4.0 * math.pi * alphaB * nH * nH)) ** (1.0 / 3.0)
	cap = alphaB * nH * nH
	kappa = 1.0 / max(box_size, 1.0e-60)

	rf = np.linspace(0.0, rmax, nr + 1)
	rc = 0.5 * (rf[:-1] + rf[1:])
	vol = (rf[1:] ** 3 - rf[:-1] ** 3) / 3.0

	src = source_top_hat(rf, Q, r_src)
	src_scale = max(np.max(src), 1.0e-60)

	# initial guess
	phi = cap * np.exp(-(rc / max(r_src, 1.0e-60)) ** 2)
	last_resid = math.inf
	n_iters = 0
	n_substeps = 0
	effective_stages = 1
	stop_reason = "max_iters"

	if method == "picard":
		omega = 0.9
		for it in range(max_iters):
			n_iters = it + 1
			n_substeps = n_iters
			phi_old = phi.copy()

			xc_old = x_eq(alphaB, nH, sigma_HI, c_light, phi_old)
			nHI_old = nH * (1.0 - xc_old)
			k_reac = c_light * sigma_HI * nHI_old

			Df = compute_face_diffusion(phi_old, rf, rc, c_light, kappa)
			lower, diag, upper = build_linear_system(rc, rf, vol, Df, k_reac)
			rhs_lin = src.copy()

			phi_lin = solve_tridiagonal(lower, diag, upper, rhs_lin)
			phi = np.maximum((1.0 - omega) * phi_old + omega * phi_lin, 0.0)

			rhs, _ = compute_rhs_dt_limit(
			    phi,
			    rf,
			    rc,
			    vol,
			    src,
			    alphaB,
			    nH,
			    sigma_HI,
			    c_light,
			    kappa,
			)
			last_resid = float(np.max(np.abs(rhs)) / src_scale)
			if log_every > 0 and (it == 0 or (it + 1) % log_every == 0):
				print(f"[iter {it + 1:7d}] residual={last_resid:.6e}")
			if last_resid < tol:
				stop_reason = "converged_tol"
				break

			delta = np.max(np.abs(phi - phi_old)) / max(np.max(np.abs(phi_old)), 1.0e-60)
			if delta > 0.5:
				omega = max(0.6, 0.9 * omega)
	elif method == "imexrk2":
		effective_stages = 2
		dt_adv_c = light_crossing_timestep(rf, c_light)
		cfl_work = max(cfl, 1.0e-16)
		cfl_floor = max(1.0e-16, cfl * 1.0e-6)
		res_hist: list[float] = []
		for it in range(max_iters):
			n_iters = it + 1
			n_substeps = n_iters * effective_stages
			rhs_picard, _ = compute_rhs_dt_limit(
			    phi,
			    rf,
			    rc,
			    vol,
			    src,
			    alphaB,
			    nH,
			    sigma_HI,
			    c_light,
			    kappa,
			)
			last_resid = float(np.max(np.abs(rhs_picard)) / src_scale)
			if log_every > 0 and (it == 0 or (it + 1) % log_every == 0):
				print(f"[iter {it + 1:7d}] residual={last_resid:.6e}")
			if last_resid < tol:
				stop_reason = "converged_tol"
				break

			dt_super = cfl_work * dt_adv_c

			max_stage_retries = 12
			accepted = False
			dt_try = max(dt_super, 1.0e-60)
			phi_new = phi
			for _ in range(max_stage_retries):
				E0, k0 = compute_explicit_and_reaction(
				    phi,
				    rf,
				    rc,
				    vol,
				    src,
				    alphaB,
				    nH,
				    sigma_HI,
				    c_light,
				    kappa,
				)
				phi1 = (phi + dt_try * E0) / np.maximum(1.0 + dt_try * k0, 1.0e-60)
				E1, k1 = compute_explicit_and_reaction(
				    phi1,
				    rf,
				    rc,
				    vol,
				    src,
				    alphaB,
				    nH,
				    sigma_HI,
				    c_light,
				    kappa,
				)
				R0 = -k0 * phi
				phi_new = (phi + 0.5 * dt_try * (E0 + E1 + R0)) / np.maximum(1.0 + 0.5 * dt_try * k1, 1.0e-60)
				if float(np.min(phi1)) >= 0.0 and float(np.min(phi_new)) >= 0.0:
					accepted = True
					break
				dt_try *= 0.5

			if not accepted:
				cfl_work *= 0.5
				if log_every > 0 and (it == 0 or (it + 1) % log_every == 0):
					print(f"[iter {it + 1:7d}] step rejected; reducing CFL to {cfl_work:.3e} and retrying")
				if cfl_work < cfl_floor:
					stop_reason = "cfl_underflow"
					if log_every > 0:
						print(f"[iter {it + 1:7d}] stopping: CFL underflow after repeated step rejections")
					break
				continue

			phi = phi_new

			res_hist.append(last_resid)
			if osc_window > 2 and len(res_hist) >= osc_window and (it + 1) >= osc_min_iters:
				seg = np.asarray(res_hist[-osc_window:], dtype=np.float64)
				seg_min = float(np.min(seg))
				seg_max = float(np.max(seg))
				rel_amp = (seg_max - seg_min) / max(seg_min, 1.0e-60)
				net_decay = (seg[0] - seg[-1]) / max(seg[0], 1.0e-60)
				if rel_amp >= osc_rel_band and net_decay <= osc_net_decay and seg_min > (10.0 * tol):
					stop_reason = "stalled_oscillatory"
					if log_every > 0:
						print(
						    f"[iter {it + 1:7d}] stopping: oscillatory residual detected "
						    f"(rel_amp={rel_amp:.3e}, net_decay={net_decay:.3e})"
						)
					break
	else:
		raise ValueError(f"Unknown method '{method}'. Expected 'picard' or 'imexrk2'.")

	xc = x_eq(alphaB, nH, sigma_HI, c_light, phi)
	idx = np.where(xc <= 0.5)[0]
	if len(idx) == 0:
		r50 = math.nan
	else:
		j = int(idx[0])
		if j == 0:
			r50 = rc[0]
		else:
			r1, r2 = rc[j - 1], rc[j]
			x1, x2 = xc[j - 1], xc[j]
			r50 = r1 + (0.5 - x1) * (r2 - r1) / (x2 - x1)

	return {
	    "rs": rs,
	    "r50": r50,
	    "resid": last_resid,
	    "iterations": n_iters,
	    "substeps": n_substeps,
	    "stages": effective_stages,
	    "stop_reason": stop_reason,
	    "r": rc,
	    "phi": phi,
	    "x": xc,
	    "cap": cap,
	}


def main():
	parser = argparse.ArgumentParser(description="Spherical FLD steady solve via pseudo-time relaxation (FV radial).")
	parser.add_argument("--Q", type=float, default=1.0e49)
	parser.add_argument("--alphaB", type=float, default=2.6e-13)
	parser.add_argument("--nH", type=float, default=1.0e3)
	parser.add_argument("--sigma-HI", type=float, default=6.3e-18)
	parser.add_argument("--c-light", type=float, default=2.99792458e10)
	parser.add_argument("--rmax", type=float, default=6.0e18)
	parser.add_argument("--r-src", type=float, default=3.0e17)
	parser.add_argument("--box-size", type=float, default=1.2e19)
	parser.add_argument("--nr", type=int, default=1024)
	parser.add_argument("--max-iters", type=int, default=20000)
	parser.add_argument("--tol", type=float, default=1.0e-6)
	parser.add_argument("--method", choices=["picard", "imexrk2"], default="picard", help="Steady-state solver algorithm.")
	parser.add_argument("--cfl", type=float, default=0.9, help="Pseudo-time CFL multiplier for IMEXRK2.")
	parser.add_argument("--log-every", type=int, default=0, help="Print residual diagnostics every N iterations (0 disables).")
	parser.add_argument("--osc-min-iters", type=int, default=1000, help="Minimum IMEXRK2 iterations before oscillation-stop checks.")
	parser.add_argument("--osc-window", type=int, default=200, help="Window size (iterations) for oscillation-stop checks.")
	parser.add_argument("--osc-rel-band", type=float, default=0.25, help="Minimum relative residual band in window to flag oscillation.")
	parser.add_argument(
	    "--osc-net-decay",
	    type=float,
	    default=0.02,
	    help="Maximum fractional net residual decay over the oscillation window to treat as stalled.",
	)
	parser.add_argument("--output", type=Path, default=Path("tests/spherical_fld_reference_relaxation.csv"))
	parser.add_argument("--plot", action="store_true", default=True, help="Write a two-panel x_H and n_gamma vs radius plot (default: on).")
	parser.add_argument("--no-plot", action="store_false", dest="plot", help="Disable plotting.")
	parser.add_argument(
	    "--plot-compare",
	    action="store_true",
	    help="Write a comparison plot with the selected method, a second method, and exact R_s.",
	)
	parser.add_argument(
	    "--compare-method",
	    choices=["picard", "imexrk2"],
	    default="imexrk2",
	    help="Second method used by --plot-compare.",
	)
	parser.add_argument(
	    "--plot-output",
	    type=Path,
	    default=Path("tests/spherical_fld_relaxation_ngamma_vs_radius.png"),
	    help="Output path for two-panel x_H and n_gamma vs radius plot.",
	)
	args = parser.parse_args()

	res = solve_relaxation(
	    Q=args.Q,
	    alphaB=args.alphaB,
	    nH=args.nH,
	    sigma_HI=args.sigma_HI,
	    c_light=args.c_light,
	    rmax=args.rmax,
	    r_src=args.r_src,
	    box_size=args.box_size,
	    nr=args.nr,
	    max_iters=args.max_iters,
	    tol=args.tol,
	    method=args.method,
	    cfl=args.cfl,
	    log_every=args.log_every,
	    osc_min_iters=args.osc_min_iters,
	    osc_window=args.osc_window,
	    osc_rel_band=args.osc_rel_band,
	    osc_net_decay=args.osc_net_decay,
	)

	print(f"R_s(analytic) = {res['rs']:.8e} cm")
	print(f"r50(relaxation) = {res['r50']:.8e} cm")
	print(f"r50 / R_s = {res['r50'] / res['rs']:.8f}")
	print(f"final rel residual = {res['resid']:.3e}")
	print(f"iterations = {res['iterations']}")
	print(f"total substeps = {res['substeps']} (stages/iter = {res['stages']})")
	print(f"stop reason = {res['stop_reason']}")

	lines = ["r_cm,phi,phi_over_cap,x_eq"]
	for rr, pp, xx in zip(res["r"], res["phi"], res["x"]):
		lines.append(f"{rr:.16e},{pp:.16e},{(pp / res['cap']):.16e},{xx:.16e}")
	args.output.write_text("\n".join(lines) + "\n")
	print(f"Wrote {args.output}")

	if args.plot_compare:
		if args.compare_method == args.method:
			raise ValueError("--compare-method must be different from --method when using --plot-compare.")

		res_cmp = solve_relaxation(
		    Q=args.Q,
		    alphaB=args.alphaB,
		    nH=args.nH,
		    sigma_HI=args.sigma_HI,
		    c_light=args.c_light,
		    rmax=args.rmax,
		    r_src=args.r_src,
		    box_size=args.box_size,
		    nr=args.nr,
		    max_iters=args.max_iters,
		    tol=args.tol,
		    method=args.compare_method,
		    cfl=args.cfl,
		    log_every=args.log_every,
		    osc_min_iters=args.osc_min_iters,
		    osc_window=args.osc_window,
		    osc_rel_band=args.osc_rel_band,
		    osc_net_decay=args.osc_net_decay,
		)

		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		pc_in_cm = 3.085677581491367e18
		r_pc = res["r"] / pc_in_cm
		rs_pc = res["rs"] / pc_in_cm

		fig, (ax_x, ax_phi) = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
		ax_x.plot(r_pc, res["x"], lw=1.9, label=args.method, color="tab:blue")
		ax_x.plot(r_pc, res_cmp["x"], lw=1.9, label=args.compare_method, color="tab:orange")
		ax_x.axvline(rs_pc, color="k", ls="--", lw=1.4, label=r"Exact $R_s$")
		ax_x.set_xlabel("Radius [pc]")
		ax_x.set_ylabel(r"$x_H$")
		ax_x.set_title(r"$x_H(r)$")
		ax_x.set_ylim(0.0, 1.02)
		ax_x.grid(True, alpha=0.25)
		ax_x.legend()

		ax_phi.plot(r_pc, res["phi"], lw=1.9, label=args.method, color="tab:blue")
		ax_phi.plot(r_pc, res_cmp["phi"], lw=1.9, label=args.compare_method, color="tab:orange")
		ax_phi.axvline(rs_pc, color="k", ls="--", lw=1.4, label=r"Exact $R_s$")
		ax_phi.set_xlabel("Radius [pc]")
		ax_phi.set_ylabel(r"$n_\gamma$ [cm$^{-3}$]")
		ax_phi.set_title(r"$n_\gamma(r)$")
		ax_phi.grid(True, alpha=0.25)
		ax_phi.legend()
		fig.savefig(args.plot_output, dpi=200)
		plt.close(fig)
		print(f"Wrote {args.plot_output}")
	elif args.plot:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		pc_in_cm = 3.085677581491367e18
		r_pc = res["r"] / pc_in_cm
		fig, (ax_x, ax_phi) = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
		ax_x.plot(r_pc, res["x"], lw=1.8, color="tab:blue")
		ax_x.axvline(res["rs"] / pc_in_cm, color="k", ls="--", lw=1.3)
		ax_x.set_xlabel("Radius [pc]")
		ax_x.set_ylabel(r"$x_H$")
		ax_x.set_title(r"Relaxation: $x_H(r)$")
		ax_x.set_ylim(0.0, 1.02)
		ax_x.grid(True, alpha=0.25)

		ax_phi.plot(r_pc, res["phi"], lw=1.8, color="tab:blue")
		ax_phi.axvline(res["rs"] / pc_in_cm, color="k", ls="--", lw=1.3)
		ax_phi.set_xlabel("Radius [pc]")
		ax_phi.set_ylabel(r"$n_\gamma$ [cm$^{-3}$]")
		ax_phi.set_title(r"Relaxation: $n_\gamma(r)$")
		ax_phi.grid(True, alpha=0.25)
		fig.savefig(args.plot_output, dpi=200)
		plt.close(fig)
		print(f"Wrote {args.plot_output}")


if __name__ == "__main__":
	main()
