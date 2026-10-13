#!/usr/bin/env python3
"""Offline reference checks. mpmath is test-only; C++ runtime is binary64.

The reference solves the coupled physical equations in gas-temperature space,
not the implementation's cancellation-resistant inner charts / outer ratios.
All decimal case literals are first rounded to binary64 and then imported
exactly into mpmath, so this tests the stated exact-stored-input target.
"""
import argparse
import dataclasses
import json
import math
import pathlib
import random
import subprocess
import sys
import time

import mpmath as mp

@dataclasses.dataclass
class Group:
    r: float = 1.0
    alpha: float = 1.0
    p: float = 1.0
    b: float = 1.0
    alpha_quarters: int = 0
    p_quarters: int = 0
    beta: int = 4
    L: float = 4.0

@dataclasses.dataclass
class Case:
    name: str
    groups: list
    A: float = 1.0
    D: float = 1.0
    T: float = 1.0
    h: float = 1.0
    chi: float = 1.0
    xmin: float = 2.0**-300
    xmax: float = 2.0**300
    tol: float = 1e-10
    margin: float = 1.0
    certified: bool = True
    variable: bool = False
    residual: bool = True
    width: bool = True
    max_outer: int = 512
    max_inner: int = 512
    max_bracket: int = 2048
    newton: bool = True
    root_in_domain: bool = True
    derivative_mode: int = 1
    expected: str = "accepted"
    expected_status: str = ""
    note: str = ""

    def wire(self):
        scalars = [self.name, len(self.groups), self.A, self.D, self.T,
                   self.h, self.chi, self.xmin, self.xmax, self.tol,
                   self.margin, int(self.certified), int(self.variable),
                   int(self.residual), int(self.width), self.max_outer,
                   self.max_inner, self.max_bracket, int(self.newton), self.derivative_mode, int(self.root_in_domain)]
        for g in self.groups:
            scalars += [g.r, g.alpha, g.p, g.b, g.alpha_quarters,
                        g.p_quarters, g.beta, g.L]
        return " ".join(str(v) for v in scalars)


def exact(x):
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / d


def cases():
    out = [
        Case("heating", [Group(r=20)]),
        Case("cooling", [Group(r=0.01)], T=3),
        Case("mixed_net_zero", [Group(r=2, b=1), Group(r=0.5, b=1.5)],
             note="Net gas exchange zero but radiation changes in both groups"),
        Case("mixed_net_zero_huge", [Group(r=2.0**261, b=2.0**260),
                                    Group(r=2.0**259, b=1.5*2.0**260)],
             note="Exactly dyadic zero net exchange with radiation/gas ratio >1e78"),
        Case("gas_dominated", [Group(r=2), Group(r=0.5)], A=1e80, D=1e40),
        Case("radiation_dominated", [Group(r=2e80, b=1e80), Group(r=3e80, b=2e80)], A=1e-20),
        Case("very_weak_collision", [Group(r=0.01), Group(r=100)], D=1e-80),
        Case("very_strong_collision", [Group(r=0.01), Group(r=100)], D=1e80),
        Case("optically_thin", [Group(r=2), Group(r=0.5)], h=1e-80),
        Case("width_only", [Group(r=2), Group(r=0.5)], h=1e-80, residual=False),
        Case("optically_thick", [Group(r=2), Group(r=0.5)], h=1e80),
        Case("tiny_surviving_group", [Group(r=1, alpha=1e100, p=0, b=0), Group(r=2)],
             note="Radiation reconstructed positively rather than r+exchange"),
        Case("pure_absorption", [Group(r=3, p=0, b=0), Group(r=4, alpha=2, p=0, b=0)]),
        Case("all_transparent", [Group(r=3, alpha=0, p=0, b=0), Group(r=4, alpha=0, p=0, b=0)]),
        Case("structural_zero_group", [Group(r=0, alpha=0, p=0, b=0), Group(r=4)]),
        Case("zero_radiation_input", [Group(r=0)], T=2),
        Case("odd_reduction", [Group(r=k+1, b=(k+1)/8) for k in range(3)]),
        Case("mixed_band_sensitivity", [Group(r=0.2, b=0.5, beta=1, L=1),
                                        Group(r=3, beta=2, L=2), Group(r=10, beta=4, L=4)]),
    ]
    for s in (1, 2):
        for T, r in ((0.2, 20), (4, 0.01), (1, 2)):
            out.append(Case(f"variable_{s}_quarters_T{T}", [
                Group(r=r, alpha_quarters=s, p_quarters=s, beta=1, L=1+s/2),
                Group(r=r/2, alpha=3, p=3, b=0.5,
                      alpha_quarters=s, p_quarters=s, beta=4, L=4+s/2)],
                T=T, variable=True, margin=1-s/4))
    for n in (8, 16, 64, 1024):
        out.append(Case(f"balanced_tree_{n}", [Group(r=(k % 7+1)/n, b=1/n) for k in range(n)]))
    # Power-of-two constants make constructed physical balances exact in binary64.
    for j, exponent in enumerate((-100, -30, 0, 30, 100)):
        out.append(Case(f"dyadic_hierarchy_{exponent}", [
            Group(r=2.0**(exponent+1), b=2.0**exponent),
            Group(r=2.0**(exponent-1), b=1.5*2.0**exponent)], D=2.0**(j*20-40)))
    rng = random.Random(2326)
    for i in range(28):
        variable = (i % 3 == 0)
        q = 1 if variable else 0
        ng = (1, 2, 3, 4, 8)[i % 5]
        gs = []
        for j in range(ng):
            alpha = 2.0**rng.randint(-20, 20)
            beta = (1, 2, 4)[rng.randrange(3)]
            gs.append(Group(r=2.0**rng.randint(-20, 20), alpha=alpha, p=alpha,
                            b=2.0**rng.randint(-20, 20), alpha_quarters=q,
                            p_quarters=q, beta=beta, L=beta+q/2))
        out.append(Case(f"seeded_{i:02d}", gs, A=2.0**rng.randint(-20, 20),
                        D=2.0**rng.randint(-20, 20), T=2.0**rng.randint(-8, 8),
                        h=2.0**rng.randint(-15, 15), chi=2.0**rng.randint(0, 12),
                        variable=variable, margin=0.75 if variable else 1))
    out.extend([
        Case("nan_derivative_fallback", [Group(r=20)], derivative_mode=2),
        Case("wrong_derivative_fallback", [Group(r=20)], derivative_mode=3),
        Case("bisection_only", [Group(r=20)], newton=False, derivative_mode=0),
        Case("missing_root_domain_contract", [Group()], root_in_domain=False, expected="failure"),
        Case("missing_contract", [Group()], certified=False, expected="failure"),
        Case("bad_margin", [Group()], variable=True, margin=0, expected="failure"),
        Case("negative_r", [Group(r=-1)], expected="failure"),
        Case("zero_A", [Group()], A=0, expected="failure"),
        Case("negative_D", [Group()], D=-1, expected="failure"),
        Case("subnormal_A", [Group()], A=math.ldexp(1.0, -1074), expected="failure"),
        Case("overflow_tau", [Group(alpha=1e200, p=1)], h=1e200, expected="failure"),
        Case("underflow_emission_factor", [Group(p=1e-200)], h=1e-200, expected="failure"),
        Case("root_outside_domain", [Group(r=1e10)], xmin=0.5, xmax=2, root_in_domain=False, expected="failure"),
        Case("requested_precision_unavailable", [Group(r=20)], tol=1e-30, expected="failure"),
        Case("exhausted_outer", [Group(r=20)], max_outer=1, expected="failure", expected_status="iteration_limit"),
        Case("exhausted_inner", [Group(r=20)], max_inner=1, expected="failure", expected_status="iteration_limit"),
        Case("exhausted_bracket", [Group(r=1e10)], max_bracket=1, expected="failure", expected_status="iteration_limit"),
        Case("zero_iteration_budget", [Group()], max_outer=0, expected="failure", expected_status="invalid_input"),
    ])
    return out


def solve_reference(case):
    # Precision covers >160 decades of cancellation in the hierarchy cases.
    with mp.workdps(260):
        A, D, T, h, chi = [exact(v) for v in (case.A, case.D, case.T, case.h, case.chi)]
        gs = [(exact(g.r), exact(g.alpha), exact(g.p), exact(g.b),
               mp.mpf(g.alpha_quarters)/4, mp.mpf(g.p_quarters)/4, g.beta)
              for g in case.groups]

        def dust_from_gas(t):
            return t + A*(t-T)/(D*mp.sqrt(t))

        def radiation(x):
            return [(r + h*p*x**ps*b*x**beta)/(1+h*a*x**aslope)
                    for r,a,p,b,aslope,ps,beta in gs]

        # The positive gas temperature at dust x=0 solves the scalar cubic
        # D*y^3 + A*y^2 - A*T = 0, y=sqrt(t). Bisection is unrelated to the
        # production implementation's three numerical gas charts.
        yl, yh = mp.mpf(0), mp.sqrt(T)
        for _ in range(940):
            y = (yl+yh)/2
            if D*y**3+A*y*y < A*T:
                yl = y
            else:
                yh = y
            if yh == yl or (yh-yl) <= mp.eps * max(yh, mp.mpf('1e-400')):
                break
        tl = yh*yh  # x >= 0, modulo highprecision roundoff
        th = T + chi*sum(g[0] for g in gs)/A
        if all(g[1] == 0 and g[2] == 0 for g in gs):
            t, x = T, T
        elif all(g[2] == 0 or g[3] == 0 for g in gs) and not case.variable:
            H = chi*sum(h*a*r/(1+h*a) for r,a,p,b,aslope,ps,beta in gs)
            t = T + H/A
            x = dust_from_gas(t)
        else:
            def F(t):
                x = max(mp.mpf(0), dust_from_gas(t))
                rs = radiation(x)
                return A*(t-T) + chi*sum(e-g[0] for e,g in zip(rs,gs))
            # Exact net-zero tests must not be spoiled by needless subtraction
            # of nearly equal highprecision numbers in the independent chart.
            if F(T) == 0:
                t, x = T, T
            else:
                for _ in range(940):
                    t = (tl+th)/2
                    if F(t) < 0:
                        tl = t
                    else:
                        th = t
                    if th == tl or (th-tl) <= mp.eps*max(abs(th), mp.mpf('1e-400')):
                        break
                t = (tl+th)/2
                x = dust_from_gas(t)
        rr = radiation(x)
        residual = A*(t-T) + chi*sum(e-g[0] for e,g in zip(rr,gs))
        scale = A*t + chi*sum(abs(e) for e in rr)
        if x <= 0 or t <= 0 or abs(residual) > mp.mpf('1e-100')*scale:
            raise AssertionError(f"{case.name}: highprecision reference did not resolve physical equations")
        return {"dust_temperature": +x, "gas_temperature": +t, "gas_energy": +(A*t),
                "radiation": [+v for v in rr]}


def rel_error(value, reference):
    if reference == 0:
        return mp.mpf(0) if value == 0 else mp.inf
    return abs(exact(value)/reference-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("reference_report.json"))
    parser.add_argument("--only", help="Run names containing this substring")
    args = parser.parse_args()
    selected = [c for c in cases() if args.only is None or args.only in c.name]
    before = time.monotonic()
    process = subprocess.run([str(args.executable.resolve())], input="\n".join(c.wire() for c in selected)+"\n",
                             text=True, capture_output=True, check=False)
    if process.returncode:
        raise RuntimeError(f"driver exited {process.returncode}: {process.stderr}")
    outputs = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if len(outputs) != len(selected):
        raise AssertionError(f"expected {len(selected)} results, got {len(outputs)}; stderr={process.stderr}")
    failures, report = [], []
    mp.mp.dps = 260
    for case, got in zip(selected, outputs):
        entry = {"name": case.name, "status": got["status"], "expected": case.expected, "note": case.note}
        if got.get("name") != case.name:
            failures.append(f"{case.name}: output identity mismatch")
        accepted = got.get("accepted", False)
        if case.expected == "failure":
            entry["passed"] = not accepted and (not case.expected_status or got["status"] == case.expected_status)
            if case.expected_status and got["status"] != case.expected_status:
                failures.append(f"{case.name}: expected {case.expected_status}, got {got['status']}")
            if accepted:
                failures.append(f"{case.name}: accepted invalid/uncertifiable test")
        elif not accepted:
            entry["passed"] = False
            failures.append(f"{case.name}: unexpected {got['status']}")
        else:
            ref = solve_reference(case)
            if not exact(case.xmin) <= ref["dust_temperature"] <= exact(case.xmax):
                raise AssertionError(f"{case.name}: test contract falsely declared root domain")
            err = {k: rel_error(got[k], ref[k]) for k in ("dust_temperature", "gas_temperature", "gas_energy")}
            group_errors = [rel_error(v, r) for v,r in zip(got["radiation"], ref["radiation"])]
            cert = got["certificate"]
            checks = [(err["dust_temperature"], exact(cert["dust_relative"])),
                      (err["gas_energy"], exact(cert["gas_relative"]))]
            checks += [(e, exact(b)) for e,b in zip(group_errors, cert["group_relative"])]
            passed = all(e <= b for e,b in checks) and all(e <= exact(case.tol) for e,_ in checks)
            passed = passed and cert["conditional"] and got["status"] == "accepted_conditional"
            passed = passed and len(got["radiation"]) == len(case.groups)
            passed = passed and len(cert["group_relative"]) == len(case.groups)
            passed = passed and all(v >= 0 and math.isfinite(v) for v in got["radiation"])
            passed = passed and got["dust_temperature"] > 0 and got["gas_energy"] > 0
            if case.name.startswith("mixed_net_zero"):
                passed = passed and all(got["radiation"][i] != case.groups[i].r for i in range(2))
            entry.update({"passed": passed, "stop": got.get("stop"),
                          "relative_errors": {k: mp.nstr(v, 14) for k,v in err.items()},
                          "max_group_relative_error": mp.nstr(max(group_errors, default=mp.mpf(0)), 14),
                          "max_error_to_reported_bound": mp.nstr(max((e/b if b else e for e,b in checks), default=mp.mpf(0)), 14),
                          "certificate": cert})
            if not passed:
                failures.append(f"{case.name}: actual error exceeds reported bound/requested tolerance or structural check failed")
        report.append(entry)
    summary = {"test_only_precision_decimal_digits": 260, "cases": len(selected),
               "passed": len(selected)-len(failures), "failures": failures,
               "elapsed_seconds": round(time.monotonic()-before, 3),
               "scope": "Synthetic analytic exact-stored-input oracle; no claim about physical Planck CDF or production Quokka",
               "results": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "results"}, indent=2))
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
