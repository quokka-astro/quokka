#!/usr/bin/env python3
"""Reference values for the ISM cooling and heating rates in src/cooling/ISMCooling.hpp.

The formulas are written once with sympy, and the partial derivatives are taken symbolically,
so this file shares no code (and no hand-derived derivative) with the C++ implementation.
Run it to regenerate ism_cooling_reference.csv:

    python3 ism_cooling_reference.py [ism_output.csv [grackle_output.csv]]

It also writes grackle_nT_reference.csv: Lambda_net and xe_CIE of CloudyData_UVB=HM2012_shielded_nT_Z1.h5 at off-grid points,
by bilinear interpolation in (log nH, log T), with the analytic derivatives of that interpolant (the C++ lookup is checked against them).

Requires numpy, sympy and h5py (e.g. ~/softwares/python-envs/base/bin/python).

All rates are volumetric (erg cm^-3 s^-1). The independent variables are (T, ne, nHI, nHp), with
n_H = nHI + nHp. The C II and O I rates are ported from Gong's Athena++ code (src/chemistry/utils/thermo.cpp:
q10CII_, CoolingCII, CoolingOI, Cooling2Level_, Cooling3Level_), BSD-3, with n_H2 = 0. Athena returns
erg s^-1 per H with abundances per H, so the result is multiplied by n_H here.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import sympy as sp

# ---------------------------------------------------------------- constants (cgs)
K_B = 1.380649e-16  # erg/K
EV = 1.602176634e-12  # erg
T_PE_MAX = 1.0e4  # K, upper limit of the PE fit
X_C_TOT = 1.6e-4  # per H, times Z'
X_O_TOT = 3.2e-4
ZD_OVER_001 = 0.68  # Z_d/0.01 = 0.5 * 0.0136 * Z' / 0.01 = 0.68 Z'
E_CR = 1.6022e-11  # erg per CR ionization
A_NEB = (0.692, -0.586, 0.816, -0.505, 0.118, 7.66e-3, -5.08e-3)
D_PE = (0.4535, 2.234, -6.266, 1.442, 0.05089)
# C+ atomic data (Athena++)
A10_CII, E10_CII, G0_CII, G1_CII = 2.3e-6, 1.26e-14, 2.0, 4.0
# O I atomic data (Athena++)
G_OI = (5.0, 3.0, 1.0)
A10_OI, A20_OI, A21_OI = 8.910e-05, 1.340e-10, 1.750e-05
E10_OI, E20_OI, E21_OI = 3.144e-14, 4.509e-14, 1.365e-14

# ---------------------------------------------------------------- symbolic formulas
T, ne, nHI, nHp, Zp, G0 = sp.symbols("T ne nHI nHp Zp G0", positive=True)
zeta, yHe = sp.symbols("zeta yHe", positive=True)
VARS = (T, ne, nHI, nHp)
nH = nHI + nHp


def lya():
    T4 = T / 1e4
    k12 = 5.31e-8 * T4**0.15 / (1 + (T4 / 5) ** 0.65) * sp.exp(-11.84 / T4)
    return 10.2 * EV * nHI * ne * k12


def nebular():
    T4 = T / 1e4
    lnT4 = sp.log(T4)
    f = 10 ** sum(a * lnT4**i for i, a in enumerate(A_NEB))
    ne2 = ne / 100
    return Zp * ne * nHp * 3.68e-23 * sp.exp(-3.86 / T4) * f / (sp.sqrt(T4) * (1 + 0.12 * ne2 ** (0.38 - 0.12 * lnT4)))


def cooling_2level(q01, q10, A10, E10, xs):
    return q01 / (q01 + q10 + A10) * A10 * E10 * xs


def cooling_3level(q01, q10, q02, q20, q12, q21, A10, A20, A21, E10, E20, E21, xs):
    R10, R20, R21 = q10 + A10, q20 + A20, q21 + A21
    a0 = R10 * R20 + R10 * R21 + q12 * R20
    a1 = q01 * R20 + q01 * R21 + R21 * q02
    a2 = q02 * R10 + q02 * q12 + q12 * q01
    de = a0 + a1 + a2
    return (a1 / de * A10 * E10 + a2 / de * (A20 * E20 + A21 * E21)) * xs


def cii():
    T2 = T / 100
    k10e = 4.53e-8 * sp.sqrt(1e4 / T)
    k10HI = 7.58e-10 * T2 ** (0.1281 + 0.0087 * sp.log(T2))
    q10 = k10e * ne + k10HI * nHI
    q01 = (G1_CII / G0_CII) * q10 * sp.exp(-E10_CII / (K_B * T))
    return nH * cooling_2level(q01, q10, A10_CII, E10_CII, X_C_TOT * Zp)


def oi():
    T2 = T / 100
    lnT2 = sp.log(T2)
    k10HI = 3.57e-10 * T2 ** (0.419 - 0.003 * lnT2)
    k20HI = 3.19e-10 * T2 ** (0.369 - 0.006 * lnT2)
    k21HI = 4.34e-10 * T2 ** (0.755 - 0.160 * lnT2)
    k10e = 5.12e-10 * T**-0.075
    k20e = 4.86e-10 * T**-0.026
    k21e = 1.08e-14 * T**0.926
    q10, q20, q21 = k10HI * nHI + k10e * ne, k20HI * nHI + k20e * ne, k21HI * nHI + k21e * ne
    q01 = (G_OI[1] / G_OI[0]) * q10 * sp.exp(-E10_OI / (K_B * T))
    q02 = (G_OI[2] / G_OI[0]) * q20 * sp.exp(-E20_OI / (K_B * T))
    q12 = (G_OI[2] / G_OI[1]) * q21 * sp.exp(-E21_OI / (K_B * T))
    n_O = X_O_TOT * Zp * nHI  # neutral O follows HI
    return cooling_3level(q01, q10, q02, q20, q12, q21, A10_OI, A20_OI, A21_OI, E10_OI, E20_OI, E21_OI, n_O)


def grain_recombination():
    x = G0 * sp.sqrt(T) / ne + 50
    lnx = sp.log(x)
    return 1e-28 * Zp * ne * nH * T ** (D_PE[0] + D_PE[1] / lnx) * sp.exp(D_PE[2] + (D_PE[3] - D_PE[4] * lnx) * lnx)


def pe_heating(Ttilde):
    psi = G0 * sp.sqrt(Ttilde) / (0.5 * ne)
    eps = 0.0487 / (1 + 0.004 * psi**0.73) + 0.0365 * (Ttilde / 1e4) ** 0.7 / (1 + 2e-4 * psi)
    return 1.3e-24 * ZD_OVER_001 * Zp * eps * G0 * nH


def cr_heating():
    return E_CR * (zeta * nHI + 1.1 * zeta * yHe * nHI)


# (name, expression(s), lambdified function of (T, ne, nHI, nHp, Zp, G0, zeta, yHe))
ARGS = (T, ne, nHI, nHp, Zp, G0, zeta, yHe)


def compile_rate(expr):
    exprs = [expr] + [sp.diff(expr, v) for v in VARS]
    f = sp.lambdify(ARGS, exprs, modules="numpy")
    return lambda *a: np.array([float(np.asarray(e)) for e in f(*[np.float64(x) for x in a])])


RATES = {
    "Lya": compile_rate(lya()),
    "Nebular": compile_rate(nebular()),
    "CII": compile_rate(cii()),
    "OI": compile_rate(oi()),
    "GrainRecombination": compile_rate(grain_recombination()),
    "PE_lowT": compile_rate(pe_heating(T)),
    "PE_highT": compile_rate(pe_heating(sp.Float(T_PE_MAX))),
    "CR": compile_rate(cr_heating()),
}


def rate(name, Tv, nev, nHIv, nHpv, Zpv=1.0, G0v=0.0, zetav=0.0, yHev=0.0):
    """Return [value, dT, dne, dnHI, dnHp]."""
    if name == "PE":
        if G0v == 0.0:  # identically zero in all four variables
            return np.zeros(5)
        name = "PE_lowT" if Tv < T_PE_MAX else "PE_highT"
    with np.errstate(all="ignore"):
        return RATES[name](Tv, nev, nHIv, nHpv, Zpv, G0v, zetav, yHev)


# ---------------------------------------------------------------- electron density (SPEC 5.4)
def xe_cie(Tv):
    """Smooth stand-in for the table value, with its exact T derivative."""
    s = np.tanh(np.log(Tv / 1e4))
    return 0.5 * (1 + s), 0.5 * (1 - s * s) / Tv


def alpha_b(Tv):
    a = 2.6e-13 * (Tv / 1e4) ** -0.76
    return a, -0.76 * a / Tv


def electron_density_max(nHIv, nHpv, Tv, xecie, dxecie_dT, dxecie_dnH, Zpv, zetav, alphab, dalphab_dT):
    """Return [ne, dT, dnHI, dnHp, branch]. Ties go to the lower branch number."""
    nHv = nHIv + nHpv
    cr = np.sqrt(zetav * nHv / alphab)
    cands = [nHpv, xecie * nHv, X_C_TOT * Zpv * nHv, cr]
    b = 0
    for i in range(1, 4):
        if cands[i] > cands[b]:
            b = i
    if b == 0:
        d = (0.0, 0.0, 1.0)
    elif b == 1:
        d = (dxecie_dT * nHv, xecie + nHv * dxecie_dnH, xecie + nHv * dxecie_dnH)
    elif b == 2:
        d = (0.0, X_C_TOT * Zpv, X_C_TOT * Zpv)
    else:
        d = (-cr / (2 * alphab) * dalphab_dT, cr / (2 * nHv), cr / (2 * nHv))
    return [cands[b], d[0], d[1], d[2], b]


# ---------------------------------------------------------------- grid
NH_GRID = [10.0**k for k in range(-3, 4)]
T_GRID = [10, 30, 100, 300, 1e3, 3e3, 8e3, 1.5e4, 3e4, 1e6, 1e9]
XHP_GRID = [1e-4, 0.5, 1 - 1e-4]
G0_GRID = [0.0, 1.0, 100.0]
Z_GRID = [0.1, 1.0]
ZETA_GRID = [1.78e-18, 1e-15]
YHE_GRID = [0.0, 0.1]

COLUMNS = ["func", "nHI", "nHp", "ne", "T", "G0", "Zp", "zeta", "yHe", "xeCIE", "dxeCIE_dT", "dxeCIE_dnH", "alphaB", "dalphaB_dT", "val", "dT", "dne", "dnHI", "dnHp", "branch"]
USES = {  # which of (Zp, G0) each function depends on
    "Lya": (False, False),
    "Nebular": (True, False),
    "CII": (True, False),
    "OI": (True, False),
    "GrainRecombination": (True, True),
    "PEHeating": (True, True),
}


def rows():
    out = []

    def add(func, nHIv, nHpv, nev, Tv, G0v=0.0, Zpv=0.0, zetav=0.0, yHev=0.0, extra=(0.0, 0.0, 0.0, 0.0, 0.0), res=(0, 0, 0, 0, 0), branch=0):
        out.append([func, nHIv, nHpv, nev, Tv, G0v, Zpv, zetav, yHev, *extra, *res, branch])

    for nHv in NH_GRID:
        for xp in XHP_GRID:
            nHpv, nHIv = xp * nHv, (1 - xp) * nHv
            nev = nHpv
            for Tv in T_GRID:
                for func, (useZ, useG) in USES.items():
                    for Zpv in Z_GRID if useZ else [1.0]:
                        for G0v in G0_GRID if useG else [0.0]:
                            key = "PE" if func == "PEHeating" else func
                            r = rate(key, Tv, nev, nHIv, nHpv, Zpv, G0v)
                            add(func, nHIv, nHpv, nev, Tv, G0v, Zpv, res=r)
            # edge block: tiny electron density, with and without radiation
            for Tv in T_GRID if (xp == 0.5 and nHv in (1e-3, 1.0, 1e3)) else []:
                for G0v in (0.0, 100.0):
                    for func in ("GrainRecombination", "PEHeating"):
                        r = rate("PE" if func == "PEHeating" else func, Tv, 1e-30, nHIv, nHpv, 1.0, G0v)
                        add(func, nHIv, nHpv, 1e-30, Tv, G0v, 1.0, res=r)
                    for func in ("Lya", "Nebular", "CII", "OI"):
                        r = rate(func, Tv, 1e-30, nHIv, nHpv, 1.0)
                        add(func, nHIv, nHpv, 1e-30, Tv, 0.0, 1.0, res=r)
            for zv in ZETA_GRID:
                for yv in YHE_GRID:
                    r = rate("CR", 100.0, nHpv, nHIv, nHpv, 1.0, 0.0, zv, yv)
                    add("CRHeating", nHIv, nHpv, nHpv, 100.0, 0.0, 1.0, zv, yv, res=r)
            for Tv in T_GRID:
                xc, dxc = xe_cie(Tv)
                dxc_dnH = -0.05 * xc / nHv  # arbitrary nonzero value, so that the n_H partial of the CIE branch is exercised
                ab, dab = alpha_b(Tv)
                for Zpv in Z_GRID:
                    for zv in ZETA_GRID:
                        e = electron_density_max(nHIv, nHpv, Tv, xc, dxc, dxc_dnH, Zpv, zv, ab, dab)
                        add("ElectronDensityMax", nHIv, nHpv, 0.0, Tv, 0.0, Zpv, zv, 0.0, (xc, dxc, dxc_dnH, ab, dab), (e[0], e[1], 0.0, e[2], e[3]), e[4])
    return out


# ---------------------------------------------------------------- Grackle (nH, T) table lookup
GRACKLE_TABLE = Path(__file__).with_name("CloudyData_UVB=HM2012_shielded_nT_Z1.h5")
GRACKLE_COLUMNS = ["nH", "T", "Lambda_net", "xe_CIE", "dLambda_dT", "dxe_dT", "dxe_dnH"]
# (i, j, fx, fy): the point lies at fractional grid index (i + fx, j + fy) in (log nH, log T). The fractions stay
# away from 0 and 1, so the centred difference of the C++ lookup (step 1e-3 in ln T or ln nH) stays inside one cell and is exact.
GRACKLE_POINTS = [(0, 0, 0.3, 0.4), (2, 20, 0.5, 0.5), (5, 45, 0.25, 0.75), (9, 70, 0.6, 0.3), (13, 92, 0.4, 0.6),
                  (16, 110, 0.7, 0.2), (19, 125, 0.35, 0.55), (22, 140, 0.5, 0.45), (25, 150, 0.65, 0.7), (27, 159, 0.8, 0.8)]


def grackle_bilinear(tab, nHv, Tv):
    """Bilinear interpolation in (log nH, log T) of both outputs; returns value and d/dT, d/dnH (physical) for each."""
    attrs = tab.attrs
    xlo, xhi, nx = np.log(attrs["xlo"]), np.log(attrs["xhi"]), attrs["Nx"]
    data = tab["data"][:]
    u = (np.log(nHv) - xlo[0]) / (xhi[0] - xlo[0]) * (nx[0] - 1)
    v = (np.log(Tv) - xlo[1]) / (xhi[1] - xlo[1]) * (nx[1] - 1)
    i, j = int(np.floor(u)), int(np.floor(v))
    fx, fy = u - i, v - j
    out = []
    for k in range(2):
        z = data[k]
        f = (1 - fx) * (1 - fy) * z[i, j] + fx * (1 - fy) * z[i + 1, j] + (1 - fx) * fy * z[i, j + 1] + fx * fy * z[i + 1, j + 1]
        dfx = (1 - fy) * (z[i + 1, j] - z[i, j]) + fy * (z[i + 1, j + 1] - z[i, j + 1])
        dfy = (1 - fx) * (z[i, j + 1] - z[i, j]) + fx * (z[i + 1, j + 1] - z[i + 1, j])
        dlnnH = dfx * (nx[0] - 1) / (xhi[0] - xlo[0])
        dlnT = dfy * (nx[1] - 1) / (xhi[1] - xlo[1])
        out.append((f, dlnT / Tv, dlnnH / nHv))
    return out


def grackle_rows():
    import h5py

    out = []
    with h5py.File(GRACKLE_TABLE, "r") as fh:
        tab = fh["tab1"]
        xlo, xhi, nx = np.log(tab.attrs["xlo"]), np.log(tab.attrs["xhi"]), tab.attrs["Nx"]
        for i, j, fx, fy in GRACKLE_POINTS:
            nHv = float(np.exp(xlo[0] + (i + fx) / (nx[0] - 1) * (xhi[0] - xlo[0])))
            Tv = float(np.exp(xlo[1] + (j + fy) / (nx[1] - 1) * (xhi[1] - xlo[1])))
            (lam, dlam_dT, _), (xe, dxe_dT, dxe_dnH) = grackle_bilinear(tab, nHv, Tv)
            assert xe > 0.0, f"reference point ({nHv}, {Tv}) has non-positive xe_CIE"
            out.append([nHv, Tv, lam, xe, dlam_dT, dxe_dT, dxe_dnH])
    return out


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("ism_cooling_reference.csv")
    grackle_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("grackle_nT_reference.csv")
    with open(grackle_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(GRACKLE_COLUMNS)
        grackle = grackle_rows()
        for r in grackle:
            w.writerow([repr(float(v)) for v in r])
    print(f"wrote {len(grackle)} rows to {grackle_path}")
    data = rows()
    branches = {r[-1] for r in data if r[0] == "ElectronDensityMax"}
    assert branches == {0, 1, 2, 3}, f"ElectronDensityMax grid misses branches: {branches}"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in data:
            w.writerow([r[0]] + [repr(float(v)) for v in r[1:-1]] + [int(r[-1])])
    print(f"wrote {len(data)} rows to {path}")


if __name__ == "__main__":
    main()
