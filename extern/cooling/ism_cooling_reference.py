#!/usr/bin/env python3
"""Reference values for the ISM cooling and heating rates in src/cooling/ISMCooling.hpp.

The formulas are written once with sympy, and the partial derivatives are taken symbolically,
so this file shares no code (and no hand-derived derivative) with the C++ implementation.
Run it to regenerate ism_cooling_reference.csv:

    python3 ism_cooling_reference.py [ism_output.csv [grackle_output.csv [equilibrium_output.csv [hot_cell_output.csv]]]]

It also writes ism_equilibrium_reference.csv, T(n_H) after 1 Gyr of the ISMCoolingOneZone network, integrated from T = 1e3 K (see equilibrium_rows),
ism_hot_cell_reference.csv, the internal energy of a hot cell after one hydro step of the hot-cell cooling (see hot_cell_reference),
and grackle_nT_reference.csv: Lambda_net and xe_CIE of CloudyData_UVB=HM2012_shielded_nT_Z1.h5 at off-grid points,
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


# ---------------------------------------------------------------- one-zone thermal equilibrium (ISMCoolingOneZone, mode "equilibrium")
# The network of src/networks/thermophotochemistry_ism_handwritten/actual_rhs.H without ionizing photons. The H rates are copied from actual_rhs.H
# (the same fits and constants), because the network's own H terms are not among the rates above.
#
# The reference is the solution of the network ODE for (x_H+, T) from the initial state of the C++ run to its stop_time, integrated with a stiff
# solver. It is not the steady state: with no photoionization, H+ recombines as x ~ 1/(alpha n t), so it has not reached its equilibrium in
# cold or dilute gas at 1 Gyr. The steady state is kept (equilibrium_temperature) for information only. The pure-H EOS of the photochemistry
# network (extern/Microphysics/EOS/photoionization) has e_vol = 3/2 k T (n_e + n_HI + n_H+) = 3/2 k T n_H (1 + x) with n_e = n_H+.
RYDBERG = 13.6 * EV  # erg
EQ_T0, EQ_X0 = 1.0e3, 1.0e-3  # initial state of the C++ run
EQ_T_END = 3.15576e16  # s, the stop_time of inputs/ISMCoolingEquilibrium.toml (1 Gyr)
EQ_G0, EQ_ZP, EQ_ZETA = 1.0, 1.0, 1.78e-18
EQ_NH = [10.0 ** (-2.0 + 4.0 * i / 24.0) for i in range(25)]  # 25 log-spaced values in [1e-2, 1e2]
EQ_T_SCAN = np.exp(np.linspace(np.log(10.0), np.log(1.0e5), 801))
EQUILIBRIUM_COLUMNS = ["nH", "T_end", "x_Hp_end", "ne_max", "branch", "T_steady", "x_Hp_steady", "n_roots"]


def alpha_rec(Tv):
    """Case-B recombination coefficient with recombination_temperature_dependent = 1 (actual_rhs.H), and its T derivative."""
    a = 2.63e-13 * (Tv / 1.0e4) ** -0.7
    return a, -0.7 * a / Tv


def k_coll_ion(Tv):
    """Collisional ionization coefficient (Krumholz et al. 2007, actual_rhs.H)."""
    return 5.84e-11 * np.sqrt(Tv) * np.exp(-2.18e-11 / (K_B * Tv))


def recombination_cooling_coeff(Tv):
    """Recombination cooling coefficient (Krumholz et al. 2007, actual_rhs.H): zero below 100 K."""
    return 0.0 if Tv < 100.0 else 6.1e-10 * K_B * Tv * Tv**-0.89


def x_hp_equilibrium(Tv):
    """d x_H+/dt = 0 with n_e = n_H+ (no photons, no CR ionization in the network): (1 - x) k_ci = alpha x."""
    k = k_coll_ion(Tv)
    return k / (k + alpha_rec(Tv)[0])


def grackle_clamped(tab, nHv, Tv):
    """(Lambda_net, xe_CIE) of the table at (nH, T), bilinear in (ln nH, ln T), clamped to the table range, and xe to >= 0, as the C++ lookup is."""
    xlo, xhi = np.log(tab.attrs["xlo"]), np.log(tab.attrs["xhi"])
    nHc = float(np.exp(min(max(np.log(nHv), xlo[0]), xhi[0] - 1e-9)))
    Tc = float(np.exp(min(max(np.log(Tv), xlo[1]), xhi[1] - 1e-9)))
    (lam, _, _), (xe, _, _) = grackle_bilinear(tab, nHc, Tc)
    return lam, max(xe, 0.0)


def grackle_xe(tab, nHv, Tv):
    """xe_CIE of the table at (nH, T), clamped as the C++ lookup is."""
    return grackle_clamped(tab, nHv, Tv)[1]


def net_heating(tab, nHv, Tv, xhp):
    """Net energy rate [erg cm^-3 s^-1] of the network at (n_H, T, x_H+) with no photons, with n_e^max and its branch."""
    nHpv, nHIv = xhp * nHv, (1.0 - xhp) * nHv
    ne = nHpv  # the network electrons
    # H terms, on the network electrons
    net = -recombination_cooling_coeff(Tv) * ne * nHpv - 1.3 * 1.427e-27 * np.sqrt(Tv) * ne * nHpv - RYDBERG * ne * nHIv * k_coll_ion(Tv)
    # ISM terms, on n_e^max; the 1e-30 floor is that of get_ism_energy_terms
    ab, dab = alpha_rec(Tv)
    emax = electron_density_max(nHIv, nHpv, Tv, grackle_xe(tab, nHv, Tv), 0.0, 0.0, EQ_ZP, EQ_ZETA, ab, dab)
    nem, branch = max(emax[0], 1e-30), emax[4]
    for name, sign in (("Lya", -1), ("Nebular", -1), ("CII", -1), ("OI", -1), ("GrainRecombination", -1), ("PE", 1), ("CR", 1)):
        net += sign * rate(name, Tv, nem, nHIv, nHpv, EQ_ZP, EQ_G0, EQ_ZETA, 0.0)[0]
    return net, nem, branch


def equilibrium_temperature(tab, nHv):
    """The equilibrium reached from T0 = 1e3 K, with H+ in equilibrium at each T.

    Returns (T_eq, x_H+, n_e^max, branch, number of roots). The net rate f(T) is scanned on a log grid. The gas heats where f > 0 and cools where
    f < 0, so from T0 it stops at the first stable root (f goes from + to - with T) on its side of T0.
    """
    from scipy.optimize import brentq

    def f_of(Tv):
        return net_heating(tab, nHv, Tv, x_hp_equilibrium(Tv))[0]

    f = np.array([f_of(Tv) for Tv in EQ_T_SCAN])
    roots = [i for i in range(len(f) - 1) if f[i] * f[i + 1] < 0.0]
    stable = [i for i in roots if f[i] > 0.0]
    i0 = int(np.searchsorted(EQ_T_SCAN, EQ_T0))
    if f_of(EQ_T0) > 0.0:  # heating: the first stable root above T0
        i = min(i for i in stable if i >= i0)
    else:  # cooling: the first stable root below T0
        i = max(i for i in stable if i < i0)
    Teq = brentq(f_of, EQ_T_SCAN[i], EQ_T_SCAN[i + 1], xtol=1e-12, rtol=1e-14)
    xhp = x_hp_equilibrium(Teq)
    _, nem, branch = net_heating(tab, nHv, Teq, xhp)
    return Teq, xhp, nem, branch, len(roots)


def integrate_cell(tab, nHv):
    """Integrate the network ODE of one cell from (T0, x0) to EQ_T_END. Returns (T, x_H+, n_e^max, branch) at the end.

    Unknowns are x = n_H+ / n_H and tau = T (1 + x) = 2 e_vol / (3 k n_H), so that
        dx/dt   = n_H x [(1 - x) k_ci(T) - alpha(T) x]        (n_e = n_H+; no photons, no CR ionization in the network)
        dtau/dt = net_heating(n_H, T, x) / (3/2 k n_H).
    """
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        x, tau = max(y[0], 0.0), y[1]
        Tv = tau / (1.0 + x)
        dx = nHv * x * ((1.0 - x) * k_coll_ion(Tv) - alpha_rec(Tv)[0] * x)
        return [dx, net_heating(tab, nHv, Tv, x)[0] / (1.5 * K_B * nHv)]

    sol = solve_ivp(rhs, (0.0, EQ_T_END), [EQ_X0, EQ_T0 * (1.0 + EQ_X0)], method="Radau", rtol=1e-10, atol=[1e-30, 1e-8])
    assert sol.success, sol.message
    x, tau = max(sol.y[0, -1], 0.0), sol.y[1, -1]
    Tv = tau / (1.0 + x)
    _, nem, branch = net_heating(tab, nHv, Tv, x)
    return Tv, x, nem, branch


def equilibrium_rows():
    """One row per cell: n_H, the ODE solution at EQ_T_END (T, x_H+, n_e^max, branch), and the steady state (T, x_H+, number of roots)."""
    import h5py

    with h5py.File(GRACKLE_TABLE, "r") as fh:
        tab = fh["tab1"]
        rows_out = []
        for nHv in EQ_NH:
            Tv, x, nem, branch = integrate_cell(tab, nHv)
            Teq, xeq, _, _, n_roots = equilibrium_temperature(tab, nHv)
            rows_out.append([nHv, Tv, x, nem, branch, Teq, xeq, n_roots])
        return rows_out


# ---------------------------------------------------------------- hot cell (ISMCoolingOneZone, mode "hot_cell")
# The hot-cell step of src/radiation/photochemistry_hot_gas.hpp: the cell is set fully ionized at fixed internal energy, then
#     dE/dt = -n_H^2 Lambda_net(n_H, T) + G_PE + G_CR
# is integrated over one hydro step at fixed density. With the pure-H EOS (e_vol = 3/2 k T (n_e + n_HI + n_H+)) and n_e = n_H+ = n_H, T = E / (3 k n_H)
# (mu = 0.5). G_PE uses n_e^max with n_HI = 0, n_H+ = n_H; G_CR is zero because n_HI = 0. The initial state and the step must match
# inputs/ISMCoolingHotCell.toml.
HOT_NH, HOT_T0, HOT_X0 = 1.0e-2, 1.0e6, 0.5  # initial state of the C++ run (T0 at x_H+ = HOT_X0, before the cell is ionized)
HOT_DT = 1.0e13  # s, constant_dt of inputs/ISMCoolingHotCell.toml
HOT_G0, HOT_ZP, HOT_ZETA = 1.0, 1.0, 1.78e-18
HOT_CELL_COLUMNS = ["nH", "T0", "x_Hp0", "G0", "dt", "Eint0", "Eint_end", "T_end"]


def hot_cell_rate(tab, nHv, Tv, G0v, Zpv, zetav):
    """dE/dt [erg cm^-3 s^-1] of a fully ionized hot cell."""
    lam, xe = grackle_clamped(tab, nHv, Tv)
    ab, dab = alpha_rec(Tv)
    nem = max(electron_density_max(0.0, nHv, Tv, xe, 0.0, 0.0, Zpv, zetav, ab, dab)[0], 1e-30)
    pe = rate("PE", Tv, nem, 0.0, nHv, Zpv, G0v, zetav, 0.0)[0]
    cr = rate("CR", Tv, nem, 0.0, nHv, Zpv, G0v, zetav, 0.0)[0]
    return -nHv * nHv * lam + pe + cr


def hot_cell_reference(tab, nHv=HOT_NH, T0=HOT_T0, x0=HOT_X0, dt=HOT_DT, G0v=HOT_G0, Zpv=HOT_ZP, zetav=HOT_ZETA):
    """Integrate the hot-cell step with solve_ivp (Radau, rtol 1e-10). Returns [nH, T0, x0, G0, dt, E0, E_end, T_end]."""
    from scipy.integrate import solve_ivp

    E0 = 1.5 * K_B * T0 * nHv * (1.0 + x0)  # the internal energy, kept when the cell is ionized

    def rhs(t, y):
        return [hot_cell_rate(tab, nHv, y[0] / (3.0 * K_B * nHv), G0v, Zpv, zetav)]

    sol = solve_ivp(rhs, (0.0, dt), [E0], method="Radau", rtol=1e-10, atol=1e-10 * E0)
    assert sol.success, sol.message
    E_end = sol.y[0, -1]
    return [nHv, T0, x0, G0v, dt, E0, E_end, E_end / (3.0 * K_B * nHv)]


def hot_cell_rows():
    import h5py

    with h5py.File(GRACKLE_TABLE, "r") as fh:
        return [hot_cell_reference(fh["tab1"])]


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
    equilibrium_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).with_name("ism_equilibrium_reference.csv")
    with open(equilibrium_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(EQUILIBRIUM_COLUMNS)
        equilibrium = equilibrium_rows()
        for r in equilibrium:
            w.writerow([repr(float(v)) for v in r[:4]] + [int(r[4])] + [repr(float(v)) for v in r[5:7]] + [int(r[7])])
    print(f"wrote {len(equilibrium)} rows to {equilibrium_path}")
    hot_path = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).with_name("ism_hot_cell_reference.csv")
    with open(hot_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HOT_CELL_COLUMNS)
        hot = hot_cell_rows()
        for r in hot:
            w.writerow([repr(float(v)) for v in r])
    print(f"wrote {len(hot)} rows to {hot_path}")
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
