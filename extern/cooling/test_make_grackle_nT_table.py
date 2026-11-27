"""Tests for make_grackle_nT_table.py. Run: python3 -m pytest test_make_grackle_nT_table.py -v

Set GRACKLE_SHIELDED_FILE to a local CloudyData_UVB=HM2012_shielded.h5 to skip the download.
"""
import os

import h5py
import numpy as np
import pytest

import make_grackle_nT_table as mk
from grackle_tables import X, m_H, read_tables


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    local = os.environ.get("GRACKLE_SHIELDED_FILE")
    if local:
        return local
    from resample_grackle_cooling_tables import DEFAULT_GRACKLE_DATA_URL_SHIELDED, download_default_grackle_tables
    path = str(tmp_path_factory.mktemp("grackle") / "CloudyData_UVB=HM2012_shielded.h5")
    return download_default_grackle_tables(DEFAULT_GRACKLE_DATA_URL_SHIELDED, local=path)


@pytest.fixture(scope="module")
def tables(source):
    return read_tables(source)


def generate(source, tmp_path_factory, zmet):
    out = str(tmp_path_factory.mktemp("out") / f"table_Z{zmet}.h5")
    mk.make_table(source, zmet, out)
    return out


@pytest.fixture(scope="module")
def table1(source, tmp_path_factory):
    return generate(source, tmp_path_factory, 1.0)


@pytest.fixture(scope="module")
def table_half(source, tmp_path_factory):
    return generate(source, tmp_path_factory, 0.5)


def read(path):
    with h5py.File(path, "r") as f:
        g = f["tab1"]
        return dict(g.attrs), g["data"][:]


def grid_index(tables, log_nH, log_T):
    return np.argmin(abs(tables.log_nH - log_nH)), np.argmin(abs(tables.log_T - log_T))


def test_attributes(table1):
    attrs, data = read(table1)
    assert attrs["zmet"] == 1.0
    assert 0.7 < attrs["H_mass_fraction"] < 0.8
    assert list(attrs["spacing"]) == [b"log", b"log"]
    assert list(attrs["input_names"]) == [b"nH", b"T"]
    assert list(attrs["output_names"]) == [b"Lambda_net", b"xe_CIE"]
    assert attrs["Ndim"] == 2 and attrs["Nout"] == 2
    assert data.shape == (2, *attrs["Nx"])


def test_lambda_matches_source(table1, tables):
    attrs, data = read(table1)
    assert np.allclose(attrs["xlo"], [10 ** tables.log_nH[0], 10 ** tables.log_T[0]], rtol=1e-12)
    assert np.allclose(attrs["xhi"], [10 ** tables.log_nH[-1], 10 ** tables.log_T[-1]], rtol=1e-12)
    for i, j in [(5, 20), (14, 80), (25, 140)]:
        lam_prim = (10 ** tables.primCooling[i, j] - 10 ** tables.primHeating[i, j]) * m_H**2
        lam_metal = (10 ** tables.metalCooling[i, j] - 10 ** tables.metalHeating[i, j]) * m_H**2
        np.testing.assert_allclose(data[0, i, j], lam_prim + lam_metal, rtol=1e-12)


def test_xe_fully_ionized(table1, tables):
    _, data = read(table1)
    i, j = grid_index(tables, 0.0, 7.0)
    y = (1.0 - X) / (4.0 * X)
    # Grackle's MMW includes metals, which puts the plateau about 1.3% below 1 + 2y; 2% covers that offset.
    np.testing.assert_allclose(data[1, i, j], 1.0 + 2.0 * y, rtol=2e-2)


def test_xe_cold_small(table1, tables):
    _, data = read(table1)
    i, j = grid_index(tables, 2.0, 2.0)
    assert data[1, i, j] < 1e-2


def test_zmet_scaling(table1, table_half, tables):
    _, d1 = read(table1)
    _, dh = read(table_half)
    lam_prim = (10 ** tables.primCooling - 10 ** tables.primHeating) * m_H**2
    np.testing.assert_allclose(dh[0] - lam_prim, 0.5 * (d1[0] - lam_prim), rtol=1e-12, atol=1e-30)
