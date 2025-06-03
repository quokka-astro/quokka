import h5py
import numpy as np
from scipy.interpolate import RectBivariateSpline

c_light_cgs_ = 2.99792458e10  # cgs
radiation_constant_cgs_ = 7.5646e-15  # cgs
boltzmann_constant_cgs_ = 1.380658e-16  # cgs
m_H = 1.672623e-24  # g

#nHe_over_nH = 0.0851 # "abundances GASS10" in Cloudy
nHe_over_nH = 0.098 # "abundances ism" in Cloudy
cloudy_H_mass_fraction = 1. / (1. + nHe_over_nH * 3.971)
X = cloudy_H_mass_fraction


class cloudyTables:
    cooling = []
    heating = []
    mmw = []
    log_nH = []
    log_T = []


def read_tables(filename):
    """"read Cloudy tables in HDF5 format."""
    f = h5py.File(filename, 'r')

    tables = cloudyTables()
    tables.log_nH = f['Parameter1']
    tables.log_T = np.log10(f['Temperature'])

    log10_or_small = lambda table: np.piecewise(table,
                                                (table > 0., table <= 0.),
                                                (np.log10, lambda x: np.NaN))
    number_or_nan = lambda table: np.piecewise(table,
        (table > 0., table <= 0.),
        (lambda x: x, lambda x: np.NaN))

    tables.cooling = log10_or_small(f['Cooling'][:, :])
    tables.heating = log10_or_small(f['Heating'][:, :])
    tables.mmw = number_or_nan(f['MMW'][:, :])
    return tables

def write_tables(newtables, filename=None, old_filename=None):
    """write Cloudy tables to filename in HDF5 format."""
    f = h5py.File(filename, 'w')

    f['Parameter1'] = newtables.log_nH
    f['Temperature'] = 10**(newtables.log_T)
    f['Cooling'] = 10**(newtables.cooling)
    f['Heating'] = 10**(newtables.heating)
    f['MMW'] = newtables.mmw

    ## must copy attributes!!
    old = h5py.File(old_filename, 'r')
    for group in ['Parameter1', 'Temperature', 'Cooling', 'Heating', 'MMW']:
        for attr in old[group].attrs.keys():
            f[group].attrs[attr] = old[group].attrs[attr]
    old.close()
    f.close()


def interpolate_mu(nH, T, tables=None):
    # given number density, temperature, return the mean mol. weight / mH
    log_nH = np.log10(nH)
    log_T = np.log10(T)
    interp_mmw = RectBivariateSpline(tables.log_nH,
                                     tables.log_T,
                                     tables.mmw,
                                     kx=1,
                                     ky=1)
    return interp_mmw(log_nH, log_T)[0][0]


def cooling_rate(nH, T, redshift=0., tables=None):
    """compute the cooling rate at a given density, redshift, and temperature.
    Note that the rate tables are C-ordered (as specified by the HDF5 standard.)"""
    log_nH = np.log10(nH)
    log_T = np.log10(T)
    interp_cooling = RectBivariateSpline(tables.log_nH,
                                         tables.log_T,
                                         tables.cooling,
                                         kx=1,
                                         ky=1)
    interp_heating = RectBivariateSpline(tables.log_nH,
                                         tables.log_T,
                                         tables.heating,
                                         kx=1,
                                         ky=1)

    metalCool = 10**interp_cooling(log_nH, log_T)[0, 0]
    metalHeat = 10**interp_heating(log_nH, log_T)[0, 0]
    rhoH = nH * m_H
    Edot = rhoH**2 * (metalHeat - metalCool)
    return Edot


def only_cooling_rate(nH, T, redshift=0., tables=None):
    """compute the cooling rate at a given density, redshift, and temperature.
    Note that the rate tables are C-ordered (as specified by the HDF5 standard.)"""
    log_nH = np.log10(nH)
    log_T = np.log10(T)
    interp_cooling = RectBivariateSpline(tables.log_nH,
                                         tables.log_T,
                                         tables.cooling,
                                         kx=1,
                                         ky=1)
    
    metalCool = 10**interp_cooling(log_nH, log_T)[0, 0]
    rhoH = nH * m_H
    Edot = rhoH**2 * (metalCool)
    return Edot