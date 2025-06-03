import h5py
import numpy as np
from scipy.interpolate import RectBivariateSpline

# From Grackle source code (initialize_chemistry_data.c, line 114):
#   In fully tabulated mode, set H mass fraction according to
#   the abundances in Cloudy, which assumes n_He / n_H = 0.1.
#   This gives a value of about 0.716. Using the default value
#   of 0.76 will result in negative electron densities at low
#   temperature. Below, we set X = 1 / (1 + hydrogen_mass_cgs_e * n_He / n_H).

c_light_cgs_ = 2.99792458e10  # cgs
radiation_constant_cgs_ = 7.5646e-15  # cgs
boltzmann_constant_cgs_ = 1.380658e-16  # cgs
m_H = 1.672623e-24  # g

cloudy_H_mass_fraction = 1. / (1. + 0.1 * 3.971)
X = cloudy_H_mass_fraction
Z = 0.02  # metal fraction by mass
Y = 1. - X - Z
mean_metals_A = 16.  # mean atomic weight of metals

sigma_T = 6.6524e-25  # Thomson cross section (cm^2)
electron_mass_cgs = 9.1093897e-28  # electron mass (g)
T_cmb = 2.725  # * (1 + z); // K
E_cmb = radiation_constant_cgs_ * (T_cmb * T_cmb * T_cmb * T_cmb)


class cloudyTables:
    metalCooling = []
    metalHeating = []
    primCooling = []
    primHeating = []
    mmw = []
    log_nH = []
    log_T = []
    redshift = []


def read_tables(filename):
    """"read Cloudy tables in HDF5 format."""
    f = h5py.File(filename, 'r')
    rates = f['CoolingRates']

    tables = cloudyTables()
    redshiftIdx = 0

    tables.log_nH = f['CoolingRates/Metals/Cooling'].attrs['Parameter1']
    tables.redshift = f['CoolingRates/Metals/Cooling'].attrs['Parameter2']
    tables.log_T = np.log10(
        f['CoolingRates/Metals/Cooling'].attrs['Temperature'])

    tbase1 = 1.0  # time units
    xbase1 = 1.0  # length units
    dbase1 = 1.0  # density units
    CoolUnit = (xbase1 * xbase1 * m_H * m_H) / (tbase1 * tbase1 * tbase1 *
                                                dbase1)
    logCoolUnit = np.log10(CoolUnit)

    log10_or_small = lambda table: np.piecewise(table,
                                                (table > 0., table <= 0.),
                                                (np.log10, lambda x: -99.0))
    tables.metalCooling = log10_or_small(
        rates['Metals/Cooling'][:, redshiftIdx, :]) - logCoolUnit
    tables.metalHeating = log10_or_small(
        rates['Metals/Heating'][:, redshiftIdx, :]) - logCoolUnit
    tables.primCooling = log10_or_small(
        rates['Primordial/Cooling'][:, redshiftIdx, :]) - logCoolUnit
    tables.primHeating = log10_or_small(
        rates['Primordial/Heating'][:, redshiftIdx, :]) - logCoolUnit

    tables.mmw = rates['Primordial/MMW'][:, redshiftIdx, :]
    return tables


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

    interp_metalCooling = RectBivariateSpline(tables.log_nH,
                                              tables.log_T,
                                              tables.metalCooling,
                                              kx=1,
                                              ky=1)
    interp_metalHeating = RectBivariateSpline(tables.log_nH,
                                              tables.log_T,
                                              tables.metalHeating,
                                              kx=1,
                                              ky=1)
    interp_primCooling = RectBivariateSpline(tables.log_nH,
                                             tables.log_T,
                                             tables.primCooling,
                                             kx=1,
                                             ky=1)
    interp_primHeating = RectBivariateSpline(tables.log_nH,
                                             tables.log_T,
                                             tables.primHeating,
                                             kx=1,
                                             ky=1)

    metalCool = 10**interp_metalCooling(log_nH, log_T)[0, 0]
    metalHeat = 10**interp_metalHeating(log_nH, log_T)[0, 0]
    primCool = 10**interp_primCooling(log_nH, log_T)[0, 0]
    primHeat = 10**interp_primHeating(log_nH, log_T)[0, 0]

    rhoH = nH * m_H
    Edot = rhoH**2 * ((metalHeat - metalCool) + (primHeat - primCool))

    # compute electron density
    # N.B. it is absolutely critical to include the metal contribution here!
    rho = rhoH / cloudy_H_mass_fraction
    mu = interpolate_mu(nH, T, tables=tables)

    n_e = (rho / m_H) * \
                     (1.0 - mu * (X + Y / 4. + Z / mean_metals_A)) / \
                     (mu - (electron_mass_cgs / m_H))
    # the approximation for the metals contribution to e- fails at high densities (~1e3 or higher)
    n_e = max(n_e, 1.0e-4 * nH)

    # photoelectric heating term
    Tsqrt = np.sqrt(T)
    phi = 0.5  # phi_PAH from Wolfire et al. (2003)
    G_0 = 1.7  # ISRF from Wolfire et al. (2003)
    epsilon = \
      4.9e-2 / (1. + 4.0e-3 * (G_0 * Tsqrt / (n_e * phi))**0.73) + \
      3.7e-2 * (T / 1.0e4)**(0.7) / \
          (1. + 2.0e-4 * (G_0 * Tsqrt / (n_e * phi)))
    Gamma_pe = 1.3e-24 * nH * epsilon * G_0
    Edot += Gamma_pe

    # Compton term (CMB photons)
    # [e.g., Hirata 2018: doi:10.1093/mnras/stx2854]
    Gamma_C = \
      (8. * sigma_T * E_cmb) / (3. * electron_mass_cgs * c_light_cgs_)
    C_n = Gamma_C * boltzmann_constant_cgs_ / (5. / 3. - 1.0)
    compton_CMB = -C_n * (T - T_cmb) * n_e
    Edot += compton_CMB

    return Edot