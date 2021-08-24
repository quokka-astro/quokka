import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # compute ODE solution for radiation-driven dusty shell
    a_rad = 7.5646e-15      # erg cm^-3 K^-4
    c = 2.99792458e10       # cm s^-1
    cs0 = 0.633e5           # (0.633 km/s) [cm s^-1]
    a0 = 2.0e5              # ('reference' sound speed) [cm s^-1]
    chat = c                # 860. * a0    # cm s^-1
    k_B = 1.380658e-16      # erg K^-1
    m_H = 1.6726231e-24     # mass of hydrogen atom [g]
    gamma_gas = 5. / 3.     # monoatomic ideal gas
    Msun = 2.0e33           # g
    parsec_in_cm = 3.086e18 # cm

    specific_luminosity = 2000.             # erg s^-1 g^-1
    GMC_mass = 1.0e6 * Msun                 # g
    epsilon = 0.5                           # dimensionless
    M_shell = (1 - epsilon) * GMC_mass      # g
    L_star = GMC_mass * specific_luminosity # erg s^-1

    r_0 = 5.0 * parsec_in_cm # cm
    sigma_star = 0.0625 * r_0
    H_shell = 0.1 * r_0 # cm

    rho_0 = M_shell / ((4. / 3.) * np.pi * r_0 * r_0 * r_0) # g cm^-3
    kappa0 = 20.0           # specific opacity [cm^2 g^-1]
    tau0 = M_shell * kappa0 / (4.0 * np.pi * r_0 * r_0)

    r = r_0
    rad_force = tau0 * (r/r_0)**(-2.) * (L_star / c)
    dt = 1.0e9  # s

    print(f"fiducial optical depth tau0 = {tau0}")
    print(f"radiation force F = {rad_force} (cgs)")
    print(f"dMomentum = {dt*rad_force}")

    Frad_shell = L_star / (4.0 * np.pi * r_0 * r_0)
    Erad = (3.0 * Frad_shell / c) * (tau0 + 2. / 3.)
    Trad = (Erad / a_rad)**(1. / 4.)
    print(f"Trad = {Trad}")