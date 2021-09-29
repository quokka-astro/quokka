import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

if __name__ == "__main__":
    # compute ODE solution to optically-thin wind
    # a_rad = 7.5646e-15      # erg cm^-3 K^-4
    # c = 2.99792458e10       # cm s^-1
    # k_B = 1.380658e-16      # erg K^-1
    # m_H = 1.6726231e-24     # mass of hydrogen atom [g]
    # mu = 2.33*m_H           # mean molecular weight
    #kappa0 = 1.0e-6          # opacity [cm^2 g^-1]
    
    Mach0 = 1.1 # Mach number at base of wind
    eta = 2.0   # no gravity
    x0 = 0.     # cm
    x1 = 1.     # cm

    chi = np.linspace(x0, x1, 1024)
    Mach = np.zeros_like(chi)
    # solve for Mach number via Bernoulli equation
    for i, chi_i in enumerate(chi):
        f = lambda M: 0.5*(M**2 - Mach0**2) + np.log(Mach0/M) + (1.0 - eta)*chi_i
        fprime = lambda M: M + (M/Mach0)*(-1.0/M**2)
        M_root = root_scalar(f, method='Newton', fprime=fprime, x0=Mach0)
        Mach[i] = M_root.root

    rho = Mach0 / Mach

    # save to file
    np.savetxt('optically_thin_wind.txt', np.c_[chi, rho, Mach], header='x density Mach')

    # plot
    plt.figure()
    #plt.plot(chi, rho, label=r'density (dimensionless)')
    plt.plot(chi, Mach, label='Mach number')
    plt.legend(loc='best')
    plt.xlim(x0, x1)
    plt.xlabel('x')
    plt.tight_layout()
    plt.savefig('density_optically_thin.pdf')
