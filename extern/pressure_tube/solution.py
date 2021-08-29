import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

if __name__ == "__main__":
    # compute ODE solution to radiation pressure tube (Krumholz et al. 2007)
    a_rad = 7.5646e-15      # erg cm^-3 K^-4
    c = 2.99792458e10       # cm s^-1
    k_B = 1.380658e-16      # erg K^-1
    m_H = 1.6726231e-24     # mass of hydrogen atom [g]
    gamma_gas = 5. / 3.     # monoatomic ideal gas
    mu = 2.33*m_H           # mean molecular weight
    kappa = 100.            # opacity [cm^2 g^-1]

    def func(t, y):
        # return dy / dt == func(t,y)
        # in this case, y[0] = rho;     dy[0]/dx = d(rho)/dx
        #               y[1] = T;       dy[1]/dx = dT/dx
        #               y[2] = dT/dx;   dy[2]/dx = d^2(T)/dx^2
        #               t == x (position)
        x = t
        rho = y[0]
        T = y[1]
        dT_dx = y[2]
        drho_dx = -(mu/(k_B*T))*((k_B/mu)*rho + (4./3.)*a_rad*T**3)*dT_dx
        d2T_dx2 = -(3./T)*dT_dx**2 + (1/rho)*drho_dx*dT_dx
        return [drho_dx, dT_dx, d2T_dx2]

    x0 = 0.     # cm
    x1 = 128.   # cm
    rho0 = 1.0  # g cm^-3
    T0 = 2.75e7  # K
    drho_dx0 = 5.0e-3  # g cm^-4
    dT_dx0 = drho_dx0 / (-(mu/(k_B*T0))*((k_B/mu)*rho0 + (4./3.)*a_rad*T0**3))

    f0 = [rho0, T0, dT_dx0]
    x_arr = np.linspace(x0, x1, 1024, endpoint=True)
    sol = solve_ivp(func, [x0, x1], f0, method='RK45', t_eval=x_arr)
    print(f"{sol}")

    x = sol.t
    rho = sol.y[0]
    T = sol.y[1]
    dT_dx = sol.y[2]
    Pgas = (k_B/mu)*rho*T
    Erad = a_rad*T**4
    Prad = (1./3.)*Erad

    # save to file

    np.savetxt('initial_conditions.txt', np.c_[
               x, rho, Pgas, Erad], header='x_cm rho_gcm3 Pgas_ergcm3 Erad_ergcm3')

    # plot

    plt.figure()
    plt.plot(sol.t, sol.y[0], label=r'density (g cm^-3)')
    plt.legend(loc='best')
    plt.xlim(x0, x1)
    plt.xlabel('x (cm)')
    plt.tight_layout()
    plt.savefig('density.pdf')

    plt.figure()
    plt.plot(sol.t, sol.y[1] / 1.0e7, label=r'temperature (10^7 K)')
    plt.legend(loc='best')
    plt.xlim(x0, x1)
    plt.xlabel('x (cm)')
    plt.tight_layout()
    plt.savefig('temperature.pdf')

    plt.figure()
    plt.plot(sol.t, Pgas, label=r'gas pressure')
    plt.plot(sol.t, Prad, label=r'radiation pressure')
    plt.legend(loc='best')
    plt.ylim(0., 2.5e15)
    plt.xlim(x0, x1)
    plt.xlabel('x (cm)')
    plt.ylabel('pressure (g cm^-3)')
    plt.tight_layout()
    plt.savefig('pressures.pdf')
