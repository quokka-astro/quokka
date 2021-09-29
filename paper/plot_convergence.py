import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    ## plot error norm as a function of spatial resolution
    ## for the Athena sound wave test

    nx, errnorm = np.loadtxt("linear_wave.csv", unpack=True)
    nx_line = np.logspace(np.log10(np.min(nx)), np.log10(np.max(nx)), num=100)
    err_line = np.max(errnorm) * (nx_line/np.min(nx))**(-2.)

    plt.figure()
    plt.scatter(nx, errnorm, color='black')
    plt.plot(nx_line, err_line, '--', color='black')
    plt.xlim(7, 1.5e3)
    plt.ylim(0.5e-11, 2e-7)
    plt.xlabel('number of grid cells $N_x$')
    plt.ylabel('error norm $||\Delta U||$')
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig("wave_convergence.pdf")