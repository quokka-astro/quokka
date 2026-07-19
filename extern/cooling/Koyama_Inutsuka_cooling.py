import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

## plot the equilibrium temperature-density curve for
## the Koyama & Inutsuka (2002) fitting function

if __name__ == '__main__':
    lambda_over_gamma = lambda T: (1.0e7 * np.exp(-114800. / (T + 1000.)) + 14.*np.sqrt(T)*np.exp(-92./T))

    # solve n_H * (Lambda/Gamma) - 1 == 0
    T_guess = 100.
    nH_array = np.logspace(-5, 2, 100)
    Teq = []
    for n_H in nH_array:
        f = lambda T: n_H * lambda_over_gamma(T) - 1.0
        root = scipy.optimize.newton(f, x0=T_guess)
        print(f"{n_H} {root}")
        Teq.append(root)

    plt.figure(figsize=(4,4))
    plt.plot(nH_array, Teq, label="equilibrium")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"density (H/cc)")
    plt.ylabel(r"temperature (K)")
    plt.title("Koyama & Inutsuka (2002) function")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cooling_curve.pdf")
