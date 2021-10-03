import numpy as np
import matplotlib.pyplot as plt
import yt
import argparse

def velocity_mag(field, data):
    vx = data["boxlib", "x-GasMomentum"] / data["boxlib", "gasDensity"]
    vy = data["boxlib", "y-GasMomentum"] / data["boxlib", "gasDensity"]
    vz = data["boxlib", "z-GasMomentum"] / data["boxlib", "gasDensity"]
    return np.sqrt(vx*vx + vy*vy + vz*vz)

if __name__ == "__main__":
    a_rad = 7.5646e-15      # erg cm^-3 K^-4
    c = 2.99792458e10       # cm s^-1
    a0 = 2.0e5              # ('reference' sound speed) [cm s^-1]
    Msun = 2.0e33           # g
    parsec_in_cm = 3.086e18  # cm

    specific_luminosity = 2000.             # erg s^-1 g^-1
    GMC_mass = 1.0e6 * Msun                 # g
    epsilon = 0.5                           # dimensionless
    M_shell = (1 - epsilon) * GMC_mass      # g
    L_star = (epsilon * GMC_mass) * specific_luminosity  # erg s^-1
    r_0 = 5.0 * parsec_in_cm  # cm
    kappa0 = 20.0           # specific opacity [cm^2 g^-1]
    t0 = r_0 / a0

    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*', help='AMReX plotfile directory')
    args = parser.parse_args()

    mach_number = []
    time = []
    for i, filename in enumerate(args.filenames):
        # read AMReX/Boxlib outputs
        ds = yt.load(filename)

        ds.add_field(
            ("gas", "velocity_magnitude"),
            units="dimensionless",
            function=velocity_mag,
            sampling_type="cell",
        )

        field = ("gas", "velocity_magnitude")
        weight = ("boxlib", "gasDensity")
        ad = ds.all_data()
        average_value = ad.quantities.weighted_average_quantity(field, weight)
        average_vel_dimensionless = average_value / a0
        mach_number.append(average_vel_dimensionless)
        time.append(ds.current_time / t0)

    M0 = np.sqrt(L_star*kappa0 / (4.0*np.pi*r_0*c)) / a0
    print(f"M0 = {M0}")
    R = np.linspace(1.0, 2.0, 100)
    Mach_analytic = np.sqrt(2) * M0 * np.sqrt(1. - 1./R)
    T = 1.0/(M0*np.sqrt(2)) * (np.sqrt(R*(R-1.0)) +
                            np.log(np.sqrt(R) + np.sqrt(R - 1.0)))

    plt.figure()
    plt.plot(T, Mach_analytic, label='thin shell solution', color='black')
    plt.scatter(time, mach_number, marker='x', label='simulation')
    plt.xlabel('time (dimensionless)')
    plt.ylabel(r'shell velocity (dimensionless)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("shell_velocity.pdf")
