import numpy as np
import sys

Myr = 1e6 * 365.25 * 24 * 3600
width = 3.018e21 # not exactly kpc
Msun = 1.988409871e+33 # g

def random_stars(N, M_min, M_max, T_min, T_max, disk_height):
    """
    N: number of stars
    M_min: minimum mass of stars, in Msun
    M_max: maximum mass of stars, in Msun
    T_min: minimum birth time of stars, in Myr
    T_max: maximum birth time of stars, in Myr
    disk_height: height of the disk of stars, as a ratio to the width of the box
    """
    x = np.random.uniform(0.0, 1.0, N) * width
    y = np.random.uniform(0.0, 1.0, N) * width
    # z = np.random.uniform(-disk_height/2., disk_height/2., N) * width
    z = np.random.normal(0.0, disk_height/2., N) * width

    M = np.random.uniform(M_min, M_max, N) * Msun
    L = np.zeros(N)
    V = np.zeros((N, 3))
    birth = np.random.uniform(T_min, T_max, N) * Myr
    lifetime = np.random.uniform(3.0, 30.0, N) * Myr
    death = birth + lifetime
    data = np.zeros((N, 7 + 3))
    data[:, :3] = np.column_stack((x, y, z))
    data[:, 3] = M
    data[:, 4:7] = V
    data[:, 7] = birth
    data[:, 8] = death
    data[:, 9] = L
    fn = f"stars_N{N}_scaleheight_{disk_height_:.3f}.txt"
    np.savetxt(fn, data, fmt="%.10e", delimiter=" ", header=f"{N}", comments="")

if __name__ == "__main__":
    M_min_ = 11
    M_max_ = 100
    T_min_ = -3.0
    T_max_ = 0.0
    np.random.seed(42)
    for N_ in [1000, 100, 10]:
        for disk_height_ in [0.05, 0.005]:
            random_stars(N_, M_min_, M_max_, T_min_, T_max_, disk_height_)
