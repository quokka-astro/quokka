import numpy as np
import sys

Myr = 1e6 * 365.25 * 24 * 3600
width = 3.018e21 # not exactly kpc
Msun = 1.988409871e+33 # g
kmpers = 1.0e5 # km/s

def random_stars(N, M_min, M_max, disk_height, T_min=None, T_max=None, prefix="stars"):
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
    ncomp = 7 + 3 if prefix == "stars" else 4 + 3
    data = np.zeros((N, ncomp))
    data[:, :3] = np.column_stack((x, y, z))
    data[:, 3] = M
    if prefix == "stars":
        birth = np.random.uniform(T_min, T_max, N) * Myr
        lifetime = np.random.uniform(3.0, 30.0, N) * Myr
        death = birth + lifetime
        data[:, 7] = birth
        data[:, 8] = death
        data[:, 9] = L
    else:
        vmag = 3.0 * kmpers
        # Generate velocity magnitudes following Gaussian distribution with cutoff at 3*vmag
        v_magnitudes = np.random.normal(0.0, vmag, N)
        v_magnitudes = np.clip(v_magnitudes, -3*vmag, 3*vmag)
        
        # Generate uniform directions on the sky (unit sphere)
        # Using spherical coordinates: phi uniform in [0, 2π], cos(theta) uniform in [-1, 1]
        phi = np.random.uniform(0, 2*np.pi, N)
        cos_theta = np.random.uniform(-1, 1, N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        # Convert to Cartesian coordinates
        V[:, 0] = v_magnitudes * sin_theta * np.cos(phi)
        V[:, 1] = v_magnitudes * sin_theta * np.sin(phi)
        V[:, 2] = v_magnitudes * cos_theta
    data[:, 4:7] = V
    fn = f"{prefix}_N{N}_scaleheight_{disk_height:.3f}.txt"
    np.savetxt(fn, data, fmt="%.10e", delimiter=" ", header=f"{N}", comments="")

def run_stars():
    M_min_ = 11
    M_max_ = 100
    T_min_ = -3.0
    T_max_ = 0.0
    np.random.seed(42)
    for N_ in [1000, 100, 10]:
        for disk_height_ in [0.05, 0.005]:
            random_stars(N_, M_min_, M_max_, disk_height_, T_min_, T_max_)

def run_CICs():
    M_min_ = 11
    M_max_ = 100
    np.random.seed(42123)
    for N_ in [1000]:
        for disk_height_ in [0.005]:
            random_stars(N_, M_min_, M_max_, disk_height_, prefix="stars_CIC")

if __name__ == "__main__":
    # run_stars()
    run_CICs()