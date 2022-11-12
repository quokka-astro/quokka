import numpy as np
from scipy.integrate import tplquad
import matplotlib.pyplot as plt

def kernel(r):
    ## Wendland kernels (Wendland 1995)
    if (r > 1.0):
        return 0.0
    else:
        return (21./(2.*np.pi)) * (1.0 - r)**4 * (4.0*r + 1.0) # Wendland C2

def integrate_cell_mass(x_bounds, y_bounds, z_bounds, r_scale=1.0):
    f = lambda z, y, x: kernel(np.sqrt(x*x + y*y + z*z) / r_scale)
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    z0, z1 = z_bounds
    lower_y = lambda x: y0
    upper_y = lambda x: y1
    lower_z = lambda x, y: z0
    upper_z = lambda x, y: z1
    res, abserr = tplquad(f, x0, x1, lower_y, upper_y, lower_z, upper_z)
    res /= r_scale**3
    return res

if __name__ == '__main__':
    ## integrate a radial kernel over a grid of 2d cells
    nx = 16
    ny = 16
    x_grid = np.linspace(-5, 5, nx+1, endpoint=True)
    y_grid = np.linspace(-5, 5, ny+1, endpoint=True)
    dx = np.diff(x_grid)[0]
    dy = np.diff(y_grid)[0]
    x_grid += 0.5 * dx
    y_grid += 0.5 * dy

    r_scale = 4.0 * dx
    zmin = -r_scale
    zmax = r_scale
    dz = zmax - zmin
    cell_vol = dx*dy*dz

    dens = np.zeros((x_grid.size - 1, y_grid.size - 1))
    for i, x0 in enumerate(x_grid[:-1]):
        x1 = x_grid[i+1]
        for j, y0 in enumerate(y_grid[:-1]):
            y1 = y_grid[j+1]
            dens[i, j] = integrate_cell_mass([x0, x1], [y0, y1], [zmin, zmax], r_scale=r_scale) / cell_vol

    total_mass = np.sum(dens) * cell_vol
    print(f"mass = {total_mass}")

    im = plt.imshow(dens)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("density.png")
