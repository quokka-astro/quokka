# note: requires yt>=4.3.0
import sys
import yt
from math import sqrt
import glob
import numpy as np

yt.set_log_level(40)

def particle_dist(plotfiles):
    t_arr = []
    err_arr = []
    err_vel_arr = []
    d0 = 2.0 * 3.125e12
    v0 = 10332860.
    m0 = 2.0e34

    for pltfile in plotfiles:
        ds = yt.load(pltfile)
        # print(ds.derived_field_list)
        Lx = ds.domain_right_edge[0] - ds.domain_left_edge[0]
        Nx = ds.domain_dimensions[0]
        cell_dx = Lx/Nx
        ad = ds.all_data()
        x = ad["CIC_particles", "particle_position_x"]
        y = ad["CIC_particles", "particle_position_y"]
        z = ad["CIC_particles", "particle_position_z"]
        vxs = ad["CIC_particles", "particle_real_comp1"]
        vys = ad["CIC_particles", "particle_real_comp2"]
        vzs = ad["CIC_particles", "particle_real_comp3"]
        ms = ad["CIC_particles", "particle_real_comp0"]
        assert ms[0] == m0 and ms[1] == m0
        dx = x[0] - x[1]
        dy = y[0] - y[1]
        dz = z[0] - z[1]
        d = sqrt(dx*dx + dy*dy + dz*dz)
        #fractional_err = (d-d0)/d0
        grid_err = (d - d0) / cell_dx.value
        vx = vxs[0]
        vy = vys[0]
        vz = vzs[0]
        v_mag = sqrt(vx*vx + vy*vy + vz*vz)
        err_vel = (v_mag - v0) / v0
        t_arr.append(float(ds.current_time) / 3.15e7)
        err_arr.append(grid_err)
        err_vel_arr.append(err_vel)

    return t_arr, err_arr, err_vel_arr

def main(pltdir):

    files = glob.glob(pltdir + "/plt*")
    files = sorted(files)
    t, err_dist, err_vel = particle_dist(files)

    print("max rel_error distance: {:.1e}".format(np.max(np.abs(err_dist))))
    print()
    # print("max error velocity: {:.1e}".format(np.max(np.abs(err_vel))))

    # print time vs err_vel as a table
    err_vel_tol = 1.0e-3
    print("time (yr) rel_err_vel within_tol_1e-3?")
    for i in range(len(t)):
        if np.abs(err_vel[i]) < err_vel_tol:
            print("{:.1e} {:.1e} yes".format(t[i], err_vel[i]))
        else:
            print("{:.1e} {:.1e} no".format(t[i], err_vel[i]))

    return

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(t[1:], np.abs(err[1:]))
    # plt.ylim(-0.1, 0.1)
    plt.grid()
    plt.xlabel("time (yr)")
    plt.ylabel(r"$(d-d_0)/\Delta x$")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("orbit.png", dpi=150)

pltdir = "."
if len(sys.argv) > 1:
    pltdir = sys.argv[1]
main(pltdir)
