# note: requires yt>=4.3.0
import os
import sys
import yt
from math import sqrt
import glob
import numpy as np

yt.set_log_level(40)

def get_particle_dist(plotfiles):
    t_arr = []
    err_arr = []
    err_vel_arr = []
    pos = []
    nx_frame0 = None
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
        pos.append((float(x[0].value), float(y[0].value), float(z[0].value)))
        vxs = ad["CIC_particles", "particle_real_comp1"]
        vys = ad["CIC_particles", "particle_real_comp2"]
        vzs = ad["CIC_particles", "particle_real_comp3"]
        ms = ad["CIC_particles", "particle_real_comp0"]
        assert np.isclose(ms[0], m0) and np.isclose(ms[1], m0)
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

        if nx_frame0 is None:
            nx_frame0 = Nx

    return t_arr, err_arr, err_vel_arr, pos, nx_frame0

def get_particle_orbit(plotfiles):
    t_arr = []
    pos1_arr = []
    pos2_arr = []
    for pltfile in plotfiles:
        ds = yt.load(pltfile)
        ad = ds.all_data()
        x = ad["CIC_particles", "particle_position_x"]
        y = ad["CIC_particles", "particle_position_y"]
        z = ad["CIC_particles", "particle_position_z"]
        t_arr.append(float(ds.current_time) / 3.15e7)
        pos1_arr.append((float(x[0].value), float(y[0].value), float(z[0].value)))
        pos2_arr.append((float(x[1].value), float(y[1].value), float(z[1].value)))
    return t_arr, pos1_arr, pos2_arr

def plot_orbit_and_error(pltdir):

    files = glob.glob(pltdir + "/plt*")
    files = sorted(files)
    t, err_dist, err_vel, pos_particle0, nx_frame0 = get_particle_dist(files)

    print("max rel_error distance: {:.1e}".format(np.max(np.abs(err_dist))))
    print("max error velocity: {:.1e}".format(np.max(np.abs(err_vel))))

    # print time vs err_vel as a table
    print("time(yr) rel_err_vel within_tol_1e-3 within_tol_1e-2?")
    for i in range(len(t)):
        print("{:.1e} {:.1e} {} {}".format(t[i], err_vel[i], np.abs(err_vel[i]) < 1.0e-3, np.abs(err_vel[i]) < 1.0e-2))

    # plot the orbit
    t, pos1_arr, pos2_arr = get_particle_orbit(files)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    Nx = nx_frame0
    plt.title(f"nx={Nx}")
    plt.scatter([pos[0] for pos in pos1_arr], [pos[1] for pos in pos1_arr], color="red", label="particle 1")
    plt.scatter([pos[0] for pos in pos2_arr], [pos[1] for pos in pos2_arr], color="blue", label="particle 2")
    plt.legend()
    ax = plt.gca()
    ax.set(xlabel="x", ylabel="y")
    ax.axis("equal")
    ax.grid()
    hw = 6.0e12
    ax.set(xlim=(-hw, hw), ylim=(-hw, hw))
    figdir = os.path.dirname(files[0])
    plt.savefig(os.path.join(figdir, "orbit.png"), dpi=300)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(t[1:], np.abs(err_dist[1:]))
    # plt.ylim(-0.1, 0.1)
    plt.grid()
    plt.xlabel("time (yr)")
    plt.ylabel(r"$(d-d_0)/\Delta x$")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "orbit_err.png"), dpi=300)

    return


def plot_debug_Poisson_rhs(pltdir):
    for pltfile in sorted(glob.glob(pltdir + "/debug_*")):
        # skip if not a directory
        if not os.path.isdir(pltfile):
            continue
        # only plot accel fields
        is_accel = "accel" in os.path.basename(pltfile)
        if not is_accel:
            continue
        ds = yt.load(pltfile)
        field = ("boxlib", "accel_x")
        slc = yt.SlicePlot(ds, "z", field)
        if is_accel:
            slc.set_zlim(field, 1e0, 3e4)
            slc.set_log(field, True)
        slc.set_width((1.3e13, "cm"))
        slc.annotate_grids()
        slc.annotate_cell_edges(line_width=0.001, color='black')
        figfn = pltfile
        if figfn[-1] == '/':
            figfn = figfn[:-1]
        figfn = figfn + "_accel_x.png"
        slc.save(figfn, mpl_kwargs={"dpi": 300})


if __name__ == "__main__":
    pltdir = "."
    if len(sys.argv) > 1:
        pltdir = sys.argv[1]

    plot_orbit_and_error(pltdir)
    # plot_debug_Poisson_rhs(pltdir)
