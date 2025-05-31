import yt
import argparse
import os.path

if __name__ == "__main__":
    ## make plot for each plotfile
    ## https://yt-project.org/doc/visualizing/callbacks.html#overplot-quivers-for-the-velocity-field

    parser = argparse.ArgumentParser()
    parser.add_argument("plotfiles", nargs='*')
    args = parser.parse_args()

    for my_plotfile in args.plotfiles:
        if my_plotfile[-4:] == ".png":
            continue

        output_file1z = my_plotfile + "_Slice_z_gasDensity.png"
        if os.path.isfile(output_file1z):
            continue
        
        # load data
        ds = yt.load(my_plotfile)
        field_prefix, field_name = zip(*ds.field_list)
        center = ds.arr([1e10, 1e10, 1e10], 'code_length')
        zoom_fac = 40

        # z-slice (face-on)
        if not os.path.isfile(output_file1z):
            plt2 = yt.ProjectionPlot(ds, 'z', ('boxlib', 'gasDensity'), center=center)
            plt2.zoom(zoom_fac)
            plt2.set_zlim(('boxlib', 'gasDensity'), 3e-4, 1.)
            if 'StochasticStellarPop_particles' in field_prefix:
                plt2.annotate_particles((1, "Mpc"), ptype="StochasticStellarPop_particles")
            plt2.annotate_scale()
            plt2.annotate_timestamp()
            plt2.save(output_file1z)
