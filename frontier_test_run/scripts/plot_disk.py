import yt
import numpy as np
import argparse
import os.path

def _velocity_x(field, data):
    return (data["boxlib", "x-GasMomentum"] / data["boxlib", "gasDensity"])

def _velocity_y(field, data):
    return (data["boxlib", "y-GasMomentum"] / data["boxlib", "gasDensity"])

def _velocity_z(field, data):
    return (data["boxlib", "z-GasMomentum"] / data["boxlib", "gasDensity"])

if __name__ == "__main__":
    ## make plot for each plotfile
    ## https://yt-project.org/doc/visualizing/callbacks.html#overplot-quivers-for-the-velocity-field

    # add derived field ('gas', 'velocity_x'), ('gas', 'velocity_y')
    yt.add_field(name=("gas", "velocity_x"),
            function=_velocity_x,
            sampling_type="local",
            units="dimensionless")
    
    yt.add_field(name=("gas", "velocity_y"),
            function=_velocity_y,
            sampling_type="local",
            units="dimensionless")
    
    yt.add_field(name=("gas", "velocity_z"),
            function=_velocity_z,
            sampling_type="local",
            units="dimensionless")

    parser = argparse.ArgumentParser()
    parser.add_argument("plotfiles", nargs='*')
    args = parser.parse_args()

    for my_plotfile in args.plotfiles:
        if my_plotfile[-4:] == ".png":
            continue

        output_file1 = my_plotfile + "_Slice_x_gasDensity.png"
        #output_file2 = my_plotfile + "_Slice_x_temperature.png"
        if os.path.isfile(output_file1):
            continue
        
        # load data
        ds = yt.load(my_plotfile)
        field_prefix, field_name = zip(*ds.field_list)
        center = ds.arr([1e10, 1e10, 1e10], 'code_length')
        zoom_fac = 40
        
        # x-slice
        if not os.path.isfile(output_file1):
            plt2 = yt.SlicePlot(ds, 'x', ('boxlib', 'gasDensity'), center=center)
            plt2.zoom(zoom_fac)
            plt2.set_zlim(('boxlib', 'gasDensity'), 1e-29, 1e-24)
            if 'StochasticStellarPop_particles' in field_prefix:
                plt2.annotate_particles((1, "Mpc"), ptype="StochasticStellarPop_particles")
            #plt2.annotate_streamlines(("gas", "velocity_y"), ("gas", "velocity_z"), color="black")
            plt2.annotate_scale()
            plt2.annotate_timestamp()
            plt2.save(output_file1)
        
        # x-slice
        #if not os.path.isfile(output_file2):
        #    plt2 = yt.SlicePlot(ds, 'x', ('boxlib', 'temperature'), center=center)
        #    plt2.zoom(zoom_fac)
        #    plt2.set_zlim(('boxlib', 'temperature'), 1e3, 2e8)
        #    if 'StochasticStellarPop_particles' in field_prefix:
        #        plt2.annotate_particles((1, "Mpc"), ptype="StochasticStellarPop_particles")
        #    #plt2.annotate_streamlines(("gas", "velocity_y"), ("gas", "velocity_z"), color="black")
        #    plt2.annotate_scale()
        #    plt2.annotate_timestamp()
        #    plt2.save(output_file2)
