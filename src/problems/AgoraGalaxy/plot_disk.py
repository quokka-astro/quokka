import yt
import argparse

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
        # load data
        ds = yt.load(my_plotfile)

        # z-projection
        plt1 = yt.SlicePlot(ds, 'z', ('boxlib', 'gasDensity'))
        plt1.zoom(20)
        plt1.annotate_particles((1, "kpc")) # line-of-sight slice width
        plt1.annotate_streamlines(("gas", "velocity_x"), ("gas", "velocity_y"), color="white")
        plt1.annotate_scale()
        plt1.annotate_timestamp()
        plt1.save()
    
        # y-projection
        plt2 = yt.SlicePlot(ds, 'x', ('boxlib', 'gasDensity'))
        plt2.zoom(20)
        plt2.annotate_particles((1, "kpc")) # line-of-sight slice width
        plt2.annotate_streamlines(("gas", "velocity_y"), ("gas", "velocity_z"), color="white")
        plt2.annotate_scale()
        plt2.annotate_timestamp()
        plt2.save()
    