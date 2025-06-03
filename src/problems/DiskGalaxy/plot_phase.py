import yt
from unyt import g
import argparse
import os.path

def _mass(field, data):
    return (data["boxlib", "gasDensity"] * data["boxlib", "cell_volume"].value) * g

if __name__ == "__main__":
    ## make plot for each plotfile
    ## https://yt-project.org/doc/visualizing/callbacks.html#overplot-quivers-for-the-velocity-field

    parser = argparse.ArgumentParser()
    parser.add_argument("plotfiles", nargs='*')
    args = parser.parse_args()

    # add derived field
    yt.add_field(name=("gas", "mass"),
            function=_mass,
            sampling_type="local",
            units="g")
    
    for my_plotfile in args.plotfiles:
        if my_plotfile[-4:] == ".png":
            continue

        output_file1 = my_plotfile + "_phaseplot.png"
        if os.path.isfile(output_file1):
            continue
        
        # load data
        ds = yt.load(my_plotfile)
        
        # phase plot
        if not os.path.isfile(output_file1):
            my_sphere = ds.sphere("c", (30.0, "kpc"))
            plt = yt.PhasePlot(my_sphere, ('boxlib','gasDensity'), ('boxlib','temperature'), ('gas','mass'), weight_field=None)
            plt.set_unit(("gas","mass"),"Msun")
            plt.set_xlim(1e-29, 1e-21)
            plt.set_ylim(1e2, 1e9)
            plt.set_xlabel("Density")
            plt.save(output_file1)
