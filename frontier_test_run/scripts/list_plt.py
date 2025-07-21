import yt
yt.set_log_level(50)
import numpy as np
import argparse
import os.path

sec_in_yr = 3.154e7

if __name__ == "__main__":
    ## list plotfiles that are at integer multiples of the given time interval

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_interval", required=True, help="the time interval between plotfiles in Myr")
    parser.add_argument("plotfiles", nargs='*')
    args = parser.parse_args()
    
    for my_plotfile in args.plotfiles:
        if my_plotfile[-4:] == ".png": # ignore generated plots
            continue

        # get time
        ds = yt.load(my_plotfile)
        time = ds.current_time.value

        # output plotfiles that match the time interval
        interval = float(args.time_interval) * 1.0e6 * sec_in_yr
        eps = 0.02
        leftover = (time % interval) / interval
        if np.abs(leftover) < eps:
            print(my_plotfile)
        
