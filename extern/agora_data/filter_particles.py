import numpy as np
import argparse

cgs_length = 1.0e3 * 3.085677587679311e18 # 1 kpc

if __name__ == "__main__":
    ## filter out particles outside of the box, save into new ASCII file
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("box_size_kpc", type=float, help='half-box size in kpc')
    args = parser.parse_args()
    box_size = args.box_size_kpc * cgs_length

    print(f"reading particles from {args.input_file}...")
    x,y,z,m,vx,vy,vz = np.loadtxt(args.input_file, unpack=True, skiprows=1)
    print("done.")
    
    # filter particles
    print("filtering particles...")
    mask = np.logical_and(np.logical_and(np.abs(x) < box_size, np.abs(y) < box_size), np.abs(z) < box_size)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    m = m[mask]
    vx = vx[mask]
    vy = vy[mask]
    vz = vz[mask]
    print("done.")
    
    ## write output ASCII file
    with open(args.output_file, 'w') as output:
        Npart = x.shape[0]
        print(f"writing {Npart} total particles...")
        output.write(f"{Npart}\n") # write *total* number of particles (required by AMReX)

        ## save to output file
        for i in range(x.shape[0]):
            output.write(f"{x[i]} {y[i]} {z[i]} {m[i]} {vx[i]} {vy[i]} {vz[i]}\n")

        print("done.")
