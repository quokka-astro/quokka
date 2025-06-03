import numpy as np
import argparse
from pathlib import Path
from itertools import (takewhile, repeat)

cgs_length = 1.0e3 * 3.086e18 # 1 kpc
cgs_vel = 1.0e5 # 1 km/s
cgs_mass = 1.0e9 * 1.989e33 # 1e9 solar masses

def count_lines_in_file(filename):
    ## return the number of lines in the ASCII file 'filename'
    ## see: https://stackoverflow.com/a/27518377
    with open(filename, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
        return sum(buf.count(b'\n') for buf in bufgen)

if __name__ == "__main__":
    ## save ASCII file in AMReX particle format
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory")
    parser.add_argument("output_file")
    args = parser.parse_args()

    input_dir = Path(args.data_directory)
    dm_file = input_dir / "halo.dat"        # Dark matter halo (halo.dat): x,y,z,vx,vy,vz,mdark
    disk_file = input_dir / "disk.dat"      # Stellar disk (disk.dat): x,y,z,vx,vy,vz,mdisk
    bulge_file = input_dir / "bulge.dat"    # Stellar bulge (bulge.dat): x,y,z,vx,vy,vz,mbulge    
    particle_file_list = [dm_file, disk_file, bulge_file]

    ## count total number of particles
    Npart = 0
    for part_file in particle_file_list:
        this_Npart = count_lines_in_file(part_file)
        print(f"found {this_Npart} particles in {part_file}.")
        Npart += this_Npart

    ## write output ASCII file
    with open(args.output_file, 'w') as output:
        print(f"writing {Npart} total particles...")
        output.write(f"{Npart}\n") # write *total* number of particles (required by AMReX)

        for part_file in particle_file_list:
            ## read data
            x,y,z,vx,vy,vz,m = np.loadtxt(part_file, unpack=True)

            ## convert to CGS units
            x *= cgs_length
            y *= cgs_length
            z *= cgs_length
            vx *= cgs_vel
            vy *= cgs_vel
            vz *= cgs_vel
            m *= cgs_mass

            ## save to output file
            print("writing", part_file, "...")
            for i in range(x.shape[0]):
                output.write(f"{x[i]} {y[i]} {z[i]} {m[i]} {vx[i]} {vy[i]} {vz[i]}\n")

        print("done.")