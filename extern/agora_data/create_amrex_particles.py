import numpy as np
import argparse
from pathlib import Path

cgs_length = 1.0e3 * 3.086e18 # 1 kpc
cgs_vel = 1.0e5 # 1 km/s
cgs_mass = 1.0e9 * 1.989e33 # 1e9 solar masses

if __name__ == "__main__":
    ## save ASCII file in AMReX particle format
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory")
    parser.add_argument("output_file")

    args = parser.parse_args()
    input_dir = Path(args.data_directory)

    # Dark matter halo (halo.dat): x,y,z,vx,vy,vz,mdark
    # Stellar disk (disk.dat): x,y,z,vx,vy,vz,mdisk
    # S tellar bulge (bulge.dat): x,y,z,vx,vy,vz,mbulge    
    dm_file = input_dir / "halo.dat"
    disk_file = input_dir / "disk.dat"
    bulge_file = input_dir / "bulge.dat"

    with open(args.output_file, 'w') as output:
        for part_file in [dm_file, disk_file, bulge_file]:            
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
            Npart = len(x)
            output.write(f"{Npart}\n")
            for i in range(Npart):
                output.write(f"{x[i]} {y[i]} {z[i]} {m[i]} {vx[i]} {vy[i]} {vz[i]}\n")
