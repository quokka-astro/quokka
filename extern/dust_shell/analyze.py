import numpy as np
import matplotlib.pyplot as plt
import yt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='AMReX plotfile directory')
    args = parser.parse_args()

    ## read AMReX/Boxlib outputs
    ds = yt.load(args.filename)

    