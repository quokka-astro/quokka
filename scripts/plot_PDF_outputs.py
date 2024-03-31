import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def float_if_possible(element: any) -> bool:
    try:
        float(element)
        return float(element)
    except ValueError:
        return element

def read_header(filename):
    f = open(filename)
    header = {}
    for line in f:
        if line.startswith('#'):
            tokens = line[1:].split()
            key = tokens[0][:-1]
            values = [float_if_possible(val) for val in tokens[1:]]
            header[key] = values
    f.close()
    return header
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='*')
    args = parser.parse_args()

    header = read_header(args.filenames[0])
    xvar, yvar = header['variables']
    islog_x, islog_y = [bool(v) for v in header['is_log_spaced']]

    for filename in args.filenames:
        header = read_header(filename)
        cycle = int(header['cycle'][0])
        time = header['time'][0]
        print(cycle)
        
        hist = pd.read_csv(filename, sep='\s+', comment='#')

        xmin = hist[xvar + '_min'].min()
        xmax = hist[xvar + '_max'].max()
        ymin = hist[yvar + '_min'].min()
        ymax = hist[yvar + '_max'].max()
        if islog_x:
            xmin = np.log10(xmin)
            xmax = np.log10(xmax)
        if islog_y:
            ymin = np.log10(ymin)
            ymax = np.log10(ymax)
            
        Xidx = hist[xvar + '_idx']
        Yidx = hist[yvar + '_idx']
        nx = Xidx.max() + 1
        ny = Yidx.max() + 1
        arr = np.ndarray((nx,ny))

        for row in range(hist.shape[0]):
            i = Xidx[row]
            j = Yidx[row]
            arr[i, j] = hist['mass_sum'][row]
        
        ## plot
        plt.figure()
        with warnings.catch_warnings(action="ignore"):
            im = plt.imshow(np.log10(arr.T), extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower')
        plt.colorbar(im)

        if islog_x:
            plt.xlabel(r"$\log_{10}$" + xvar)
        else:
            plt.xlabel(xvar)
        
        if islog_y:
            plt.ylabel(r"$\log_{10}$" + yvar)
        else:
            plt.ylabel(yvar)

        plt.title(f"cycle {cycle:06d} time {time:.3g}")
        plt.savefig(filename + ".png")
        plt.close()
        
