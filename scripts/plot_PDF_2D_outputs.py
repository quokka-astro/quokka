import warnings
import numpy as np
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
    parser = argparse.ArgumentParser(description="This script reads and plots a 2D histogram/PDF output produced using the DiagPDF diagnostic.")
    parser.add_argument("filenames", nargs='*', help="A list of *.dat histogram output files.")
    args = parser.parse_args()

    for filename in args.filenames:
        header = read_header(filename)
        xvar, yvar = header['variables']
        islog_x, islog_y = [bool(v) for v in header['is_log_spaced']]

        cycle = int(header['cycle'][0])
        time = header['time'][0]
        print(cycle)
        
        hist = np.genfromtxt(filename, names=True, skip_header=4)

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
            
        Xidx = hist[xvar + '_idx'].astype(int)
        Yidx = hist[yvar + '_idx'].astype(int)
        nx = Xidx.max() + 1
        ny = Yidx.max() + 1
        arr = np.ndarray((nx,ny))

        for row in range(hist.shape[0]):
            i = Xidx[row]
            j = Yidx[row]
            arr[i, j] = hist['mass_sum'][row]
        
        ## plot
        plt.figure()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore exceptions due to applying np.log10 to zero values
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
        
