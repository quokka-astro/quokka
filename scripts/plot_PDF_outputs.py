import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xvar")
    parser.add_argument("yvar")
    parser.add_argument("islog_x", type=bool)
    parser.add_argument("islog_y", type=bool)
    parser.add_argument("filenames", nargs='*')
    args = parser.parse_args()

    xvar = args.xvar
    yvar = args.yvar
    islog_x = args.islog_x
    islog_y = args.islog_y
    
    for filename in args.filenames:
        hist = pd.read_csv(filename, sep='\s+')

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
        
        plt.savefig(filename + ".png")
