#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2022

import sys
import argparse
import numpy as np
import cfpack as cfp
from cfpack import hdfio, print, stop

# =========================================================
def Gaussian_func(x, mean, sigma):
    ret = 1.0/np.sqrt(2.0*np.pi*sigma**2) * np.exp(-0.5*(x-mean)**2/sigma**2)
    return ret
# =========================================================

# =========================================================
def get_number_of_components(ndim):
    if ndim == 1: ncmp = 1
    if ndim == 2: ncmp = 2
    if ndim == 1.5 or ndim == 2.5 or ndim == 3: ncmp = 3
    return ncmp
# =========================================================

# === generate turbulent field by calling C++ TurbGen ===
def generate(args, parser):
    # error checking on arguments
    if len(sys.argv) == 2: # if user does not specify any arguments, print help
        parser.print_help()
        print("\nSpecify at least 1 argument.", error=True)
    if len(args.N) > 3 or len(args.L) > 3: # N and L args take only up to 3 values
        parser.print_help()
        print("\n-N and -L only allow up to 3 arguments (x, y, z)", error=True)
    # construct command line for call to TurbGen
    arg_ndim = " -ndim "+str(args.ndim)
    arg_N = " -N"
    arg_L = " -L"
    for i in range(int(args.ndim)):
        arg_N += " "+str(args.N[i])
        arg_L += " "+str(args.L[i])
    arg_kmin = " -kmin "+str(args.kmin)
    arg_kmid = " -kmid "+str(args.kmid)
    arg_kmax = " -kmax "+str(args.kmax)
    args_spect_form = " -spect_form "+str(args.spect_form)
    args_power_law_exp   = ""
    args_power_law_exp_2 = ""
    if args.spect_form == 2:
        args_power_law_exp   = " -power_law_exp "  +str(args.power_law_exp)
        args_power_law_exp_2 = " -power_law_exp_2 "+str(args.power_law_exp_2)
    args_angles_exp = " -angles_exp "+str(args.angles_exp)
    args_sol_weight = " -sol_weight "+str(args.sol_weight)
    args_random_seed = " -random_seed "+str(args.random_seed)
    args_verbose = " -verbose "+str(args.verbose)
    args_outputfile = " -o "+args.outputfile
    args_write_modes = ""
    if args.write_modes: args_write_modes = " -write_modes"
    args_mpirun = ""
    if args.num_procs: args_mpirun = "mpirun -np "+str(args.num_procs)+" "
    cmd = args_mpirun+"./TurbGen"+arg_ndim+arg_N+arg_L+arg_kmin+arg_kmid+arg_kmax
    cmd += args_spect_form+args_power_law_exp+args_power_law_exp_2+args_angles_exp
    cmd += args_sol_weight+args_random_seed+args_verbose+args_outputfile+args_write_modes
    cfp.run_shell_command(cmd) # run TurbGen
# =========================================================

# === analyse turbulent field in HDF5 file ===
def analyse(args, parser):
    # read file and analyse
    ndim = float(hdfio.read(args.inputfile, 'ndim'))
    ncmp = int(hdfio.read(args.inputfile, 'ncmp'))
    # read components
    dirs = ['x', 'y', 'z']
    dsetnamebase = 'turb_field_'
    dat = []
    for ic in range(ncmp): # loop over components
        dat.append(hdfio.read(args.inputfile, dsetnamebase+dirs[ic]))
    dat = np.array(dat)
    # statistics
    mean = np.array([dat[d].mean() for d in range(ncmp)])
    std = np.array([dat[d].std() for d in range(ncmp)])
    print("mean = ", mean, highlight=1)
    print("standard deviation = ", std, highlight=1)
    # PDF plot
    for d in range(ncmp):
        pdf_obj = cfp.get_pdf(dat[d], bins=min(100,int(0.1*dat[d].size)))
        pdf = pdf_obj.pdf
        q = pdf_obj.bin_center
        cfp.plot(x=q, y=pdf, label="$"+dirs[d]+"$")
        # fit and plot fit
        #print("fit for "+dirs[d]+" component:")
        #fit_result = cfp.fit(Gaussian_func, q, pdf)
        #xfit = np.linspace(q.min(),q.max(),num=500)
        #yfit = Gaussian_func(xfit, *fit_result.popt)
        #cfp.plot(x=xfit, y=yfit, label="$\mathrm{fit:}\:"+dirs[d]+"$")
    # finally, plot everything
    cfp.plot(xlabel='$q$', ylabel='PDF', ylog=True, save=args.inputfile+"_pdf.pdf")
    # Fourier spectra power comparison
    sp = cfp.get_spectrum(dat, ncmp=ncmp)
    tot_power = sp["P_tot"].sum()/ncmp
    print("total power = "+cfp.eform(tot_power), highlight=2)
    if "P_lgt" in sp:
        lgt_power = sp["P_lgt"].sum()/ncmp
        print("logitudinal power = "+cfp.eform(lgt_power)+", relative to total: "+cfp.round(100*lgt_power/tot_power,str_ret=True)+"%", highlight=2)
    if "P_trv" in sp:
        trv_power = sp["P_trv"].sum()/ncmp
        print("transverse power  = "+cfp.eform(trv_power)+", relative to total: "+cfp.round(100*trv_power/tot_power,str_ret=True)+"%", highlight=2)
    # Fourier spectra plot
    k = sp["k"]
    ind = (k > 0) & (sp["P_tot"] > 1e-15)
    cfp.plot(x=k[ind], y=sp["P_tot"][ind], label="total")
    if "P_lgt" in sp: cfp.plot(x=k[ind], y=sp["P_lgt"][ind], label="long")
    if "P_trv" in sp: cfp.plot(x=k[ind], y=sp["P_trv"][ind], label="trans")
    spect_form = int(hdfio.read(args.inputfile, 'spect_form'))
    if spect_form == 2: # power law
        power_law_exp   = float(hdfio.read(args.inputfile, 'power_law_exp'))
        power_law_exp_2 = float(hdfio.read(args.inputfile, 'power_law_exp_2'))
        kmin = float(hdfio.read(args.inputfile, 'kmin'))
        kmid = float(hdfio.read(args.inputfile, 'kmid'))
        kmax = float(hdfio.read(args.inputfile, 'kmax'))
        x = cfp.get_1d_coords(cmin=kmin, cmax=kmax, ndim=2000, cell_centred=False)
        y = sp["P_tot"][ind][0] * x**power_law_exp
        ind = x >= kmid
        if np.sum(ind) == 0:
            ind = x == kmax
        y[ind] = y[ind][0] * (x[ind]/x[ind][0])**power_law_exp_2
        cfp.plot(x=x, y=y, label="power-law slopes: "+str(power_law_exp)+", "+str(power_law_exp_2), linestyle='dashed', color='black')
    cfp.plot(xlabel='$k$', ylabel='$P(k)$', xlim=[0.9,1.1*np.max(k)], xlog=True, ylog=True, legend_loc='lower left', save=args.inputfile+"_spectrum.pdf")

# =========================================================

def testing():

    from numpy import fft

    iter_max = 10
    iter_tol = 0.01

    eta = 0.5 # correction factor

    PL_exponent = -5/3
    kmin = 2
    kmax = 20
    mean = 1.0
    sigma = 1.0
    sigma_ln = np.sqrt(np.log((1+sigma/mean)**2))
    mean_ln = -0.5*sigma_ln**2
    shape = (10000)

    # Gaussian random field
    data = np.random.normal(mean, sigma, shape)
    obj_pdf = cfp.get_pdf(data)
    cfp.plot(obj_pdf.pdf, obj_pdf.bin_edges, type="pdf")
    cfp.plot(xlabel="x", ylabel="PDF", save="test_PDF1.pdf")

    # Target spectrum
    k = np.linspace(0, shape-1, num=shape)
    target_spec = np.zeros(shape)
    target_spec[kmin:kmax+1] = k[kmin:kmax+1]**PL_exponent
    target_spec /= np.sum(target_spec)
    cfp.plot(target_spec, k, label="PS data")
    cfp.plot(k[kmin:kmax+1]**PL_exponent, k[kmin:kmax+1], linestyle="--", label="PS target")
    cfp.plot(xlog=True, ylog=True, legend_loc="upper right", save="test_PS1.pdf")

    # N-dimensional FFT of lognormal cube
    data_ft = fft.fftn(data, norm="forward")

    # Multiply with power law spectrum
    data_ft[1:] *= np.sqrt(target_spec[1:])
    cfp.plot(np.abs(data_ft)**2, k, label="PS data")
    cfp.plot(k[kmin:kmax+1]**PL_exponent, k[kmin:kmax+1], linestyle="--", label="PS target")
    cfp.plot(xlog=True, ylog=True, legend_loc="upper right", save="test_PS2.pdf")

    # check PDF again
    data_after_scaling = np.real(fft.fftn(data, norm="backward"))
    obj_pdf = cfp.get_pdf(data_after_scaling)
    cfp.plot(obj_pdf.pdf, obj_pdf.bin_edges, type="pdf")
    cfp.plot(xlabel="x", ylabel="PDF", type="pdf", save="test_PDF2.pdf")

    stop()

    # Loop begins here
    convergence, iiter = 1, 1
    while convergence > iter_tol and iiter <= iter_max:

        # Fourier transform back (the imaginary part is negligible)
        data_new = np.real(fft.ifftn(data_ft))

        # Create lognormal
        data_ln_new = np.exp(data_new)

        # Power spectrum of lognormal is not desired power law
        data_ln_new = fft.fftn(data_ln_new)
        data_ln_spect = cfp.get_spectrum(data_ln_new)

        # Fits to the isotropic spectra
        cube_a = mt.zero_log10(cube_a[sf_lk_iso])

        # Zeroth order fit, to find best height for target spectrum
        # (kind of normalization). The multiplication by 10**fit0
        # retains zeroes in log space.
        fit0 = np.average(cube_a - target_spec_iso,
                            weights=np.r_[np.diff(k_iso), np.diff(k_iso[-2:])][sf_lk_iso])

        # Fit power spec of lognormal with second order polynomial
        fit2 = np.polyfit(mt.zero_log10(k_iso)[sf_lk_iso], cube_a, 2)

        # Corrections based on fits. fit0 needs to be multiplied
        # with the func_target_spec, otherwise 0s are not preserved.
        #p2 = np.polyval(fit2, mt.zero_log10(kmag))
        #p0 = mt.zero_log10(10 ** fit0 * self.func_target_spec(kmag))
        #corr = np.where(sf_lkmag, eta * (p0 - p2), 0)

        # Apply correction (in log space) to apodization spectrum
        #corr = target_spec * 10 ** corr

        # Re-Apodize with normalized power law
        # From here, it's the same as before the loop.
        #corr = self.norm_spec(corr * corr)
        #target_spec_old = target_spec.copy()
        #target_spec = np.sqrt(corr)
        #cube_a = cube * target_spec

        # Estimate convergence
        #convergence = np.average(mt.zero_div(abs(self._power_spec(target_spec_old) -
        #                                            self._power_spec(target_spec)),
        #                                        self._power_spec(target_spec)))
        # Some messages
        print('iteration ' + str(iiter))
        print('convergence = ' + str(convergence))
        print('')

        # Get ready for next iteration
        iiter += 1

    # A last conversion

    # Fourier transform back (the imaginary part is negligible)
    cube = np.real(fft.ifftn(cube_a))

    # Create lognormal
    cube = np.exp(cube)

    return



# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    print("test")
    testing()
    print("test done")
    exit()

    # read arguments
    parser = argparse.ArgumentParser(description='Runs and anayses output from TurbGen.')
    subparsers = parser.add_subparsers(title='sub-commands', dest='subcommand', help='see additional help with -h')

    # sub-command 'generate'
    parser_generate = subparsers.add_parser('generate', help="Call C++ TurbGen to generate turbulent field.")
    parser_generate.add_argument('-ndim', type=float, choices=[1, 1.5, 2, 2.5, 3], default=3,
        help="number of spatial dimensions; (default: %(default)s)")
    parser_generate.add_argument('-N', nargs='+', type=int, default=[64, 64, 64],
        help='number of grid cells for generated turbulent field; (default: %(default)s)')
    parser_generate.add_argument('-L', nargs='+', type=int, default=[1.0, 1.0, 1.0],
        help='physical size of box in which to generate turbulent field; (default: %(default)s)')
    parser_generate.add_argument('-kmin', type=float, default=2.0,
        help='minimum wavenumber of generated field in units of 2pi/L[X]; (default: %(default)s)')
    parser_generate.add_argument('-kmid', type=float, default=1e38,
        help='middle  wavenumber for optional 2nd power-law section (only for spect_form=2); (default: %(default)s)')
    parser_generate.add_argument('-kmax', type=float, default=20.0,
        help='maximum wavenumber of generated field in units of 2pi/L[X]; (default: %(default)s)')
    parser_generate.add_argument('-spect_form', type=int, choices=[0, 1, 2], default=2,
        help='spectral form: 0 (band/rectangle/constant), 1 (paraboloid), 2 (power law); (default: %(default)s)')
    parser_generate.add_argument('-power_law_exp', type=float, default=-2.0,
        help='if spect_form 2: power-law exponent of power-law spectrum (e.g. Kolmogorov: -5/3, Burgers: -2, Kazantsev: 1.5); (default: %(default)s)')
    parser_generate.add_argument('-power_law_exp_2', type=float, default=-2.0,
        help='if spect_form 2: power-law exponent of 2nd power-law section; (default: %(default)s)')
    parser_generate.add_argument('-angles_exp', type=float, default=1.0,
        help='if spect_form 2: angles exponent for sparse sampling (e.g., 2.0: full sampling, 0.0: healpix-like sampling); (default: %(default)s)')
    parser_generate.add_argument('-sol_weight', type=float, default=0.5,
        help='solenoidal weight: 1.0 (divergence-free field), 0.5 (natural mix), 0.0 (curl-free field); (default: %(default)s)')
    parser_generate.add_argument('-random_seed', type=int, default=140281,
        help='random seed for turbulent field; (default: %(default)s)')
    parser_generate.add_argument('-o', '--outputfile', type=str, default='TurbGen_output.h5',
        help="output filename (for HDF5 output); (default: %(default)s)")
    parser_generate.add_argument('-write_modes', default=False, action='store_true',
        help="write generating Fourier modes and amplitudes to output file; (default: %(default)s)")
    parser_generate.add_argument('-np', '--num_procs', type=int,
        help='number of cores for mpirun -np <NP> TurbGen...')

    # sub-command 'analyse'
    parser_analyse = subparsers.add_parser('analyse', help="Analyse turbulent field in file.")
    parser_analyse.add_argument('-i', '--inputfile', type=str, default='TurbGen_output.h5',
        help="input filename (HDF5); (default: %(default)s)")

    # arguments shared by all sub-commands
    for name, subp in subparsers.choices.items():
        subp.add_argument("-verbose", "--verbose", type=int, choices=[0, 1, 2], default=1,
            help="0 (no shell output), 1 (standard shell output), 2 (more shell output); (default: %(default)s)")

    args = parser.parse_args()

    # need to specify at least one sub-command
    if not args.subcommand:
        parser.print_help()
        exit()

    if args.subcommand == 'generate':
        generate(args, parser_generate)

    if args.subcommand == 'analyse':
        analyse(args, parser_analyse)
