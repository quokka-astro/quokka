import h5py
import numpy as np
import numpy.fft
from math import *
from optparse import OptionParser
import sys


def init_perturbations(n, kmin, kmax, dtype):
    kx = np.zeros(n, dtype=dtype)
    ky = np.zeros(n, dtype=dtype)
    kz = np.zeros(n, dtype=dtype)
    # perform fft k-ordering convention shifts
    for j in range(0,n[1]):
        for k in range(0,n[2]):
            kx[:,j,k] = n[0]*np.fft.fftfreq(n[0])
    for i in range(0,n[0]):
        for k in range(0,n[2]):
            ky[i,:,k] = n[1]*np.fft.fftfreq(n[1])
    for i in range(0,n[0]):
        for j in range(0,n[1]):
            kz[i,j,:] = n[2]*np.fft.fftfreq(n[2])
            
    kx = np.array(kx, dtype=dtype)
    ky = np.array(ky, dtype=dtype)
    kz = np.array(kz, dtype=dtype)
    k = np.sqrt(np.array(kx**2+ky**2+kz**2, dtype=dtype))

    # only use the positive frequencies
    inds = np.where(np.logical_and(k**2 >= kmin**2, k**2 < (kmax+1)**2))
    nr = len(inds[0])

    phasex = np.zeros(n, dtype=dtype)
    phasex[inds] = 2.*pi*np.random.uniform(size=nr)
    fx = np.zeros(n, dtype=dtype)
    fx[inds] = np.random.normal(size=nr)
    
    phasey = np.zeros(n, dtype=dtype)
    phasey[inds] = 2.*pi*np.random.uniform(size=nr)
    fy = np.zeros(n, dtype=dtype)
    fy[inds] = np.random.normal(size=nr)
    
    phasez = np.zeros(n, dtype=dtype)
    phasez[inds] = 2.*pi*np.random.uniform(size=nr)
    fz = np.zeros(n, dtype=dtype)
    fz[inds] = np.random.normal(size=nr)

    # rescale perturbation amplitude so that low number statistics
    # at low k do not throw off the desired power law scaling.
    for i in range(kmin, kmax+1):
        slice_inds = np.where(np.logical_and(k >= i, k < i+1))
        rescale = sqrt(np.sum(np.abs(fx[slice_inds])**2 + np.abs(fy[slice_inds])**2 + np.abs(fz[slice_inds])**2))
        fx[slice_inds] = fx[slice_inds]/rescale
        fy[slice_inds] = fy[slice_inds]/rescale
        fz[slice_inds] = fz[slice_inds]/rescale

    # set the power law behavior
    # wave number bins
    fx[inds] = fx[inds]*k[inds]**-(0.5*alpha)
    fy[inds] = fy[inds]*k[inds]**-(0.5*alpha)
    fz[inds] = fz[inds]*k[inds]**-(0.5*alpha)

    # add in phases
    fx = np.cos(phasex)*fx + 1j*np.sin(phasex)*fx
    fy = np.cos(phasey)*fy + 1j*np.sin(phasey)*fy
    fz = np.cos(phasez)*fz + 1j*np.sin(phasez)*fz

    return fx, fy, fz, kx, ky, kz


def normalize(fx, fy, fz):
    norm = np.sqrt(np.sum(fx**2 + fy**2 + fz**2)/np.product(n))
    fx = fx/norm
    fy = fy/norm
    fz = fz/norm
    return fx, fy, fz


def make_perturbations(n, kmin, kmax, f_solenoidal):
    fx, fy, fz, kx, ky, kz = init_perturbations(n, kmin, kmax, dtype)
    if f_solenoidal != None:
        k2 = kx**2+ky**2+kz**2
        # solenoidal part
        fxs = 0.; fys =0.; fzs = 0.
        if f_solenoidal != 0.0:
            fxs = np.real(fx - kx*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fys = np.real(fy - ky*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fzs = np.real(fz - kz*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            ind = np.where(k2 == 0)
            fxs[ind] = 0.; fys[ind] = 0.; fzs[ind] = 0.
            # need to normalize this before applying relative weighting of solenoidal / compressive components
            norm = np.sqrt(np.sum(fxs**2+fys**2+fzs**2))
            fxs = fxs/norm
            fys = fys/norm
            fzs = fzs/norm
        # compressive part
        # get a different random cube for the compressive part
        # so that we can target the RMS solenoidal fraction,
        # instead of setting a constant solenoidal fraction everywhere.
        fx, fy, fz, kx, ky, kz = init_perturbations(n, kmin, kmax, dtype)
        fxc = 0.; fyc =0.; fzc = 0.
        if f_solenoidal != 1.0:
            fxc = np.real(kx*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fyc = np.real(ky*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fzc = np.real(kz*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            ind = np.where(k2 == 0)
            fxc[ind] = 0.; fyc[ind] = 0.; fzc[ind] = 0.
            # need to normalize this before applying relative weighting of solenoidal / compressive components
            norm = np.sqrt(np.sum(fxc**2+fyc**2+fzc**2))
            fxc = fxc/norm
            fyc = fyc/norm
            fzc = fzc/norm
        # back to real space
        pertx = np.real(np.fft.ifftn(f_solenoidal*fxs + (1.-f_solenoidal)*fxc))
        perty = np.real(np.fft.ifftn(f_solenoidal*fys + (1.-f_solenoidal)*fyc))
        pertz = np.real(np.fft.ifftn(f_solenoidal*fzs + (1.-f_solenoidal)*fzc))
    else:
        # just convert to real space
        pertx = np.real(np.fft.ifftn(fx))
        perty = np.real(np.fft.ifftn(fy))
        pertz = np.real(np.fft.ifftn(fz))
    
    # subtract off COM (assuming uniform density)
    pertx = pertx-np.average(pertx)
    perty = perty-np.average(perty)
    pertz = pertz-np.average(pertz)
    # scale RMS of perturbation cube to unity
    pertx, perty, pertz = normalize(pertx, perty, pertz)
    return pertx, perty, pertz


def cut_sphere(pertx, perty, pertz, rad):
    # Make radial array
    x, y, z = np.mgrid[0:n[0], 0:n[1], 0:n[2]]
    x = x - (n[0]-1)/2.
    y = y - (n[1]-1)/2.
    z = z - (n[2]-1)/2.
    r2 = x**2+y**2+z**2
    # Get range of indices we want to set to zero, and those we want to keep
    idx0 = r2 > (rad*n[0]/2.0)**2
    idx1 = np.logical_not(idx0)
    # Zero outside the desired region
    pertx[idx0] = 0.0
    perty[idx0] = 0.0
    pertz[idx0] = 0.0
    # Recompute COM velocity, and renormalize
    pertx[idx1] = pertx[idx1]-np.average(pertx[idx1])
    perty[idx1] = perty[idx1]-np.average(perty[idx1])
    pertz[idx1] = pertz[idx1]-np.average(pertz[idx1])
    pertx, perty, pertz = normalize(pertx, perty, pertz)
    return pertx, perty, pertz


def get_erot_ke_ratio(pertx, perty, pertz, rad=-1.0):
    x, y, z = np.mgrid[0:n[0], 0:n[1], 0:n[2]]
    x = x - (n[0]-1)/2.
    y = y - (n[1]-1)/2.
    z = z - (n[2]-1)/2.
    r2 = x**2+y**2+z**2
    if rad > 0:
        idx0 = r2 > (rad*n[0]/2.0)**2
        r2[idx0] = 0.0
    erot_ke_ratio = (np.sum(y*pertz-z*perty)**2 +
                     np.sum(z*pertx-x*pertz)**2 +
                     np.sum(x*perty-y*pertx)**2)/(np.sum(r2)*np.product(n)) 
    return erot_ke_ratio


def plot_spectrum1D(pertx, perty, pertz):
    # plot the 1D power to check the scaling.
    fx = np.abs(np.fft.fftn(pertx))
    fy = np.abs(np.fft.fftn(perty))
    fz = np.abs(np.fft.fftn(pertz))
    fx = np.abs(fx)
    fy = np.abs(fy)
    fz = np.abs(fz)
    kx = np.zeros(n, dtype=dtype)
    ky = np.zeros(n, dtype=dtype)
    kz = np.zeros(n, dtype=dtype)
    # perform fft k-ordering convention shifts
    for j in range(0,n[1]):
        for k in range(0,n[2]):
            kx[:,j,k] = n[0]*np.fft.fftfreq(n[0])
    for i in range(0,n[0]):
        for k in range(0,n[2]):
            ky[i,:,k] = n[1]*np.fft.fftfreq(n[1])
    for i in range(0,n[0]):
        for j in range(0,n[1]):
            kz[i,j,:] = n[2]*np.fft.fftfreq(n[2])
    k = np.sqrt(np.array(kx**2+ky**2+kz**2,dtype=dtype))
    k1d = []
    power = []
    for i in range(kmin,kmax+1):
        slice_inds = np.where(np.logical_and(k >= i, k < i+1))
        k1d.append(i+0.5)
        power.append(np.sum(fx[slice_inds]**2 + fy[slice_inds]**2 + fz[slice_inds]**2))
        print(i,power[-1])
    import matplotlib.pyplot as plt
    plt.loglog(k1d, power)
    plt.show()


###################
# input parameters, read from command line
###################
parser = OptionParser()
parser.add_option('--kmin', dest='kmin', 
                  help='minimum wavenumber.', 
                  default=-1)
parser.add_option('--kmax', dest='kmax', 
                  help='maximum wavenumber.', 
                  default=-1)
parser.add_option('--size', dest='size', 
                  help='size of each direction of data cube.  default=256', 
                  default=256)
parser.add_option('--alpha', dest='alpha', 
                  help='negative of power law slope.  (Power ~ k^-alpha) '+
                  'supersonic turbulence is near alpha=2.  '+
                  'driving over a narrow band of two modes is often done with alpha=0', 
                  default = None)
parser.add_option('--seed', dest='seed', 
                  help='seed for random # generation.  default=0', 
                  default = 0)
parser.add_option('--f_solenoidal', dest='f_solenoidal', 
                  help='volume RMS fraction of solenoidal component of the perturbations relative to the total.  ' + 
                       'If --f_solenoidal=None, the motions are purely random.  For low wave numbers ' +
                       'the relative imporance of solenoidal to compressive may be sensitive to the ' +
                       'choice of radom seed.  It has been suggested (Federrath 2008) that ' +
                       'f_solenoidal=2/3 is the most natural driving mode and this is currently' + 
                       'the suggested best-practice.', 
                  default = -1)
parser.add_option('--sphererad', dest='rad',
                  help='if set, perturbations are set to zero outside spherical region, '+
                  'and the perturbation field is shifted and renormalized to keep the '+
                  'center of mass velocity at zero and the variance at unity; the '+
                  'spherical region cut out is centered at the center of the perturbation '+
                  'cube, and has a radius given by the value of this parameter, with sphererad = '+
                  '1 corresponding to the spherical region going all the way to the edge of the '+
                  'perturbation cube',
                  default = -1.0)

(options, args) = parser.parse_args()

# size of the data domain
n = [int(options.size), int(options.size), int(options.size)]
# range of perturbation length scale in units of the smallest side of the domain
kmin = int(options.kmin)
kmax = int(options.kmax)
print(kmin, kmax)
if kmin > kmax or kmin < 0 or kmax < 0:
    print("kmin must be < kmax, with kmin > 0, kmax > 0.  See --help.")
    sys.exit(0)
if kmax > floor(np.min(n))/2:
    print("kmax must be <= floor(size/2).  See --help.")
    sys.exit(0)
f_solenoidal = options.f_solenoidal
if f_solenoidal == "None" or f_solenoidal == "none":
    f_solenoidal = None
else:
    f_solenoidal = float(options.f_solenoidal)
    if f_solenoidal > 1. or f_solenoidal < 0.:
        print("You must choose f_solenoidal.  See --help.")
        sys.exit(0)
alpha = options.alpha
if alpha==None:
    print("You must choose a power law slope, alpha.  See --help.")
    sys.exit(0)
alpha = float(options.alpha)
if alpha < 0.:
    print("alpha is less than zero. That's probably not what you want.  See --help.")
    sys.exit(0)
seed = int(options.seed)
# data precision
dtype = np.float64
# ratio of solenoidal to compressive components
if options.f_solenoidal=="None" or options.f_solenoidal==None:
    f_solenoidal = None
else:
    f_solenoidal = min(max(float(options.f_solenoidal), 0.), 1.)
rad = float(options.rad)
if rad > 1:
    raise ValueError('sphererad is '+options.rad+', must be from 0 to 1')

###################
# begin computation
###################

np.random.seed(seed=seed)
pertx, perty, pertz = make_perturbations(n, kmin, kmax, f_solenoidal)
if rad > 0:
    pertx, perty, pertz = cut_sphere(pertx, perty, pertz, rad)
erot_ke_ratio = get_erot_ke_ratio(pertx, perty, pertz, rad)
print("erot_ke_ratio = ", erot_ke_ratio)

# hdf5 output
f = h5py.File('zdrv.hdf5', 'w')
ds = f['/'].create_dataset('pertx', n, dtype=np.float64)
ds[:] = pertx
ds = f['/'].create_dataset('perty', n, dtype=np.float64)
ds[:] = perty
ds = f['/'].create_dataset('pertz', n, dtype=np.float64)
ds[:] = pertz
f['/'].attrs['kmin'] = kmin
f['/'].attrs['kmax'] = kmax
f['/'].attrs['alpha'] = alpha
if f_solenoidal!=None: f['/'].attrs['f_solenoidal'] = f_solenoidal
if rad > 0: f['/'].attrs['sphererad'] = rad
f['/'].attrs['erot_ke_ratio'] = erot_ke_ratio
f['/'].attrs['seed'] = seed
f.close()

