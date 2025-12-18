# TurbGen - Turbulence Generator #

This tool generates a time sequence of random Fourier modes via an Ornstein-Uhlenbeck (OU) process, used to drive turbulence in hydrodynamical simulation codes. It can also generate single turbulent realisations.

Turbulence driving based on this method is currently supported by implementations in AREPO, FLASH, GADGET, PHANTOM, PLUTO.

Main files:

* 'TurbGen.h' contains the main C++ class with functions and data structures used by the generator.
* 'TurbGen.cpp' is an MPI-parallelised program that computes turbulent field(s) with specified parameters and writes the field(s) to an HDF5 file.
* 'TurbGenDemo.cpp' contains 3 basic examples for how to include and use the generator, including the generation of driving via an OU process and the generation of single turbulent fields.
* 'TurbGen.par' is the parameter file that controls the turbulence driving.

Directory 'plugins' contains examples of how TurbGen.h is used in hydrodynamical codes.

Directory 'python' contains a python frontend, which calls TurbGen to generate single turbulent fields and to analyse their properties (statistics, PDFs, power spectra).

### What are the first steps when working with this repository? ###

* Create a fork of this repo to your own bitbucket account.
* Clone the fork from your bitbucket account to your local computer.
* Make your own modifications, commit, push, etc.
* If you want (some of) your own changes to go into this main repo, please create a pull request.

### Main contact ###

* Christoph Federrath (christoph.federrath@anu.edu.au)

### Citation ###

If you use TurbGen for scientific work, please cite the following paper:

**Federrath et al. (2010), A&A 512, A81**

"Comparing the statistics of interstellar turbulence in simulations and observations: Solenoidal versus compressive turbulence forcing"

DOI: https://doi.org/10.1051/0004-6361/200912437

ADS: https://ui.adsabs.harvard.edu/abs/2010A%26A...512A..81F

BibTeX:
```bibtex
@ARTICLE{2010A&A...512A..81F,
  author = {{Federrath}, C. and {Roman-Duval}, J. and {Klessen}, R.~S. and 
            {Schmidt}, W. and {Mac Low}, M.-M.},
  title = "{Comparing the statistics of interstellar turbulence in simulations and 
            observations. Solenoidal versus compressive turbulence forcing}",
  journal = {Astronomy and Astrophysics},
  year = 2010,
  month = mar,
  volume = {512},
  eid = {A81},
  pages = {A81},
  doi = {10.1051/0004-6361/200912437},
  eprint = {0905.1060},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.SR}
}
```

### Copyright notes ###

See LICENSE file.

### Disclaimer ###

This software is provided "as is", with use at your own risk. No warranties as to performance, fitness for a particular purpose, or any other warranties are made, whether expressed or implied.
