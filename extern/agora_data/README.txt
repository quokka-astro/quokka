UNITS
-----------
Velocity: km/s
Mass: 10^9 Msun
Length: kpc


Disk properties (in above units)
----------------------------
Disk scale length: 3.43218
Disk scale height: 0.343218 (10% of scale length)
M_DISK: 42.9661
M_GAS: 8.59322  (i.e. 20 % gas fraction)
M_200: 1074.15
R_200: 205.479
Total mass: 1301.7

Halo
-----
c=10
v_200=150 km/s
lambda=0.04

For other quantities, see main.c


Number of particles and particle mass resolution
------------------------------------------------

LOW
-----
NGAS=1e5	----> mgas_res = 8.593e4 Msun
NHALO=1e5 	----> mdark_res = 1.254e7 Msun	
NDISK=1e5 	----> mdisk_res = 3.4373e5 Msun
NBULGE=1.25e4 	----> mbulge_res = 3.4373e5 Msun   (Bulge to total disk ratio=0.1, Bulge to stellar disk=0.125)

MED
-----
NGAS=1e6	----> mgas_res = 8.593e3 Msun
NHALO=1e6 	----> mdark_res = 1.254e6 Msun	
NDISK=1e6 	----> mdisk_res = 3.4373e4 Msun
NBULGE=1,25e5 	----> mbulge_res = 3.4373e4 Msun   (Bulge to total disk ratio=0.1, Bulge to stellar disk=0.125)

HIGH
-------
NGAS=1e7	----> mgas_res = 8.593e2 Msun
NHALO=1e7 	----> mdark_res = 1.254e5 Msun	
NDISK=1e7 	----> mdisk_res = 3.4373e3 Msun
NBULGE=1,25e6 	----> mbulge_res = 3.4373e3 Msun   (Bulge to total disk ratio=0.1, Bulge to stellar disk=0.125)


Structure of ASCII files
------------------------------
Gas particle (gas.dat): x,y,z,vx,vy,vz,mgas,u_gas
Dark matter halo (halo.dat): x,y,z,vx,vy,vz,mdark
Stellar disk (disk.dat): x,y,z,vx,vy,vz,mdisk
Stellar bulge (bulge.dat): x,y,z,vx,vy,vz,mbulge

Note that u_gas is the specific internal energy of the gas, see Springel (2000). The initial specific energy has been computed by MakeDisk to ensure a global equilibrium. We suggest here that all codes use 10^4 K as initial temperature instead.


Initial stellar ages
-------------------------------------
Assume the initial stellar ages to be infinite, i.e. they contribute nothing to the feedback budget.


For AMR codes
-----------------------------
The gas can be initialized analytically following the profile adopted by MakeDisk:

rho(r,z)=rho_0*exp(-r/r_d)*exp(-abs(z)/z_d)

where 

rho_0=M_GAS/4./pi/r_d^2/z_d

The scale radius r_d, scale height z_d and mass M_GAS of the gas disk are provided in the "Disk properties" section. In RAMSES we truncate the disk at a maximum radius r_max and z_max, where we typically choose r_max=20 kpc and z_max=3 kpc. If chosen large enough, the results are very insensitive to the latter parameters.

In RAMSES we set the initial gas temperature to T=1e4 K and at solar metallicty. We impose a very tenuous gas halo of uniform density (d_halo ~ 10^-6 cm-3). The initial temperature should be set to T=10^6 K, and the metal content should be primordial (Z=0 Z_odot). The gas velocity should be zero in the halo.

The circular velocity profile of the system is provided in the file "vcirc.dat" where the first column is the radius in kpc and the second vcirc in km/s. This circular velocity has to be used to set your disk initial rotational velocity profile in centrifugal equilibrium.

UPDATE: The circular velocity from the IC generator is not exactly equal to the final rotational velocity of the SPH particles (differences of ~5% in the central few kpc). The new file vcirc_SPH.dat is the rotational velocity profiles binned from the SPH particle distribution within |z|<h_disk. 


Suggested run parameters and Jeans support
--------------------------------------------
For all codes, the gas mass resolution has to be as close as possible to, in the case of the low resolution run, 8.593e4 Msun. Using a Jeans mass 64x larger and the corresponding Jeans length, this results in a spatial resolution of 80 pc for the low resolution run. 

LOW: 	N_gas=1d5, mgas_res=8.593d4 msun, Delta x=80 pc
MED:	N_gas=1d6, mgas_res=8.593d3 msun, Delta x=40 pc
HIGH:	N_gas=1d7, mgas_res=8.593d2 msun, Delta x=20 pc

We suggest that all codes, if possible, use this spatial resolution, or as close to it as possible. We also suggest, when applicable, to use a pressure floor, so that the Jeans length is resolved at all time by at least N_jeans cells (N_jeans=4 was suggested by Truelove et al. 1997, and is also what we suggest here).

Assuming \lambda_Jeans = N_jeans*\Delta x, this gives:  P_jeans = G * N_jeans^2 * \rho^2 * \Delta x^2 / gamma / pi

where Delta x refers to the cell size on the finest refinement level (in AMR codes).

NB! (7/7-2015): The equation for P_jeans had a typo in a previous version of this document (multiplication by pi rather than division by pi).


Refinement
--------------------------------------------
What we have recommended to AMR users is that we want refinement to be done in a Lagrangian fashion similar to SPH. What we mean by this is that the fundamental mass to refine on is mgas_res (i.e. a cell is split into an Oct, 8 cells) where mgas_res is the gas SPH particle mass for the different runs. This is a good starting point for AMR codes to attempt to do something similar to SPH methods. We continue to refine the grid down to the suggested resolution limit where the pressure floor kicks in. Once we get into details and start comparing our simulations we will study if a more suitable refinement criteria is necessary to match results. 

In RAMSES, we refine, in addition to the above baryon criterion, if the number of dark matter/star particle particles present in a cell exceeds 8 (keeps the particle number/cell to ~1 on average).


Star formation 
—————————————————
We have agreed to use a “Schmidt-like” star formation law:

\dot{rho}_*=\epsilon_ff \rho_gas/t_ff for \rho_gas > \rho_SF

- t_ff is the gas free-fall time
- \epsilon_ff is the efficiency per free-fall time 
- \rho_SF is the star formation density threshold 

Which values to pick for \epsilon_ff and \rho_SF is a difficult subject: For the low resolution run we recommend a fiducial setting of \epsilon_ff=1% and rho_SF=10 H/cc. The threshold is where the Jeans polytrope intersects a typical equilibrium T-rho equation of state in the RAMSES disks (T_J~1800 K, n_J~8 H/cc). In the higher resolution runs this should be increased. In addition to these fiducial settings, the groups are encouraged to experiment with different values, especially in conjunction with their feedback recipes.
