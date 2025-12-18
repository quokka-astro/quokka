!***************************************************************************
! User changes file 'forcing_generator.inp'                                !
! see the MAIN routine for further details (last routine)                  !
! Please see Federrath et al. (2010, A&A 512, A81) for details and cite :) !
!***************************************************************************

Module Stir_data
!!
!! DESCRIPTION
!!
!!  Writes the turbulence driving mode sequence to a file that is then read by
!!  the hydro code (e.g., AREPO, FLASH, GADGET, PHANTOM, PLUTO, QUOKKA, etc.)
!!  to generate the physical turbulent acceleration field as a time sequence.
!!  The driving sequence follows an Ornstein-Uhlenbeck (OU) process.
!!
!!  For example applications see Federrath et al. (2008, ApJ 688, L79);
!!  Federrath et al. (2010, A&A 512, A81); Federrath (2013, MNRAS 436, 1245);
!!  Federrath et al. (2021, Nature Astronomy 5, 365)
!!
!! AUTHOR: Christoph Federrath, 2008-2022
!!
!!***

integer, parameter :: st_maxmodes = 100000
real, parameter    :: twopi = 8 * atan(1.0)

! OU variance corresponding to decay time and energy input rate
real, save :: st_OUvar

! number of modes
integer, save :: st_nmodes

real, save, dimension(3,st_maxmodes) :: st_mode, st_aka, st_akb
real, save, dimension(6*st_maxmodes) :: st_OUphases
real, save, dimension(  st_maxmodes) :: st_ampl

integer, save  :: ndim ! number of spatial dimensions
real, save     :: xmin, xmax, ymin, ymax, zmin, zmax
real, save     :: st_decay
real, save     :: st_energy
real, save     :: st_stirmin, st_stirmax
real, save     :: st_solweight, st_solweightnorm
real, save     :: velocity
real, save     :: st_power_law_exp
real, save     :: st_angles_exp
integer, save  :: st_seed
integer, save  :: st_spectform

contains

!!******************************************************
subroutine init_stir()
!!******************************************************

  implicit none

  logical, parameter :: Debug = .false.

  integer :: ikxmin, ikxmax, ikymin, ikymax, ikzmin, ikzmax
  integer :: ikx, iky, ikz, st_tot_nmodes, orig_seed
  real    :: kx, ky, kz, k, kc, Lx, Ly, Lz, amplitude, parab_prefact

  ! applies in case of power law (st_spectform .eq. 2)
  integer :: iang, nang, ik, ikmin, ikmax
  real :: rand, phi, theta

  orig_seed = st_seed ! save the original seed for printing below

  ! initialize some variables, allocate random seed
  st_OUvar = sqrt(st_energy/st_decay)

  ! this is for st_spectform = 1 (paraboloid) only
  ! prefactor for amplitude normalistion to 1 at kc = 0.5*(st_stirmin+st_stirmax)
  parab_prefact = -4.0 / (st_stirmax-st_stirmin)**2

  ! characteristic k for scaling the amplitude below
  kc = st_stirmin
  if (st_spectform .eq. 1) kc = 0.5*(st_stirmin+st_stirmax)

  ! this makes the rms force constant, irrespective of the solenoidal weight
  if (ndim .eq. 3) st_solweightnorm = sqrt(3.0/3.0)*sqrt(3.0)*1.0/sqrt(1.0-2.0*st_solweight+3.0*st_solweight**2.0)
  if (ndim .eq. 2) st_solweightnorm = sqrt(3.0/2.0)*sqrt(3.0)*1.0/sqrt(1.0-2.0*st_solweight+2.0*st_solweight**2.0)
  if (ndim .eq. 1) st_solweightnorm = sqrt(3.0/1.0)*sqrt(3.0)*1.0/sqrt(1.0-2.0*st_solweight+1.0*st_solweight**2.0)

  ikxmin = 0
  ikymin = 0
  ikzmin = 0

  ikxmax = 256
  ikymax = 0
  ikzmax = 0
  if (ndim .gt. 1) ikymax = 256
  if (ndim .gt. 2) ikzmax = 256

  Lx = xmax - xmin
  Ly = ymax - ymin
  Lz = zmax - zmin

  ! determine the number of required modes (in case of full sampling)
  st_nmodes = 0
  do ikx = ikxmin, ikxmax
      kx = twopi * ikx / Lx
      do iky = ikymin, ikymax
          ky = twopi * iky / Ly
          do ikz = ikzmin, ikzmax
              kz = twopi * ikz / Lz
              k = sqrt( kx**2 + ky**2 + kz**2 )
              if ((k .ge. st_stirmin) .and. (k .le. st_stirmax)) then
                 st_nmodes = st_nmodes + 1
                 if (ndim .gt. 1) st_nmodes = st_nmodes + 1
                 if (ndim .gt. 2) st_nmodes = st_nmodes + 2
              endif
          enddo
      enddo
  enddo
  st_tot_nmodes = st_nmodes
  if (st_spectform .ne. 2) write(*,'(A,I8,A)') 'Generating ', st_tot_nmodes, ' driving modes...'

  st_nmodes = 0

  ! ===================================================================
  ! === for band and parabolic spectrum, use the standard full sampling
  if (st_spectform .ne. 2) then

      open(13, file='power.txt', action='write')
      write(13, '(6A16)') 'k', 'power', 'amplitude', 'kx', 'ky', 'kz'
      close(13)

      ! loop over all kx, ky, kz to generate driving modes
      do ikx = ikxmin, ikxmax
          kx = twopi * ikx / Lx

          do iky = ikymin, ikymax
              ky = twopi * iky / Ly

              do ikz = ikzmin, ikzmax
                  kz = twopi * ikz / Lz

                  k = sqrt( kx**2 + ky**2 + kz**2 )

                  if ((k .ge. st_stirmin) .and. (k .le. st_stirmax)) then

                     if ((st_nmodes + 2**(ndim-1)) .gt. st_maxmodes) then

                        print *, 'init_stir:  st_nmodes = ', st_nmodes, ' maxstirmodes = ',st_maxmodes
                        print *, 'Too many stirring modes'
                        exit

                     endif

                     if (st_spectform .eq. 0) amplitude = 1.0                                ! Band
                     if (st_spectform .eq. 1) amplitude = abs(parab_prefact*(k-kc)**2.0+1.0) ! Parabola

                     ! note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D) 
                     amplitude = sqrt(amplitude) * (kc/k)**((ndim-1)/2.0)

                     st_nmodes = st_nmodes + 1

                     st_ampl(st_nmodes) = amplitude
                     if (Debug) print *, "init_stir:  st_ampl(",st_nmodes,") = ", st_ampl(st_nmodes)

                     st_mode(1,st_nmodes) = kx
                     st_mode(2,st_nmodes) = ky
                     st_mode(3,st_nmodes) = kz

                     if (ndim.gt.1) then

                       st_nmodes = st_nmodes + 1

                       st_ampl(st_nmodes) = amplitude
                       if (Debug) print *, "init_stir:  st_ampl(",st_nmodes,") = ", st_ampl(st_nmodes)

                       st_mode(1,st_nmodes) = kx
                       st_mode(2,st_nmodes) =-ky
                       st_mode(3,st_nmodes) = kz

                     endif

                     if (ndim.gt.2) then

                       st_nmodes = st_nmodes + 1

                       st_ampl(st_nmodes) = amplitude
                       if (Debug) print *, "init_stir:  st_ampl(",st_nmodes,") = ", st_ampl(st_nmodes)

                       st_mode(1,st_nmodes) = kx
                       st_mode(2,st_nmodes) = ky
                       st_mode(3,st_nmodes) =-kz

                       st_nmodes = st_nmodes + 1

                       st_ampl(st_nmodes) = amplitude
                       if (Debug) print *, "init_stir:  st_ampl(",st_nmodes,") = ", st_ampl(st_nmodes)

                       st_mode(1,st_nmodes) = kx
                       st_mode(2,st_nmodes) =-ky
                       st_mode(3,st_nmodes) =-kz

                     endif

                     open(13, file='power.txt', action='write', access='append')
                     write(13, '(6E16.6)') k, amplitude**2*(k/kc)**(ndim-1), amplitude, kx, ky, kz
                     close(13)

                     if (mod(st_nmodes,1000) .eq. 0) &
                          & write(*,'(A,I6,A,I6,A)') ' ...', st_nmodes, ' of total ', st_tot_nmodes, ' modes generated...'

                  endif ! in k range
              enddo ! ikz
          enddo ! iky
      enddo ! ikx
  endif ! st_spectform .ne. 2

  ! ===============================================================================
  ! === for power law, generate modes that are distributed randomly on the k-sphere
  ! === with the number of angles growing ~ k^st_angles_exp
  if (st_spectform .eq. 2) then

      write(*,'(A,I8,A)') 'There would be ', st_tot_nmodes, ' driving modes, if k-space were fully sampled (st_angles_exp = 2.0)...'
      write(*,'(A,F4.2)') 'Here we are using st_angles_exp = ', st_angles_exp

      ! initialize additional random numbers (uniformly distributed) to randomise angles
      rand = ran2(-st_seed) ! initialise Numerical Recipes rand gen (call with negative integer)

      ! loop between smallest and largest k
      ikmin = max(1, nint(st_stirmin*Lx/twopi))
      ikmax =        nint(st_stirmax*Lx/twopi)

      write(*,'(A,I4,A,I4,A)') 'Generating driving modes within k = [', ikmin, ', ', ikmax, ']'

      do ik = ikmin, ikmax

          nang = 2**ndim * ceiling(ik**st_angles_exp)
          print *, 'ik, number of angles = ', ik, nang
          do iang = 1, nang

            phi = twopi * ran2(st_seed) ! phi = [0,2pi] sample the whole sphere
            if (ndim .eq. 1) then
                if (phi .lt. twopi/2) phi = 0.0
                if (phi .ge. twopi/2) phi = twopi/2.0
            endif
            theta = twopi/4.0
            if (ndim .gt. 2) theta = acos(1.0 - 2.0*ran2(st_seed)) ! theta = [0,pi] sample the whole sphere

            if (Debug) print *, 'entering: theta=', theta, ', phi=', phi

            rand = ik + ran2(st_seed) - 0.5
            kx = twopi * nint(rand*sin(theta)*cos(phi)) / Lx
            ky = 0.0
            if (ndim .gt. 1) ky = twopi * nint(rand*sin(theta)*sin(phi)) / Ly
            kz = 0.0
            if (ndim .gt. 2) kz = twopi * nint(rand*cos(theta)         ) / Lz

            k = sqrt( kx**2 + ky**2 + kz**2 )

            if ((k .ge. st_stirmin) .and. (k .le. st_stirmax)) then

               if ((st_nmodes + 2**(ndim-1)) .gt. st_maxmodes) then

                  print *, 'init_stir:  st_nmodes = ', st_nmodes, ' maxstirmodes = ',st_maxmodes
                  print *, 'Too many stirring modes'
                  exit

               endif

               amplitude = (k/kc)**st_power_law_exp ! Power law

               ! note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
               ! ...and correct for the number of angles sampled relative to the full sampling (k^2 per k-shell in 3D)
               amplitude = sqrt( amplitude * (ik**(ndim-1)/real(nang)*4.0*sqrt(3.0)) ) * (kc/k)**((ndim-1)/2.0)

               st_nmodes = st_nmodes + 1

               st_ampl(st_nmodes) = amplitude
               if (Debug) print *, "init_stir:  st_ampl(",st_nmodes,") = ", st_ampl(st_nmodes)

               st_mode(1,st_nmodes) = kx
               st_mode(2,st_nmodes) = ky
               st_mode(3,st_nmodes) = kz

               open(13,file='power.txt',action='write',access='append')
               write(13,'(6E16.6)') k, amplitude**2*(k/kc)**(ndim-1), amplitude, kx, ky, kz
               close(13)

               if (mod(st_nmodes,1000) .eq. 0) &
                    & write(*,'(A,I6,A)') ' ...', st_nmodes, ' modes generated...'

            endif ! in k range

          enddo ! loop over angles
      enddo ! loop over k
  endif ! st_spectform .eq. 2

  write (*,'(A,I6,A)') 'Initialized ',st_nmodes,' modes for stirring.'
  if (st_spectform.eq.0) write (*,'(A,I2,A)') ' spectral form         = ', st_spectform, ' (Band)'
  if (st_spectform.eq.1) write (*,'(A,I2,A)') ' spectral form         = ', st_spectform, ' (Parabola)'
  if (st_spectform.eq.2) write (*,'(A,I2,A)') ' spectral form         = ', st_spectform, ' (Power Law)'
  write (*,'(A,ES10.3)') ' solenoidal weight     = ', st_solweight
  write (*,'(A,ES10.3)') ' st_solweightnorm      = ', st_solweightnorm
  write (*,'(A,ES10.3)') ' velocity              = ', velocity
  write (*,'(A,ES10.3)') ' auto-correlation time = ', st_decay
  write (*,'(A,ES10.3)') ' stirring energy       = ', st_energy
  write (*,'(A,ES10.3)') ' minimum wavenumber    = ', st_stirmin
  write (*,'(A,ES10.3)') ' maximum wavenumber    = ', st_stirmax
  write (*,'(A,I8)')    ' random seed           = ', orig_seed

  return

end subroutine init_stir


!!******************************************************
subroutine st_ounoiseinit(vector, vectorlength, variance)
!!******************************************************
!! initialize pseudo random sequence for the OU process

  implicit none

  real, intent (INOUT)    :: vector (:)
  integer, intent (IN)    :: vectorlength
  real, intent (IN)       :: variance
  real                    :: grnval
  integer                 :: i

  do i = 1, vectorlength
     call st_grn (grnval)
     vector (i) = grnval * variance
  end do

  return

end subroutine st_ounoiseinit


!!******************************************************
subroutine st_ounoiseupdate(vector, vectorlength, variance, dt, ts)
!!******************************************************
!! update Ornstein-Uhlenbeck sequence
!!
!! The sequence x_n is a Markov process that takes the previous value,
!! weights by an exponential damping factor with a given correlation
!! time 'ts', and drives by adding a Gaussian random variable with
!! variance 'variance', weighted by a second damping factor, also
!! with correlation time 'ts'. For a timestep of dt, this sequence
!! can be written as
!!
!!     x_n+1 = f x_n + sigma * sqrt (1 - f**2) z_n,
!!
!! where f = exp (-dt / ts), z_n is a Gaussian random variable drawn
!! from a Gaussian distribution with unit variance, and sigma is the
!! desired variance of the OU sequence.
!!
!! The resulting sequence should satisfy the properties of zero mean,
!! and stationarity (independent of portion of sequence) RMS equal to
!! 'variance'. Its power spectrum in the time domain can vary from
!! white noise to "brown" noise (P (f) = const. to 1 / f^2).
!!
!! References:
!!   Bartosch (2001)
!!   Eswaran & Pope (1988)
!!   Schmidt et al. (2009)
!!   Federrath et al. (2010, A&A 512, A81)
!!
!! ARGUMENTS
!!
!!   vector :       vector to be updated
!!   vectorlength : length of vector to be updated
!!   variance :     variance of the distribution
!!   dt :           timestep
!!   ts :           autocorrelation time
!!
!!***

  implicit none

  real, intent (INOUT) :: vector (:)
  integer, intent (IN) :: vectorlength
  real, intent (IN)    :: variance, dt, ts
  real                 :: grnval, damping_factor
  integer              :: i

  damping_factor = exp(-dt/ts)

  do i = 1, vectorlength
     call st_grn (grnval)
     vector (i) = vector (i) * damping_factor + variance *   &
          sqrt(1.0 - damping_factor**2.0) * grnval
  end do

  return

end subroutine st_ounoiseupdate


!!******************************************************
subroutine st_grn(grnval)
!!******************************************************
!!
!! DESCRIPTION
!!
!!  Subroutine draws a number randomly from a Gaussian distribution
!!  with the standard uniform distribution function "random_number"
!!  using the Box-Muller transformation in polar coordinates. The
!!  resulting Gaussian has unit variance.
!!
!!***

  implicit none

  real, intent (OUT) :: grnval
  real               :: r1, r2, g1

  r1 = 0.0; r2 = 0.0;
  r1 = ran1s(st_seed)
  r2 = ran1s(st_seed)
  g1 = sqrt(2.0*log(1.0/r1))*cos(twopi*r2)

  grnval = g1

  return

end subroutine st_grn


!!************** Numerical recipes ran1s ***************
function ran1s(idum)
!!******************************************************
  integer idum, IA, IM, IQ, IR, NTAB
  real ran1s, AM, EPS, RNMX
  parameter (IA=16807, IM=2147483647, AM=1.0/IM, IQ=127773, IR=2836, &
             NTAB=32, EPS=1.2e-7, RNMX=1.0-EPS)
  integer k, iy
  if (idum .le. 0) then
    idum = max(-idum, 1)
  endif
  k = idum/IQ
  idum = IA*(idum-k*IQ)-IR*k
  if (idum .lt. 0) idum = idum+IM
  iy = idum
  ran1s = min(AM*iy, RNMX)
  return
end function ran1s


!!************** Numerical recipes ran2 ****************
!! Long period (> 2 x 10^18) random number generator of L'Ecuyer
!! with Bays-Durham shuffle and added safeguards.
!! Returns a uniform random deviate between 0.0 and 1.0 (exclusive of the endpoint values).
!! Call with idum a negative integer to initialize; thereafter,
!! do not alter idum between successive deviates in a sequence.
!! RNMX should approximate the largest floating value that is less than 1.
FUNCTION ran2(idum)
!!******************************************************
    INTEGER idum, IM1, IM2, IMM1, IA1, IA2, IQ1, IQ2, IR1, IR2, NTAB, NDIV
    REAL ran2, AM, EPS, RNMX
    PARAMETER (IM1=2147483563, IM2=2147483399, AM=1.0/IM1, IMM1=IM1-1, &
               IA1=40014, IA2=40692, IQ1=53668, IQ2=52774, IR1=12211, IR2=3791, &
               NTAB=32, NDIV=1+IMM1/NTAB, EPS=1.2e-7, RNMX=1.0-EPS)
    INTEGER idum2, j, k, iv(NTAB), iy
    SAVE iv, iy, idum2
    DATA idum2/123456789/, iv/NTAB*0/, iy/0/
    if (idum .le. 0) then
        idum = max(-idum, 1)
        idum2 = idum
        do j = NTAB+8, 1, -1
            k = idum/IQ1
            idum = IA1*(idum-k*IQ1)-k*IR1
            if (idum .lt. 0) idum = idum+IM1
            if (j .le. NTAB) iv(j) = idum
        end do
        iy = iv(1)
    endif
    k = idum/IQ1
    idum = IA1*(idum-k*IQ1)-k*IR1
    if (idum .lt. 0) idum = idum+IM1
    k = idum2/IQ2
    idum2 = IA2*(idum2-k*IQ2)-k*IR2
    if (idum2 .lt. 0) idum2 = idum2+IM2
    j = 1+iy/NDIV
    iy = iv(j)-idum2
    iv(j) = idum
    if (iy .lt. 1) iy = iy+IMM1
    ran2 = min(AM*iy, RNMX)
    return

END FUNCTION ran2


!!******************************************************
subroutine st_calcPhases()
!!******************************************************
!!
!! DESCRIPTION
!!
!!     This routine updates the stirring phases from the OU phases.
!!     It copies them over and applies the projection operator.
!!
!!***

  implicit none

  real                 :: ka, kb, kk, diva, divb, curla, curlb
  integer              :: i,j
  logical, parameter   :: Debug = .false.

  do i = 1, st_nmodes
     ka = 0.0
     kb = 0.0
     kk = 0.0
     do j=1, ndim
        kk = kk + st_mode(j,i)*st_mode(j,i)
        ka = ka + st_mode(j,i)*st_OUphases(6*(i-1)+2*(j-1)+1+1)
        kb = kb + st_mode(j,i)*st_OUphases(6*(i-1)+2*(j-1)+0+1)
     enddo
     do j=1, ndim
         diva  = st_mode(j,i)*ka/kk
         divb  = st_mode(j,i)*kb/kk
         curla = st_OUphases(6*(i-1)+2*(j-1) + 0 + 1) - divb
         curlb = st_OUphases(6*(i-1)+2*(j-1) + 1 + 1) - diva
         st_aka(j,i) = st_solweight*curla+(1.0-st_solweight)*divb
         st_akb(j,i) = st_solweight*curlb+(1.0-st_solweight)*diva
! purely compressive
!         st_aka(j,i) = st_mode(j,i)*kb/kk
!         st_akb(j,i) = st_mode(j,i)*ka/kk
! purely solenoidal
!         st_aka(j,i) = bjiR - st_mode(j,i)*kb/kk
!         st_akb(j,i) = bjiI - st_mode(j,i)*ka/kk
        if (Debug) then
            print *, 'st_mode(dim=',j,',mode=',i,') = ', st_mode(j,i)
            print *, 'st_aka (dim=',j,',mode=',i,') = ', st_aka(j,i)
            print *, 'st_akb (dim=',j,',mode=',i,') = ', st_akb(j,i)
        endif
     enddo
  enddo

  return

end subroutine st_calcPhases


!!******************************************************
subroutine write_forcing_file(outfile, nsteps, step, time, end_time)
!!******************************************************
!!
!! DESCRIPTION
!!
!!  Writes time dependent modes, phases and amplitudes to file.
!!  This file is ultimately read by a hydro code to generate the physical
!!  turbulence acceleration field from the Fourier mode sequence in this file.
!!
!!***

  implicit none

  character (len=80), intent(in) :: outfile
  integer, intent(in)            :: nsteps, step
  real, intent(in)               :: time, end_time
  logical, save                  :: first_call = .true.
  integer                        :: iostat

  if (first_call) then
     open (unit=42, file=outfile, iostat=iostat, status='REPLACE', action='WRITE', &
           access='SEQUENTIAL', form='UNFORMATTED')
     ! header contains number of times and number of modes, end time, autocorrelation time, ...
     if (iostat.eq.0) then
        write (unit=42) nsteps, st_nmodes, end_time, st_decay, st_energy, st_solweight, &
                        velocity, st_stirmin, st_stirmax, st_spectform
     else
        write (*,'(2A)') 'could not create file for write. filename: ', trim(outfile)
     endif
     close (unit=42)
     first_call = .false.
  endif

  open (unit=42, file=outfile, iostat=iostat, status='OLD', action='WRITE', &
        position='APPEND', access='SEQUENTIAL', form='UNFORMATTED')

  if (iostat.eq.0) then
     write (unit=42) step, time, &
                     st_mode    (:, 1:  st_nmodes), &
                     st_aka     (:, 1:  st_nmodes), &
                     st_akb     (:, 1:  st_nmodes), &
                     st_ampl    (   1:  st_nmodes), &
                     st_OUphases(   1:6*st_nmodes)
  else
     write (*,'(2A)') 'could not open file for write. filename: ', trim(outfile)
  endif

  close (unit=42)

  return

end subroutine write_forcing_file


end Module Stir_data


!!******************************************************
!!*********************** MAIN *************************
!!******************************************************
program generate_forcing_file

  use Stir_data

  implicit none

  real, save                :: end_time, time, dt
  integer, save             :: step, nsteps, m, nsteps_per_turnover_time
  character (len=200), save :: outfilename
  character (len=80), save  :: power_law_exp_string, vel_string, solweight_string, seed_string
  logical                   :: print_stuff_to_shell = .true.
  real, parameter           :: eps = tiny(0.0)
  real                      :: Lx, k_driv, k_min, k_max, st_energy_coeff

  ! ******** READING NAMELIST for user input ******** !
  namelist /input/ ndim, xmin, xmax, ymin, ymax, zmin, zmax, velocity, k_driv, k_min, k_max, &
                   st_solweight, st_spectform, st_power_law_exp, st_angles_exp, &
                   st_energy_coeff, st_seed, end_time, nsteps_per_turnover_time
  open(1, file='forcing_generator.inp', status='old')
  read(1, input)
  close(1)
  ! ************************************************* !

  Lx = xmax-xmin                                 ! Length of box in x; used for normalisations below
  st_stirmin = (k_min-eps) * twopi / Lx          ! Minimum driving wavenumber <~  k_min * 2pi / Lx
  st_stirmax = (k_max+eps) * twopi / Lx          ! Maximum driving wavenumber >~  k_max * 2pi / Lx
  st_decay = Lx / k_driv / velocity              ! Auto-correlation time, t_turb = Lx / k_driv / velocity;
                                                 ! aka turbulent turnover (crossing) time; note that k_driv is in units of 2pi/Lx
  st_energy = st_energy_coeff * velocity**3 / Lx ! Energy input rate => driving amplitude ~ sqrt(energy/decay)
                                                 ! Note that energy input rate ~ velocity^3 * L_box^-1
                                                 ! st_energy_coeff needs to be adjusted to approach actual target velocity dispersion
  end_time = end_time * st_decay                 ! Forcing sequence end time (in units of turnover times);
                                                 ! increase, if more turnover times are needed
  nsteps = nsteps_per_turnover_time * &          ! Total number of time dumps to update the driving patterns
                    nint(end_time/st_decay)      ! until end_time is reached (default: 10 samples per turnover time)

  power_law_exp_string = ''
  if (st_spectform .eq. 2) then
    write (power_law_exp_string, '(SP,F5.2)') st_power_law_exp
    power_law_exp_string = '_pl'//trim(power_law_exp_string)
  endif
  write (vel_string, '(ES8.2)') velocity
  write (solweight_string, '(F3.1)') st_solweight
  write (seed_string, '(I6.6)') st_seed
  ! output filename read by FLASH
  outfilename  = 'turb'//trim(power_law_exp_string)//'_v'//trim(adjustl(vel_string))// &
    & '_zeta'//trim(solweight_string)//'_seed'//trim(seed_string)//'.dat'

  print_stuff_to_shell = .false. ! print additional information
  ! =====================================================================

  call init_stir()
  call st_ounoiseinit(st_OUphases, 6*st_nmodes, st_OUvar)

  write(*,'(A)') '**********************************************************************************************************'

  dt = end_time / nsteps
  time = 0.0
  do step = 0, nsteps
     call st_ounoiseupdate(st_OUphases, 6*st_nmodes, st_OUvar, dt, st_decay)
     call st_calcPhases()
     call write_forcing_file(outfilename, nsteps, step, time, end_time)
     if (print_stuff_to_shell) then
        write(*,'(A,I6,A,ES10.3)') 'step = ', step, '  time = ', time
        do m = 1, st_nmodes
           write(*,'(A,I4,A,3(1X,ES9.2),A,3(1X,ES9.2),A,3(1X,ES9.2),A,1(1X,ES9.2))') &
                 'm=', m, '  st_mode:', st_mode(:,m), '  st_aka:', st_aka(:,m), '  st_akb:', st_akb(:,m), '  st_ampl:', st_ampl(m)
        enddo
        write(*,'(A)') '**********************************************************************************************************'
     endif
     time = time + dt
  enddo

  write(*,'(3A,I6,A,ES10.3,A)') 'Outputfile "', trim(outfilename), &
        '" containing ', nsteps, ' stirring times with end time ', end_time, ' written. Finished.'

end program generate_forcing_file

!!******************************************************
