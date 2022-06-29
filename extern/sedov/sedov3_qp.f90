      program sedov3
      implicit none

! exercises the sedov solver

! declare
      character*80       :: outfile,string
      integer            :: i,nstep,iargc
      integer, parameter :: nmax = 1000
      real*16            :: time,zpos(nmax), &
                            eblast,rho0,omega,vel0,ener0,pres0,cs0,gamma, &
                            xgeom,rshock,rho2,u2,e2,p2,cs2,rvacuum, &
                            den(nmax),ener(nmax),pres(nmax),vel(nmax), &
                            cs(nmax), &
                            zlo,zhi,zstep,value


! popular formats
 01   format(1x,t5,a,t8,a,t32,a,t56,a,t80,a,t104,a,t128,a,t132,a)
 02   format(1x,i4,1p8e12.4)
 03   format(1x,i4,1p8e24.16)



! if your compiler/linker handles command line arguments
! get the number of spatial points, blast energy, geometry type,
! density exponent,  and output file name

      i = iargc()

      if (i .eq. 0) then
       nstep = 120
       eblast = 0.851072q0
       xgeom  = 3.0q0
       omega  = 0.0q0
       gamma  = 1.4q0
       outfile = 'spherical_standard_omega0p00.dat'

      else if (i .eq. 6) then

      call getarg(1,string)
      nstep = int(value(string))

      call getarg(2,string)
      eblast = value(string)

      call getarg(3,string)
      xgeom = value(string)

      call getarg(4,string)
      omega = value(string)

      call getarg(5,string)
      gamma = value(string)

      call getarg(6,outfile)

     else
      stop 'pass in 6 parameters: nstep eblast xgeom omega gamma outfile'
     end if

! input parameters in cgs
      time   = 1.0q0
      rho0   = 1.0q0
      vel0   = 0.0q0
      ener0  = 0.0q0
      pres0  = 0.0q0
      cs0    = 0.0q0



! or explicitly set stuff
! standard cases
! spherical constant density should reach r=1 at t=1
      nstep = 120
      eblast = 0.851072q0
      xgeom  = 3.0q0
      omega  = 0.0q0
      outfile = 'spherical_standard_omega0p00.dat'

! cylindrical constant density should reach r=1 at t=1
!      nstep = 120
!      eblast = 0.311357q0
!      xgeom  = 2.0q0
!      omega  = 0.0q0
!      outfile = 'cylindrical_standard_omega0p00.dat'

! planar constant density should reach x=1 at t=1
!      nstep = 120
!      eblast = 0.0673185q0
!      xgeom  = 1.0q0
!      omega  = 0.0q0
!      outfile = 'planar_standard_omega0p00.dat'


! singular cases
! spherical should reach r=1 at t=1
!      nstep = 120
!      eblast = 4.90875q0
!      xgeom  = 3.0q0
!      omega  = 7.0q0/3.0q0
!      outfile = 'spherical_singular_omega2p33.dat'

! cylindrical should reach r=0.75 at t=1
!      nstep = 120
!      eblast = 2.45749q0
!      xgeom  = 2.0q0
!      omega  = 5.0q0/3.0q0
!      outfile = 'cylindrical_singular_omega1p66.dat'


! vacuum cases
! spherical should reach r=1 at t=1
!      nstep = 120
!      eblast = 5.45670q0
!      xgeom  = 3.0q0
!      omega  = 2.4q0
!      outfile = 'spherical_vacuum_omega2p40.dat'

! cylindrical should reach r=0.75 at t=1
!      nstep = 120
!      eblast = 2.67315q0
!      xgeom  = 2.0q0
!      omega  = 1.7q0
!      outfile = 'cylindrical_vacuum_omega1p70.dat'



! number of grid points, spatial domain, spatial step size
! to match rage output, use the mid-cell points
      zlo   = 0.0q0
      zhi   = 1.2q0
      zstep = (zhi - zlo)/float(nstep-1)
      do i=1,nstep
!       zpos(i)   = zlo + 0.5q0*zstep + float(i-1)*zstep
       zpos(i)   = zlo + float(i-1)*zstep
      enddo


! get the solution for all spatial points at once

       call sed_1d(time,nstep,zpos, &
                   eblast,omega,xgeom, &
                   rho0,vel0,ener0,pres0,cs0,gamma, &
                   rshock,rho2,u2,e2,p2,cs2,rvacuum, &
                   den,ener,pres,vel,cs)



! output file
      open(unit=2,file=outfile,status='unknown')
      write(2,02) nstep,time
      write(2,01) 'i','x','den','ener','pres','vel','cs'
      do i=1,nstep

       write(2,03) i,zpos(i),den(i),ener(i),pres(i),vel(i),cs(i)

!       write(2,03) i,zpos(i),den(i)/rho2,ener(i)/e2, &
!                   pres(i)/p2,vel(i)/u2,cs(i)/cs2

      enddo
      close(unit=2)

      write(6,*)

! close up shop
      end program sedov3



      include 'sedov_library_qp.f90'
