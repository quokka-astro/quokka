       program ppm
       implicit none
       include 'ppm.dek'

! one-dimensional lagrangian + remap hydrodynamics using the piecewise-parabolic method
!
! original code from bruce fryxell, december 1992
! tweaked by fxt 1998, 2000, 2006, 2016

! set up grid and initial conditions

      call grid
      call init
      call tstep

! begin main loop
      do nstep = 1, nmax
       write (6,*) nstep, t, dt
       if (t + dt .gt. tmax) dt = tmax - t
       t = t + dt

       call get
       call hydro
       call store
       call tstep

       if (t .ge. tmax) go to 100
      enddo 


! write out final results
100   call output
      end program ppm






      subroutine grid
      implicit none
      include 'ppm.dek'

! compute grid coordinates

! local variables
      integer  ::  i
      real*8   ::  grdlnr,delr,delr2


! choose number of zones

      nr = 500

! set up grid in x direction, grdlnr is the length of the spatial domain.
! the following code assumes a uniformly spaced grid.
! in practice, any grid spacing may be used.

      grdlnr = 5.0d0
      delr   = grdlnr / nr
      delr2  = 0.5d0  * delr

      do i = 1, nr
       rr(i) =   i   * delr
       rl(i) = rr(i) - delr
       r (i) = rl(i) + delr2
      enddo

      return
      end subroutine grid





      subroutine init
      implicit none
      include 'ppm.dek'

! set initial conditions

! local variables
      integer           ::  i,njmpl,njmpr
      real*8            ::  rorite,roleft,urite,uleft,pleft,prite


! nmax         = number of time steps allowed before code stops
! t            = initial time
! dt           = guess for initial time step value
! dtmin, dtmax = minimum and maximum values allowed for time step
! cfl          = fraction of maximum stable time step to use
! nriem        = maximum number of iterations in riemann solution
! gamma        = the ratio of specific heats in the polytopic equation of state

      nmax   = 1000000
      t      = 0.0d0
      tmax   = 0.40d0
      dt     = 1.0d-3
      dtmin  = 1.0d-10
      dtmax  = tmax
      cfl    = 0.8d0
      nriem  = 40
      gamma  = 1.4d0


! the following constants are 'small' values for dimensionless
! numbers, density, pressure, and energy.  they should be set to 
! values a few orders of magnitude less  than values expected for 
! the corresponding variable.  they are used primarily to prevent 
! errors such as dividing by zero which can result if a variable 
! undershoots its correct value.

      small  = 1.0d-10
      smlrho = 1.0d-10
      smallp = 1.0d-10
      smalle = 1.0d-10

! common gamma related quantities
      gamm1  = gamma - 1.0d0
      gamp1  = gamma + 1.0d0
      gmfc   = 0.5d0 * gamm1
      gamfac = 0.5d0 * gamp1 / gamma

! gr is the gravitational accelerations

      gr = 0.0d0

! define the values of the conserved quantities at each grid point

      pleft  = 100.0d0
      uleft  = 0.0d0
      roleft = 10.d0

      prite  = 1.0d0
      urite  = 0.0d0
      rorite = 1.0d0


! part of the the grid in the left state, part in the right state
      njmpl = nr / 2
      njmpr = nr - njmpl

      do i = 1, nr
       if ( r(i) .le. 2.0d0) then
        densty(i) = roleft
        vr(i)     = uleft
        press(i)  = pleft 
        energy(i) = press(i) / (gamm1 * densty(i))
       else
        densty(i) = rorite
        vr(i)     = urite
        press(i)  = prite
        energy(i) = press(i) / (gamm1 * densty(i))
       endif
      enddo

      return
      end subroutine init



      subroutine get
      implicit none
      include 'ppm.dek'

! copy a row of the grid into the working arrays, and
! apply the boundary conditions

! local variables
      integer  :: i,i4,i5,i9

      n   = nr
      np1 = n + 1
      np2 = n + 2
      np3 = n + 3
      np4 = n + 4
      np5 = n + 5
      np6 = n + 6
      np7 = n + 7
      np8 = n + 8

      do i = 1, nr
       i4 = i + 4
       rho(i4)  = densty(i)
       u(i4)    = vr(i)
       e(i4)    = energy(i)
       p(i4)    = press(i)
       xl(i4)   = rl(i)
       x(i4)    = r(i)
       xr(i4)   = rr(i)
       dx(i4)   = xr(i4) - xl(i4)
      enddo

! reflecting boundary conditions

      do i = 1, 4
       i9 = 9 - i
       rho(i) = rho(i9)
       u(i)   = -u(i9)
       e(i)   = e(i9)
       p(i)   = p(i9)
       dx(i)  = dx(i9)
      enddo

      do i = 1, 4
       i5 = 5 - i
       xl(i5) = xl(i5+1) - dx(i5)
       xr(i5) = xr(i5+1) - dx(i5+1)
       x(i5)  = 0.5d0 * (xl(i5) + xr(i5))
      enddo

      do i = np5, np8
       i9 = 2 * n + 9 - i
       rho(i) = rho(i9)
       u(i)   = -u(i9)
       e(i)   = e(i9)
       p(i)   = p(i9)
       dx(i)  = dx(i9)
      enddo

      do i = np5, np8
       xl(i) = xl(i-1) + dx(i-1)
       xr(i) = xr(i-1) + dx(i)
       x(i)  = 0.5d0 * (xl(i) + xr(i))
      enddo

      do i = 1, np8
       tau(i) = 1.0d0 / rho(i)
       c(i)   = sqrt (gamma * p(i) * rho(i))
       dm(i)  = rho(i) * dx(i)
      enddo

      return
      end subroutine get




      subroutine store
      implicit none
      include 'ppm.dek'

! store the updated values of the working arrays into answer arrays

! local variables
      integer  :: i,i4

      do i = 1, nr
       i4 = i + 4
       densty(i) = rhonu(i4)
       vr(i)     = unu(i4)
       energy(i) = enu(i4)
       press(i)  = pnu(i4)
       rl(i)     = xlnu(i4)
       r(i)      = xnu(i4)
       rr(i)     = xrnu(i4)
     end do

      return
      end subroutine store




      subroutine hydro
      implicit none
      include 'ppm.dek'

! solve the lagrangian hydrodynamic equations on a single row of the grid
!
! local variables
      integer     :: i,imap
      real*8      :: dvol(ndp8), al(ndp8), ar(ndp8), ei


      do i = 1, np8
       dtdx(i) = dt   / dx(i)
       dtdm(i) = dt   / dm(i)
       ce(i)   = c(i) / rho(i)
      enddo

      imap = 0
      call intrfc (imap)
      call states
      call rieman

      do i = 5, np5
       xlnu(i)   = xl(i) + dt * ustar(i)
       xrnu(i-1) = xlnu(i)
       upstar(i) = ustar(i) * pstar(i)
      enddo

      do i = 5, np4
       dxnu(i) = xrnu(i) - xlnu(i)
       xnu(i)  = 0.5d0 * (xrnu(i) + xlnu(i))
      enddo

      do i = 5, np4
       dvol(i) = dxnu(i)
       al(i)   = 1.0d0
       ar(i)   = 1.0d0
      enddo

      do i = 5, np4
       rhonu(i) = dm(i) / dvol(i)
       taunu(i) = 1.0d0  / rhonu(i)
       unu(i)   = u(i)  - dtdm(i) * 0.5d0 * (al(i) + ar(i)) &
                        * (pstar(i+1)  - pstar(i)) + dt * gr
       enu(i)   = e(i)  - dtdm(i) &
                * (ar(i) * upstar(i+1) - al(i) * upstar(i)) &
                + 0.5d0 * dt * (u(i) + unu(i)) * gr
       ei       = enu(i) - 0.5d0 * unu(i)**2
       ei       = max(ei, smalle)
       pnu(i)   = gamm1 * rhonu(i) * ei
      enddo

      return
      end subroutine hydro





      subroutine intrfc (imap)
      implicit none
      include 'ppm.dek'

!
! calculate interface values of density, pressure, and velocity
!

! local variables
      integer :: imap,i


! calculate interpolation coefficients

      call coeff

! perform interpolation

      call interp (rhol, rho, rhor)
      call interp (  ul,   u,   ur)
      call interp (  pl,   p,   pr)

! in remap step only, interpolate transverse velocity and steepen contact discontinuities

      if (imap .eq. 1) call detect


! apply monotonicity constraints on interpolation parabolae

      call monot (rhol, rho, rhor, drho, rho6)
      call monot (  ul,   u,   ur,   du,   u6)
      call monot (  pl,   p,   pr,   dp,   p6)


! flatten interpolation parabolae near shocks which are too thin

      call flaten

      do i = 4, np5
       rhol(i) = fshk(i) * rho(i) + fshk1(i) * rhol(i)
       rhor(i) = fshk(i) * rho(i) + fshk1(i) * rhor(i)
       ul(i)   = fshk(i) * u(i)   + fshk1(i) * ul(i)
       ur(i)   = fshk(i) * u(i)   + fshk1(i) * ur(i)
       pl(i)   = fshk(i) * p(i)   + fshk1(i) * pl(i)
       pr(i)   = fshk(i) * p(i)   + fshk1(i) * pr(i)
     enddo

      do i = 4, np5
       drho(i) = rhor(i) - rhol(i)
       du(i)   = ur(i) - ul(i)
       dp(i)   = pr(i) - pl(i)
       rho6(i) = 6.d0 * rho(i) - 3.d0 * (rhol(i) + rhor(i))
       u6(i) = 6.d0 * u(i) - 3.d0 * (ul(i) + ur(i))
       p6(i) = 6.d0 * p(i) - 3.d0 * (pl(i) + pr(i))
      enddo

      return
      end subroutine intrfc




      subroutine coeff
      implicit none
      include 'ppm.dek'

! calculate coefficients of interpolation polynomial

! local variables
      integer  :: i
      real*8   :: s1(ndp8), s2(ndp8), s3(ndp8), s4(ndp8)


      do i = 2, np8
       s1(i) = dx(i) + dx(i-1)
       s2(i) = s1(i) + dx(i)
       s3(i) = s1(i) + dx(i-1)
      enddo

      do i = 2, np7
       s4(i) = dx(i) / (s1(i) + dx(i+1))
       c1(i) = s4(i) * s3(i)   / s1(i+1)
       c2(i) = s4(i) * s2(i+1) / s1(i)
      enddo

      do i = 2, np6
       s4(i) =  1.0d0  / (s1(i) + s1(i+2))
       c3(i) = -s4(i) * dx(i)   * s1(i)   / s3(i+1)
       c4(i) =  s4(i) * dx(i+1) * s1(i+2) / s2(i+1)
       c5(i) =  (dx(i) - 2.0d0*(dx(i+1)*c3(i) + dx(i)*c4(i))) / s1(i+1)
      enddo

      return
      end subroutine coeff





      subroutine interp (al, a, ar)
      implicit none
      include 'ppm.dek'


! interpolate interface values and monotonize


! declare the pass
      real*8     :: al(ndp8),  a(ndp8), ar(ndp8)

! local variables
      integer    :: i
      real*8     :: s1(ndp8), s2(ndp8), armax, armin


      do i = 2, np8
       s1(i) = a(i) - a(i-1)
       s2(i) = 2.0d0 * abs(s1(i))
      enddo

      do i = 2, np7
       dela(i) = c1(i) * s1(i+1) + c2(i) * s1(i)
       dela(i) = min(abs(dela(i)),s2(i),s2(i+1)) * sign(1.0d0, dela(i))
       if (s1(i) * s1(i+1) .le. 0.0d0) dela(i) = 0.0d0
      enddo

      do i = 2, np6
       ar(i) = a(i) + c5(i)*s1(i+1) + c3(i)*dela(i+1) + c4(i)*dela(i)
      enddo

      do i = 2, np6
       armin   = min(a(i), a(i+1))
       armax   = max(a(i), a(i+1))
       ar(i)   = min(ar(i), armax)
       ar(i)   = max(ar(i), armin)
       al(i+1) = ar(i)
      enddo

      return
      end subroutine interp





      subroutine monot (al, a, ar, da, a6)
      implicit none
      include 'ppm.dek'

! apply monotonicity constraint to interpolation parabolae

! declare the pass
      real*8   :: al(ndp8), a(ndp8), ar(ndp8), da(ndp8), a6(ndp8)

! local variables
      integer  :: i
      real*8   :: s1,s2,a3

      do i = 4, np5
       if ((ar(i) - a(i)) * (a(i) - al(i)) .le. 0.0d0) then
          al(i) = a(i)
          ar(i) = a(i)
       endif

       da(i) = ar(i) - al(i)
       a3    = 3.0d0 * a(i)
       s1    = a3 - 2.0d0 * ar(i)
       s2    = a3 - 2.0d0 * al(i)

       if (da(i) * (al(i) - s1   ) .lt. 0.0d0) al(i) = s1
       if (da(i) * (s2    - ar(i)) .lt. 0.0d0) ar(i) = s2

       da(i) = ar(i) - al(i)
       a6(i) = 6.0d0 * a(i) - 3.0d0 * (al(i) + ar(i))
      enddo

      return
      end subroutine monot




      subroutine flaten
      implicit none
      include 'ppm.dek'

! flaten zone structure in regions where shocks are too thin
!

! local variables
      integer           :: i
      real*8            :: divu(ndp8), f(ndp8), delp1(ndp8), &
                           dmch0,dmch,s0,sp,delu1,delp2
      real*8, parameter :: dpf = 1.0d0/3.0d0

      dmch0 = 2.0d0 * dpf**2 / (gamma * (2.0d0 * gamma + gamp1 * dpf))
      dmch0 = sqrt(dmch0)

      do i = 3, np6
       delu1    = u(i+1) - u(i-1)
       delp1(i) = p(i+1) - p(i-1)
       delp2    = p(i+2) - p(i-2)
       divu(i)  = delu1 / (x(i+1) - x(i-1))

       sp   = abs(delp1(i) / max(delp2, smallp))
       s0   = max(0.0d0, min(0.5d0, 5.d0 * (sp - 0.75d0)))
       dmch = abs(delu1 / min(ce(i+1), ce(i-1)))

       if (dmch .gt. dmch0 .and. divu(i) .lt. 0.0d0) then
          f(i) = s0
       else
          f(i) = 0.0d0
       endif
      enddo

      do i = 4, np5
       if (delp1(i) .lt. 0.0d0) then
        fshk(i) = max(f(i), f(i-1))
       else
        fshk(i) = max(f(i), f(i+1))
       endif
       fshk1(i) = 1.0d0 - fshk(i)
      enddo

      return
      end subroutine flaten



      subroutine detect
      implicit none
      include 'ppm.dek'

! search for contact discontinuities and steepen the structure if necessary

! local variables
      integer           :: i
      real*8            :: s1(ndp8), s2(ndp8), s3(ndp8), s4(ndp8), &
                           d2rho(ndp8), eta(ndp8), rhodr,rhodl
      real*8, parameter :: eta1  = 20.0d0, eta2 = 0.05d0, epsln = 0.01d0, k0 = 0.1d0

      do i = 2, np7
       s1(i) = dx(i) + dx(i-1)
       s2(i) = s1(i) + dx(i+1)
       s3(i) = rho(i) - rho(i-1)
       s1(i) = s3(i) / s1(i)
      enddo

      do i = 2, np6
       d2rho(i) = (s1(i+1) - s1(i)) / s2(i)
      enddo

      do i = 2, np8
       s1(i) = x(i) - x(i-1)
       s1(i) = s1(i) * s1(i) * s1(i)
      enddo

      do i = 4, np5
       s3(i) = - (d2rho(i+1) - d2rho(i-1)) * (s1(i) + s1(i+1))
       if (rho(i+1) - rho(i-1) .eq. 0.0d0) then
        s4(i) = small * smlrho
       else
        s4(i) = rho(i+1) - rho(i-1)
       endif
       eta(i) = s3(i) / ((x(i+1) - x(i-1)) * s4(i))
       if (d2rho(i-1) * d2rho(i+1) .gt. 0.0d0) eta(i) = 0.0d0
      enddo

      do i = 4, np5
       s4(i)  = epsln * min(abs(rho(i+1)), abs(rho(i-1))) &
                - abs(rho(i+1) - rho(i-1))
       if (s4(i) .ge. 0.0d0) eta(i) = 0.0d0
       eta(i) = max(0.0d0, min(eta1 * (eta(i) - eta2), 1.0d0))
       s1(i)  = abs(p(i+1) - p(i-1)) / min(p(i+1),   p(i-1)  )
       s2(i)  = abs(rho(i+1) - rho(i-1)) / min(rho(i+1), rho(i-1))
       if (gamma * k0 * s2(i) .lt. s1(i)) eta(i) = 0.0d0
      enddo

      do i = 4, np5
       rhodl   = rho(i-1) + 0.5d0 * dela(i-1)
       rhodr   = rho(i+1) - 0.5d0 * dela(i+1)
       rhol(i) = rhol(i)  * (1.0d0 - eta(i)) + rhodl * eta(i)
       rhor(i) = rhor(i)  * (1.0d0 - eta(i)) + rhodr * eta(i)
      enddo

      return
      end subroutine detect




      subroutine states
      implicit none
      include 'ppm.dek'

! compute left and right states for input to riemann problem

! local variables
      integer            :: i
      real*8             :: s1(ndp8), s2(ndp8), apls, amns, a
      real*8, parameter  :: forthd = 4.0d0/3.0d0

      do i = 1, np8
       s1(i) = 0.5d0 * min(1.0d0, ce(i) * dtdx(i))
       s2(i) = 1.0d0 - forthd * s1(i)
      enddo

      do i = 5, np5
       upls(i) = ur(i-1) - s1(i-1) * (du(i-1) - s2(i-1) * u6(i-1))
      enddo

      do i = 5, np5
       upls(i)   = upls(i) + 0.5d0 * dt * gr
       taupls(i) = rhor(i-1) &
                   - s1(i-1) * (drho(i-1) - s2(i-1) * rho6(i-1))
       taupls(i) = 1.0d0 / taupls(i)
       ppls(i)   = pr(i-1) - s1(i-1) * (dp(i-1) - s2(i-1) * p6(i-1))
       ppls(i)   = max(smallp, ppls(i))
       cpls(i)   = sqrt(gamma * ppls(i) / taupls(i))
      enddo

      do i = 5, np5
       umns(i) = ul(i) + s1(i) * (du(i) + s2(i) * u6(i))
      enddo

      do i = 5, np5
       umns(i)   = umns(i) + 0.5d0 * dt * gr
       taumns(i) = rhol(i) + s1(i) * (drho(i) + s2(i) * rho6(i))
       taumns(i) = 1.0d0 / taumns(i)
       pmns(i)   = pl(i) + s1(i) * (dp(i) + s2(i) * p6(i))
       pmns(i)   = max(smallp, pmns(i))
       cmns(i)   = sqrt(gamma * pmns(i) / taumns(i))
      enddo

      return
      end subroutine states





      subroutine rieman
      implicit none
      include 'ppm.dek'

! solve riemann shock tube problem

! local variables
      integer           :: i,imax, nnn
      real*8            :: pstr1(ndp8), ustarm, ustarp, emax, zp, zm, &
                           error, s1, s2
      real*8, parameter :: tol = 1.0d-4


! first guess  (wpls = cpls , wmns = cmns)

      do i = 5, np5
       pstar(i) = pmns(i) - ppls(i)  - cmns(i) * (umns(i) - upls(i))
       pstar(i) = ppls(i) + pstar(i) * cpls(i) / (cpls(i) + cmns(i))
       pstar(i) = max(smallp, pstar(i))
      enddo

! begin iteration loop
      do nnn = 1, nriem

       do i = 5, np5
        s1      = 1.0d0 + gamfac * (pstar(i) - ppls(i)) / ppls(i)
        s2      = 1.0d0 + gamfac * (pstar(i) - pmns(i)) / pmns(i)
        wpls(i) = cpls(i) * sqrt(s1)
        wmns(i) = cmns(i) * sqrt(s2)
       enddo

       do i = 5, np5
        zp = 4.0d0 * taupls(i) * wpls(i) * wpls(i)
        zp = -zp * wpls(i) / (zp - gamp1 * (pstar(i) - ppls(i)))
        zm = 4.0d0 * taumns(i) * wmns(i) * wmns(i)
        zm = zm * wmns(i)  / (zm - gamp1 * (pstar(i) - pmns(i)))
        ustarp = upls(i) - (pstar(i) - ppls(i)) / wpls(i)
        ustarm = umns(i) + (pstar(i) - pmns(i)) / wmns(i)
        pstr1(i) = pstar(i)
        pstar(i) = pstar(i) + (ustarm - ustarp) * (zm*zp) / (zm-zp)
        pstar(i) = max(smallp, pstar(i))
       enddo

! test for convergence
       emax = 0.0d0
       do i = 5, np5
        error = abs(pstar(i) - pstr1(i)) / pstar(i)
        if (error .gt. emax) then
         emax = error
         imax = i
        endif
       enddo

       if (emax .le. tol) go to 60
      enddo

! end of iteration loop

60    continue

      if (emax .gt. tol) then
         write (6,*) '***Warning***  Convergence failure in rieman.'
         write (6,*) 'Maximum error = ', emax, ' in zone ', imax
      endif

! calculate velocity

      do i = 5, np5
       ustarp   = upls(i) - (pstar(i) - ppls(i)) / wpls(i)
       ustarm   = umns(i) + (pstar(i) - pmns(i)) / wmns(i)
       ustar(i) = 0.5d0 * (ustarp + ustarm)
      enddo

      return
      end subroutine rieman




      subroutine tstep
      implicit none
      include 'ppm.dek'

! compute new time step value using cfl condition

! local variables
      integer :: i
      real*8  :: cflmax,s1,olddt,ceul

      cflmax = 0.0d0
      do i = 1, nr
       ceul = sqrt (gamma * press(i) / densty(i))
       s1   = ceul / (rr(i) - rl(i))
       if (s1 .gt. cflmax) cflmax = s1
      enddo

      olddt = dt
      dt    = cfl / cflmax
      dt    = min (dt, 1.2d0 * olddt)

      return
      end subroutine tstep




      subroutine output
      implicit none
      include 'ppm.dek'

! store results

! local variables
      integer  :: i

! popular formats
10    format (i6, 2x, 1pe12.5, 2x, e10.3, 2x, 0pf9.6, 2x, i4//)
20    format (i4, 5(2x, 1pe11.4))

      open (3, file = 'results')
      write(3,10) nstep, t, dt, gamma, nr
      do i = 1, nr
        write(3,20) i, r(i), densty(i), vr(i), press(i), energy(i)
      enddo
      close(3)

      return
      end subroutine output

