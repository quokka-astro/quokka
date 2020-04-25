      program riemann
      implicit none

c..solves the exact riemann problem
c..original code from bruce fryxell

c..declare
      integer    n,npts,itmax,iter,i
      parameter (n = 1000)
      real       x(n),rho(n),u(n),p(n),
     1           rhol,pl,ul,rhor,pr,ur,gamma,xi,t,xl,xr,
     2           rho1,p1,u1,rho5,p5,u5,p40,p41,f0,f,eps,
     3           f1,p4,error,z,c5,gm1,gp1,gmfac1,gmfac2,fact,
     4           u4,rho4,w,p3,u3,rho3,c1,c3,xsh,xcd,xft,xhd,dx


c..define initial conditions
c..state at left of discontinuity
         rhol = 10.0
         pl   = 100.0
         ul   = 0.

c..state at right of discontinuity
         rhor = 1.0
         pr   = 1.0
         ur   = 0.

         if (ul .ne. 0. .or. ur .ne. 0.) then
          write (6,*) 'must have ul = ur = 0.'
          stop
         endif

c.. equation of state
         gamma = 1.4

c..location of discontinuity at t = 0
         xi = 2.0

c..time at which solution is desired
         t = 0.40

c..number of points in solution
         npts = 500
         if (npts .gt. n) then
          write (6,*) 'number of points exceeds array size'
          stop
         endif

c..spatial interval over which to compute solution
         xl = 0.0
         xr = 5.0
         if (xr .lt. xl) then
          write (6,*) 'xr must be greater than xl'
          stop
         endif



c..begin solution
      if (pl .gt. pr) then
         rho1 = rhol
         p1   = pl
         u1   = ul
         rho5 = rhor
         p5   = pr
         u5   = ur
      else
         rho1 = rhor
         p1   = pr
         u1   = ur
         rho5 = rhol
         p5   = pl
         u5   = ul
      endif


c..solve for post-shock pressure by secant method
c..initial guesses

         p40 = p1
         p41 = p5
         f0  = f(p40, p1, p5, rho1, rho5, gamma)

c..maximum number of iterations and maxium allowable relative error
         itmax = 20
         eps   = 1.e-5

         do iter = 1, itmax
          f1 = f(p41, p1, p5, rho1, rho5, gamma)
          if (f1 .eq. f0) go to 10

          p4 = p41 - (p41 - p40) * f1 / (f1 - f0)

          error = abs (p4 - p41) / p41
          if (error .lt. eps) go to 10

          p40 = p41
          p41 = p4
          f0  = f1
         enddo
         write (6,*) 'iteration failed to converge'
         stop 'abnormal termination'

10       continue


c..compute post-shock density and velocity
         z  = (p4 / p5 - 1.)
         c5 = sqrt (gamma * p5 / rho5)

         gm1 = gamma - 1.
         gp1 = gamma + 1.
         gmfac1 = 0.5 * gm1 / gamma
         gmfac2 = 0.5 * gp1 / gamma

         fact = sqrt (1. + gmfac2 * z)

         u4 = c5 * z / (gamma * fact)
         rho4 = rho5 * (1. + gmfac2 * z)
     1               / (1. + gmfac1 * z)

c..shock speed
         w = c5 * fact


c..compute values at foot of rarefaction
         p3 = p4
         u3 = u4
         rho3 = rho1 * (p3 / p1)**(1. /gamma)

c..compute positions of waves
      if (pl .gt. pr) then
       c1 = sqrt (gamma * p1 / rho1)
       c3 = sqrt (gamma * p3 / rho3)

       xsh = xi + w * t
       xcd = xi + u3 * t
       xft = xi + (u3 - c3) * t
       xhd = xi - c1 * t

c..and do say what we found
       write (6, 500)
       write (6, 501) rho1, p1, u1
       write (6, 502)
       write (6, 503) rho3, p3, u3
       write (6, 504) rho4, p4, u4
       write (6, 505) rho5, p5, u5

       write (6, 506) xhd
       write (6, 507) xft
       write (6, 508) xcd
       write (6, 509) xsh

500    format (// 2x, 'Region', 4x, 'Density', 8x, 'Pressure',
     1                          8x, 'Velocity')
501    format (5x, '1', 3(2x,1pe14.7))
502    format (5x, '2' ,20x, 'RAREFACTION')
503    format (5x, '3', 3(2x,1pe14.7))
504    format (5x, '4', 3(2x,1pe14.7))
505    format (5x, '5', 3(2x,1pe14.7)//)

506    format (2x, 'Head Of Rarefaction    x = ', 1pe14.7)
507    format (2x, 'Foot Of Rarefaction    x = ', 1pe14.7)
508    format (2x, 'Contact Discontinuity  x = ', 1pe14.7)
509    format (2x, 'Shock                  x = ', 1pe14.7//)




c..compute solution as a function of position
       dx = (xr - xl) / (npts - 1)
       do i = 1, npts
        x(i) = xl + dx * (i - 1)
       enddo

       do i = 1, npts
        if (x(i) .lt. xhd) then
         rho(i) = rho1
         p(i)   = p1
         u(i)   = u1
        else if (x(i) .lt. xft) then
         u(i)   = 2. / gp1 * (c1 + (x(i) - xi) / t)
         fact   = 1. - 0.5 * gm1 * u(i) / c1
         rho(i) = rho1 * fact ** (2. / gm1)
         p(i)   = p1 * fact ** (2. * gamma / gm1)
        else if (x(i) .lt. xcd) then
         rho(i) = rho3
         p(i)   = p3
         u(i)   = u3
        else if (x(i) .lt. xsh) then
         rho(i) = rho4
         p(i)   = p4
         u(i)   = u4
        else
         rho(i) = rho5
         p(i)   = p5
         u(i)   = u5
        endif
       enddo
      endif


c..if pr > pl, reverse solution
      if (pr .gt. pl) then
       c1 = sqrt (gamma * p1 / rho1)
       c3 = sqrt (gamma * p3 / rho3)

       xsh = xi - w * t
       xcd = xi - u3 * t
       xft = xi - (u3 - c3) * t
       xhd = xi + c1 * t


c..and do say what we found
       write (6, 500)
       write (6, 501) rho5, p5, u5
       write (6, 602) rho4, p4, u4
       write (6, 503) rho3, p3, u3
       write (6, 604)
       write (6, 505) rho1, p1, u1

       write (6, 609) xsh
       write (6, 508) xcd
       write (6, 507) xft
       write (6, 606) xhd

602    format (5x, '2', 3(2x,1pe14.7))
604    format (5x, '4' ,20x, 'RAREFACTION')
606    format (2x, 'Head Of Rarefaction    x = ', 1pe14.7//)
609    format (2x, 'Shock                  x = ', 1pe14.7)

       dx = (xr - xl) / (npts - 1)
       do i = 1, npts
         x(i) = xl + dx * (i - 1)
       enddo

       do i = 1, npts
        if (x(i) .lt. xsh) then
           rho(i) = rho5
           p(i)   = p5
           u(i)   = -u5
        else if (x(i) .lt. xcd) then
           rho(i) = rho4
           p(i)   = p4
           u(i)   = -u4
        else if (x(i) .lt. xft) then
           rho(i) = rho3
           p(i)   = p3
           u(i)   = -u3
        else if (x(i) .lt. xhd) then
           u(i)   = -2. / gp1 * (c1 + (xi - x(i)) / t)
           fact   = 1. + 0.5 * gm1 * u(i) / c1
           rho(i) = rho1 * fact ** (2. / gm1)
           p(i)   = p1 * fact ** (2. * gamma / gm1)
        else
           rho(i) = rho1
           p(i)   = p1
           u(i)   = -u1
        endif
       enddo
      endif


      open (unit=1, file = 'output',status='unknown')
      write (1, 1000)
1000  format (2x, 'i', 10x, 'x', 12x, 'density', 8x, 'pressure',
     1             8x, 'velocity'/)
1001  format (i4, 4(2x, 1pe14.7))
      do i = 1, npts
       write (1,1001) i, x(i), rho(i), p(i), u(i)
      enddo
      close (1)
      stop 'normal termination'
      end




      real function f(p4, p1, p5, rho1, rho5, gamma)
      implicit none

c..shock tube equation

c..declare the pass
      real p4,p1,p5,rho1,rho5,gamma

c..local variables
      real z,c1,c5,gm1,gp1,g2,fact

      z = (p4 / p5 - 1.)
      c1 = sqrt (gamma * p1 / rho1)
      c5 = sqrt (gamma * p5 / rho5)

      gm1 = gamma - 1.
      gp1 = gamma + 1.
      g2  = 2. * gamma

      fact = gm1 / g2 * (c5 / c1) * z
     1     / sqrt (1. + gp1 / g2 * z)
      fact = (1. - fact) ** (g2 / gm1)

      f = p1 * fact - p4

      return
      end
