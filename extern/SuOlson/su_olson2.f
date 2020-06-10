      program suo
      implicit none

c..solves the su-olson problem
c..cgs units throughout,
c..except some duly noted temperatures that are in ev

c..declare
      character*80     outfile,string
      integer          i,nstep,iargc
      double precision time,zpos,trad_bc_ev,opac,alpha,
     1                 erad,trad,trad_ev,tmat,tmat_ev,
     2                 zlo,zhi,zstep,value,uans,vans,tau


c..some physics
      double precision clight,ssol,asol
      parameter        (clight  = 2.99792458d10,
     1                  ssol     = 5.67051d-5,
     2                  asol    = 4.0d0 * ssol / clight)


c..popular formats
 01   format(1x,t4,a,t10,a,t22,a,t34,a,t46,a,t58,a,t70,a)
 02   format(1x,i4,1p8e12.4)



c..input parameters
      trad_bc_ev = 1.0d0
      opac       = 1.0d0
      alpha      = 4.0d0*asol
      tau        = 1.0d0
      time       = tau * alpha / (4.0d0*asol*clight*opac)


c..number of grid points, spatial domain, spatial step size
      nstep = 100
      zlo   = 0.0d0
      zhi   = 20.0d0
      zstep = (zhi - zlo)/float(nstep)


c..output file
      outfile = '100pt_tau1p0.dat'
      open(unit=2,file=outfile,status='unknown')
c.....write(2,02) nstep,tau
      write(2,01) 'i','x','U','V','Trad/T_H','Tmat/T_H'


c..use the mid-cell points to match various eularian hydrocodes
      do i=1,nstep
       zpos = zlo + 0.5d0*zstep + float(i-1)*zstep

       call so_wave(time,zpos,trad_bc_ev,opac,alpha,
     1              erad,trad,trad_ev,tmat,tmat_ev,uans,vans)

       write(6,40) i,zpos,uans,vans,trad_ev,tmat_ev
       write(2,40) i,zpos,uans,vans,trad_ev,tmat_ev
 40    format(1x,i4,1p8e14.6)

      enddo

c..close up stop
      close(unit=2)
      end






      subroutine so_wave(time,zpos,trad_bc_ev,opac,alpha,
     1                   erad,trad,trad_ev,tmat,tmat_ev,uans,vans)
      implicit none
      save

c..provides solution to the su-olson problem

c..input:
c..time       = time point where solution is desired
c..zpos       = spaatial point where solution is desired
c..trad_bc_ev = boundary condition temperature in electron volts
c..opac       = the opacity in cm**2/g
c..alpha      = coefficient of the material equation of state c_v = alpha T_mat**3

c..output:
c..erad       = energy desnity of radiation field erg/cm**3
c..trad       = temperature of radiation field kelvin
c..trad_ev    = temperature of radiation field electron volts
c..trad       = temperature of material field kelvin
c..trad_ev    = temperature of material field electron volts


c..declare the input
      double precision time,zpos,trad_bc_ev,opac,alpha,
     1                 erad,trad,trad_ev,tmat,tmat_ev,uans,vans


c..local variables
      double precision trad_bc,ener_in,xpos,epsilon,tau,ialpha,
     1                 usolution,vsolution


c..some physics
      double precision clight,ssol,asol,kev,rt3,a4,a4c
      parameter        (clight    = 2.99792458d10,
     1                  ssol      = 5.67051d-5,
     2                  asol      = 4.0d0 * ssol / clight,
     3                  kev       = 8.617385d-5,
     4                  rt3       = 1.7320508075688772d0,
     5                  a4        = 4.0d0*asol,
     6                  a4c       = a4 * clight)



c..derived parameters and conversion factors
      trad_bc = trad_bc_ev/kev
      ener_in = asol * trad_bc**4
      xpos    = rt3 * opac * zpos
      ialpha  = 1.0d0/alpha
      tau     = a4c * opac * ialpha * time
      epsilon = a4 * ialpha


c..get the dimensionless solutions
      uans = usolution(xpos,tau,epsilon)
      vans = vsolution(xpos,tau,epsilon,uans)


c..compute the physical solution

      erad    = uans * ener_in
      trad    = (erad/asol)**(0.25d0)
      trad_ev = trad * kev

      tmat    = (vans*ener_in/asol)**(0.25d0)
      tmat_ev = tmat * kev

      return
      end






      double precision function usolution(posx_in,tau_in,epsilon_in)
      implicit none
      save

c..computes the u solution for the su-olson problem

c..declare the pass
      double precision posx_in,tau_in,epsilon_in

c..common block communication
      double precision posx,tau,epsilon
      common  /bdoor/  posx,tau,epsilon
      integer         jwant
      common /rots/   jwant


c..local variables
      external         midpnt,upart1,upart2,
     1                 gamma_one_root,gamma_two_root
      logical          bracket
      integer          i,niter
      double precision xi1,xi2,midpnt,upart1,upart2,sum1,sum2,
     1                 gamma_one_root,gamma_two_root,
     2                 zbrent,eta_hi,eta_lo,eta_int

      double precision tol,eps,eps2,pi,rt3,rt3opi
      parameter        (tol    = 1.0d-6,
     &                  eps    = 1.0d-10,
     &                  eps2   = 1.0d-8,
     &                  pi     = 3.1415926535897932384d0,
     &                  rt3    = 1.7320508075688772d0,
     &                  rt3opi = rt3/pi)


c..transfer input to common block
      posx    = posx_in
      tau     = tau_in
      epsilon = epsilon_in


c..integrand may not oscillate for small values of posx
      eta_lo  = 0.0d0
      eta_hi  = 1.0d0
      sum1    = 0.0d0
      jwant   = 1
      bracket = (gamma_one_root(eta_lo)*gamma_one_root(eta_hi).le.0.0)
      if (.not.bracket) then
       call qromo(upart1,eta_lo,eta_hi,eps,sum1,midpnt)

c..integrate over each oscillitory piece
      else
       do i=1,100
        jwant = i
        eta_int = zbrent(gamma_one_root,eta_lo,eta_hi,tol,niter)
        call qromo(upart1,eta_lo,eta_int,eps,xi1,midpnt)
        sum1  = sum1 + xi1
        eta_lo = eta_int
        if (abs(xi1) .le. eps2) goto 10
       enddo
 10    continue
      end if


c..integrand may not oscillate for small values of posx
      eta_lo = 0.0d0
      eta_hi = 1.0d0
      sum2   = 0.0d0
      jwant   = 1
      bracket = (gamma_two_root(eta_lo)*gamma_two_root(eta_hi).le.0.0)
      if (.not.bracket) then
       call qromo(upart2,eta_lo,eta_hi,eps,sum2,midpnt)

c..integrate from hi to lo on this piece
      else
       do i=1,100
        jwant = i
        eta_int = zbrent(gamma_two_root,eta_hi,eta_lo,tol,niter)
        call qromo(upart2,eta_hi,eta_int,eps,xi2,midpnt)
        sum2  = sum2 + xi2
        eta_hi = eta_int
        if (abs(xi2) .le. eps2) goto 20
       enddo
 20    continue
       sum2 = -sum2
      endif


c..done
      usolution = 1.0d0 - 2.0d0*rt3opi*sum1 - rt3opi*exp(-tau)*sum2
      return
      end





      double precision function vsolution(posx_in,tau_in,epsilon_in,
     1                                    uans)
      implicit none
      save

c..computes the v solution for the su-olson problem

c..declare the pass
      double precision posx_in,tau_in,epsilon_in,uans

c..common block communication
      double precision posx,tau,epsilon
      common  /bdoor/  posx,tau,epsilon
      integer         jwant
      common /rots/   jwant


c..local variables
      external         midpnt,vpart1,vpart2,
     1                 gamma_two_root,gamma_three_root
      logical          bracket
      integer          i,niter
      double precision xi1,xi2,midpnt,vpart1,vpart2,sum1,sum2,
     1                 gamma_two_root,gamma_three_root,
     2                 zbrent,eta_hi,eta_lo,eta_int

      double precision tol,eps,eps2,pi,rt3,rt3opi
      parameter        (tol    = 1.0d-6,
     &                  eps    = 1.0d-10,
     &                  eps2   = 1.0d-8,
     &                  pi  = 3.1415926535897932384d0,
     &                  rt3 = 1.7320508075688772d0,
     &                  rt3opi = rt3/pi)



c..transfer input to common block
      posx    = posx_in
      tau     = tau_in
      epsilon = epsilon_in


c..integrand may not oscillate for small values of posx
      eta_lo  = 0.0d0
      eta_hi  = 1.0d0
      sum1    = 0.0d0
      jwant   = 1
      bracket = (gamma_three_root(eta_lo)*gamma_three_root(eta_hi)
     1           .le. 0.0)
      if (.not.bracket) then
       call qromo(vpart1,eta_lo,eta_hi,eps,sum1,midpnt)

c..integrate over each oscillitory piece
c..from 1 to 0 on this part; this one really oscillates
      else
       do i=1,100
        jwant = i
        eta_int = zbrent(gamma_three_root,eta_hi,eta_lo,tol,niter)
        call qromo(vpart1,eta_hi,eta_int,eps,xi1,midpnt)
        sum1  = sum1 + xi1
        eta_hi = eta_int
        if (abs(xi1) .le. eps2) goto 10
       enddo
 10    continue
       sum1 = -sum1
      end if



c..integrand may not oscillate for small values of posx
      eta_lo = 0.0d0
      eta_hi = 1.0d0
      sum2   = 0.0d0
      jwant   = 1
      bracket = (gamma_two_root(eta_lo)*gamma_two_root(eta_hi).le.0.0)
      if (.not.bracket) then
       call qromo(vpart2,eta_lo,eta_hi,eps,sum2,midpnt)

c..integrate over each oscillitory piece
c..from 1 to 0 on this part; this one really oscillates
      else
       do i=1,100
        jwant = i
        eta_int = zbrent(gamma_two_root,eta_hi,eta_lo,tol,niter)
        call qromo(vpart2,eta_hi,eta_int,eps,xi2,midpnt)
        sum2  = sum2 + xi2
        eta_hi = eta_int
        if (abs(xi2) .le. eps2) goto 20
       enddo
 20    continue
       sum2 = -sum2
      endif


c..done
      vsolution = uans - 2.0d0*rt3opi*sum1 + rt3opi*exp(-tau)*sum2
      return
      end








      double precision function upart1(eta)
      implicit none
      save

c..equation 36 of su & olson jqsrt 1996, first integrand

c..declare the pass
      double precision eta

c..common block communication
      double precision posx,tau,epsilon
      common  /bdoor/  posx,tau,epsilon


c..local variables
      double precision numer,denom,gamma_one,theta_one,tiny
      parameter        (tiny = 1.0d-14)

      numer = sin(posx*gamma_one(eta,epsilon) + theta_one(eta,epsilon))

      denom = eta * sqrt(3.0d0 + 4.0d0*gamma_one(eta,epsilon)**2)
      denom = max(tiny,denom)

      upart1= exp(-tau*eta*eta) * numer/denom

      return
      end



      double precision function upart2(eta)
      implicit none
      save

c..equation 36 of su & olson jqsrt 1996, second integrand

c..declare the pass
      double precision eta

c..common block communication
      double precision posx,tau,epsilon
      common  /bdoor/  posx,tau,epsilon


c..local variables
      double precision numer,denom,gamma_two,theta_two,tiny
      parameter        (tiny = 1.0d-14)

      numer = sin(posx*gamma_two(eta,epsilon) + theta_two(eta,epsilon))

      denom = eta * (1.0d0 + epsilon*eta) *
     &        sqrt(3.0d0 + 4.0d0*gamma_two(eta,epsilon)**2)
      denom = max(tiny,denom)

      upart2= exp(-tau/(max(tiny,eta*epsilon))) * numer/denom

      return
      end





      double precision function vpart1(eta)
      implicit none
      save

c..equation 42 of su & olson jqsrt 1996, first integrand

c..declare the pass
      double precision eta

c..common block communication
      double precision posx,tau,epsilon
      common  /bdoor/  posx,tau,epsilon


c..local variables
      double precision numer,denom,gamma_two,theta_two,
     &                 gamma_three,theta_three,eta2,tiny
      parameter        (tiny = 1.0d-14)

      eta2   = eta * eta

      numer  = sin(posx*gamma_three(eta,epsilon) +
     &             theta_three(eta,epsilon))

      denom  = sqrt(4.0d0 - eta2 + 4.0d0*epsilon*eta2*(1.0d0 - eta2))
      denom  = max(tiny,denom)

      vpart1 = exp(-tau*(1.0d0 - eta2)) * numer/denom

      return
      end



      double precision function vpart2(eta)
      implicit none
      save

c..equation 42 of su & olson jqsrt 1996, second integrand

c..declare the pass
      double precision eta

c..common block communication
      double precision posx,tau,epsilon
      common  /bdoor/  posx,tau,epsilon


c..local variables
      double precision numer,denom,gamma_two,theta_two,tiny
      parameter        (tiny = 1.0d-14)

      numer = sin(posx*gamma_two(eta,epsilon) + theta_two(eta,epsilon))

      denom = eta * sqrt(3.0d0 + 4.0d0*gamma_two(eta,epsilon)**2)
      denom = max(tiny,denom)

      vpart2= exp(-tau/(max(tiny,eta*epsilon))) * numer/denom

      return
      end




      double precision function gamma_one_root(eta_in)
      implicit none
      save

c..used by a root finder to determine the integration inveral

c..declare the pass
      double precision eta_in


c..common block communication
      double precision posx,tau,epsilon
      common  /bdoor/  posx,tau,epsilon
      integer         jwant
      common /rots/   jwant


c..local variables
      double precision gamma_one,theta_one,pi,twopi
      parameter        (pi  = 3.1415926535897932384d0,
     1                  twopi = 2.0d0*pi)



c      write(6,*) eta_in,posx,tau,epsilon
c      write(6,*) jwant
c      write(6,*) gamma_one(eta_in,epsilon),theta_one(eta_in,epsilon)

c..go
      gamma_one_root = gamma_one(eta_in,epsilon)*posx
     &                 + theta_one(eta_in,epsilon)
     &                 - jwant * twopi


c      write(6,*) gamma_one_root
c      read(5,*)

      return
      end



      double precision function gamma_two_root(eta_in)
      implicit none
      save

c..used by a root finder to determine the integration inveral

c..declare the pass
      double precision eta_in


c..common block communication
      double precision posx,tau,epsilon
      common  /bdoor/  posx,tau,epsilon

      integer         jwant
      common /rots/   jwant


c..local variables
      double precision gamma_two,theta_two,pi,twopi
      parameter        (pi  = 3.1415926535897932384d0,
     1                  twopi = 2.0d0*pi)


c..go
      gamma_two_root = gamma_two(eta_in,epsilon)*posx
     &                 + theta_two(eta_in,epsilon)
     &                 - jwant * twopi

      return
      end




      double precision function gamma_three_root(eta_in)
      implicit none
      save

c..used by a root finder to determine the integration inveral

c..declare the pass
      double precision eta_in


c..common block communication
      double precision posx,tau,epsilon
      common  /bdoor/  posx,tau,epsilon

      integer         jwant
      common /rots/   jwant


c..local variables
      double precision gamma_three,theta_three,pi,twopi
      parameter        (pi  = 3.1415926535897932384d0,
     1                  twopi = 2.0d0*pi)


c..go
      gamma_three_root = gamma_three(eta_in,epsilon)*posx
     &                 + theta_three(eta_in,epsilon)
     &                 - jwant * twopi

      return
      end




      double precision function theta_one(eta,epsilon)
      implicit none
      save

c..equation 38 of su & olson jqsrt 1996

c..declare the pass
      double precision eta,epsilon

c..local variables
      double precision gamma_one

      theta_one = acos(sqrt(3.0d0 /
     &                 (3.0d0 + 4.0d0*gamma_one(eta,epsilon)**2)))

      return
      end



      double precision function theta_two(eta,epsilon)
      implicit none
      save

c..equation 38 of su & olson jqsrt 1996

c..declare the pass
      double precision eta,epsilon

c..local variables
      double precision gamma_two

      theta_two = acos(sqrt(3.0d0 /
     &                 (3.0d0 + 4.0d0*gamma_two(eta,epsilon)**2)))

      return
      end


      double precision function theta_three(eta,epsilon)
      implicit none
      save

c..equation 43 of su & olson jqsrt 1996

c..declare the pass
      double precision eta,epsilon

c..local variables
      double precision gamma_three

      theta_three = acos(sqrt(3.0d0 /
     &                 (3.0d0 + 4.0d0*gamma_three(eta,epsilon)**2)))

      return
      end







      double precision function gamma_one(eta,epsilon)
      implicit none
      save

c..equation 37 of su & olson jqsrt 1996

c..declare the pass
      double precision eta,epsilon

c..local variables
      double precision ein,tiny
      parameter        (tiny = 1.0d-14)

      ein = max(tiny,min(eta,1.0d0-tiny))
      gamma_one = ein * sqrt(epsilon + 1.0d0/(1.0d0 - ein*ein))

      return
      end




      double precision function gamma_two(eta,epsilon)
      implicit none
      save

c..equation 37 of su & olson jqsrt 1996

c..declare the pass
      double precision eta,epsilon

c..local variables
      double precision ein,tiny
      parameter        (tiny = 1.0d-14)

      ein = max(tiny,min(eta,1.0d0-tiny))
      gamma_two = sqrt((1.0d0 - ein) * (epsilon + 1.0d0/ein))

      return
      end



      double precision function gamma_three(eta,epsilon)
      implicit none
      save

c..equation 43 of su & olson jqsrt 1996

c..declare the pass
      double precision eta,epsilon

c..local variables
      double precision ein,tiny
      parameter        (tiny = 1.0d-14)

      ein = max(tiny,min(eta,1.0d0-tiny))
      gamma_three = sqrt((1.0d0 - ein*ein)*(epsilon + 1.0d0/(ein*ein)))

      return
      end






      subroutine midpnt(func,a,b,s,n)
      implicit none
      save

c..this routine computes the n'th stage of refinement of an extended midpoint
c..rule. func is input as the name of the function to be integrated between
c..limits a and b. when called with n=1, the routine returns as s the crudest
c..estimate of the integralof func from a to b. subsequent calls with n=2,3...
c..improve the accuracy of s by adding 2/3*3**(n-1) addtional interior points.

c..declare
      external          func
      integer           n,it,j
      double precision  func,a,b,s,tnm,del,ddel,x,sum

      if (n.eq.1) then
       s  = (b-a) * func(0.5d0*(a+b))
      else
       it   = 3**(n-2)
       tnm  = it
       del  = (b-a)/(3.0d0*tnm)
       ddel = del + del
       x    = a + (0.5d0 * del)
       sum  = 0.0d0
       do j=1,it
        sum = sum + func(x)
        x   = x + ddel
        sum = sum + func(x)
        x   = x + del
       enddo
       s  = (s + ((b-a) * sum/tnm)) / 3.0d0
      end if
      return
      end




      subroutine qromo(func,a,b,eps,ss,choose)
      implicit none
      save

c..this routine returns as s the integral of the function func from a to b
c..with fractional accuracy eps.
c..jmax limits the number of steps; nsteps = 3**(jmax-1)
c..integration is done via romberg algorithm.

c..it is assumed the call to choose triples the number of steps on each call
c..and that its error series contains only even powers of the number of steps.
c..the external choose may be any of the above drivers, i.e midpnt,midinf...


c..declare
      external          choose,func
      integer           j,jmax,jmaxp,k,km
      parameter         (jmax=14, jmaxp=jmax+1, k=5, km=k-1)
      double precision  a,b,ss,s(jmaxp),h(jmaxp),eps,dss,func

      h(1) = 1.0d0
      do j=1,jmax
       call choose(func,a,b,s(j),j)
       if (j .ge. k) then
        call polint(h(j-km),s(j-km),k,0.0d0,ss,dss)
        if (abs(dss) .le. eps*abs(ss)) return
       end if
       s(j+1) = s(j)
       h(j+1) = h(j)/9.0d0
      enddo
c      write(6,*)  'too many steps in qromo'
      return
      end








      subroutine polint(xa,ya,n,x,y,dy)
      implicit none
      save

c..given arrays xa and ya of length n and a value x, this routine returns a
c..value y and an error estimate dy. if p(x) is the polynomial of degree n-1
c..such that ya = p(xa) ya then the returned value is y = p(x)

c..declare
      integer          n,nmax,ns,i,m
      parameter        (nmax=10)
      double precision xa(n),ya(n),x,y,dy,c(nmax),d(nmax),dif,dift,
     1                 ho,hp,w,den

c..find the index ns of the closest table entry; initialize the c and d tables
      ns  = 1
      dif = abs(x - xa(1))
      do i=1,n
       dift = abs(x - xa(i))
       if (dift .lt. dif) then
        ns  = i
        dif = dift
       end if
       c(i)  = ya(i)
       d(i)  = ya(i)
      enddo

c..first guess for y
      y = ya(ns)

c..for each column of the table, loop over the c's and d's and update them
      ns = ns - 1
      do m=1,n-1
       do i=1,n-m
        ho   = xa(i) - x
        hp   = xa(i+m) - x
        w    = c(i+1) - d(i)
        den  = ho - hp
        if (den .eq. 0.0) stop ' 2 xa entries are the same in polint'
        den  = w/den
        d(i) = hp * den
        c(i) = ho * den
       enddo

c..after each column is completed, decide which correction c or d, to add
c..to the accumulating value of y, that is, which path to take in the table
c..by forking up or down. ns is updated as we go to keep track of where we
c..are. the last dy added is the error indicator.
       if (2*ns .lt. n-m) then
        dy = c(ns+1)
       else
        dy = d(ns)
        ns = ns - 1
       end if
       y = y + dy
      enddo
      return
      end






      double precision function zbrent(func,x1,x2,tol,niter)
      implicit none
      save


c..using brent's method this routine finds the root of a function func
c..between the limits x1 and x2. the root is when accuracy is less than tol.
c..
c..note: eps the the machine floating point precision

c..declare
      external          func
      integer           niter,itmax,iter
      parameter         (itmax = 100)
      double precision  func,x1,x2,tol,a,b,c,d,e,fa,
     1                  fb,fc,xm,tol1,p,q,r,s,eps
      parameter         (eps=3.0d-15)

c..initialize
      niter = 0
      a     = x1
      b     = x2
      fa    = func(a)
      fb    = func(b)
      if ( (fa .gt. 0.0  .and. fb .gt. 0.0)  .or.
     1     (fa .lt. 0.0  .and. fb .lt. 0.0)       ) then
       write(6,100) x1,fa,x2,fb
100    format(1x,' x1=',1pe11.3,' f(x1)=',1pe11.3,/,
     1        1x,' x2=',1pe11.3,' f(x2)=',1pe11.3)
       stop 'root not bracketed in routine zbrent'
      end if
      c  = b
      fc = fb

c..rename a,b,c and adjusting bound interval d
      do iter =1,itmax
       niter = niter + 1
       if ( (fb .gt. 0.0  .and. fc .gt. 0.0)  .or.
     1      (fb .lt. 0.0  .and. fc .lt. 0.0)      ) then
        c  = a
        fc = fa
        d  = b-a
        e  = d
       end if
       if (abs(fc) .lt. abs(fb)) then
        a  = b
        b  = c
        c  = a
        fa = fb
        fb = fc
        fc = fa
       end if
       tol1 = 2.0d0 * eps * abs(b) + 0.5d0 * tol
       xm   = 0.5d0 * (c-b)

c..convergence check
       if (abs(xm) .le. tol1 .or. fb .eq. 0.0) then
        zbrent = b
        return
       end if

c..attempt quadratic interpolation
       if (abs(e) .ge. tol1 .and. abs(fa) .gt. abs(fb)) then
        s = fb/fa
        if (a .eq. c) then
         p = 2.0d0 * xm * s
         q = 1.0d0 - s
        else
         q = fa/fc
         r = fb/fc
         p = s * (2.0d0 * xm * q *(q-r) - (b-a)*(r - 1.0d0))
         q = (q - 1.0d0) * (r - 1.0d0) * (s - 1.0d0)
        end if

c..check if in bounds
        if (p .gt. 0.0) q = -q
        p = abs(p)

c..accept interpolation
        if (2.0d0*p .lt. min(3.0d0*xm*q - abs(tol1*q),abs(e*q))) then
         e = d
         d = p/q

c..or bisect
        else
         d = xm
         e = d
        end if

c..bounds decreasing to slowly use bisection
       else
        d = xm
        e = d
       end if

c..move best guess to a
       a  = b
       fa = fb
       if (abs(d) .gt. tol1) then
        b = b + d
       else
        b = b + sign(tol1,xm)
       end if
       fb = func(b)
      enddo
      stop 'too many iterations in routine zbrent'
      end





      double precision function value(string)
      implicit none
      save


c..this routine takes a character string and converts it to a real number.
c..on error during the conversion, a fortran stop is issued

c..declare
      logical          pflag
      character*(*)    string
      character*1      plus,minus,decmal,blank,se,sd,se1,sd1
      integer          noblnk,long,ipoint,power,psign,iten,j,z,i
      double precision x,sign,factor,rten,temp
      parameter        (plus = '+'  , minus = '-' , decmal = '.'   ,
     1                  blank = ' ' , se = 'e'    , sd = 'd'       ,
     2                  se1 = 'E'   , sd1 = 'D'   , rten =  10.0,
     3                  iten = 10                                   )

c..initialize
      x      =  0.0d0
      sign   =  1.0d0
      factor =  rten
      pflag  =  .false.
      noblnk =  0
      power  =  0
      psign  =  1
      long   =  len(string)


c..remove any leading blanks and get the sign of the number
      do z = 1,7
       noblnk = noblnk + 1
       if ( string(noblnk:noblnk) .eq. blank) then
        if (noblnk .gt. 6 ) goto  30
       else
        if (string(noblnk:noblnk) .eq. plus) then
         noblnk = noblnk + 1
        else if (string(noblnk:noblnk) .eq. minus) then
         noblnk = noblnk + 1
         sign =  -1.0d0
        end if
        goto 10
       end if
      enddo


c..main number conversion loop
 10   continue
      do i = noblnk,long
       ipoint = i + 1


c..if a blank character then we are done
       if ( string(i:i) .eq. blank ) then
        x     = x * sign
        value = x
        return


c..if an exponent character, process the whole exponent, and return
       else if (string(i:i).eq.se  .or. string(i:i).eq.sd .or.
     1          string(i:i).eq.se1 .or. string(i:i).eq.sd1   ) then
        if (x .eq. 0.0 .and. ipoint.eq.2)     x = 1.0d0
        if (sign .eq. -1.0 .and. ipoint.eq.3) x = 1.0d0
        if (string(ipoint:ipoint) .eq. plus) ipoint = ipoint + 1
        if (string(ipoint:ipoint) .eq. minus) then
         ipoint = ipoint + 1
         psign = -1
        end if
        do z = ipoint,long
         if (string(z:z) .eq. blank)  then
          x = sign * x * rten**(power*psign)
          value = x
          return
         else
          j = ichar(string(z:z)) - 48
          if ( (j.lt.0) .or. (j.gt.9) ) goto 30
          power= (power * iten)  + j
         end if
        enddo


c..if an ascii number character, process ie
       else if (string(i:i) .ne. decmal) then
        j = ichar(string(i:i)) - 48
        if ( (j.lt.0) .or. (j.gt.9) ) goto 30
        if (.not.(pflag) ) then
         x = (x*rten) + j
        else
         temp   = j
         x      = x + (temp/factor)
         factor = factor * rten
         goto 20
        end if

c..must be a decimal point if none of the above
c..check that there are not two decimal points!
       else
        if (pflag) goto 30
        pflag = .true.
       end if
 20   continue
      end do

c..if we got through the do loop ok, then we must be done
      x     = x * sign
      value = x
      return


c..error processing the number
 30   write(6,40) long,string(1:long)
 40   format(' error converting the ',i4,' characters ',/,
     1       ' >',a,'< ',/,
     2       ' into a real number in function value')
      stop ' error in routine value'
      end

