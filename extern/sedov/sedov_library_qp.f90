      subroutine sed_1d(time,nstep,xpos, &
                        eblast,omega_in,xgeom_in, &
                        rho0,vel0,ener0,pres0,cs0,gam0, &
                        rshock,rho2,u2,e2,p2,cs2,rvacuum, &
                        den,ener,pres,vel,cs)
      implicit none


! this routine produces 1d solutions for a sedov blast wave propagating
! through a density gradient rho = rho**(-omega)
! in planar, cylindrical or spherical geometry
! for the standard, transitional and vaccum cases:

! standard     : a nonzero solution extends from the shock to the origin, where the pressure is finite.
! transitional : a nonzero solution extends from the shock to the origin, where the pressure vanishes.
! vacuum       : a nonzero solution extends from the shock to a boundary point, where the density vanishes.


! input:
! time     = temporal point where solution is desired seconds
! xpos(i)  = spatial points where solution is desired cm
! eblast   = energy of blast erg
! rho0     = ambient density g/cm**3    rho = rho0 * r**(-omega_in)
! omegain  = density power law exponent rho = rho0 * r**(-omega_in)
! vel0     = ambient material speed cm/s
! pres0    = ambient pressure erg/cm**3
! cs0      = ambient sound speed cm/s
! gam0     = gamma law equation of state
! xgeom_in = geometry factor, 3=spherical, 2=cylindircal, 1=planar


! for efficiency (i.e., doing the energy integrals only once),
! this routine returns the solution for an array of spatial points
! at the desired time point.

! output:
! den(i)  = density  g/cm**3
! ener(i) = specific internal energy erg/g
! pres(i) = presssure erg/cm**3
! vel(i)  = velocity cm/s
! cs(i)   = sound speed cm/s


! this routine is based upon two papers:
! "evaluation of the sedov-von neumann-taylor blast wave solution"
! jim kamm, la-ur-00-6055
! "the sedov self-similiar point blast solutions in nonuniform media"
! david book, shock waves, 4, 1, 1994

! although the ordinary differential equations are analytic,
! the sedov expressions appear to become singular for various
! combinations of parameters and at the lower limits of the integration
! range. all these singularies are removable and done so by this routine.

! real*16 because the real*8 implementations run out of precision, for example,
! "near" the origin in the standard case or the transition region in the vacuum case.


! declare the pass
      integer      ::  nstep
      real*16      ::  time,xpos(*), &
                       eblast,rho0,omega_in,vel0,ener0,pres0,cs0, &
                       gam0,xgeom_in,rshock,rho2,u2,e2,p2,cs2,rvacuum, &
                       den(*),ener(*),pres(*),vel(*),cs(*)

! local variables
      external     ::  midpnt,midpowl,midpowl2,sed_v_find,sed_r_find, &
                       efun01,efun02
      integer      ::  i
      real*16      ::  efun01,efun02,eval1,eval2
      real*16      ::  v0,v2,vstar,vmin,alpha,us, &
                       zeroin,sed_v_find,sed_r_find, &
                       vat,l_fun,dlamdv,f_fun,g_fun,h_fun, &
                       denom2,denom3,rho1


! eps controls the integration accuracy, don't get too greedy or the number
! of function evaluations required kills.
! eps2 controls the root find accuracy
! osmall controls the size of transition regions

      integer, parameter :: iprint = 1
      real*16, parameter :: eps    = 1.0q-10, &
                            eps2   = 1.0q-28, &
                            osmall = 1.0q-4, &
                            pi     = 3.1415926535897932384626433832795029q0


! common block communication
      include 'sedov3_qp.dek'


! common block communication with the integration stepper
      real*16        :: gam_int
      common /cmidp/    gam_int


! popular formats
 87   format(1x,1p10e14.6)
 88   format(1x,8(a7,1pe14.6,' '))


! initialize the solution
      do i=1,nstep
       den(i)  = 0.0q0
       vel(i)  = 0.0q0
       pres(i) = 0.0q0
       ener(i) = 0.0q0
       cs(i)   = 0.0q0
      end do


! return on unphysical cases
! infinite mass
      if (omega_in .ge. xgeom_in) return



! transfer the pass to common block and create some frequent combinations
      gamma  = gam0
      gamm1  = gamma - 1.0q0
      gamp1  = gamma + 1.0q0
      gpogm  = gamp1 / gamm1
      xgeom  = xgeom_in
      omega  = omega_in
      xg2    = xgeom + 2.0q0 - omega
      denom2 = 2.0q0*gamm1 + xgeom - gamma*omega
      denom3 = xgeom * (2.0q0 - gamma) - omega


! post shock location v2 and location of transitional point vstar
! kamm equation 18 and 19

      v2    = 4.0q0 / (xg2 * gamp1)
      vstar = 2.0q0 / (gamm1*xgeom + 2.0q0)


! set logicals that determines the type of solution

      lstandard     = .false.
      ltransitional = .false.
      lvacuum       = .false.

      if (abs(v2 - vstar) .le. osmall) then
       ltransitional = .true.
       if (iprint .eq. 1) write(6,*) 'transitional'
      else if (v2 .lt. vstar - osmall) then
       lstandard = .true.
       if (iprint .eq. 1) write(6,*) 'standard'
      else if (v2 .gt. vstar + osmall) then
       lvacuum = .true.
       if (iprint .eq. 1) write(6,*) 'vacuum'
      end if

! two apparent singularies, book's notation for omega2 and omega3
! resetting denom2 and denom3 is harmless as these two singularities are explicitely addressed

      lomega2 = .false.
      lomega3 = .false.

      if (abs(denom2) .le. osmall) then
       if (iprint .eq. 1) write(6,'(a,1pe12.4)') 'omega = omega2 ',denom2
       lomega2 = .true.
       denom2  = 1.0q-8
       
      else if (abs(denom3) .le. osmall) then
       if (iprint .eq. 1) write(6,'(a,1pe12.4)') 'omega = omega3 ',denom3
       lomega3 = .true.
       denom3  = 1.0q-8
      end if


! various exponents, kamm equations 42-47
! in terms of book's notation:
! a0=beta6 a1=beta1  a2=-beta2 a3=beta3 a4=beta4 a5=-beta5

      a0  = 2.0q0/xg2
      a2  = -gamm1/denom2
      a1  =  xg2*gamma/(2.0q0 + xgeom*gamm1) * &
            (((2.0q0*(xgeom*(2.0q0-gamma) - omega))/(gamma*xg2*xg2))-a2)
      a3  = (xgeom - omega) / denom2
      a4  = xg2 * (xgeom - omega) * a1 /denom3
      a5  = (omega*gamp1 - 2.0q0*xgeom)/denom3


! frequent combinations, kamm equations 33-37
      a_val = 0.25q0 * xg2 * gamp1
      b_val = gpogm
      c_val = 0.5q0 * xg2 * gamma
      d_val = (xg2 * gamp1)/(xg2*gamp1 - 2.0q0*(2.0q0 + xgeom*gamm1))
      e_val = 0.5q0 * (2.0q0 + xgeom * gamm1)



! evaluate the energy integrals
! the transitional case can be done by hand; save some cpu cycles
! kamm equations 80, 81, and 85

      if (ltransitional) then

       eval2 = gamp1/(xgeom*(gamm1*xgeom + 2.0q0)**2)
       eval1 = 2.0q0/gamm1 * eval2
       alpha = gpogm * 2**(xgeom)/(xgeom*(gamm1*xgeom + 2.0q0)**2)
       if (int(xgeom) .ne. 1) alpha = pi * alpha



! for the standard or vacuum cases
! v0 = post-shock origin v0 and vv = vacuum boundary vv
! set the radius corespondin to vv to zero for now
! kamm equations 18, and 28.

      else
       v0  = 2.0q0 / (xg2 * gamma)
       vv  = 2.0q0 / xg2
       rvv = 0.0q0
       if (lstandard) vmin = v0
       if (lvacuum)   vmin  = vv



! the first energy integral
! in the standard case the term (c_val*v - 1) might be singular at v=vmin

       if (lstandard) then
        gam_int = a3 - a2*xg2 - 1.0q0
        if (gam_int .ge. 0) then
         call qromo(efun01,vmin,v2,eps,eval1,midpnt)
        else
         gam_int = abs(gam_int)
         call qromo(efun01,vmin,v2,eps,eval1,midpowl)
        end if

! in the vacuum case the term (1 - c_val/gamma*v) might be singular at v=vmin

       else if (lvacuum) then
        gam_int = a5
        if (gam_int .ge. 0) then
         call qromo(efun01,vmin,v2,eps,eval1,midpnt)
        else
         gam_int = abs(gam_int)
         call qromo(efun01,vmin,v2,eps,eval1,midpowl2)
        end if
       end if



! the second energy integral
! in the standard case the term (c_val*v - 1) might be singular at v=vmin

       if (lstandard) then
        gam_int = a3 - a2*xg2 - 2.0q0
        if (gam_int .ge. 0) then
         call qromo(efun02,vmin,v2,eps,eval2,midpnt)
        else
         gam_int = abs(gam_int)
         call qromo(efun02,vmin,v2,eps,eval2,midpowl)
        end if

! in the vacuum case the term (1 - c_val/gamma*v) might be singular at v=vmin

       else if (lvacuum) then
        gam_int = a5
        if (gam_int .ge. 0) then
         call qromo(efun02,vmin,v2,eps,eval2,midpnt)
        else
         gam_int = abs(gam_int)
         call qromo(efun02,vmin,v2,eps,eval2,midpowl2)
        end if
       end if


! kamm, bolstad & timmes equation 57 for alpha
       if (int(xgeom) .eq. 1) then

! bug reported Irina Sagert march 2019
!        alpha = 0.5q0*eval1 + eval2/gamm1

        alpha = eval1 + 2.0q0*eval2/gamm1

       else
        alpha = (xgeom - 1.0q0) * pi * (eval1 + 2.0q0 * eval2/gamm1)
       end if
      end if


! write what we have for the energy integrals
      if (iprint .eq. 1) &
          write(6,88) 'xgeom =',xgeom,'gamma=',gamma, &
                      'omega =',omega,'alpha =',alpha, &
                      'j1    =',eval1,'j2    =',eval2




! immediate post-shock values
! kamm page 14 or equations 14, 16, 5, 13
! r2 = shock position, u2 = shock speed, rho1 = pre-shock density,
! u2 = post-shock material speed, rho2 = post-shock density,
! p2 = post-shock pressure, e2 = post-shoock specific internal energy,
! and cs2 = post-shock sound speed

      r2   = (eblast/(alpha*rho0))**(1.0q0/xg2) * time**(2.0q0/xg2)
      rshock = r2
      us   = (2.0q0/xg2) * r2 / time
      rho1 = rho0 * r2**(-omega)
      u2   = 2.0q0 * us / gamp1
      rho2 = gpogm * rho1
      p2   = 2.0q0 * rho1 * us**2 / gamp1
      e2   = p2/(gamm1*rho2)
      cs2  = sqrt(gamma*p2/rho2)


! find the radius corresponding to vv
       if (lvacuum)   then
        vwant = vv
        rvv = zeroin(0.0q0,r2,sed_r_find,eps2)
        rvacuum = rvv
       end if

! write a summary
      if (lstandard .and. iprint .eq. 1) &
           write(6,88) 'r2    =',r2,'rho2  =',rho2, &
                       'u2    =',u2,'e2    =',e2, &
                       'p2    =',p2,'cs2   =',cs2

      if (lvacuum .and. iprint .eq. 1) &
           write(6,88) &
                       'rv    =',rvv, &
                       'r2    =',r2,'rho2  =',rho2, &
                       'u2    =',u2,'e2    =',e2, &
                       'p2    =',p2,'cs2   =',cs2




! now start the loop over spatial positions
      do i=1,nstep
       rwant  = xpos(i)


! if we are upstream from the shock front
       if (rwant .gt. r2) then
        den(i)  = rho0 * rwant**(-omega)
        vel(i)  = vel0
        pres(i) = pres0
        ener(i) = ener0
        cs(i)   = cs0


! if we are between the origin and the shock front
! find the correct similarity value for this radius in the standard or vacuum cases
       else
        if (lstandard) then
         vat = zeroin(0.90q0*v0,v2,sed_v_find,eps2)
        else if (lvacuum) then
         vat = zeroin(v2,1.2q0*vv,sed_v_find,eps2)
        end if

! the physical solution
        call sedov_funcs(vat,l_fun,dlamdv,f_fun,g_fun,h_fun)
        den(i)   = rho2 * g_fun
        vel(i)   = u2   * f_fun
        pres(i)  = p2   * h_fun
!        den(i)   = g_fun
!        vel(i)   = f_fun
!        pres(i)  = h_fun
        ener(i)  = 0.0q0
        cs(i)    = 0.0q0
        if (den(i) .ne. 0.0) then
         ener(i)  = pres(i) / (gamm1 * den(i))
         cs(i)    = sqrt(gamma * pres(i)/den(i))
        end if
       end if

! end of loop over positions
      enddo

      return
      end subroutine sed_1d






      real*16 function efun01(v)
      implicit none
      save

! evaluates the first energy integrand, kamm equations 67 and 10.
! the (c_val*v - 1) term might be singular at v=vmin in the standard case.
! the (1 - c_val/gamma * v) term might be singular at v=vmin in the vacuum case.
! due care should be taken for these removable singularities by the integrator.

! declare the pass
      real*16 :: v

! common block communication
      include 'sedov3_qp.dek'

! local variables
      real*16 :: l_fun,dlamdv,f_fun,g_fun,h_fun

! go
      call sedov_funcs(v,l_fun,dlamdv,f_fun,g_fun,h_fun)
      efun01 = dlamdv * l_fun**(xgeom + 1.0q0) * gpogm * g_fun * v**2

      return
      end function efun01






      real*16 function efun02(v)
      implicit none
      save

! evaluates the second energy integrand, kamm equations 68 and 11.
! the (c_val*v - 1) term might be singular at v=vmin in the standard case.
! the (1 - c_val/gamma * v) term might be singular at v=vmin in the vacuum case.
! due care should be taken for these removable singularities by the integrator.

! declare the pass
      real*16  :: v


! common block communication
      include 'sedov3_qp.dek'

! local variables
      real*16 :: l_fun,dlamdv,f_fun,g_fun,h_fun,z

! go
      call sedov_funcs(v,l_fun,dlamdv,f_fun,g_fun,h_fun)
      z = 8.0q0/( (xgeom + 2.0q0 - omega)**2 * gamp1)
      efun02 = dlamdv * l_fun**(xgeom - 1.0q0 ) * h_fun * z

      return
      end function efun02







      real*16 function sed_v_find(v)
      implicit none
      save

! given corresponding physical distances, find the similarity variable v
! kamm equation 38 as a root find

! declare the pass
      real*16  :: v


! common block communication
      include 'sedov3_qp.dek'

! local variables
      real*16 :: l_fun,dlamdv,f_fun,g_fun,h_fun


      call sedov_funcs(v,l_fun,dlamdv,f_fun,g_fun,h_fun)
      sed_v_find = r2*l_fun - rwant

      return
      end function sed_v_find





      real*16 function sed_r_find(r)
      implicit none
      save

! given the similarity variable v, find the corresponding physical distance
! kamm equation 38 as a root find

! declare the pass
      real*16  :: r


! common block communication
      include 'sedov3_qp.dek'


! local variables
      real*16 :: l_fun,dlamdv,f_fun,g_fun,h_fun

      call sedov_funcs(vwant,l_fun,dlamdv,f_fun,g_fun,h_fun)
      sed_r_find = r2*l_fun - r

      return
      end function sed_r_find








      real*16 function sed_lam_find(v)
      implicit none
      save

! given the similarity variable v, find the corresponding physical distance
! kamm equation 38 as a root find

! declare the pass
      real*16  :: v


! common block communication
      include 'sedov3_qp.dek'


! local variables
      real*16 :: l_fun,dlamdv,f_fun,g_fun,h_fun


      call sedov_funcs(v,l_fun,dlamdv,f_fun,g_fun,h_fun)
      sed_lam_find = l_fun - xlam_want

      return
      end function sed_lam_find







      subroutine sedov_funcs(v,l_fun,dlamdv,f_fun,g_fun,h_fun)
      implicit none
      save

! given the similarity variable v, returns functions
! lambda, f, g, and h and the derivative of lambda with v dlamdv

! although the ordinary differential equations are analytic,
! the sedov expressions appear to become singular for various
! combinations of parameters and at the lower limits of the integration
! range. all these singularies are removable and done so by this routine.


! declare the pass
      real*16            :: v,l_fun,dlamdv,f_fun,g_fun,h_fun


! common block communication
      include 'sedov3_qp.dek'


! local variables
      real*16            :: x1,x2,x3,x4,dx1dv,dx2dv,dx3dv,dx4dv, &
                            cbag,ebag,beta0,pp1,pp2,pp3,pp4,c2,c6,y,z, &
                            dpp2dv
      real*16, parameter :: eps = 1.0q-30


! frequent combinations and their derivative with v
! kamm equation 29-32, x4 a bit different to save a divide
! x1 is book's F

       x1 = a_val * v
       dx1dv = a_val

       cbag = max(eps, c_val * v - 1.0q0)
       x2 = b_val * cbag
       dx2dv = b_val * c_val

       ebag = 1.0q0 - e_val * v
       x3 = d_val * ebag
       dx3dv = -d_val * e_val

       x4 = b_val * (1.0q0 - 0.5q0 * xg2 *v)
       dx4dv = -b_val * 0.5q0 * xg2


! transition region between standard and vacuum cases
! kamm page 15 or equations 88-92
! lambda = l_fun is book's zeta
! f_fun is books V, g_fun is book's D, h_fun is book's P

      if (ltransitional) then
       l_fun  = rwant/r2
       dlamdv = 0.0q0
       f_fun  = l_fun
       g_fun  = l_fun**(xgeom - 2.0q0)
       h_fun  = l_fun**xgeom



! for the vacuum case in the hole
      else if (lvacuum .and. rwant .lt. rvv) then

       l_fun  = 0.0q0
       dlamdv = 0.0q0
       f_fun  = 0.0q0
       g_fun  = 0.0q0
       h_fun  = 0.0q0



! omega = omega2 = (2*(gamma -1) + xgeom)/gamma case, denom2 = 0
! book expressions 20-22

      else if (lomega2) then

       beta0 = 1.0q0/(2.0q0 * e_val)
       pp1   = gamm1 * beta0
       c6    = 0.5q0 * gamp1
       c2    = c6/gamma
       y     = 1.0q0/(x1 - c2)
       z     = (1.0q0 - x1)*y
       pp2   = gamp1 * beta0 * z
       dpp2dv = -gamp1 * beta0 * dx1dv * y * (1.0q0 + z)
       pp3   = (4.0q0 - xgeom - 2.0q0*gamma) * beta0
       pp4   = -xgeom * gamma * beta0

       l_fun = x1**(-a0) * x2**(pp1) * exp(pp2)
       dlamdv = (-a0*dx1dv/x1 + pp1*dx2dv/x2 + dpp2dv) * l_fun
       f_fun = x1 * l_fun
       g_fun = x1**(a0*omega) * x2**pp3 * x4**a5 * exp(-2.0q0*pp2)
       h_fun = x1**(a0*xgeom) * x2**pp4 * x4**(1.0q0 + a5)



! omega = omega3 = xgeom*(2 - gamma) case, denom3 = 0
! book expressions 23-25

      else if (lomega3) then

       beta0 = 1.0q0/(2.0q0 * e_val)
       pp1   = a3 + omega * a2
       pp2   = 1.0q0 - 4.0q0 * beta0
       c6    = 0.5q0 * gamp1
       pp3   = -xgeom * gamma * gamp1 * beta0 * (1.0q0 - x1)/(c6 - x1)
       pp4   = 2.0q0 * (xgeom * gamm1 - gamma) * beta0

       l_fun = x1**(-a0) * x2**(-a2) * x4**(-a1)
       dlamdv = -(a0*dx1dv/x1 + a2*dx2dv/x2 + a1*dx4dv/x4) * l_fun
       f_fun = x1 * l_fun
       g_fun = x1**(a0*omega) * x2**pp1 * x4**pp2 * exp(pp3)
       h_fun = x1**(a0*xgeom) * x4**pp4 * exp(pp3)


! for the standard or vacuum case not in the hole
! kamm equations 38-41

      else
       l_fun = x1**(-a0) * x2**(-a2) * x3**(-a1)
       dlamdv = -(a0*dx1dv/x1 + a2*dx2dv/x2 + a1*dx3dv/x3) * l_fun
       f_fun = x1 * l_fun
       if (x4 .eq. 0.0 .and. a5 .lt. 0.0) then
        g_fun = 1.0q30
       else
        g_fun = x1**(a0*omega)*x2**(a3+a2*omega)*x3**(a4+a1*omega)*x4**a5
       end if
       h_fun = x1**(a0*xgeom)*x3**(a4+a1*(omega-2.0q0))*x4**(1.0q0 + a5)
      end if

      return
      end subroutine sedov_funcs








      subroutine midpnt(func,a,b,s,n)
      implicit none
      save

! this routine computes the n'th stage of refinement of an extended midpoint
! rule. func is input as the name of the function to be integrated between
! limits a and b. when called with n=1, the routine returns as s the crudest
! estimate of the integralof func from a to b. subsequent calls with n=2,3...
! improve the accuracy of s by adding 2/3*3**(n-1) addtional interior points.

! declare the pass
      external       :: func
      integer        :: n
      real*16        :: func,a,b,s

! local variables

      integer        :: it,j
      real*16        :: tnm,del,ddel,x,asum

      if (n.eq.1) then
       s  = (b-a) * func(0.5q0*(a+b))
      else
       it   = 3**(n-2)
       tnm  = it
       del  = (b-a)/(3.0q0*tnm)
       ddel = del + del
       x    = a + (0.5q0 * del)
       asum  = 0.0q0
       do j=1,it
        asum = asum + func(x)
        x   = x + ddel
        asum = asum + func(x)
        x   = x + del
       enddo
       s  = (s + ((b-a) * asum/tnm)) / 3.0q0
      end if
      return
      end  subroutine midpnt







      subroutine midpowl(funk,aa,bb,s,n)
      implicit none
      save

! this routine is an exact replacement for midpnt, except that it allows for
! an integrable power-law singularity of the form (x - a)**(-gam_int)
! at the lower limit aa for 0 < gam_int < 1.

! declare the pass
      external       :: funk
      integer        :: n
      real*16        :: funk,aa,bb,s

! local variables
      integer        :: it,j
      real*16        :: func,a,b,tnm,del,ddel,x,asum


! common block communication
      real*16          gam_int
      common /cmidp/   gam_int


! statement function,  recipe equation 4.4.3
      func(x) = 1.0q0/(1.0q0 - gam_int) * x**(gam_int/(1.0q0 - gam_int)) &
                * funk(x**(1.0q0/(1.0q0 - gam_int)) + aa)

      b = (bb - aa)**(1.0q0 - gam_int)
      a = 0.0q0

! now exactly as midpnt
      if (n .eq. 1) then
       s = (b-a) * func(0.5q0*(a+b))
      else
       it   = 3**(n-2)
       tnm = it
       del = (b-a)/(3.0q0*tnm)
       ddel = del + del
       x = a + (0.5q0 * del)
       asum = 0.0q0
       do j=1,it
        asum = asum + func(x)
        x   = x + ddel
        asum = asum + func(x)
        x   = x + del
       enddo
       s = (s + ((b-a) * asum/tnm)) / 3.0q0
      end if
      return
      end subroutine midpowl




      subroutine midpowl2(funk,aa,bb,s,n)
      implicit none
      save

! this routine is an exact replacement for midpnt, except that it allows for
! an integrable power-law singularity of the form (a - x)**(-gam_int)
! at the lower limit aa for 0 < gam_int < 1.

! declare the pass
      external       :: funk
      integer        :: n
      real*16        :: funk,aa,bb,s

! local variables

      integer        :: it,j
      real*16        :: func,a,b,tnm,del,ddel,x,asum

! common block communication
      real*16          gam_int
      common /cmidp/   gam_int


! statement function and recipe equation 4.4.3
      func(x) = 1.0q0/(gam_int - 1.0q0) * x**(gam_int/(1.0q0 - gam_int)) &
                * funk(aa - x**(1.0q0/(1.0q0 - gam_int)))

      b = (aa - bb)**(1.0q0 - gam_int)
      a = 0.0q0

! now exactly as midpnt
      if (n .eq. 1) then
       s = (b-a) * func(0.5q0*(a+b))
      else
       it   = 3**(n-2)
       tnm = it
       del = (b-a)/(3.0q0*tnm)
       ddel = del + del
       x = a + (0.5q0 * del)

       asum = 0.0q0
       do j=1,it
        asum = asum + func(x)
        x   = x + ddel
        asum = asum + func(x)
        x   = x + del
       enddo
       s = (s + ((b-a) * asum/tnm)) / 3.0q0
      end if
      return
      end subroutine midpowl2






      subroutine qromo(func,a,b,eps,ss,choose)
      implicit none
      save

! this routine returns as s the integral of the function func from a to b
! with fractional accuracy eps.
! jmax limits the number of steps; nsteps = 3**(jmax-1)
! integration is done via romberg algorithm.

! it is assumed the call to choose triples the number of steps on each call
! and that its error series contains only even powers of the number of steps.
! the external choose may be any of the above drivers, i.e midpnt,midinf...

! declare the pass
      external           :: choose,func
      real*16            :: a,b,ss,eps,func

! local variables
      integer            :: i,j
      integer, parameter :: jmax=15, jmaxp=jmax+1, k=5, km=k-1
      real*16            :: s(jmaxp),h(jmaxp),dss


      h(1) = 1.0q0
      do j=1,jmax
       call choose(func,a,b,s(j),j)
       if (j .ge. k) then
        call polint(h(j-km),s(j-km),k,0.0q0,ss,dss)
        if (abs(dss) .le. eps*abs(ss)) return
       end if
       s(j+1) = s(j)
       h(j+1) = h(j)/9.0q0
      enddo
      write(6,*)  'too many steps in qromo'
      return
      end subroutine qromo








      subroutine polint(xa,ya,n,x,y,dy)
      implicit none
      save

! given arrays xa and ya of length n and a value x, this routine returns a
! value y and an error estimate dy. if p(x) is the polynomial of degree n-1
! such that ya = p(xa) ya then the returned value is y = p(x)

! declare the pass
      integer            :: n
      real*16            :: xa(n),ya(n),x,y,dy

! local variables
      integer            :: ns,i,m
      integer, parameter :: nmax=20
      real*16            :: c(nmax),d(nmax),dif,dift,ho,hp,w,den

! find the index ns of the closest table entry; initialize the c and d tables
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

! first guess for y
      y = ya(ns)

! for each column of the table, loop over the c's and d's and update them
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

! after each column is completed, decide which correction c or d, to add
! to the accumulating value of y, that is, which path to take in the table
! by forking up or down. ns is updated as we go to keep track of where we
! are. the last dy added is the error indicator.
       if (2*ns .lt. n-m) then
        dy = c(ns+1)
       else
        dy = d(ns)
        ns = ns - 1
       end if
       y = y + dy
      enddo
      return
      end subroutine polint







      real*16 function zeroin( ax, bx, f, tol)
      implicit real*16 (a-h,o-z)

!-----------------------------------------------------------------------
!
! This subroutine solves for a zero of the function  f(x)  in the
! interval ax,bx.
!
!  input..
!
!  ax     left endpoint of initial interval
!  bx     right endpoint of initial interval
!  f      function subprogram which evaluates f(x) for any x in
!         the interval  ax,bx
!  tol    desired length of the interval of uncertainty of the
!         final result ( .ge. 0.0q0)
!
!
!  output..
!
!  zeroin abcissa approximating a zero of  f  in the interval ax,bx
!
!
!      it is assumed  that   f(ax)   and   f(bx)   have  opposite  signs
!  without  a  check.  zeroin  returns a zero  x  in the given interval
!  ax,bx  to within a tolerance  4*macheps*abs(x) + tol, where macheps
!  is the relative machine precision.
!      this function subprogram is a slightly  modified  translation  of
!  the algol 60 procedure  zero  given in  richard brent, algorithms for
!  minimization without derivatives, prentice - hall, inc. (1973).
!
!-----------------------------------------------------------------------

! .. call list variables

      real*16  :: ax, bx, f, tol, a, b, c, d, e, eps, fa, fb, fc, tol1, xm, p, q, r, s
      external :: f

!----------------------------------------------------------------------

!
!  compute eps, the relative machine precision
!
      eps = 1.0q0
   10 eps = eps/2.0q0
      tol1 = 1.0q0 + eps
      if (tol1 .gt. 1.0q0) go to 10
!
! initialization
!
      a = ax
      b = bx
      fa = f(a)
      fb = f(b)

!      write(6,'(a,1p2e40.32)') ' fa fb ',fa, fb

!
! begin step
!
   20 c = a
      fc = fa
      d = b - a
      e = d
   30 if ( abs(fc) .ge.  abs(fb)) go to 40
      a = b
      b = c
      c = a
      fa = fb
      fb = fc
      fc = fa
!
! convergence test
!
   40 tol1 = 2.0q0*eps*abs(b) + 0.5q0*tol
      xm = 0.5q0*(c - b)

!      write(6,'(a,1p2e40.32)') ' xm tol1 ',abs(xm), tol1
!      write(6,*)

      if (abs(xm) .le. tol1) go to 90
      if (fb .eq. 0.0q0) go to 90
!
! is bisection necessary?
!
      if (abs(e) .lt. tol1) go to 70
      if (abs(fa) .le. abs(fb)) go to 70
!
! is quadratic interpolation possible?
!
      if (a .ne. c) go to 50
!
! linear interpolation
!
      s = fb/fa
      p = 2.0q0*xm*s
      q = 1.0q0 - s
      go to 60
!
! inverse quadratic interpolation
!
   50 q = fa/fc
      r = fb/fc
      s = fb/fa
      p = s*(2.0q0*xm*q*(q - r) - (b - a)*(r - 1.0q0))
      q = (q - 1.0q0)*(r - 1.0q0)*(s - 1.0q0)
!
! adjust signs
!
   60 if (p .gt. 0.0q0) q = -q
      p = abs(p)
!
! is interpolation acceptable?
!
      if ((2.0q0*p) .ge. (3.0q0*xm*q - abs(tol1*q))) go to 70
      if (p .ge. abs(0.5q0*e*q)) go to 70
      e = d
      d = p/q
      go to 80
!
! bisection
!
   70 d = xm
      e = d
!
! complete step
!
   80 a = b
      fa = fb
      if (abs(d) .gt. tol1) b = b + d
      if (abs(d) .le. tol1) b = b + Sign(tol1, xm)
      fb = f(b)
      if ((fb*(fc/abs(fc))) .gt. 0.0q0) go to 20
      go to 30
!
! done
!
   90 zeroin = b

      return
      end function zeroin









      real*16 function value(string)
      implicit none
      save


! this routine takes a character string and converts it to a real number.
! on error during the conversion, a fortran stop is issued

! declare
      logical            :: pflag
      character*(*)      :: string
      integer            :: noblnk,long,ipoint,power,psign,j,z,i
      integer, parameter :: iten = 10
      real*16            :: x,sign,factor,temp
      real*16, parameter :: rten =  10.0q0
      character*1, parameter :: plus = '+'  , minus = '-' , decmal = '.'   , &
                                blank = ' ' , &
                                se  = 'e'   , sd  = 'd'   , sq = 'q'       , &
                                se1 = 'E'   , sd1 = 'D'   , sq1 ='Q'
! initialize
      x      =  0.0q0
      sign   =  1.0q0
      factor =  rten
      pflag  =  .false.
      noblnk =  0
      power  =  0
      psign  =  1
      long   =  len(string)


! remove any leading blanks and get the sign of the number
      do z = 1,7
       noblnk = noblnk + 1
       if ( string(noblnk:noblnk) .eq. blank) then
        if (noblnk .gt. 6 ) goto  30
       else
        if (string(noblnk:noblnk) .eq. plus) then
         noblnk = noblnk + 1
        else if (string(noblnk:noblnk) .eq. minus) then
         noblnk = noblnk + 1
         sign =  -1.0q0
        end if
        goto 10
       end if
      enddo


! main number conversion loop
 10   continue
      do i = noblnk,long
       ipoint = i + 1


! if a blank character then we are done
       if ( string(i:i) .eq. blank ) then
        x     = x * sign
        value = x
        return


! if an exponent character, process the whole exponent, and return
       else if (string(i:i) .eq. se .or. &
                string(i:i) .eq. sd .or. &
                string(i:i) .eq. sq .or. &
                string(i:i) .eq. se1 .or. &
                string(i:i) .eq. sd1 .or. &
                string(i:i) .eq. sq1   ) then
        if (x .eq. 0.0 .and. ipoint.eq.2)     x = 1.0q0
        if (sign .eq. -1.0 .and. ipoint.eq.3) x = 1.0q0
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


! if an ascii number character, process ie
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

! must be a decimal point if none of the above
! check that there are not two decimal points!
       else
        if (pflag) goto 30
        pflag = .true.
       end if
 20   continue
      end do

! if we got through the do loop ok, then we must be done
      x     = x * sign
      value = x
      return


! error processing the number
 30   write(6,40) long,string(1:long)
 40   format(' error converting the ',i4,' characters ',/, &
             ' >',a,'< ',/, &
             ' into a real number in function value')
      stop ' error in routine value'
      end function value

