!!****if* source/physics/sourceTerms/Stir/StirFromFileMain/st_calcAccel
!!
!! NAME
!!  st_calcAccel
!!
!! SYNOPSIS
!!  st_calcAccel(integer, intent(in) :: blockid,
!!               integer, dimension(2,MDIM), intent(in) :: blkLimitsGC,
!!               real(:), intent(in) :: xcoord,
!!               real(:), intent(in) :: ycoord,
!!               real(:), intent(in) :: zcoord)
!!
!! DESCRIPTION
!!   Computes components of the driving force from Fourier modes
!!
!! ARGUMENTS
!!   blockID : ID of block in current processor
!!   blkLimitsGC : array holding the upper and lower index limits
!!                 of an entire block (including GCs)
!!   xcoord : real array containg the coordinates along IAXIS
!!   ycoord : real array containg the coordinates along JAXIS
!!   zcoord : real array containg the coordinates along KAXIS
!!
!! AUTHOR
!!  Christoph Federrath, 2008-2022
!!
!!***

subroutine st_calcAccel(blockID, blkLimits, blkLimitsGC, xCoord, yCoord, zCoord)

  use Stir_data
  use Grid_interface, ONLY : Grid_getBlkPtr, Grid_releaseBlkPtr

  implicit none

#include "constants.h"
#include "Flash.h"

  integer, dimension(2,MDIM), intent(in) :: blkLimits, blkLimitsGC
  integer, intent(in)                    :: blockID

  real, dimension(blkLimitsGC(LOW,IAXIS):blkLimitsGC(HIGH,IAXIS)), intent(in) :: xCoord
  real, dimension(blkLimitsGC(LOW,JAXIS):blkLimitsGC(HIGH,JAXIS)), intent(in) :: yCoord
  real, dimension(blkLimitsGC(LOW,KAXIS):blkLimitsGC(HIGH,KAXIS)), intent(in) :: zCoord

  ! pre-compute some mode info
  real, dimension(st_nmodes,blkLimitsGC(LOW,IAXIS):blkLimitsGC(HIGH,IAXIS)) :: cosxi, sinxi
  real, dimension(st_nmodes,blkLimitsGC(LOW,JAXIS):blkLimitsGC(HIGH,JAXIS)) :: cosxj, sinxj
  real, dimension(st_nmodes,blkLimitsGC(LOW,KAXIS):blkLimitsGC(HIGH,KAXIS)) :: cosxk, sinxk

  real, pointer, dimension(:,:,:,:) :: solndata
  real                              :: realtrigterms, imtrigterms
  integer                           :: ib, ie
  integer                           :: i, j, k, m
  real, dimension(st_nmodes)        :: ampl

  ! pre-compute some trigonometry
  do m = 1, st_nmodes

    cosxk(m,:) = cos(st_mode(3,m)*zCoord(:))
    sinxk(m,:) = sin(st_mode(3,m)*zCoord(:))

    cosxj(m,:) = cos(st_mode(2,m)*yCoord(:))
    sinxj(m,:) = sin(st_mode(2,m)*yCoord(:))

    cosxi(m,:) = cos(st_mode(1,m)*xCoord(:))
    sinxi(m,:) = sin(st_mode(1,m)*xCoord(:))

    ampl(m) = 2.0*st_ampl(m)

  enddo

  ib = blkLimits(LOW, IAXIS)
  ie = blkLimits(HIGH, IAXIS)

  call Grid_getBlkPtr(blockID,solnData)

  !! loop over all cells
  do k = blkLimits(LOW, KAXIS), blkLimits(HIGH, KAXIS)
    do j = blkLimits(LOW, JAXIS), blkLimits(HIGH, JAXIS)

#ifdef ACCX_VAR
      solnData(ACCX_VAR,ib:ie,j,k) = 0.0
#else
      accx(ib:ie,j,k) = 0.0
#endif
#ifdef ACCY_VAR
      solnData(ACCY_VAR,ib:ie,j,k) = 0.0
#else
      accy(ib:ie,j,k) = 0.0
#endif
#ifdef ACCZ_VAR
      solnData(ACCZ_VAR,ib:ie,j,k) = 0.0
#else
      accz(ib:ie,j,k) = 0.0
#endif

      do i = blkLimits(LOW, IAXIS), blkLimits(HIGH, IAXIS)

        do m = 1, st_nmodes

          !  these are the real and imaginary parts, respectively, of
          !     e^{ i \vec{k} \cdot \vec{x} }
          !          = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)

          realtrigterms =    ( cosxi(m,i)*cosxj(m,j) - sinxi(m,i)*sinxj(m,j) ) * cosxk(m,k)  &
                           - ( sinxi(m,i)*cosxj(m,j) + cosxi(m,i)*sinxj(m,j) ) * sinxk(m,k)

          imtrigterms   =    cosxi(m,i) * ( cosxj(m,j)*sinxk(m,k) + sinxj(m,j)*cosxk(m,k) ) &
                           + sinxi(m,i) * ( cosxj(m,j)*cosxk(m,k) - sinxj(m,j)*sinxk(m,k) )
#ifdef ACCX_VAR
          solnData(ACCX_VAR,i,j,k) = solnData(ACCX_VAR,i,j,k) + ampl(m)*(st_aka(IAXIS,m)*realtrigterms-st_akb(IAXIS,m)*imtrigterms)
#else
          accx(i,j,k)              = real(accx(i,j,k) + ampl(m)*(st_aka(IAXIS,m)*realtrigterms-st_akb(IAXIS,m)*imtrigterms),kind=4)
#endif
#ifdef ACCY_VAR
          solnData(ACCY_VAR,i,j,k) = solnData(ACCY_VAR,i,j,k) + ampl(m)*(st_aka(JAXIS,m)*realtrigterms-st_akb(JAXIS,m)*imtrigterms)
#else
          accy(i,j,k)              = real(accy(i,j,k) + ampl(m)*(st_aka(JAXIS,m)*realtrigterms-st_akb(JAXIS,m)*imtrigterms),kind=4)
#endif
#ifdef ACCZ_VAR
          solnData(ACCZ_VAR,i,j,k) = solnData(ACCZ_VAR,i,j,k) + ampl(m)*(st_aka(KAXIS,m)*realtrigterms-st_akb(KAXIS,m)*imtrigterms)
#else
          accz(i,j,k)              = real(accz(i,j,k) + ampl(m)*(st_aka(KAXIS,m)*realtrigterms-st_akb(KAXIS,m)*imtrigterms),kind=4)
#endif

        enddo ! end loop over modes

#ifdef ACCX_VAR
        solnData(ACCX_VAR,i,j,k) = st_solweightnorm*solnData(ACCX_VAR,i,j,k)
#else
        accx(i,j,k)              = real(st_solweightnorm,kind=4)*accx(i,j,k)
#endif
#ifdef ACCY_VAR
        solnData(ACCY_VAR,i,j,k) = st_solweightnorm*solnData(ACCY_VAR,i,j,k)
#else
        accy(i,j,k)              = real(st_solweightnorm,kind=4)*accy(i,j,k)
#endif
#ifdef ACCZ_VAR
        solnData(ACCZ_VAR,i,j,k) = st_solweightnorm*solnData(ACCZ_VAR,i,j,k)
#else
        accz(i,j,k)              = real(st_solweightnorm,kind=4)*accz(i,j,k)
#endif

      enddo ! end loop over i
    enddo ! end loop over j
  enddo ! end loop over k

  call Grid_releaseBlkPtr(blockID,solnData)

  return

end subroutine st_calcAccel
