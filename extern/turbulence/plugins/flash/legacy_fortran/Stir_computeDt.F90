!!****if* source/physics/sourceTerms/Stir/StirFromFile/Stir_computeDt
!!
!! NAME
!!  Stir_computeDt
!!
!! SYNOPSIS
!!  Stir_computeDt(integer(IN)   :: blockID
!!                 integer(IN)   :: blkLimits(2,MDIM)
!!                 integer(IN)   :: blkLimitsGC(2,MDIM)
!!                 real, pointer :: solnData(:,:,:,:)
!!                 real(OUT)     :: dt_stir
!!                 real(OUT)     :: dt_minloc(5))
!!
!! DESCRIPTION
!!  compute a turbulent stirring timestep limiter
!!
!! ARGUMENTS
!!  blockID       --  local block ID
!!  blkLimits     --  the indices for the interior endpoints of the block
!!  blkLimitsGC   --  the indices for endpoints including the guardcells
!!  solnData      --  the physical, solution data from grid
!!  dt_stir       --  variable to hold timestep constraint
!!  dt_minloc(5)  --  array to hold limiting zone info:  zone indices
!!                    (i,j,k), block ID, PE number
!!
!! SEE ALSO
!!  Driver_computeDt
!!
!!***

subroutine Stir_computeDt(blockID, blkLimits, blkLimitsGC, solnData, dt_stir, dt_minloc)

  use Stir_data
  use Driver_interface, ONLY: Driver_getSimTime
  use Driver_data, ONLY : dr_globalMe
  use Grid_interface, ONLY : Grid_getDeltas

#include "constants.h"
#include "Flash.h"

  implicit none

  !! arguments
  integer, intent(in) :: blockID
  integer, intent(in), dimension(2,MDIM)::blkLimits, blkLimitsGC
  real, pointer :: solnData(:,:,:,:)
  real, intent(inout) :: dt_stir
  integer, intent(inout) :: dt_minloc(5)

  real, dimension(MDIM) :: delta
  real :: dt_temp, time
  integer :: i,j,k

  !!===================================================================

  call Driver_getSimTime(time)
  if ((.not.st_useStir).or.(.not.st_computeDt).or.(time.ge.st_stop_driving_time)) return

  ! initialize the timestep from this block to some high number
  dt_temp = HUGE(0.0)
  call Grid_getDeltas(blockID, delta)

  do k = blkLimitsGC(LOW,KAXIS), blkLimitsGC(HIGH,KAXIS)
    do j = blkLimitsGC(LOW,JAXIS), blkLimitsGC(HIGH,JAXIS)
      do i = blkLimitsGC(LOW,IAXIS), blkLimitsGC(HIGH,IAXIS)

#ifdef ACCX_VAR
        dt_temp = min(dt_temp,abs(delta(IAXIS)/solnData(ACCX_VAR,i,j,k)))
#else
        dt_temp = min(dt_temp,abs(delta(IAXIS)/accx(i,j,k)))
#endif
#ifdef ACCY_VAR
        dt_temp = min(dt_temp,abs(delta(JAXIS)/solnData(ACCY_VAR,i,j,k)))
#else
        dt_temp = min(dt_temp,abs(delta(JAXIS)/accy(i,j,k)))
#endif
#ifdef ACCZ_VAR
        dt_temp = min(dt_temp,abs(delta(KAXIS)/solnData(ACCZ_VAR,i,j,k)))
#else
        dt_temp = min(dt_temp,abs(delta(KAXIS)/accz(i,j,k)))
#endif
        if (dt_temp < dt_stir*dt_stir) then
          dt_stir = sqrt(dt_temp)
          dt_minloc(1) = i
          dt_minloc(2) = j
          dt_minloc(3) = k
          dt_minloc(4) = blockID
          dt_minloc(5) = dr_globalMe
        end if

      enddo
    enddo
  enddo

  ! make sure that we go through all forcing patterns in the forcingfile
  if (dt_update_accel < dt_temp) then
    dt_stir = 0.5*dt_update_accel
    dt_minloc(1) = blkLimits(LOW,IAXIS)
    dt_minloc(2) = blkLimits(LOW,JAXIS)
    dt_minloc(3) = blkLimits(LOW,KAXIS)
    dt_minloc(4) = blockID
    dt_minloc(5) = dr_globalMe
  end if

  return

end subroutine Stir_computeDt
