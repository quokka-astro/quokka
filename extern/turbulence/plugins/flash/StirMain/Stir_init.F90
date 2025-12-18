!!****if* source/physics/sourceTerms/Stir/StirMain/Stir_init
!!
!! NAME
!!  Stir_init
!!
!! SYNOPSIS
!!  Stir_init(logical(in) :: restart)
!!
!! DESCRIPTION
!!  Initialise turbulence driving
!!
!! ARGUMENTS
!!   restart - restarting from checkpoint?
!!
!! PARAMETERS
!!   These are the runtime parameters used in the Stir unit.
!!
!!    useStir        [BOOLEAN]
!!        Switch to turn stirring on or off at runtime.
!!    st_infilename  [CHARACTER]
!!        file containing the turbulence stirring parameters
!!    st_computeDt   [BOOLEAN]
!!        whether to restrict timestep based on stirring
!!
!! AUTHOR
!!  Christoph Federrath
!!
!!***

subroutine Stir_init(restart)

  use Stir_data
  use Driver_data, ONLY : dr_globalMe
  use Driver_interface, ONLY : Driver_getSimTime
  use RuntimeParameters_interface, ONLY : RuntimeParameters_get
  use iso_c_binding, ONLY : c_double

  implicit none

#include "constants.h"
#include "Flash.h"

  logical, intent(in) :: restart
  real(c_double) :: time

  call RuntimeParameters_get('useStir', st_useStir)
  call RuntimeParameters_get('st_infilename', st_infilename)
  call RuntimeParameters_get('st_computeDt', st_computeDt)
  call RuntimeParameters_get('st_stop_driving_time', st_stop_driving_time)

  call Driver_getSimTime(time)

  if (.not. st_useStir) then
    if (dr_globalMe == MASTER_PE) &
      write(*,'(A)') 'Stir_init: WARNING: You have included the Stir unit but useStir = .false.'
    return
  endif

  ! initialise the turbulence generator based on parameter file provided in st_infilename
  ! and time (in case of restart and automatic amplitude adjustment; ampl_auto_adjust = 1)
  call st_stir_init_driving_c(trim(st_infilename)//char(0), real(time,kind=c_double), dt_update_accel);

  return

end subroutine Stir_init
