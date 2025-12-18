!!****if* source/physics/sourceTerms/Stir/StirFromFileMain/Stir_init
!!
!! NAME
!!  Stir_init
!!
!! SYNOPSIS
!!  Stir_init(logical(in) :: restart)
!!
!! DESCRIPTION
!!  Initialise turbulence driving; read header of stirring modes file
!!
!! ARGUMENTS
!!   restart -restarting from checkpoint?
!!
!! PARAMETERS
!!   These are the runtime parameters used in the Stir unit.
!!
!!    useStir        [BOOLEAN]
!!        Switch to turn stirring on or off at runtime.
!!    st_infilename  [CHARACTER]
!!        file containing the stirring modes time sequence
!!    st_computeDt   [BOOLEAN]
!!        whether to restrict timestep based on stirring
!!
!! AUTHOR
!!  Christoph Federrath, 2008-2022
!!
!!***

subroutine Stir_init(restart)

  use Stir_data
  use Driver_data, ONLY : dr_globalMe
  use Driver_interface, ONLY : Driver_getSimTime
  use RuntimeParameters_interface, ONLY : RuntimeParameters_get
  use Grid_data, ONLY : gr_imin, gr_imax

  implicit none

#include "constants.h"
#include "Flash.h"

  logical, intent(in) :: restart
  real                :: time, Lx
  real(kind=8)        :: timeinfile

  call RuntimeParameters_get('useStir', st_useStir)
  call RuntimeParameters_get('st_infilename', st_infilename)
  call RuntimeParameters_get('st_computeDt', st_computeDt)
  call RuntimeParameters_get('st_stop_driving_time', st_stop_driving_time)

  if (.not. st_useStir) then
    if (dr_globalMe .eq. MASTER_PE) &
      write(*,'(A)') 'Stir_init: WARNING: You have included the StirFromFile unit but useStir = .false.'
    return
  endif

  ! this call sets dt_update_accel and reads general information from the driving modes file
  call Driver_getSimTime(time)
  call st_read_modes_file(st_infilename, real(time,kind=8), timeinfile)

  ! this makes the rms force constant for 1D, 2D, 3D, irrespective of the solenoidal weight
  if (NDIM .eq. 3) st_solweightnorm = sqrt(3.0/3.0)*sqrt(3.0)*1.0/sqrt(1.0-2.0*st_solweight+3.0*st_solweight**2.0)
  if (NDIM .eq. 2) st_solweightnorm = sqrt(3.0/2.0)*sqrt(3.0)*1.0/sqrt(1.0-2.0*st_solweight+2.0*st_solweight**2.0)
  if (NDIM .eq. 1) st_solweightnorm = sqrt(3.0/1.0)*sqrt(3.0)*1.0/sqrt(1.0-2.0*st_solweight+1.0*st_solweight**2.0)

  ! box size (in x)
  Lx = gr_imax - gr_imin

  if (dr_globalMe .eq. MASTER_PE) then
    write(*,'(A)') " Stir_init: =================================================================================="
    write(*,'(A,I6,A)') ' Using ', st_nmodes, ' modes for turbulence driving from file "'//trim(st_infilename)//'"'
    if (st_spectform .eq. 0) &
      write(*,'(A,I2,A)') ' spectral form                                     = ', st_spectform, ' (Band)'
    if (st_spectform .eq. 1) &
      write(*,'(A,I2,A)') ' spectral form                                     = ', st_spectform, ' (Paraboloid)'
    if (st_spectform .eq. 2) &
      write(*,'(A,I2,A)') ' spectral form                                     = ', st_spectform, ' (Power Law)'
    write(*,'(A,ES10.3)') ' simulation box size Lx                            = ', Lx
    write(*,'(A,ES10.3)') ' velocity dispersion                               = ', st_velocity
    write(*,'(A,ES10.3)') ' auto-correlation time                             = ', st_decay
    write(*,'(A,ES10.3)') '  -> characteristic driving wavenumber (in 2pi/Lx) = ', Lx / st_velocity / st_decay
    write(*,'(A,ES10.3)') ' minimum driving wavenumber (in 2pi/Lx)            = ', st_stirmin / (2*PI) * Lx
    write(*,'(A,ES10.3)') ' maximum driving wavenumber (in 2pi/Lx)            = ', st_stirmax / (2*PI) * Lx
    write(*,'(A,ES10.3)') ' driving energy (injection rate)                   = ', st_energy
    write(*,'(A,ES10.3)') '  -> energy coefficient (energy / velocity^3 * Lx) = ', st_energy / st_velocity**3 * Lx
    write(*,'(A,ES10.3)') ' solenoidal weight (0.0: comp, 0.5: mix, 1.0: sol) = ', st_solweight
    write(*,'(A,I1,A,ES10.3)') '  -> st_solweightnorm (set based on NDIM=', NDIM, ')        = ', st_solweightnorm
    write(*,'(A)') " ============================================================================================="
  endif

  return

end subroutine Stir_init
