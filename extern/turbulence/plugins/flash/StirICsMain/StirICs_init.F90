!!****if* source/physics/sourceTerms/StirICs/StirICsMain/StirICs_init
!!
!! NAME
!!  StirICs_init
!!
!! SYNOPSIS
!!  StirICs_init(logical(in) :: restart)
!!
!! DESCRIPTION
!!  Initialize turbulent stirring unit
!!
!! ARGUMENTS
!!   restart - restarting from checkpoint?
!!
!! PARAMETERS
!!   These are the runtime parameters used in the Stir unit.
!!
!!   To see the default parameter values and all the runtime parameters
!!   specific to your simulation check the "setup_params" file in your
!!   object directory.
!!   You might have overwritten these values with the flash.par values
!!   for your specific run.
!!
!!    useStirICs   [BOOLEAN]
!!        Switch to turn initial conditions stirring on or off at runtime.
!!
!!    st_rmsVelocity    [REAL]
!!        the target turbulent RMS velocity
!!    st_solWeight      [REAL]
!!        solenoidal weight (1: purely solenoidal, 0: purely compressive)
!!    st_stirMin        [REAL]
!!        minimum stirring wavenumber (in units of 2pi/L_box)
!!    st_stirMax        [REAL]
!!        maximum stirring wavenumber (in units of 2pi/L_box)
!!    st_spectForm      [INTEGER]
!!        spectral form of amplitude (0: band, 1: paraboloid, 2: power law)
!!    st_powerLawExp    [REAL]
!!        power law exponent in case of st_spectForm = 2
!!    st_seed           [INETGER]
!!        random number generator seed
!!
!!    st_rmsMagneticField     [REAL]
!!        the target turbulent RMS magnetic field
!!    st_stirMagneticKMin     [REAL]
!!        minimum magnetic field wavenumber (in units of 2pi/L_box)
!!    st_stirMagneticKMax     [REAL]
!!        maximum magnetic field wavenumber (in units of 2pi/L_box)
!!    st_MagneticSpectForm    [INTEGER]
!!        spectral form of amplitude (0: band, 1: paraboloid, 2: power law)
!!    st_MagneticPowerLawExp  [REAL]
!!        magnetic spectral power-law exponent in case of st_spectForm = 2
!!    st_MagneticSeed         [INETGER]
!!        random number generator seed for magentic field
!!
!! AUTHOR
!!  Christoph Federrath
!!
!!***

subroutine StirICs_init(restart)

  use StirICs_data
  use Driver_data, ONLY : dr_globalMe
  use RuntimeParameters_interface, ONLY : RuntimeParameters_get

  implicit none

#include "constants.h"
#include "Flash.h"

  logical, intent(in) :: restart

  call RuntimeParameters_get('useStirICs', st_useStirICs)

  ! for initial turbulent velocity field
  call RuntimeParameters_get('st_rmsVelocity', st_rmsVelocity)
  call RuntimeParameters_get('st_solWeight', st_solWeight)
  call RuntimeParameters_get('st_stirMin', st_stirMin)
  call RuntimeParameters_get('st_stirMax', st_stirMax)
  call RuntimeParameters_get('st_spectForm', st_spectForm)
  call RuntimeParameters_get('st_powerLawExp', st_powerLawExp)
  call RuntimeParameters_get('st_seed', st_seed)

  ! this controls the sampling in k space (currently applies to both velocity and magnetic field generation)
  call RuntimeParameters_get('st_anglesExp', st_anglesExp)

  ! for initial turbulent magnetic field
  call RuntimeParameters_get('st_rmsMagneticField', st_rmsMagneticField)
  call RuntimeParameters_get('st_stirMagneticKMin', st_stirMagneticKMin)
  call RuntimeParameters_get('st_stirMagneticKMax', st_stirMagneticKMax)
  call RuntimeParameters_get('st_MagneticSpectForm', st_MagneticSpectForm)
  call RuntimeParameters_get('st_MagneticPowerLawExp', st_MagneticPowerLawExp)
  call RuntimeParameters_get('st_MagneticSeed', st_MagneticSeed)

  if (restart) return ! return on restart, so we don't get confused with the messages below

  if ((dr_globalMe == MASTER_PE) .and. (.not. st_useStirICs)) &
      & write(*,'(A)') 'WARNING: You have included the StirInitialConditions unit but useStirICs = .false.'

  ! for initial turbulent velocity field
  if ((dr_globalMe == MASTER_PE) .and. st_useStirICs .and. (st_rmsVelocity > 0.0)) then
     write (*,'(A)') '--- Turbulent initial conditions for velocity field ---'
     write (*,'(A,ES10.3)') ' target RMS turbulent velocity = ', st_rmsVelocity
     write (*,'(A,ES10.3)') ' solenoidal weight    = ', st_solWeight
     write (*,'(A,ES10.3)') ' minimum wavenumber   = ', st_stirMin
     write (*,'(A,ES10.3)') ' maximum wavenumber   = ', st_stirMax
     if (st_spectform == 0) write (*,'(A,I2,A)') ' spectral form        = ', st_spectform, ' (band)'
     if (st_spectform == 1) write (*,'(A,I2,A)') ' spectral form        = ', st_spectform, ' (paraboloid)'
     if (st_spectform == 2) write (*,'(A,I2,A)') ' spectral form        = ', st_spectform, ' (power law)'
     if (st_spectform == 2) write (*,'(A,ES10.3)') ' power-law exponent   = ', st_powerLawExp
     write (*,'(A,I7)') ' random seed          = ', st_seed
     write (*,'(A)') '-------------------------------------------------------'
  endif

  ! for initial turbulent magnetic field
  if ((dr_globalMe == MASTER_PE) .and. st_useStirICs .and. (st_rmsMagneticField > 0.0)) then
     write (*,'(A)') '--- Turbulent initial conditions for magnetic field ---'
     write (*,'(A,ES10.3)') ' target RMS magnetic field = ', st_rmsMagneticField
     write (*,'(A,ES10.3)') ' minimum wavenumber   = ', st_stirMagneticKMin
     write (*,'(A,ES10.3)') ' maximum wavenumber   = ', st_stirMagneticKMax
     if (st_MagneticSpectForm == 0) write (*,'(A,I2,A)') ' spectral form        = ', st_MagneticSpectForm, ' (band)'
     if (st_MagneticSpectForm == 1) write (*,'(A,I2,A)') ' spectral form        = ', st_MagneticSpectForm, ' (paraboloid)'
     if (st_MagneticSpectForm == 2) write (*,'(A,I2,A)') ' spectral form        = ', st_MagneticSpectForm, ' (power law)'
     if (st_MagneticSpectForm == 2) write (*,'(A,ES10.3)') ' power-law exponent   = ', st_MagneticPowerLawExp
     write (*,'(A,I7)') ' random seed          = ', st_MagneticSeed
     write (*,'(A)') '-------------------------------------------------------'
  endif

  return

end subroutine StirICs_init
