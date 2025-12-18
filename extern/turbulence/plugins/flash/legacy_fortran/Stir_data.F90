!!****if* source/physics/sourceTerms/Stir/StirFromFileMain/Stir_data
!!
!! NAME
!!  Stir_data
!!
!! SYNOPSIS
!!  Stir_data()
!!
!! DESCRIPTION
!!  Stores the local data for Source Term: StirFromFile
!!
!! AUTHOR
!!  Christoph Federrath, 2008-2022
!!
!!***

Module Stir_data

#include "Flash.h"

  ! number of modes
  integer, save :: st_nmodes

  real(kind=8), save, allocatable, dimension(:,:) :: st_mode, st_aka, st_akb
  real(kind=8), save, allocatable, dimension(:)   :: st_ampl
  real(kind=8), save, allocatable, dimension(:)   :: st_OUphases

  character (len=80), save :: st_infilename
  real(kind=8), save :: last_time_updated_accel, dt_update_accel
  real, save :: st_stop_driving_time

  logical, save  :: st_useStir, st_computeDt
  real(kind=8), save :: st_decay, st_velocity, st_energy
  real(kind=8), save :: st_stirmin, st_stirmax
  real(kind=8), save :: st_solweight, st_solweightnorm
  integer, save  :: st_spectform

#if !defined(ACCX_VAR) && !defined(ACCY_VAR) && !defined(ACCZ_VAR)
  real(kind=4), save, dimension(GRID_ILO_GC:GRID_IHI_GC, &
                                GRID_JLO_GC:GRID_JHI_GC, &
                                GRID_KLO_GC:GRID_KHI_GC) :: accx, accy, accz
#endif

end Module Stir_data
