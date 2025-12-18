!!****if* source/physics/sourceTerms/Stir/StirMain/Stir_data
!!
!! NAME
!!  Stir_data
!!
!! SYNOPSIS
!!  Stir_data()
!!
!! DESCRIPTION
!!  Stores the local data for source term: Stir
!!
!! AUTHOR
!!  Christoph Federrath
!!
!!***

Module Stir_data

  use iso_c_binding, ONLY : c_double, c_float

  implicit none

#include "Flash.h"

  character(len=80), save :: st_infilename
  logical, save  :: st_useStir, st_computeDt
  real, save :: st_stop_driving_time
  real(c_double), save :: dt_update_accel

  ! local container of type c_float to receive the turbulent acceleration field
  real(c_float), save, dimension(NXB, NYB, NZB) :: accx, accy, accz

end Module Stir_data
