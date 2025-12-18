!!****if* source/physics/sourceTerms/StirICs/StirICsMain/StirICs_data
!!
!! NAME
!!  StirICs_data
!!
!! SYNOPSIS
!!  StirICs_data()
!!
!! DESCRIPTION
!!  Stores local data for StirICs
!!
!! AUTHOR
!!  Christoph Federrath
!!
!!***

Module StirICs_data

  use iso_c_binding, ONLY : c_double

  implicit none

  logical, save :: st_useStirICs

  ! for initial turbulent velocity field
  real(c_double), save :: st_rmsVelocity, st_solWeight, st_stirMin, st_stirMax, st_powerLawExp
  integer, save        :: st_spectForm, st_seed

  ! currently applies to both velocity and magnetic field generation
  real(c_double), save :: st_anglesExp

  ! for initial turbulent magnetic field
  real(c_double), save :: st_rmsMagneticField, st_stirMagneticKMin, st_stirMagneticKMax, st_MagneticPowerLawExp
  integer, save        :: st_MagneticSpectForm, st_MagneticSeed

end Module StirICs_data
