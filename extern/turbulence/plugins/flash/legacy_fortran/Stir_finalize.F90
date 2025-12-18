!!****f* source/physics/sourceTerms/Stir/StirFromFileMain/Stir_finalize
!!
!! NAME
!!
!!  Stir_finalize
!!
!! SYNOPSIS
!!
!!  Stir_finalize()
!!
!! DESCRIPTION
!!
!!  Clean up the Stir unit
!!
!!***

subroutine Stir_finalize()

  use Stir_data

  implicit none

  if (st_useStir) then
    deallocate(st_mode)
    deallocate(st_aka)
    deallocate(st_akb)
    deallocate(st_ampl)
    deallocate(st_OUphases)
  endif

  return

end subroutine Stir_finalize
