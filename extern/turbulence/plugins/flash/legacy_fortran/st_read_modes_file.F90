!!****if* source/physics/sourceTerms/Stir/StirFromFileMain/st_read_modes_file
!!
!! NAME
!!  st_read_modes_file()
!!
!! DESCRIPTION
!!  reads the stirring modes necessary to construct the physical force field
!!
!! ARGUMENTS
!!  infile     : name of file containing the forcing sequence
!!  time       : current simulation time
!!  timeinfile : time in forcing file for which the modes/aka/akb are to be updated
!!
!! AUTHOR: Christoph Federrath, 2008-2022
!!
!!***

subroutine st_read_modes_file(infile, time, timeinfile)

  use Stir_data
  use Driver_data, ONLY : dr_globalMe

  implicit none

#include "constants.h"
#include "Flash.h"

  character (len=80), intent(in) :: infile
  real(kind=8), intent(in)       :: time
  real(kind=8), intent(out)      :: timeinfile

  logical, parameter  :: Debug = .false.
  logical             :: opened_successful = .false.
  integer             :: nsteps, step, stepinfile, desired_step, istat
  real(kind=8)        :: end_time
  integer, save       :: n_failed_reads = 0
  logical, save       :: first_call = .true.

  opened_successful = .false.
  do while (.not. opened_successful)
    open (unit=42, file=infile, iostat=istat, status='OLD', action='READ', &
          access='SEQUENTIAL', form='UNFORMATTED')
    ! header contains number of times and number of modes, end time, autocorrelation time, ...
    if (istat.eq.0) then
      if (Debug) write (*,'(A)') 'reading header...'
      read (unit=42) nsteps, st_nmodes, end_time, st_decay, st_energy, st_solweight, &
                    st_velocity, st_stirmin, st_stirmax, st_spectform
      if (Debug) write (*,'(A)') '...finished reading header'
      opened_successful = .true.
    else
      write (*,'(A,I6,2A)') '[',dr_globalMe,'] st_read_modes_file: could not open file for read. filename: ', trim(infile)
      write (*,'(A,I6,A)') '[',dr_globalMe,'] Trying again...'
      call sleep(1) ! wait a second...
      ! stop if there was an excessive number of failed read attempts
      n_failed_reads = n_failed_reads + 1
      if (n_failed_reads .gt. 100) call Driver_abortFlash("st_read_modes_file: Too many failed attempts at reading forcing file.")
    endif
  enddo

  if (first_call) then
    ! allocate arrays
    allocate(st_mode(3,st_nmodes),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("st_read_modes_file: could not allocate st_mode.")
    allocate(st_aka(3,st_nmodes),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("st_read_modes_file: could not allocate st_aka.")
    allocate(st_akb(3,st_nmodes),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("st_read_modes_file: could not allocate st_akb.")
    allocate(st_ampl(st_nmodes),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("st_read_modes_file: could not allocate st_ampl.")
    allocate(st_OUphases(6*st_nmodes),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("st_read_modes_file: could not allocate st_OUphases.")
    first_call = .false.
  endif

  ! these are in the global contex
  dt_update_accel = end_time/nsteps
  desired_step = floor(time/dt_update_accel)

  do step = 0, nsteps

    if (Debug) write (*,'(A,I6)') 'step = ', step
    read (unit=42) stepinfile, timeinfile, &
                  st_mode    (:, 1:  st_nmodes), &
                  st_aka     (:, 1:  st_nmodes), &
                  st_akb     (:, 1:  st_nmodes), &
                  st_ampl    (   1:  st_nmodes), &
                  st_OUphases(   1:6*st_nmodes)

    if (step.ne.stepinfile) write(*,'(A,I6)') 'st_read_modes_file: something wrong! step = ', step
    if (desired_step.eq.step) then
      if (dr_globalMe == MASTER_PE) write(*,'(A,I5,2(A,ES12.5))') ' --- Read new turbulence driving pattern: #', step, &
                                                                  ', time(sim) =', time, ', time(file) =', timeinfile
      close (unit=42)
      exit ! the loop
    endif

  enddo

  return

end subroutine st_read_modes_file
