!!****if* source/physics/sourceTerms/Stir/StirMain/Stir
!!
!! NAME
!!  Stir
!!
!! SYNOPSIS
!!  Stir(integer(in) :: blockCount,
!!       integer(in) :: blockList(blockCount),
!!       real(in)    :: dt)
!!
!! DESCRIPTION
!!   Apply the turbulence driving operator on the list of blocks provided as input
!!
!! ARGUMENTS
!!   blockCount   : The number of blocks in the list
!!   blockList(:) : The list of blocks on which to apply turbulence driving
!!   dt           : the current timestep
!!   pass         : optional: dt-pass, in case of split solver
!!
!! AUTHOR
!!   Christoph Federrath
!!
!!    (Aug 2013: added a write-out of the correction for the center-of-mass motion)
!!    (2012: added a pre-proc statement to treat the acceleration field as a force for testing; not the default)
!!    (2011/2012: added a write-out of the energy injection rate)
!!    (2017: added dynamic allocation of st_nmodes arrays)
!!    (2022: use of turbulence_generator C++ library/header, TurbGen.h)
!!    (2022: support for automatic amplitude adjustment)
!!
!!***

subroutine Stir(blockCount, blockList, dt, pass)

#include "constants.h"
#include "Flash.h"

  use Stir_data
  use Driver_interface, ONLY : Driver_getSimTime
  use Timers_interface, ONLY : Timers_start, Timers_stop
  use Grid_interface,   ONLY : Grid_getBlkIndexLimits, Grid_getBlkPtr, Grid_releaseBlkPtr, &
                               Grid_getDeltas, Grid_getBlkPhysicalSize, Grid_getBlkCenterCoords
  use Driver_data, ONLY : dr_nstep, dr_globalMe
#ifdef FLASH_IO
  use IO_data, ONLY : io_integralFreq
#endif
  use iso_c_binding, ONLY : c_double

  implicit none

#include "Flash_mpi.h"

  integer, intent(in)                        :: blockCount
  integer, dimension(blockCount), intent(in) :: blockList
  real, intent(in)                           :: dt
  integer, intent(in), optional              :: pass

  logical, save              :: firstCall=.true., driving_stopped=.false.
  real, save                 :: last_time = huge(real(1.0))
  integer                    :: blockID, i, j, k, ii, jj, kk
  integer, dimension(2,MDIM) :: blkLimits, blkLimitsGC
  logical                    :: update_accel = .true.
  integer                    :: update_accel_int, error
  real, dimension(MDIM)      :: del, blockSize, blockCenter ! MDIM is always 3
  real(c_double)             :: pos_beg(MDIM), pos_end(MDIM)
  integer                    :: ncells(MDIM)
  real                       :: ekin_old, ekin_new, d_ekin
  real(c_double)             :: time
  real                       :: volume, mass, momentum(MDIM), force(MDIM)
  real(c_double)             :: v_turb(MDIM) ! turbulent velocity dispersion (for amplitude auto adjustment in TurbGen)

  integer, parameter :: funit = 22
  character(len=80)  :: outfile = "stir.dat"
  logical            :: check_for_io
#ifndef FLASH_IO
  integer, parameter :: io_integralFreq = -1
#endif

  real, DIMENSION(:,:,:,:), POINTER :: solnData

! this is to determine whether we remove the average force and average momentum in every timestep
#define CORRECT_BULK_MOTION
#ifdef CORRECT_BULK_MOTION
  real(c_double) :: correction
#else
  real(c_double), parameter :: correction = 0.0
#endif

#ifdef HYBRID_PRECISION
  integer, parameter :: flash_real_dtype = FLASH_DOUBLE
  real(c_double) :: locSumVars(0:7), globSumVars(0:7) ! locally and globally summed variables
  real(c_double) :: ekin_added, ekin_added_red, dvol, dmass, accel
#else
  integer, parameter :: flash_real_dtype = FLASH_REAL
  real         :: locSumVars(0:7), globSumVars(0:7) ! locally and globally summed variables
  real         :: ekin_added, ekin_added_red, dvol, dmass, accel
#endif

  if (firstCall) then
    if (dr_globalMe == MASTER_PE) then
      open(funit, file=trim(outfile), position='APPEND') ! write header
      write(funit,'(13(1X,A16))') '[00]time', '[01]dt', '[02]d(Ekin)', '[03]d(Ekin)/dt', &
                                  '[04]xForce', '[05]yForce', '[06]zForce', &
                                  '[07]xMomentum', '[08]yMomentum', '[09]zMomentum', &
                                  '[10]xVturb', '[11]yVturb', '[12]zVturb'
      close(funit)
    endif
    firstCall = .false.
  endif

  ! =====================================================================

  ! if not using stirring or stirring is turned off after some time (st_stop_driving_time), then return
  if ((.not. st_useStir) .or. driving_stopped) return

  call Driver_getSimTime(time)

  if (time >= st_stop_driving_time) then
    driving_stopped = .true.
    ! clear the acceleration field
    accx(:,:,:) = 0.0
    accy(:,:,:) = 0.0
    accz(:,:,:) = 0.0
#ifdef ACCX_VAR
    do blockID = 1, blockCount
      call Grid_getBlkPtr(blockList(blockID),solnData)
      solnData(ACCX_VAR,:,:,:) = 0.0
      solnData(ACCY_VAR,:,:,:) = 0.0
      solnData(ACCZ_VAR,:,:,:) = 0.0
      call Grid_releaseBlkPtr(blockList(blockID),solnData)
    enddo
#endif
    if (dr_globalMe == MASTER_PE) print *, 'Stir: TURBULENCE DRIVING STOPPED!'
    return
  endif

  call Timers_start("Stir")

#ifdef CORRECT_BULK_MOTION

  ! local and global sum containers
  locSumVars (:) = 0.0
  globSumVars(:) = 0.0

  ! sum quantities over list of blocks (to determine global mean force and momentum)
  do blockID = 1, blockCount
    ! get the index limits of the block
    call Grid_getBlkIndexLimits(blockList(blockID), blkLimits, blkLimitsGC)
    ! getting the dx's
    call Grid_getDeltas(blocklist(blockID), del)
#if NDIM == 1
    dvol = del(IAXIS)
#endif
#if NDIM == 2
    dvol = del(IAXIS) * del(JAXIS)
#endif
#if NDIM == 3
    dvol = del(IAXIS) * del(JAXIS) * del(KAXIS)
#endif
    ! get a pointer to the current block of data
    call Grid_getBlkPtr(blockList(blockID), solnData)
    ! loop over all grid cells and sum local contributions to global mean force and momentum
    do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
      do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
        do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)
          ! volume
          locSumVars(0) = locSumVars(0) + dvol
          ! cell mass
          dmass = solnData(DENS_VAR,i,j,k)*dvol
          ! mass
          locSumVars(1) = locSumVars(1) + dmass
#ifdef VELX_VAR
          ! momentum x
          locSumVars(2) = locSumVars(2) + solnData(VELX_VAR,i,j,k)*dmass
          ! vx**2
          locSumVars(3) = locSumVars(3) + solnData(VELX_VAR,i,j,k)**2*dvol
#endif
#ifdef VELY_VAR
          ! momentum y
          locSumVars(4) = locSumVars(4) + solnData(VELY_VAR,i,j,k)*dmass
          ! vy**2
          locSumVars(5) = locSumVars(5) + solnData(VELY_VAR,i,j,k)**2*dvol
#endif
#ifdef VELZ_VAR
          ! momentum z
          locSumVars(6) = locSumVars(6) + solnData(VELZ_VAR,i,j,k)*dmass
          ! vz**2
          locSumVars(7) = locSumVars(7) + solnData(VELZ_VAR,i,j,k)**2*dvol
#endif
        enddo ! i
      enddo ! j
    enddo ! k
    call Grid_releaseBlkPtr(blockList(blockID), solnData)
  enddo ! blocks

  ! now communicate all global summed quantities to all processors
  call MPI_AllReduce(locSumVars(0:7), globSumVars(0:7), 8, flash_real_dtype, MPI_Sum, MPI_Comm_World, error)

  volume = globSumVars(0) ! total volume
  mass = globSumVars(1) ! gas mass
  momentum(1:3) = globSumVars(2:6:2) ! gas momentum
  ! turbulent velocity dispersion for call to st_stir_check_for_update_of_turb_pattern_c
  v_turb(1:3) = sqrt( globSumVars(3:7:2)/volume - (momentum(1:3)/mass)**2 + tiny(0.0) )

#else
  v_turb(1:3) = -1.0 ! no amplitude auto adjustment in this case
#endif
! ifdef CORRECT_BULK_MOTION

  ! check if we need to update the turbulent acceleration field
  update_accel = .false.
  call st_stir_check_for_update_of_turb_pattern_c(real(time,kind=c_double), update_accel_int, v_turb)
  if (update_accel_int .ne. 0) update_accel = .true.

#ifdef CORRECT_BULK_MOTION

  ! local and global sum containers
  locSumVars (:) = 0.0
  globSumVars(:) = 0.0

  ! sum quantities over list of blocks (to determine global mean force and momentum)
  do blockID = 1, blockCount
    ! get the index limits of the block
    call Grid_getBlkIndexLimits(blockList(blockID), blkLimits, blkLimitsGC)
    ! getting the dx's
    call Grid_getDeltas(blocklist(blockID), del)
#if NDIM == 1
    dvol = del(IAXIS)
#endif
#if NDIM == 2
    dvol = del(IAXIS) * del(JAXIS)
#endif
#if NDIM == 3
    dvol = del(IAXIS) * del(JAXIS) * del(KAXIS)
#endif
    ! update turbulent acceleration field, otherwise use previous acceleration field
    if (update_accel) then
      call Grid_getBlkPhysicalSize(blockList(blockID), blockSize)
      call Grid_getBlkCenterCoords(blockList(blockID), blockCenter)
      pos_beg = blockCenter - 0.5*blockSize + del/2.0 ! first active cell coordinate in block (x,y,z)
      pos_end = blockCenter + 0.5*blockSize - del/2.0 ! last  active cell coordinate in block (x,y,z)
      ncells = blkLimits(HIGH,:)-blkLimits(LOW,:)+1 ! number of active cells in (x,y,z)
      call st_stir_get_turb_vector_unigrid_c(pos_beg, pos_end, ncells, accx, accy, accz)
    endif
    ! get a pointer to the current block of data
    call Grid_getBlkPtr(blockList(blockID), solnData)
    ! loop over all grid cells and sum local contributions to global mean force and momentum
    do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
      kk = k-blkLimits(LOW,KAXIS)+1 ! z index of accx, accy, accz starts at 1 and goes to NZB
      do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
        jj = j-blkLimits(LOW,JAXIS)+1 ! y index of accx, accy, accz starts at 1 and goes to NYB
        do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)
          ii = i-blkLimits(LOW,IAXIS)+1 ! x index of accx, accy, accz starts at 1 and goes to NXB
#ifdef ACCX_VAR
          ! if we use ACCX_VAR, ..Y, ..Z (usually when using AMR because of re-gridding), copy from accx, ..y, ..z
          if (update_accel) then
            solnData(ACCX_VAR,i,j,k) = accx(ii,jj,kk)
            solnData(ACCY_VAR,i,j,k) = accy(ii,jj,kk)
            solnData(ACCZ_VAR,i,j,k) = accz(ii,jj,kk)
          endif
#endif
          ! cell mass
          dmass = solnData(DENS_VAR,i,j,k)*dvol
          ! driving force
#ifdef ACCX_VAR
          locSumVars(1) = locSumVars(1) + solnData(ACCX_VAR,i,j,k)*dmass
#else
          locSumVars(1) = locSumVars(1) + accx(ii,jj,kk)*dmass
#endif
#ifdef ACCY_VAR
          locSumVars(2) = locSumVars(2) + solnData(ACCY_VAR,i,j,k)*dmass
#else
          locSumVars(2) = locSumVars(2) + accy(ii,jj,kk)*dmass
#endif
#ifdef ACCZ_VAR
          locSumVars(3) = locSumVars(3) + solnData(ACCZ_VAR,i,j,k)*dmass
#else
          locSumVars(3) = locSumVars(3) + accz(ii,jj,kk)*dmass
#endif
        enddo ! i
      enddo ! j
    enddo ! k
    call Grid_releaseBlkPtr(blockList(blockID), solnData)
  enddo ! blocks

  ! now communicate all global summed quantities to all processors
  call MPI_AllReduce(locSumVars(1:3), globSumVars(1:3), 3, flash_real_dtype, MPI_Sum, MPI_Comm_World, error)

  force(1:3) = globSumVars(1:3) ! driving force

#endif
! ifdef CORRECT_BULK_MOTION

  ! set to zero for adding block and cell contributions below
  ekin_added = 0.0

  ! Loop over local blocks again to actually apply the turbulent acceleration
  do blockID = 1, blockCount

    ! get the index limits of the block
    call Grid_getBlkIndexLimits(blockList(blockID), blkLimits, blkLimitsGC)

    ! getting the dx's
    call Grid_getDeltas(blocklist(blockID), del)

#if NDIM == 1
    dvol = del(IAXIS)
#endif
#if NDIM == 2
    dvol = del(IAXIS) * del(JAXIS)
#endif
#if NDIM == 3
    dvol = del(IAXIS) * del(JAXIS) * del(KAXIS)
#endif

! update acceleration if CORRECT_BULK_MOTION is not used (otherwise, st_get_turb_vector_unigrid_c was called above)
#ifndef CORRECT_BULK_MOTION
    ! update turbulent acceleration field, otherwise use previous acceleration field
    if (update_accel) then
      call Grid_getBlkPhysicalSize(blockList(blockID), blockSize)
      call Grid_getBlkCenterCoords(blockList(blockID), blockCenter)
      pos_beg = blockCenter - 0.5*blockSize + del/2.0 ! first active cell coordinate in block (x,y,z)
      pos_end = blockCenter + 0.5*blockSize - del/2.0 ! last  active cell coordinate in block (x,y,z)
      ncells = blkLimits(HIGH,:)-blkLimits(LOW,:)+1 ! number of active cells in (x,y,z)
      call st_stir_get_turb_vector_unigrid_c(pos_beg, pos_end, ncells, accx, accy, accz)
    endif
#endif
! ifndef CORRECT_BULK_MOTION

    ! get a pointer to the current block of data
    call Grid_getBlkPtr(blockList(blockID), solnData)

    ! loop over all grid cells and apply turbulent acceleration
    do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
      kk = k-blkLimits(LOW,KAXIS)+1 ! z index of accx, accy, accz starts at 1 and goes to NZB
      do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
        jj = j-blkLimits(LOW,JAXIS)+1 ! y index of accx, accy, accz starts at 1 and goes to NYB
        do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)
          ii = i-blkLimits(LOW,IAXIS)+1 ! x index of accx, accy, accz starts at 1 and goes to NXB

#ifndef CORRECT_BULK_MOTION
#ifdef ACCX_VAR
          ! if we use ACCX_VAR, ..Y, ..Z (usually when using AMR because of re-gridding), copy from accx, ..y, ..z
          if (update_accel) then
            solnData(ACCX_VAR,i,j,k) = accx(ii,jj,kk)
            solnData(ACCY_VAR,i,j,k) = accy(ii,jj,kk)
            solnData(ACCZ_VAR,i,j,k) = accz(ii,jj,kk)
          endif
#endif
#endif

#ifdef VELX_VAR
          ekin_old = 0.5*(solnData(VELX_VAR,i,j,k)**2+solnData(VELY_VAR,i,j,k)**2+solnData(VELZ_VAR,i,j,k)**2)
#endif

#ifdef VELX_VAR
#ifdef ACCX_VAR
          accel = solnData(ACCX_VAR,i,j,k)
#else
          accel = accx(ii,jj,kk)
#endif
#ifdef CORRECT_BULK_MOTION
          correction = force(1)/mass*dt + momentum(1)/mass
#endif
          solnData(VELX_VAR,i,j,k) = solnData(VELX_VAR,i,j,k) + accel*dt - correction
#endif

#ifdef VELY_VAR
#ifdef ACCY_VAR
          accel = solnData(ACCY_VAR,i,j,k)
#else
          accel = accy(ii,jj,kk)
#endif
#ifdef CORRECT_BULK_MOTION
          correction = force(2)/mass*dt + momentum(2)/mass
#endif
          solnData(VELY_VAR,i,j,k) = solnData(VELY_VAR,i,j,k) + accel*dt - correction
#endif

#ifdef VELZ_VAR
#ifdef ACCZ_VAR
          accel = solnData(ACCZ_VAR,i,j,k)
#else
          accel = accz(ii,jj,kk)
#endif
#ifdef CORRECT_BULK_MOTION
          correction = force(3)/mass*dt + momentum(3)/mass
#endif
          solnData(VELZ_VAR,i,j,k) = solnData(VELZ_VAR,i,j,k) + accel*dt - correction
#endif

#ifdef VELX_VAR
          ekin_new = 0.5*(solnData(VELX_VAR,i,j,k)**2+solnData(VELY_VAR,i,j,k)**2+solnData(VELZ_VAR,i,j,k)**2)
#endif
          ! compute energy injection
          d_ekin = ekin_new - ekin_old
#ifdef ENER_VAR
          ! update the total energy
          solnData(ENER_VAR,i,j,k) = solnData(ENER_VAR,i,j,k) + d_ekin
#endif
#ifdef EKIN_VAR
          ! update the kinetic energy
          solnData(EKIN_VAR,i,j,k) = solnData(EKIN_VAR,i,j,k) + d_ekin
#endif
#ifdef DENS_VAR
          ! add up total injected kinetic energy
          ekin_added = ekin_added + d_ekin * solnData(DENS_VAR,i,j,k) * dvol
#endif
#ifdef INJR_VAR
          ! fill kinetic energy injection rate d(0.5 rho v^2) / dt; note that rho=const here
          solnData(INJR_VAR,i,j,k) = d_ekin * solnData(DENS_VAR,i,j,k) / dt
#endif
        enddo ! i
      enddo ! j
    enddo ! k

    call Grid_releaseBlkPtr(blockList(blockID), solnData)

  enddo ! loop over blocks

  ! write time evolution of ekin_added to file
  if (io_integralFreq > 0) then
    ! synchronise output frequency with IO_output's use of io_integralFreq
    check_for_io = .true.
    if (present(pass)) then
      if (pass == 1) check_for_io = .false.
    endif
    if ((check_for_io) .and. (mod(dr_nstep+1, io_integralFreq) == 0)) then
      if (abs(time - last_time) > tiny(time)) then
        last_time = time
        ! sum up injected kinetic energy contributions from all blocks and processors
        ekin_added_red = 0.0
        call MPI_Reduce(ekin_added, ekin_added_red, 1, flash_real_dtype, MPI_Sum, MASTER_PE, MPI_Comm_World, error)
        ekin_added = ekin_added_red
        ! only MASTER_PE writes
        if (dr_globalMe == MASTER_PE) then
          open(funit, file=trim(outfile), position='APPEND')
          write(funit,'(13(1X,ES16.9))') time, dt, ekin_added, ekin_added/dt, force(1:3), momentum(1:3), v_turb(1:3)
          close(funit)
        endif
      endif
    endif
  endif

  call Timers_stop("Stir")

  return

end subroutine Stir
