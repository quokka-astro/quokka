!!****if* source/physics/sourceTerms/Stir/StirFromFileMain/Stir
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
!!   Christoph Federrath, 2008-2022
!!
!!    (Aug 2013: added a write-out of the correction for the center-of-mass motion)
!!    (2012: added a pre-proc statement to treat the acceleration field as a force for testing; not the default)
!!    (2011/2012: added a write-out of the energy injection rate)
!!    (2017: added dynamic allocation of st_nmodes arrays)
!!
!!***

subroutine Stir(blockCount, blockList, dt, pass)

  use Stir_data
  use Driver_interface, ONLY : Driver_getSimTime
  use Timers_interface, ONLY : Timers_start, Timers_stop
  use Grid_interface,   ONLY : Grid_getBlkIndexLimits, Grid_getBlkPtr, &
                               Grid_getDeltas, Grid_releaseBlkPtr, Grid_getCellCoords
  use Driver_data, ONLY : dr_restart, dr_nstep, dr_globalMe
  use IO_data, ONLY : io_integralFreq

  implicit none

#include "constants.h"
#include "Flash.h"
#include "Flash_mpi.h"

  integer, intent(in)                        :: blockCount
  integer, dimension(blockCount), intent(IN) :: blockList
  real, intent(in)                           :: dt
  integer, intent(in), optional              :: pass

  logical, save              :: firstCall=.true., driving_stopped=.false., just_restarted=.false.
  real, save                 :: last_time = HUGE(real(1.0))
  integer                    :: blockID, i, j, k
  integer, dimension(2,MDIM) :: blkLimits, blkLimitsGC
  logical, parameter         :: gcell = .true.
  logical                    :: update_accel = .true.
  integer                    :: sizeZ, sizeY, sizeX, error
  real                       :: del(MDIM), time, ekin_old, ekin_new, d_ekin
  real                       :: mass, xMomentum, yMomentum, zMomentum, xForce, yForce, zForce

  integer, parameter :: funit = 22
  character(len=80)  :: outfile = "stir.dat"
  logical            :: check_for_io

  integer, parameter :: nGlobalSum = 7                  ! Number of globally-summed quantities
  real(kind=8)       :: globalSumQuantities(nGlobalSum) ! Global summed quantities
  real(kind=8)       :: localSumQuantities(nGlobalSum)  ! Global summed quantities
  real(kind=8)       :: timeinfile, ekin_added, ekin_added_red, dvol, dmass, accel

  real, DIMENSION(:,:,:,:), POINTER :: solnData

#ifdef FIXEDBLOCKSIZE
  real, dimension(GRID_IHI_GC) :: iCoord
  real, dimension(GRID_JHI_GC) :: jCoord
  real, dimension(GRID_KHI_GC) :: kCoord
#else
  real, allocatable, dimension(:) :: iCoord, jCoord, kCoord
  integer :: istat
#endif

! this is to determine whether we remove the average force and average momentum in every timestep
#define CORRECT_BULK_MOTION
#ifdef CORRECT_BULK_MOTION
  real(kind=8) :: correction
#else
  real(kind=8), parameter :: correction = 0.0
#endif

  if (firstCall) then
    call Driver_getSimTime(time)
    last_time_updated_accel = time-0.9999*dt_update_accel ! first time stirring
    if (dr_restart) just_restarted = .true.
    if (dr_globalMe == MASTER_PE) then
      open(funit, file=trim(outfile), position='APPEND')
      write(funit,'(10(1X,A16))') '[00]time', '[01]dt', '[02]d(Ekin)', '[03]d(Ekin)/dt', &
                                  '[04]xForce', '[05]yForce', '[06]zForce', &
                                  '[07]xMomentum', '[08]yMomentum', '[09]zMomentum'
      close(funit)
    endif
    firstCall = .false.
  endif

  ! =====================================================================

  ! if not using stirring or stirring is turned off after some time (st_stop_driving_time), then return
  if ((.not.st_useStir).or.driving_stopped) return

  call Driver_getSimTime(time)

  if (time .ge. st_stop_driving_time) then
    driving_stopped = .true.
    ! clear the acceleration field
    do blockID = 1, blockCount
      call Grid_getBlkPtr(blockList(blockID),solnData)
#ifdef ACCX_VAR
      solnData(ACCX_VAR,:,:,:) = 0.0
#else
                   accx(:,:,:) = 0.0
#endif
#ifdef ACCY_VAR
      solnData(ACCY_VAR,:,:,:) = 0.0
#else
                   accy(:,:,:) = 0.0
#endif
#ifdef ACCZ_VAR
      solnData(ACCZ_VAR,:,:,:) = 0.0
#else
                   accz(:,:,:) = 0.0
#endif
      call Grid_releaseBlkPtr(blockList(blockID),solnData)
    enddo
    if (dr_globalMe == MASTER_PE) print *, 'Stir: TURBULENCE DRIVING STOPPED!'
    return
  endif

  call Timers_start("Stir")

  ! check if we need to update acceleration field
  update_accel = .false.
  if (time .ge. last_time_updated_accel+dt_update_accel) then
    call st_read_modes_file(st_infilename, real(time,kind=8), timeinfile)
    last_time_updated_accel = timeinfile
    update_accel = .true.
  endif

  ! in case of restart only
  if (just_restarted) then
    update_accel = .true.
    just_restarted = .false.
  endif

  globalSumQuantities = 0.0
  localSumQuantities  = 0.0

#ifdef CORRECT_BULK_MOTION

  ! sum quantities over list of blocks
  do blockID = 1, blockCount

    !get the index limits of the block
    call Grid_getBlkIndexLimits(blockList(blockID), blkLimits, blkLimitsGC)

    ! get a pointer to the current block of data
    call Grid_getBlkPtr(blockList(blockID), solnData)

    !getting the dx's
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

    sizeX = blkLimitsGC(HIGH,IAXIS)-blkLimitsGC(LOW,IAXIS)+1
    sizeY = blkLimitsGC(HIGH,JAXIS)-blkLimitsGC(LOW,JAXIS)+1
    sizeZ = blkLimitsGC(HIGH,KAXIS)-blkLimitsGC(LOW,KAXIS)+1

#ifndef FIXEDBLOCKSIZE
    allocate(iCoord(sizeX),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("could not allocate iCoord in Stir.F90")
    allocate(jCoord(sizeY),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("could not allocate jCoord in Stir.F90")
    allocate(kCoord(sizeZ),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("could not allocate kCoord in Stir.F90")
#endif
    ! x coordinates
    call Grid_getCellCoords(IAXIS,blockList(blockID),CENTER,gcell,iCoord,sizeX)
#if NDIM > 1
    ! y coordinates
    call Grid_getCellCoords(JAXIS,blockList(blockID),CENTER,gcell,jCoord,sizeY)
#endif
#if NDIM > 2
    ! z coordinates
    call Grid_getCellCoords(KAXIS,blockList(blockID),CENTER,gcell,kCoord,sizeZ)
#endif
    ! update forcing pattern, otherwise use previous forcing pattern
    if (update_accel) call st_calcAccel(blockList(blockID),blkLimits,blkLimitsGC,iCoord,jCoord,kCoord)

    ! Sum contributions from the indicated blkLimits of cells.
    do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
      do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
        do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

          ! cell mass
          dmass = solnData(DENS_VAR,i,j,k)*dvol

          ! mass
#ifdef DENS_VAR
          localSumQuantities(1) = localSumQuantities(1) + dmass

          ! momentum
#ifdef VELX_VAR
          localSumQuantities(2) = localSumQuantities(2) + solnData(VELX_VAR,i,j,k)*dmass
#endif
#ifdef VELY_VAR
          localSumQuantities(3) = localSumQuantities(3) + solnData(VELY_VAR,i,j,k)*dmass
#endif
#ifdef VELZ_VAR
          localSumQuantities(4) = localSumQuantities(4) + solnData(VELZ_VAR,i,j,k)*dmass
#endif
          ! driving force
#ifdef ACCX_VAR
          localSumQuantities(5) = localSumQuantities(5) + solnData(ACCX_VAR,i,j,k)*dmass
#else
          localSumQuantities(5) = localSumQuantities(5) +              accx(i,j,k)*dmass
#endif
#ifdef ACCY_VAR
          localSumQuantities(6) = localSumQuantities(6) + solnData(ACCY_VAR,i,j,k)*dmass
#else
          localSumQuantities(6) = localSumQuantities(6) +              accy(i,j,k)*dmass
#endif
#ifdef ACCZ_VAR
          localSumQuantities(7) = localSumQuantities(7) + solnData(ACCZ_VAR,i,j,k)*dmass
#else
          localSumQuantities(7) = localSumQuantities(7) +              accz(i,j,k)*dmass
#endif
#endif
! ifdef DENS_VAR
        enddo
      enddo
    enddo

    call Grid_releaseBlkPtr(blockList(blockID),solnData)

#ifndef FIXEDBLOCKSIZE
    deallocate(iCoord)
    deallocate(jCoord)
    deallocate(kCoord)
#endif

  enddo ! blocks

  ! now communicate all global summed quantities to all processors
  call MPI_AllReduce(localSumQuantities, globalSumQuantities, nGlobalSum, &
                      MPI_DOUBLE_PRECISION, MPI_Sum, MPI_Comm_World, error)

  mass      = globalSumQuantities(1)
  xMomentum = globalSumQuantities(2)
  yMomentum = globalSumQuantities(3)
  zMomentum = globalSumQuantities(4)
  xForce    = globalSumQuantities(5)
  yForce    = globalSumQuantities(6)
  zForce    = globalSumQuantities(7)

#endif
! ifdef CORRECT_BULK_MOTION

  ! set to zero for adding block and cell contributions below
  ekin_added = 0.0

  ! Loop over local blocks
  do blockID = 1, blockCount

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

    ! Get cell coordinates for this block
    call Grid_getBlkIndexLimits(blockList(blockID),blkLimits,blkLimitsGC)

! update acceleration if CORRECT_BULK_MOTION is not used (otherwise, st_calcAccel was called above)
#ifndef CORRECT_BULK_MOTION
    sizeX = blkLimitsGC(HIGH,IAXIS) - blkLimitsGC(LOW,IAXIS) + 1
    sizeY = blkLimitsGC(HIGH,JAXIS) - blkLimitsGC(LOW,JAXIS) + 1
    sizeZ = blkLimitsGC(HIGH,KAXIS) - blkLimitsGC(LOW,KAXIS) + 1
#ifndef FIXEDBLOCKSIZE
    allocate(iCoord(sizeX),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("could not allocate iCoord in Stir.F90")
    allocate(jCoord(sizeY),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("could not allocate jCoord in Stir.F90")
    allocate(kCoord(sizeZ),stat=istat)
    if (istat .ne. 0) call Driver_abortFlash("could not allocate kCoord in Stir.F90")
#endif
    ! x coordinates
    call Grid_getCellCoords(IAXIS,blockList(blockID),CENTER,gcell,iCoord,sizeX)
#if NDIM > 1
    ! y coordinates
    call Grid_getCellCoords(JAXIS,blockList(blockID),CENTER,gcell,jCoord,sizeY)
#endif
#if NDIM > 2
    ! z coordinates
    call Grid_getCellCoords(KAXIS,blockList(blockID),CENTER,gcell,kCoord,sizeZ)
#endif
    ! update forcing pattern, otherwise use previous forcing pattern
    if (update_accel) call st_calcAccel(blockList(blockID),blkLimits,blkLimitsGC,iCoord,jCoord,kCoord)
#endif
! ifndef CORRECT_BULK_MOTION

    call Grid_getBlkPtr(blockList(blockID),solnData,CENTER)

    do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
      do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
        do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

#ifdef VELX_VAR
          ekin_old = 0.5*(solnData(VELX_VAR,i,j,k)**2+solnData(VELY_VAR,i,j,k)**2+solnData(VELZ_VAR,i,j,k)**2)
#endif

#ifdef VELX_VAR
#ifdef ACCX_VAR
          accel = solnData(ACCX_VAR,i,j,k)
#else
          accel = accx(i,j,k)
#endif
#ifdef CORRECT_BULK_MOTION
          correction = xForce/mass*dt + xMomentum/mass
#endif
          solnData(VELX_VAR,i,j,k) = solnData(VELX_VAR,i,j,k) + accel*dt - correction
#endif

#if NDIM > 1
#ifdef VELY_VAR
#ifdef ACCY_VAR
          accel = solnData(ACCY_VAR,i,j,k)
#else
          accel = accy(i,j,k)
#endif
#ifdef CORRECT_BULK_MOTION
          correction = yForce/mass*dt + yMomentum/mass
#endif
          solnData(VELY_VAR,i,j,k) = solnData(VELY_VAR,i,j,k) + accel*dt - correction
#endif
#endif

#if NDIM > 2
#ifdef VELZ_VAR
#ifdef ACCZ_VAR
          accel = solnData(ACCZ_VAR,i,j,k)
#else
          accel = accz(i,j,k)
#endif
#ifdef CORRECT_BULK_MOTION
          correction = zForce/mass*dt + zMomentum/mass
#endif
          solnData(VELZ_VAR,i,j,k) = solnData(VELZ_VAR,i,j,k) + accel*dt - correction
#endif
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
#ifdef DENS_VAR
          ! add up total injected kinetic energy
          ekin_added = ekin_added + d_ekin * solnData(DENS_VAR,i,j,k) * dvol
#endif
#ifdef INJR_VAR
          ! fill kinetic energy injection rate d(0.5 rho v^2) / dt; note that rho=const here
          solnData(INJR_VAR,i,j,k) = d_ekin * solnData(DENS_VAR,i,j,k) / dt
#endif
        enddo
      enddo
    enddo

    call Grid_releaseBlkPtr(blockList(blockID),solnData)

#ifndef CORRECT_BULK_MOTION
#ifndef FIXEDBLOCKSIZE
    deallocate(iCoord)
    deallocate(jCoord)
    deallocate(kCoord)
#endif
#endif

  enddo ! loop over blocks

  ! write time evolution of ekin_added to file
  if (io_integralFreq .gt. 0) then
    ! synchronise output frequency with IO_output's use of io_integralFreq
    check_for_io = .true.
    if (PRESENT(pass)) then
      if (pass .eq. 1) check_for_io = .false.
    endif
    if ((check_for_io) .and. (mod(dr_nstep+1, io_integralFreq) .eq. 0)) then
      if (abs(time - last_time) .gt. TINY(time)) then
        last_time = time
        ! sum up injected kinetic energy contributions from all blocks and processors
        ekin_added_red = 0.0
        call MPI_Reduce(ekin_added, ekin_added_red, 1, MPI_DOUBLE_PRECISION, MPI_Sum, MASTER_PE, MPI_Comm_World, error)
        ekin_added = ekin_added_red
        ! only MASTER_PE writes
        if (dr_globalMe == MASTER_PE) then
          open(funit, file=trim(outfile), position='APPEND')
          write(funit,'(10(1X,ES16.9))') time, dt, ekin_added, ekin_added/dt, &
                                        xForce, yForce, zForce, xMomentum, yMomentum, zMomentum
          close(funit)
        endif
      endif
    endif
  endif

  call Timers_stop ("Stir")

  return

end subroutine Stir
