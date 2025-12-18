!!****if* source/physics/sourceTerms/StirICs/StirICsMain/StirICs
!!
!! NAME
!!  StirICs
!!
!! SYNOPSIS
!!  StirICs(integer(IN) :: blockCount,
!!          integer(IN) :: blockList(blockCount))
!!
!! DESCRIPTION
!!   This adds turbulent velocities and/or turbulent magnetic field components to the
!!   initial conditions of a simulation.
!!   Apply the stirring operator on the list of blocks provided as input.
!!   Do this only once when time = 0.0; i.e., as an initial condition.
!!   Here we also initialize turbulent magnetic fields.
!!   In case of magnetic fields, we also provide an option to maintain the mean field
!!   in every time step, exactly, by enforcing that the total flux remains constant in time
!!
!! ARGUMENTS
!!   blockCount   : The number of blocks in the list
!!   blockList(:) : The list of blocks on which to apply the stirring operator
!!
!! AUTHOR
!!   Christoph Federrath
!!
!!***

subroutine StirICs(blockCount, blockList)

#include "constants.h"
#include "Flash.h"

  use StirICs_data
  use Driver_data, ONLY : dr_globalMe, dr_restart
  use Driver_interface, ONLY : Driver_getNStep, Driver_getMype, Driver_abortFlash
  use Timers_interface, ONLY : Timers_start, Timers_stop
  use Grid_interface,   ONLY : Grid_getBlkIndexLimits, Grid_getBlkPtr, Grid_releaseBlkPtr, &
                               Grid_getDeltas, Grid_getBlkPhysicalSize, Grid_getBlkCenterCoords
  use Grid_data, ONLY : gr_imin, gr_imax, gr_jmin, gr_jmax, gr_kmin, gr_kmax
  use PhysicalConstants_interface, ONLY : PhysicalConstants_get
  use iso_c_binding, ONLY : c_double, c_float

  implicit none

#include "Flash_mpi.h"

  integer, intent(IN)                        :: blockCount
  integer, dimension(blockCount), intent(IN) :: blockList

  logical, save                :: called_already = .false.
  integer                      :: blockID, i, j, k, ii, jj, kk, nstep
  integer, dimension(2,MDIM)   :: blkLimits, blkLimitsGC
  integer                      :: ib, ie, error, istat
  real, dimension(MDIM)        :: del, blockSize, blockCenter ! MDIM is always 3
  real(c_double)               :: pos_beg(MDIM), pos_end(MDIM)
  integer                      :: ncells(MDIM)
  real                         :: mu0, dvol, RmsVelNorm
  real                         :: mass, totvol, xMomentum, yMomentum, zMomentum, xVelSq, yVelSq, zVelSq
  real                         :: xBmean, yBmean, zBmean, RmsMagNorm
  real(c_double)               :: L(3) ! domain length

  real, dimension(:,:,:,:), POINTER :: solnData

  real(c_float), allocatable, dimension(:,:,:) :: vx, vy, vz

  real, dimension(NXB) :: ke_old, ke_new
  real, dimension(NXB) :: me_old, me_new

  integer, parameter :: nGlobalSum = 8 ! Number of globally-summed quantities

#ifdef HYBRID_PRECISION
  integer, parameter :: flash_real_dtype = FLASH_DOUBLE
  real(c_double)     :: globalSumQuantities(nGlobalSum) ! Global summed quantities
  real(c_double)     :: localSumQuantities(nGlobalSum)  ! Global summed quantities
  real(c_double)     :: ekin_added, ekin_added_red, emag_added, emag_added_red
#else
  integer, parameter :: flash_real_dtype = FLASH_REAL
  real               :: globalSumQuantities(nGlobalSum) ! Global summed quantities
  real               :: localSumQuantities(nGlobalSum)  ! Global summed quantities
  real               :: ekin_added, ekin_added_red, emag_added, emag_added_red
#endif

  ! if not using intial conditions stirring, return
  if ((.not. st_useStirICs) .or. dr_restart) return

  ! return if nstep > 1 or if already called
  call Driver_getNStep(nstep)
  if ((nstep > 1) .or. called_already) return

  called_already = .true.

  call Timers_start("StirICs")

  call PhysicalConstants_get("mu0", mu0)

  ! set domain length
  L(1) = gr_imax - gr_imin
  L(2) = gr_jmax - gr_jmin
  L(3) = gr_kmax - gr_kmin

  !! =============================================
  !! Generate initial turbulent VELOCITY field
  !! =============================================
  if (st_rmsVelocity > 0.0) then

    if (dr_globalMe == MASTER_PE) then
      write (*,'(A,ES10.3)') 'StirICs: === Generating initial turbulent velocity field; st_rmsVelocity = ', st_rmsVelocity
    endif

    ! initialise the turbulence generator based on input parameters for single realisation
    call st_stirics_init_single_realisation_c(NDIM, L, st_stirMin, st_stirMax, &
          st_spectForm, st_powerLawExp, st_anglesExp, st_solWeight, st_seed);

    globalSumQuantities(:) = 0.0
    localSumQuantities(:)  = 0.0

    ! sum quantities over list of blocks
    do BlockID = 1, blockCount

      ! getting the dx's
      call Grid_getDeltas(blocklist(BlockID), del)

#if NDIM == 1
      dvol = del(IAXIS)
#endif
#if NDIM == 2
      dvol = del(IAXIS) * del(JAXIS)
#endif
#if NDIM == 3
      dvol = del(IAXIS) * del(JAXIS) * del(KAXIS)
#endif

      ! get the index limits of the block
      call Grid_getBlkIndexLimits(blockList(BlockID), blkLimits, blkLimitsGC)

      ! get turbulent pattern
      allocate(vx(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vx in StirICs.F90")
      allocate(vy(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vy in StirICs.F90")
      allocate(vz(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vz in StirICs.F90")

      ! get turbulent vector field for this block
      call Grid_getBlkPhysicalSize(blockList(BlockID), blockSize)
      call Grid_getBlkCenterCoords(blockList(BlockID), blockCenter)
      pos_beg = blockCenter - 0.5*blockSize + del/2.0 ! first active cell coordinate in block (x,y,z)
      pos_end = blockCenter + 0.5*blockSize - del/2.0 ! last  active cell coordinate in block (x,y,z)
      ncells = blkLimits(HIGH,:)-blkLimits(LOW,:)+1 ! number of active cells in (x,y,z)
      call st_stirics_get_turb_vector_unigrid_c(pos_beg, pos_end, ncells, vx, vy, vz)

      ! get a pointer to the current block of data
      call Grid_getBlkPtr(blockList(BlockID), solnData)

      ! Sum contributions from the indicated blkLimits of cells.
      do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
        kk = k-blkLimits(LOW,KAXIS)+1 ! z index of vx, vy, vz starts at 1 and goes to NZB
        do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
          jj = j-blkLimits(LOW,JAXIS)+1 ! y index of vx, vy, vz starts at 1 and goes to NYB
          do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)
            ii = i-blkLimits(LOW,IAXIS)+1 ! x index of vx, vy, vz starts at 1 and goes to NXB

            ! volume
            localSumQuantities(1) = localSumQuantities(1) + dvol
#ifdef DENS_VAR
            ! mass
            localSumQuantities(2) = localSumQuantities(2) + solnData(DENS_VAR,i,j,k)*dvol

            ! momentum and RMS velocity
#ifdef VELX_VAR
            localSumQuantities(3) = localSumQuantities(3) + solnData(DENS_VAR,i,j,k)*vx(ii,jj,kk)*dvol
            localSumQuantities(6) = localSumQuantities(6) + vx(ii,jj,kk)**2*dvol
#endif
#ifdef VELY_VAR
            localSumQuantities(4) = localSumQuantities(4) + solnData(DENS_VAR,i,j,k)*vy(ii,jj,kk)*dvol
            localSumQuantities(7) = localSumQuantities(7) + vy(ii,jj,kk)**2*dvol
#endif
#ifdef VELZ_VAR
            localSumQuantities(5) = localSumQuantities(5) + solnData(DENS_VAR,i,j,k)*vz(ii,jj,kk)*dvol
            localSumQuantities(8) = localSumQuantities(8) + vz(ii,jj,kk)**2*dvol
#endif
#endif
! ifdef DENS_VAR

          enddo
        enddo
      enddo

      call Grid_releaseBlkPtr(blockList(BlockID),solnData)

      deallocate(vx)
      deallocate(vy)
      deallocate(vz)

      if (mod(100*blockID,blockCount) == 0) then
        write(*,'(A,I5,A,F6.1,A)') '[', dr_globalMe, '] StirInitialConditions: 1st loop (kinetic): done: ', &
                                    100.0*blockID/real(blockCount), '% of my blocks.'
      endif

    enddo ! loop over blocks

    ! now communicate all global summed quantities to all processors
    call MPI_AllReduce(localSumQuantities, globalSumQuantities, nGlobalSum, &
                       flash_real_dtype, MPI_Sum, MPI_Comm_World, error)

    totvol    = globalSumQuantities(1)
    mass      = globalSumQuantities(2)
    xMomentum = globalSumQuantities(3)
    yMomentum = globalSumQuantities(4)
    zMomentum = globalSumQuantities(5)
    xVelSq    = globalSumQuantities(6) / totvol
    yVelSq    = globalSumQuantities(7) / totvol
    zVelSq    = globalSumQuantities(8) / totvol
    RmsVelNorm = sqrt(xVelSq + yVelSq + zVelSq)

    ! set to zero for adding processor contributions below
    ekin_added = 0.0

    ! Loop over local blocks
    do blockID = 1, blockCount

      ! getting the dx's
      call Grid_getDeltas(blocklist(BlockID), del)

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
      call Grid_getBlkIndexLimits(blockList(BlockID),blkLimits,blkLimitsGC)

      ! get turbulent pattern
      allocate(vx(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vx in StirICs.F90")
      allocate(vy(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vy in StirICs.F90")
      allocate(vz(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vz in StirICs.F90")

      ! get turbulent vector field for this block
      call Grid_getBlkPhysicalSize(blockList(BlockID), blockSize)
      call Grid_getBlkCenterCoords(blockList(BlockID), blockCenter)
      pos_beg = blockCenter - 0.5*blockSize + del/2.0 ! first active cell coordinate in block (x,y,z)
      pos_end = blockCenter + 0.5*blockSize - del/2.0 ! last  active cell coordinate in block (x,y,z)
      ncells = blkLimits(HIGH,:)-blkLimits(LOW,:)+1 ! number of active cells in (x,y,z)
      call st_stirics_get_turb_vector_unigrid_c(pos_beg, pos_end, ncells, vx, vy, vz)

      call Grid_getBlkPtr(blockList(blockID),solnData,CENTER)
      ib = blkLimits(LOW, IAXIS)
      ie = blkLimits(HIGH, IAXIS)
      do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
        kk = k-blkLimits(LOW,KAXIS)+1 ! z index of vx, vy, vz starts at 1 and goes to NZB
        do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
          jj = j-blkLimits(LOW,JAXIS)+1 ! y index of vx, vy, vz starts at 1 and goes to NYB

#ifdef VELX_VAR
          ke_old(:) = 0.5 * ( solnData(VELX_VAR,ib:ie,j,k)**2 + &
                              solnData(VELY_VAR,ib:ie,j,k)**2 + &
                              solnData(VELZ_VAR,ib:ie,j,k)**2 )
#endif
#ifdef VELX_VAR
          solnData(VELX_VAR,ib:ie,j,k) = solnData(VELX_VAR,ib:ie,j,k) + &
            ( vx(1:NXB,jj,kk) - xMomentum/mass ) / RmsVelNorm * st_rmsVelocity
#endif
#if NDIM > 1
#ifdef VELY_VAR
          solnData(VELY_VAR,ib:ie,j,k) = solnData(VELY_VAR,ib:ie,j,k) + &
            ( vy(1:NXB,jj,kk) - yMomentum/mass ) / RmsVelNorm * st_rmsVelocity
#endif
#if NDIM > 2
#ifdef VELZ_VAR
          solnData(VELZ_VAR,ib:ie,j,k) = solnData(VELZ_VAR,ib:ie,j,k) + &
            ( vz(1:NXB,jj,kk) - zMomentum/mass ) / RmsVelNorm * st_rmsVelocity
#endif
#endif
#endif
#ifdef VELX_VAR
          ke_new(:) = 0.5 * ( solnData(VELX_VAR,ib:ie,j,k)**2 + &
                              solnData(VELY_VAR,ib:ie,j,k)**2 + &
                              solnData(VELZ_VAR,ib:ie,j,k)**2 )
#endif
#ifdef ENER_VAR
          ! update the total energy
          solnData(ENER_VAR,ib:ie,j,k) = solnData(ENER_VAR,ib:ie,j,k) + ke_new(:)-ke_old(:)
#endif
#ifdef EKIN_VAR
          ! update the kinetic energy
          solnData(EKIN_VAR,ib:ie,j,k) = solnData(EKIN_VAR,ib:ie,j,k) + ke_new(:)-ke_old(:)
#endif
#ifdef DENS_VAR
          ! compute injected kinetic energy and add it to the sum
          ekin_added = ekin_added + SUM((ke_new(:)-ke_old(:))*solnData(DENS_VAR,ib:ie,j,k))*dvol
#endif
        enddo ! j
      enddo ! k

      call Grid_releaseBlkPtr(blockList(blockID),solnData)

      deallocate(vx)
      deallocate(vy)
      deallocate(vz)

      if (mod(100*blockID,blockCount) == 0) then
        write(*,'(A,I5,A,F6.1,A)') '[', dr_globalMe, '] StirInitialConditions: 2nd loop (kinetic): done: ', &
                                    100.0*blockID/real(blockCount), '% of my blocks.'
      endif

    enddo ! loop over blocks

    ! sum up injected kinetic energy contributions from all blocks and processors
    ekin_added_red = 0.0
    call MPI_Reduce(ekin_added, ekin_added_red, 1, flash_real_dtype, MPI_Sum, MASTER_PE, MPI_Comm_World, error)
    if (dr_globalMe == MASTER_PE) print *, 'StirICs: E_kin added  = ', ekin_added_red
    if (dr_globalMe == MASTER_PE) print *, 'StirICs: rms(v) added = ', st_rmsVelocity

  endif ! st_rmsVelocity > 0
  !! =============================================

  call MPI_Barrier(MPI_Comm_World, error)

  !! =============================================
  !! Generate initial turbulent MAGNETIC field
  !! =============================================
  if (st_rmsMagneticField > 0.0) then

    if (dr_globalMe == MASTER_PE) then
      write (*,'(A,ES10.3)') &
        'StirICs: === Generating initial turbulent magnetic field; st_rmsMagneticField = ', st_rmsMagneticField
    endif

    ! initialise the turbulence generator based on input parameters for single realisation
    call st_stirics_init_single_realisation_c(NDIM, L, st_stirMagneticKMin, st_stirMagneticKMax, &
          st_MagneticSpectForm, st_MagneticPowerLawExp, st_anglesExp, 1d0, st_MagneticSeed);

    globalSumQuantities(:) = 0.0
    localSumQuantities(:)  = 0.0

    ! sum quantities over list of blocks
    do BlockID = 1, blockCount

      ! getting the dx's
      call Grid_getDeltas(blocklist(BlockID), del)

#if NDIM == 1
      dvol = del(IAXIS)
      del(JAXIS) = 1.0
      del(KAXIS) = 1.0
#endif
#if NDIM == 2
      dvol = del(IAXIS) * del(JAXIS)
      del(KAXIS) = 1.0
#endif
#if NDIM == 3
      dvol = del(IAXIS) * del(JAXIS) * del(KAXIS)
#endif

      ! Get cell coordinates for this block
      call Grid_getBlkIndexLimits(blockList(BlockID),blkLimits,blkLimitsGC)

      ! get turbulent pattern
      allocate(vx(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vx in StirICs.F90")
      allocate(vy(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vy in StirICs.F90")
      allocate(vz(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vz in StirICs.F90")

      ! get turbulent vector field for this block
      call Grid_getBlkPhysicalSize(blockList(BlockID), blockSize)
      call Grid_getBlkCenterCoords(blockList(BlockID), blockCenter)
      pos_beg = blockCenter - 0.5*blockSize + del/2.0 ! first active cell coordinate in block (x,y,z)
      pos_end = blockCenter + 0.5*blockSize - del/2.0 ! last  active cell coordinate in block (x,y,z)
      ncells = blkLimits(HIGH,:)-blkLimits(LOW,:)+1 ! number of active cells in (x,y,z)
      call st_stirics_get_turb_vector_unigrid_c(pos_beg, pos_end, ncells, vx, vy, vz)

      ! get a pointer to the current block of data
      call Grid_getBlkPtr(blockList(BlockID), solnData)

      ! Sum contributions from the indicated blkLimits of cells.
      do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
        kk = k-blkLimits(LOW,KAXIS)+1 ! z index of vx, vy, vz starts at 1 and goes to NZB
        do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
          jj = j-blkLimits(LOW,JAXIS)+1 ! y index of vx, vy, vz starts at 1 and goes to NYB
          do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)
            ii = i-blkLimits(LOW,IAXIS)+1 ! x index of vx, vy, vz starts at 1 and goes to NXB

            ! area in x
            localSumQuantities(1) = localSumQuantities(1) + del(JAXIS)*del(KAXIS)
            ! area in y
            localSumQuantities(2) = localSumQuantities(2) + del(IAXIS)*del(KAXIS)
            ! area in z
            localSumQuantities(3) = localSumQuantities(3) + del(IAXIS)*del(JAXIS)

            ! mean B field (sum over local fluxes, and divide by total area below)
            localSumQuantities(4) = localSumQuantities(4) + vx(ii,jj,kk)*del(JAXIS)*del(KAXIS)
            localSumQuantities(5) = localSumQuantities(5) + vy(ii,jj,kk)*del(IAXIS)*del(KAXIS)
            localSumQuantities(6) = localSumQuantities(6) + vz(ii,jj,kk)*del(IAXIS)*del(JAXIS)

            ! rms B field
            localSumQuantities(7) = localSumQuantities(7) + &
                                ( vx(ii,jj,kk)**2 + vy(ii,jj,kk)**2 + vz(ii,jj,kk)**2 ) * dvol

            ! volume
            localSumQuantities(8) = localSumQuantities(8) + dvol

          enddo
        enddo
      enddo

      call Grid_releaseBlkPtr(blockList(BlockID),solnData)

      deallocate(vx)
      deallocate(vy)
      deallocate(vz)

      if (mod(100*blockID,blockCount) == 0) then
        write(*,'(A,I5,A,F6.1,A)') '[', dr_globalMe, '] StirInitialConditions: 1st loop (magnetic): done: ', &
                                    100.0*blockID/real(blockCount), '% of my blocks.'
      endif

    enddo ! loop over blocks

    ! now communicate all global summed quantities to all processors
    call MPI_AllReduce(localSumQuantities, globalSumQuantities, nGlobalSum, &
                       flash_real_dtype, MPI_Sum, MPI_Comm_World, error)

    xBmean = globalSumQuantities(4) / globalSumQuantities(1)
    yBmean = globalSumQuantities(5) / globalSumQuantities(2)
    zBmean = globalSumQuantities(6) / globalSumQuantities(3)
    RmsMagNorm = sqrt( globalSumQuantities(7) / globalSumQuantities(8) - xBmean**2 - yBmean**2 - zBmean**2 )

    ! set to zero for adding processor contributions below
    emag_added = 0.0

    ! Loop over local blocks
    do blockID = 1, blockCount

      ! getting the dx's
      call Grid_getDeltas(blocklist(BlockID), del)

#if NDIM == 1
      dvol = del(IAXIS)
      del(JAXIS) = 1.0
      del(KAXIS) = 1.0
#endif
#if NDIM == 2
      dvol = del(IAXIS) * del(JAXIS)
      del(KAXIS) = 1.0
#endif
#if NDIM == 3
      dvol = del(IAXIS) * del(JAXIS) * del(KAXIS)
#endif

      ! Get cell coordinates for this block
      call Grid_getBlkIndexLimits(blockList(BlockID),blkLimits,blkLimitsGC)

      ! get turbulent pattern
      allocate(vx(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vx in StirICs.F90")
      allocate(vy(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vy in StirICs.F90")
      allocate(vz(NXB,NYB,NZB),stat=istat)
      if (istat .ne. 0) call Driver_abortFlash("could not allocate vz in StirICs.F90")

      ! get turbulent vector field for this block
      call Grid_getBlkPhysicalSize(blockList(BlockID), blockSize)
      call Grid_getBlkCenterCoords(blockList(BlockID), blockCenter)
      pos_beg = blockCenter - 0.5*blockSize + del/2.0 ! first active cell coordinate in block (x,y,z)
      pos_end = blockCenter + 0.5*blockSize - del/2.0 ! last  active cell coordinate in block (x,y,z)
      ncells = blkLimits(HIGH,:)-blkLimits(LOW,:)+1 ! number of active cells in (x,y,z)
      call st_stirics_get_turb_vector_unigrid_c(pos_beg, pos_end, ncells, vx, vy, vz)

      call Grid_getBlkPtr(blockList(blockID),solnData,CENTER)
      ib = blkLimits(LOW, IAXIS)
      ie = blkLimits(HIGH, IAXIS)
      do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
        kk = k-blkLimits(LOW,KAXIS)+1 ! z index of vx, vy, vz starts at 1 and goes to NZB
        do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
          jj = j-blkLimits(LOW,JAXIS)+1 ! y index of vx, vy, vz starts at 1 and goes to NYB

#ifdef MAGZ_VAR
          me_old(:) = solnData(MAGX_VAR,ib:ie,j,k)**2 + &
                      solnData(MAGY_VAR,ib:ie,j,k)**2 + &
                      solnData(MAGZ_VAR,ib:ie,j,k)**2
#endif
#ifdef MAGX_VAR
          solnData(MAGX_VAR,ib:ie,j,k) = solnData(MAGX_VAR,ib:ie,j,k) + &
            (vx(1:NXB,jj,kk) - xBmean) / RmsMagNorm * st_rmsMagneticField
#endif
#ifdef MAGY_VAR
          solnData(MAGY_VAR,ib:ie,j,k) = solnData(MAGY_VAR,ib:ie,j,k) + &
            (vy(1:NXB,jj,kk) - yBmean) / RmsMagNorm * st_rmsMagneticField
#endif
#ifdef MAGZ_VAR
          solnData(MAGZ_VAR,ib:ie,j,k) = solnData(MAGZ_VAR,ib:ie,j,k) + &
            (vz(1:NXB,jj,kk) - zBmean) / RmsMagNorm * st_rmsMagneticField
#endif
#ifdef MAGZ_VAR
          me_new(:) = solnData(MAGX_VAR,ib:ie,j,k)**2 + &
                      solnData(MAGY_VAR,ib:ie,j,k)**2 + &
                      solnData(MAGZ_VAR,ib:ie,j,k)**2
#endif
          ! compute injected magnetic energy and add it to the sum
          emag_added = emag_added + SUM(me_new(:)-me_old(:))/(2*mu0)*dvol

        enddo ! j
      enddo ! k

      call Grid_releaseBlkPtr(blockList(blockID),solnData)

      deallocate(vx)
      deallocate(vy)
      deallocate(vz)

      if (mod(100*blockID,blockCount) == 0) then
        write(*,'(A,I5,A,F6.1,A)') '[', dr_globalMe, '] StirInitialConditions: 2nd loop (magnetic): done: ', &
                                    100.0*blockID/real(blockCount), '% of my blocks.'
      endif

    enddo ! loop over blocks

    ! sum up injected kinetic energy contributions from all blocks and processors
    emag_added_red = 0.0
    call MPI_Reduce(emag_added, emag_added_red, 1, flash_real_dtype, MPI_Sum, MASTER_PE, MPI_Comm_World, error)
    if (dr_globalMe == MASTER_PE) print *, 'StirICs: E_mag added  = ', emag_added_red
    if (dr_globalMe == MASTER_PE) print *, 'StirICs: rms(B) added = ', &
                                        & sqrt( emag_added_red*(2*mu0) / globalSumQuantities(8) )

  endif ! st_rmsMagneticField > 0
  !! =============================================

  call Timers_stop ("StirICs")

  return

end subroutine StirICs
