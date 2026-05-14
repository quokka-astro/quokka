# Install script for directory: /Users/meow/quokka/extern/amrex

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/meow/quokka/build-chuhan/amrex/Src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX" TYPE FILE FILES
    "/Users/meow/quokka/build-chuhan/amrex/lib/cmake/AMReX/AMReXConfig.cmake"
    "/Users/meow/quokka/build-chuhan/amrex/lib/cmake/AMReX/AMReXConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/meow/quokka/build-chuhan/amrex/Src/libamrex_3d.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libamrex_3d.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libamrex_3d.a")
    execute_process(COMMAND "/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libamrex_3d.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ccse-mpi.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Math.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Algorithm.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Any.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Array.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BlockMutex.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Enum.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuComplex.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Order.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_SmallMatrix.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ConstexprFor.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Vector.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_TableData.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Tuple.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_TypeList.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Demangle.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Exception.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Extension.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_PODVector.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ParmParse.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Functional.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Stack.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_String.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Utility.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FileSystem.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ValLocPair.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Reduce.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Scan.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Partition.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Morton.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Random.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_RandomEngine.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BLassert.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ArrayLim.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_REAL.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_INT.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_CONSTANTS.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_SPACE.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_DistributionMapping.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ParallelDescriptor.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_OpenMP.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ParallelReduce.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ForkJoin.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ParallelContext.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_VisMFBuffer.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_VisMF.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_AsyncOut.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BackgroundThread.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Arena.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BArena.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_CArena.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_PArena.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_SArena.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_DataAllocator.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BLProfiler.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BLBackTrace.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BLFort.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_NFiles.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_parstream.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ANSIEscCode.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FabConv.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FPC.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_VectorIO.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Print.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_IntConv.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_IOFormat.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Box.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BoxIterator.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Dim3.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_IntVect.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_IndexType.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Loop.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Loop.nolint.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Orientation.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Periodicity.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_RealBox.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_RealVect.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BoxList.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BoxArray.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BoxDomain.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FArrayBox.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_IArrayBox.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BaseFab.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Array4.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MakeType.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_TypeTraits.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FabDataType.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FabFactory.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BaseFabUtility.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MultiFab.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MFCopyDescriptor.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_iMultiFab.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FabArrayBase.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MFIter.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FabArray.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FACopyDescriptor.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FabArrayCommI.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FBI.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_PCI.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FabArrayUtility.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_LayoutData.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_CoordSys.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_COORDSYS_3D_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_COORDSYS_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Geometry.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MultiFabUtil.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MultiFabUtilI.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MultiFabUtil_3D_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MultiFabUtil_nd_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MultiFabUtil_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BCRec.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_PhysBCFunct.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BCUtil.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BC_TYPES.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FilCC_3D_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FilCC_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FilFC_3D_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FilFC_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FilND_C.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_NonLocalBC.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_NonLocalBCImpl.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_PlotFileUtil.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_PlotFileDataImpl.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_FEIntegrator.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_IntegratorBase.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_RKIntegrator.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_TimeIntegrator.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_RungeKutta.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Gpu.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuQualifiers.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuKernelInfo.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuPrint.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuAssert.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuTypes.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuControl.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunch.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunch.nolint.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunchGlobal.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunchMacrosG.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunchMacrosG.nolint.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunchMacrosC.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunchMacrosC.nolint.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunchFunctsG.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunchFunctsC.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuLaunchFunctsSIMD.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuError.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuDevice.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuBuffer.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuAtomic.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuUtility.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuAsyncArray.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuElixir.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuMemory.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuRange.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuReduce.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuAllocators.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_GpuContainers.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MFParallelFor.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MFParallelForC.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MFParallelForG.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_SIMD.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_TagParallelFor.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_CTOParallelForImpl.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_ParReduce.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_CudaGraph.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Machine.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MemPool.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/AMReX_Parser.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/AMReX_Parser_Exe.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/AMReX_Parser_Y.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/amrex_parser.lex.nolint.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/amrex_parser.tab.nolint.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/AMReX_IParser.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/AMReX_IParser_Exe.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/AMReX_IParser_Y.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/amrex_iparser.lex.nolint.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/Parser/amrex_iparser.tab.nolint.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_LUSolver.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_Slopes_K.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_BaseFwd.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_TinyProfiler.H"
    "/Users/meow/quokka/extern/amrex/Src/Base/AMReX_MPMD.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_FabSet.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_BndryRegister.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_Mask.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_MultiMask.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_BndryData.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_BoundCond.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_InterpBndryData.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_LO_BCTYPES.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_InterpBndryData_K.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_InterpBndryData_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_LOUtil_K.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_YAFluxRegister.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_YAFluxRegister_K.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_YAFluxRegister_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_BoundaryFwd.H"
    "/Users/meow/quokka/extern/amrex/Src/Boundary/AMReX_EdgeFluxRegister.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_AmrCore.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_Cluster.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_ErrorList.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_FillPatchUtil.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_FillPatchUtil_I.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_FillPatcher.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_FluxRegister.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_InterpBase.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_MFInterpolater.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_Interpolater.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_TagBox.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_AmrMesh.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_FluxReg_3D_C.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_FluxReg_C.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_Interp_C.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_Interp_3D_C.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_MFInterp_C.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_MFInterp_3D_C.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_InterpFaceRegister.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_InterpFaceReg_C.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_InterpFaceReg_3D_C.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_AmrCoreFwd.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_AmrParGDB.H"
    "/Users/meow/quokka/extern/amrex/Src/AmrCore/AMReX_AmrParticles.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLMG.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLMG_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLMG_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLMGBndry.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLLinOp.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLLinOp_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLCellLinOp.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLNodeLinOp.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLNodeLinOp_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLNodeLinOp_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLCellABecLap.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLCellABecLap_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLCellABecLap_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLCGSolver.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_PCGSolver.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLABecLaplacian.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLABecLap_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLABecLap_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLALaplacian.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLALap_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLALap_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLPoisson.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLPoisson_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLPoisson_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_GMRES.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_GMRES_MLMG.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_GMRES_MV.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_Smoother_MV.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_Algebra.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_AlgPartition.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_AlgVector.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_AlgVecUtil.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_CSR.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_SpMatrix.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_SpMatUtil.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/AMReX_SpMV.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLMG_2D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLPoisson_2D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLALap_2D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLNodeABecLaplacian.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLNodeABecLap_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLNodeABecLap_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLNodeLaplacian.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLTensorOp.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLTensor_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/MLMG/AMReX_MLTensor_3D_K.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/OpenBC/AMReX_OpenBC.H"
    "/Users/meow/quokka/extern/amrex/Src/LinearSolvers/OpenBC/AMReX_OpenBC_K.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_Particles.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleContainer.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_SparseBins.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParGDB.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_Particle_mod_K.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_TracerParticles.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_NeighborParticles.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_NeighborParticlesI.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_NeighborList.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_Particle.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleInit.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleContainerI.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParIter.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleMPIUtil.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleUtil.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_StructOfArrays.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ArrayOfStructs.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleTile.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_MakeParticle.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_NeighborParticlesCPUImpl.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_NeighborParticlesGPUImpl.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleBufferMap.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleCommunication.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleInterpolators.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleReduce.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleMesh.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleLocator.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleIO.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_DenseBins.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_BinIterator.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleTransformation.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_WriteBinaryParticleData.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleContainerBase.H"
    "/Users/meow/quokka/extern/amrex/Src/Particle/AMReX_ParticleArray.H"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX/AMReXTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX/AMReXTargets.cmake"
         "/Users/meow/quokka/build-chuhan/amrex/CMakeFiles/Export/2260e541ece776bcef17e59de6c71ec8/AMReXTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX/AMReXTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX/AMReXTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX" TYPE FILE FILES "/Users/meow/quokka/build-chuhan/amrex/CMakeFiles/Export/2260e541ece776bcef17e59de6c71ec8/AMReXTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX" TYPE FILE FILES "/Users/meow/quokka/build-chuhan/amrex/CMakeFiles/Export/2260e541ece776bcef17e59de6c71ec8/AMReXTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(CREATE_LINK
           libamrex_3d.a
           "/usr/local/lib/libamrex.a"
           COPY_ON_ERROR SYMBOLIC)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/amrex" TYPE DIRECTORY FILES
    "/Users/meow/quokka/extern/amrex/Tools/C_scripts"
    "/Users/meow/quokka/extern/amrex/Tools/typechecker"
    USE_SOURCE_PERMISSIONS)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/AMReX/AMReXCMakeModules" TYPE DIRECTORY FILES "/Users/meow/quokka/extern/amrex/Tools/CMake/" USE_SOURCE_PERMISSIONS)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/meow/quokka/build-chuhan/amrex/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
