option(QUOKKA_PARTICLE_BENCHMARKS "Build optional particle migration benchmarks" OFF)
if(QUOKKA_PARTICLE_BENCHMARKS AND AMReX_SPACEDIM EQUAL 3)
  add_executable(ParticleKernelBenchmark benchmarkParticleKernels.cpp)
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(ParticleKernelBenchmark)
  endif()
endif()

add_executable(ParticleSchema testParticleSchema.cpp)
if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(ParticleSchema)
endif()
add_test(NAME ParticleSchema COMMAND ParticleSchema
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
if(AMReX_MPI)
  add_test(NAME ParticleSchemaMPI
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ParticleSchema> ${MPIEXEC_POSTFLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(ParticleSchema ParticleSchemaMPI PROPERTIES RUN_SERIAL TRUE)
endif()

add_executable(ParticleTracer testParticleTracer.cpp)
if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(ParticleTracer)
endif()
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tracer)
add_test(NAME ParticleTracer COMMAND ParticleTracer
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tracer)
if(AMReX_MPI)
  add_test(NAME ParticleTracerMPI
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ParticleTracer> ${MPIEXEC_POSTFLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tracer)
  set_tests_properties(ParticleTracer ParticleTracerMPI PROPERTIES RUN_SERIAL TRUE)
endif()

if(QUOKKA_OPENPMD)
  add_executable(ParticleOpenPMD testParticleOpenPMD.cpp ${openPMDSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(ParticleOpenPMD)
  endif()
  add_test(NAME ParticleOpenPMD COMMAND ParticleOpenPMD
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test(NAME ParticleOpenPMDHostFallback COMMAND ParticleOpenPMD host_fallback=1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(ParticleOpenPMD ParticleOpenPMDHostFallback PROPERTIES RUN_SERIAL TRUE TIMEOUT 60)
  if(AMReX_MPI AND openPMD_USE_ADIOS2)
    add_test(NAME ParticleOpenPMDMPI
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ParticleOpenPMD> ${MPIEXEC_POSTFLAGS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME ParticleOpenPMDEmptyRank
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ParticleOpenPMD> ${MPIEXEC_POSTFLAGS} empty_rank=1
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME ParticleOpenPMDMixedFallback
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ParticleOpenPMD> ${MPIEXEC_POSTFLAGS} host_fallback=2
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(ParticleOpenPMDMPI ParticleOpenPMDEmptyRank ParticleOpenPMDMixedFallback PROPERTIES RUN_SERIAL TRUE TIMEOUT 60)
  endif()
endif()

if(QUOKKA_OPENPMD)
  add_executable(OpenPMDCheckpoint testOpenPMDCheckpoint.cpp ${openPMDSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(OpenPMDCheckpoint)
  endif()
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/checkpoint)
  add_test(NAME OpenPMDCheckpoint COMMAND OpenPMDCheckpoint
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/checkpoint)
  add_test(NAME ParticleOpenPMDCheckpoint COMMAND ParticleOpenPMD checkpoint=1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/checkpoint)
  if(AMReX_MPI)
    add_test(NAME OpenPMDCheckpointMPI COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:OpenPMDCheckpoint> ${MPIEXEC_POSTFLAGS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/checkpoint)
    add_test(NAME ParticleOpenPMDCheckpointMPI COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:ParticleOpenPMD> ${MPIEXEC_POSTFLAGS} checkpoint=1 empty_rank=1
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/checkpoint)
    set_tests_properties(OpenPMDCheckpointMPI ParticleOpenPMDCheckpointMPI PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)
  endif()
  set_tests_properties(OpenPMDCheckpoint ParticleOpenPMDCheckpoint PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)
endif()

if(QUOKKA_OPENPMD AND AMReX_SPACEDIM EQUAL 3)
  set(checkpoint_mpiexec "")
  if(AMReX_MPI)
    set(checkpoint_mpiexec "${MPIEXEC_EXECUTABLE}")
  endif()
  add_test(NAME OpenPMDRestartIntegration
    COMMAND ${CMAKE_COMMAND}
      -DSIM=$<TARGET_FILE:HydroBlast3D>
      -DCOMPARE=$<TARGET_FILE:OpenPMDCheckpoint>
      -DINPUT=${CMAKE_SOURCE_DIR}/inputs/HydroBlast3D.toml
      -DRUN_DIR=${CMAKE_CURRENT_BINARY_DIR}/restart_integration
      -DMPIEXEC=${checkpoint_mpiexec} -DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/testOpenPMDRestart.cmake)
  set_tests_properties(OpenPMDRestartIntegration PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)
endif()
