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
