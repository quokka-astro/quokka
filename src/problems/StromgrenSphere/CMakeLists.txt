if(AMReX_SPACEDIM EQUAL 3)
  set(microphysics_network_name "photoionization_H_caseB_handwritten") # this will override
                                                  # network_name to
                                                  # photoionization for this
                                                  # directory only

  set(JOB_NAME StromgrenSphere)

  # NSPEC/SPECIES_ENUM/SPEC_NAMES/AION/ZION/CHEM_BANDS/... come from the
  # 'photoionization' entry in src/networks/NetworkRegistry.cmake, and
  # network_properties.H is generated from them inside
  # setup_target_for_microphysics_compilation() below.
  setup_target_for_microphysics_compilation(${microphysics_network_name}
                                            "${CMAKE_CURRENT_BINARY_DIR}/"
                                            INTEGRATOR "Rosenbrock")

  # two executables built from StromgrenSphere.hpp, each with its own test and input
  foreach(target ${JOB_NAME} ${JOB_NAME}TempIndependentRecombination)
    add_executable(
      ${target} test${target}.cpp "${QuokkaSourcesNoEOS}"
                ${${microphysics_network_name}_sources})

    # this will add #define PHOTOCHEMISTRY
    target_compile_definitions(${target} PUBLIC PHOTOCHEMISTRY SKIP_PHOTOCHEMFLUX)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
      setup_target_for_cuda_compilation(${target})
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")

    add_test(NAME ${target} COMMAND ${target} ../inputs/${target}.toml ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  endforeach()
endif()