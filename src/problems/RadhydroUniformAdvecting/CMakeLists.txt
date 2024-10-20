# Include all subdirectories under ../../../extern/Microphysics/integration
#include_directories(../../../extern/Microphysics)
include_directories(../../../extern/Microphysics/integration)
include_directories(../../../extern/Microphysics/EOS)
include_directories(../../../extern/Microphysics/interfaces)
include_directories(../../../extern/Microphysics/networks)
#include_directories(../../../extern/Microphysics/integration/VODE)
include_directories(../../../extern/Microphysics/integration/BackwardEuler)
include_directories(../../../extern/Microphysics/integration/utils)

# set(CMAKE_CURRENT_FUNCTION_LIST_DIR "../../../extern/Microphysics")

# set(networkparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/_parameters")
# set(VODEparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/integration/VODE/_parameters")
# set(integrationparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/integration/BackwardEuler/_parameters")
# set(unittestparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/unit_test/_parameters")

# set(paramfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/unit_test/burn_cell_primordial_chem/_parameters")
# set(EOSparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS/primordial_chem/_parameters")
# set(networkpcparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/primordial_chem/_parameters")

add_executable(test_radhydro_uniform_advecting test_radhydro_uniform_advecting.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_uniform_advecting)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroUniformAdvecting COMMAND test_radhydro_uniform_advecting RadhydroUniformAdvecting.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
