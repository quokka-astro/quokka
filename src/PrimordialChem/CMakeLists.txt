set(microphysics_network_name "primordial_chem") #this will override network_name to primordial_chem for this directory only
setup_target_for_microphysics_compilation(${microphysics_network_name} "${CMAKE_CURRENT_BINARY_DIR}/")

# use the BEFORE keyword so that these files get priority in compilation for targets in this directory
# this is critical to ensure the correct Microphysics files are linked to primordial chem target
include_directories(BEFORE ${primordial_chem_dirs} "${CMAKE_CURRENT_BINARY_DIR}/" "includes/extern_parameters.H" "includes/network_properties.H")

add_executable(test_primordial_chem test_primordial_chem.cpp "${QuokkaSourcesNoEOS}" ${primordial_chem_sources})

# this will add #define PRIMORDIAL_CHEM
target_compile_definitions(test_primordial_chem PUBLIC PRIMORDIAL_CHEM) 

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_primordial_chem)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME PrimordialChem COMMAND test_primordial_chem primordial_chem.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
