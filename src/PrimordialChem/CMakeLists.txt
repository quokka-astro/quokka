set(microphysics_network_name "primordial_chem") #this will override network_name to primordial_chem for this directory only
setup_target_for_microphysics_compilation(${microphysics_network_name} "${CMAKE_CURRENT_BINARY_DIR}/")

#use the BEFORE keyword so that these files get priority in compilation for targets in this directory
#this is critical to ensure the correct Microphysics files are linked to primordial chem target
include_directories(BEFORE ${primordial_chem_dirs} "${CMAKE_CURRENT_BINARY_DIR}/" "includes/extern_parameters.H" "includes/network_properties.H")

add_executable(test_primordial_chem test_primordial_chem.cpp ../main.cpp "${openPMDSources}" ../GrackleDataReader.cpp ../CloudyCooling.cpp ../Chemistry.cpp ${primordial_chem_sources})
target_compile_definitions(test_primordial_chem PUBLIC PRIMORDIAL_CHEM) #this will add #define PRIMORDIAL_CHEM

#de-link QuokkaObj from this target to avoid multiple definitions of same variables
#this is also why we build main.cpp CloudyCooling.cpp GrackleDataReader.cpp again for this target
#get_target_property(tlink_libs test_primordial_chem LINK_LIBRARIES)
#list(REMOVE_ITEM tlink_libs QuokkaObj)
#set_property(TARGET test_primordial_chem PROPERTY LINK_LIBRARIES ${tlink_libs})


if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_primordial_chem)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME PrimordialChem COMMAND test_primordial_chem primordial_chem.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
