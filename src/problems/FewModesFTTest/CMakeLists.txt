# Test for FewModesFT Gaussian random vector field generator

if(AMReX_SPACEDIM EQUAL 3)
    add_executable(test_few_modes_ft test_few_modes_ft.cpp ${QuokkaObjSources})
    target_link_libraries(test_few_modes_ft PUBLIC AMReX::amrex ${gamma_law_libs})
    target_compile_features(test_few_modes_ft PUBLIC cxx_std_17)
endif()