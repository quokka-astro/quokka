# Quokka subset of Microphysics; see README.quokka.md for provenance and scope.
cmake_minimum_required(VERSION 3.20)

project(Microphysics
    VERSION 1.0.0
    DESCRIPTION "Quokka vendored Microphysics EOS and chemistry"
    LANGUAGES CXX C)

#----------------------------------------------------------------------------------------------------------------------
# general settings and options
#----------------------------------------------------------------------------------------------------------------------

find_package (Python COMPONENTS Interpreter Development)

include(CTest)

# Reject obsolete backend selections instead of silently changing the solver.
if(DEFINED DTypeFront_INTEGRATOR AND NOT DTypeFront_INTEGRATOR STREQUAL "Rosenbrock")
  message(FATAL_ERROR "DTypeFront_INTEGRATOR is no longer supported; Rosenbrock is the only ODE backend. Remove this cache entry.")
endif()
unset(DTypeFront_INTEGRATOR CACHE)

#use this function to build Microphysics using different EOS for different test problems in external codes
function(setup_target_for_microphysics_compilation network_name output_dir)

  set(networkparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/_parameters")
  if(NOT ARGC EQUAL 2)
    message(FATAL_ERROR "setup_target_for_microphysics_compilation takes only network_name and output_dir; Rosenbrock is the only ODE backend")
  endif()
  set(integrator_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/integration/Rosenbrock")
  set(integratorparamfile "${integrator_dir}/_parameters")
  set(integrationparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/integration/_parameters")


  if (${network_name} STREQUAL "gamma_law")
    set(EOSparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS/gamma_law/_parameters")

    set(networkpropfile "${output_dir}/network_properties.H")
    set(networkfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/general_null/gammalaw.net")
    set(networkheadertemplatefile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/general_null/network_header.template")

    set (gamma_law_dirs ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/util ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/util/gcem/include
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/interfaces
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS/gamma_law
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/general_null
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/constants PARENT_SCOPE)

    execute_process(COMMAND python3 "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/util/build_scripts/write_probin.py" --pa "${EOSparamfile} ${networkparamfile}
                    ${integratorparamfile} ${integrationparamfile}" --use_namespace WORKING_DIRECTORY ${output_dir}/)

    set(gamma_law_sources ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/interfaces/eos_data.cpp
                          ${output_dir}/extern_parameters.cpp PARENT_SCOPE)
    execute_process(COMMAND python3 "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/general_null/write_network.py" --header_template "${networkheadertemplatefile}" --header_output "${networkpropfile}" -s "${networkfile}" WORKING_DIRECTORY ${output_dir}/)

  elseif (${network_name} STREQUAL "primordial_chem")
    #need these to write extern_parameters.H
    set(EOSparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS/primordial_chem/_parameters")
    set(networkpcparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/primordial_chem/_parameters")

    #similarly, we want network_properties.H
    set(networkpropfile "${output_dir}/network_properties.H")
    set(networkfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/primordial_chem/pynucastro.net")
    set(networkheadertemplatefile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/general_null/network_header.template")

    #DO NOT change the order of the directories below!
    set (primordial_chem_dirs ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/util ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/util/gcem/include
                             ${integrator_dir}
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/integration ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/interfaces
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS/primordial_chem
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/primordial_chem ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/constants
                             PARENT_SCOPE)

    #we need to have extern_parameters.cpp be available at configure time
    #the script write_probin.py writes this .cpp file so we call it here
    #note, execute_process only works on 'cmake' and not 'make'
    #so, if any of the _parameter files are changed, one needs to re-run 'cmake'
    #to generate updated header files

    execute_process(COMMAND python3 "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/util/build_scripts/write_probin.py" --pa "${EOSparamfile} ${networkpcparamfile}
                    ${networkparamfile} ${integratorparamfile} ${integrationparamfile} " --use_namespace WORKING_DIRECTORY ${output_dir}/)


    set(primordial_chem_sources ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/interfaces/eos_data.cpp
                                ${output_dir}/extern_parameters.cpp PARENT_SCOPE)

    execute_process(COMMAND python3 "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks/general_null/write_network.py" --header_template "${networkheadertemplatefile}" --header_output "${networkpropfile}" -s "${networkfile}" WORKING_DIRECTORY ${output_dir}/)

  elseif (${network_name} STREQUAL "photoionization")
    #need these to write extern_parameters.H
    set(EOSparamfile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS/photoionization/_parameters")
    set(networkpcparamfile "${CMAKE_SOURCE_DIR}/src/networks/photoionization/_parameters")

    # network_properties.H for photoionization is not generated from a .net
    # file via write_network.py -- each problem's own CMakeLists.txt writes
    # it directly (via configure_file of network_header.template) with
    # hand-specified species, since there is no general_null-style .net file
    # for this network.

    #DO NOT change the order of the directories below!
    set (photoionization_dirs ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/util ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/util/gcem/include
                             ${integrator_dir}
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/integration ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/interfaces
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS/photoionization
                             ${CMAKE_SOURCE_DIR}/src/networks/photoionization ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/networks
                             ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/constants
                             PARENT_SCOPE)

    #we need to have extern_parameters.cpp be available at configure time
    #the script write_probin.py writes this .cpp file so we call it here
    #note, execute_process only works on 'cmake' and not 'make'
    #so, if any of the _parameter files are changed, one needs to re-run 'cmake'
    #to generate updated header files

    execute_process(COMMAND python3 "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/util/build_scripts/write_probin.py" --pa "${EOSparamfile} ${networkpcparamfile}
                    ${networkparamfile} ${integratorparamfile} ${integrationparamfile} " --use_namespace WORKING_DIRECTORY ${output_dir}/)


    set(photoionization_sources ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/interfaces/eos_data.cpp
                                ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EOS/photoionization/actual_eos_data.cpp
                                ${output_dir}/extern_parameters.cpp PARENT_SCOPE)

  else()
    message(FATAL_ERROR "Given network_name " ${network_name} " currently not supported. Use 'gamma_law', 'primordial_chem', or 'photoionization'. ")

  endif()


endfunction()
