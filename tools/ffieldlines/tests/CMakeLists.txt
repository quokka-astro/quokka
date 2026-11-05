add_executable(ffieldlines_make_test_plotfile MakeTestPlotfile.cpp)
target_link_libraries(ffieldlines_make_test_plotfile PRIVATE AMReX::amrex)

find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(WARNING "ffieldlines: Python 3 not found; regression tests are disabled")
  return()
endif()
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy" RESULT_VARIABLE _ffieldlines_numpy)
if(NOT _ffieldlines_numpy EQUAL 0)
  message(WARNING "ffieldlines: numpy not found for ${Python3_EXECUTABLE}; regression tests are disabled")
  return()
endif()

set(_ffieldlines_mpiexec "")
if(AMReX_MPI AND MPIEXEC_EXECUTABLE)
  set(_ffieldlines_mpiexec "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4")
endif()

foreach(_case circle decomposition mpi twolevel straight periodic terminations legacy no_temperature validation)
  add_test(NAME ffieldlines_${_case}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_fieldlines.py
      --ffieldlines $<TARGET_FILE:ffieldlines>
      --make-plotfile $<TARGET_FILE:ffieldlines_make_test_plotfile>
      --workdir ${CMAKE_CURRENT_BINARY_DIR}/work_${_case}
      --mpiexec "${_ffieldlines_mpiexec}"
      ${_case})
endforeach()
