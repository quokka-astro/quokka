# Builds against the vendored Microphysics EOS: a backend (EOS/<backend>), the shim's parameter reader for that backend
# (shim/<backend>) and the shim's species table for a network (shim/networks/<network>).
# Transitional: part of the EOS shim, which goes when the EOS is ported to Quokka.

set(QUOKKA_EOS_DIR ${CMAKE_CURRENT_LIST_DIR})

# The include directories for a backend and a network, most specific first.
function(quokka_eos_include_dirs out backend network)
  set(${out}
      ${QUOKKA_EOS_DIR}/shim/networks/${network}
      ${QUOKKA_EOS_DIR}/shim/${backend}
      ${QUOKKA_EOS_DIR}/EOS/${backend}
      ${QUOKKA_EOS_DIR}/EOS
      ${QUOKKA_EOS_DIR}/interfaces
      PARENT_SCOPE)
endfunction()

# The sources for a backend.
function(quokka_eos_sources out backend)
  set(_sources ${QUOKKA_EOS_DIR}/interfaces/eos_data.cpp ${QUOKKA_EOS_DIR}/shim/${backend}/eos_parameters.cpp)
  if(backend STREQUAL "photoionization")
    list(APPEND _sources ${QUOKKA_EOS_DIR}/EOS/photoionization/actual_eos_data.cpp)
  endif()
  set(${out} ${_sources} PARENT_SCOPE)
endfunction()

# Puts a backend and a network's species table on a target, ahead of the directory-wide default (gamma_law with
# general_null, set in src/CMakeLists.txt). Call it right after add_executable(), before
# setup_target_for_cuda_compilation(), so that the EOS sources are compiled as CUDA too.
function(quokka_add_eos target backend network)
  quokka_eos_include_dirs(_dirs ${backend} ${network})
  target_include_directories(${target} BEFORE PRIVATE ${_dirs})
  quokka_eos_sources(_sources ${backend})
  target_sources(${target} PRIVATE ${_sources})
endfunction()
