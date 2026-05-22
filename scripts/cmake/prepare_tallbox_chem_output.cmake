if(NOT DEFINED OUTDIR)
  message(FATAL_ERROR "OUTDIR is required")
endif()

if(NOT OUTDIR MATCHES "/tests/tallbox_chem$")
  message(FATAL_ERROR "Refusing to remove unexpected OUTDIR='${OUTDIR}'")
endif()

file(REMOVE_RECURSE "${OUTDIR}")
file(MAKE_DIRECTORY "${OUTDIR}")
