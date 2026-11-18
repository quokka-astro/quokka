# needs a checkpoint converted from ORION2 data (scripts/python/orion2_to_quokka_checkpoint.py), so no ctest
if (AMReX_SPACEDIM EQUAL 3)
    quokka_add_problem(JOB_NAME OrionRestart ADD_TEST OFF)
endif()
