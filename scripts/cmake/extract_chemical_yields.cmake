include_guard(GLOBAL)

set(QUOKKA_YIELD_ARCHIVE "${CMAKE_SOURCE_DIR}/extern/yields/quokka_yield_tables.tar.gz")
set(QUOKKA_YIELD_DESTINATION "${CMAKE_SOURCE_DIR}/extern/yields")

file(ARCHIVE_EXTRACT INPUT "${QUOKKA_YIELD_ARCHIVE}" DESTINATION "${QUOKKA_YIELD_DESTINATION}")

foreach(required_file AGB_yield_table.csv SNII_yield_table.csv WR_yield_table.csv WR_mass_loss_distribution_table.csv)
  if(NOT EXISTS "${QUOKKA_YIELD_DESTINATION}/${required_file}")
    message(FATAL_ERROR "Quokka chemical yield datatable extraction did not create ${required_file}")
  endif()
endforeach()
