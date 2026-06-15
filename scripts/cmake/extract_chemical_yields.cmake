if(NOT DEFINED QUOKKA_YIELD_ARCHIVE)
  message(FATAL_ERROR "QUOKKA_YIELD_ARCHIVE is required")
endif()
if(NOT DEFINED QUOKKA_YIELD_DESTINATION)
  message(FATAL_ERROR "QUOKKA_YIELD_DESTINATION is required")
endif()
if(NOT DEFINED QUOKKA_YIELD_SENTINEL)
  message(FATAL_ERROR "QUOKKA_YIELD_SENTINEL is required")
endif()
if(NOT DEFINED QUOKKA_YIELD_REQUIRED_FILES)
  set(QUOKKA_YIELD_REQUIRED_FILES
      "${QUOKKA_YIELD_DESTINATION}/AGB_yield_table.csv"
      "${QUOKKA_YIELD_DESTINATION}/SNII_yield_table.csv"
      "${QUOKKA_YIELD_DESTINATION}/WR_yield_table.csv"
      "${QUOKKA_YIELD_DESTINATION}/WR_mass_loss_distribution_table.csv")
endif()

set(QUOKKA_YIELD_TABLES_AVAILABLE TRUE)
foreach(required_file IN LISTS QUOKKA_YIELD_REQUIRED_FILES)
  if(NOT EXISTS "${required_file}")
    set(QUOKKA_YIELD_TABLES_AVAILABLE FALSE)
  endif()
endforeach()

if(QUOKKA_YIELD_TABLES_AVAILABLE)
  return()
endif()

if(NOT EXISTS "${QUOKKA_YIELD_ARCHIVE}")
  message(FATAL_ERROR "Missing Quokka chemical yield datatable archive: ${QUOKKA_YIELD_ARCHIVE}")
endif()

file(ARCHIVE_EXTRACT INPUT "${QUOKKA_YIELD_ARCHIVE}" DESTINATION "${QUOKKA_YIELD_DESTINATION}")

foreach(required_file IN LISTS QUOKKA_YIELD_REQUIRED_FILES)
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR "Quokka chemical yield datatable extraction did not create ${required_file}")
  endif()
endforeach()
