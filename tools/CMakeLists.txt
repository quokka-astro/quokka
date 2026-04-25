add_executable(quokka_diag_from_plotfile diag_from_plotfile.cpp)

target_sources(quokka_diag_from_plotfile
  PRIVATE
    ${QuokkaCode_SOURCE_DIR}/src/io/DiagBase.cpp
    ${QuokkaCode_SOURCE_DIR}/src/io/DiagFilter.cpp
    ${QuokkaCode_SOURCE_DIR}/src/io/DiagFramePlane.cpp
    ${QuokkaCode_SOURCE_DIR}/src/io/DiagPDF.cpp
)

if(AMReX_SPACEDIM EQUAL 3)
  target_sources(quokka_diag_from_plotfile PRIVATE ${QuokkaCode_SOURCE_DIR}/src/io/DiagVolumeRender.cpp)
endif()
