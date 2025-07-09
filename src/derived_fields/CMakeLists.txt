# Derived Fields Library

add_library(derived_fields
	DerivedFieldFactory.cpp
	BuiltinFields.cpp
	ExpressionField.cpp
	RegisterBuiltinFields.cpp
)

target_include_directories(derived_fields PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(derived_fields PUBLIC
	AMReX::amrex
	fmt::fmt
)

# Install headers
install(FILES
	DerivedFieldFactory.H
	BuiltinFields.H
	ExpressionField.H
	DESTINATION include/derived_fields
)