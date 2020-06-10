## Makefile for TwoMomentRad library

KOKKOS_PATH = ../extern/kokkos
KOKKOS_DEVICES = serial

FMT_PATH = ../extern/fmt/include

CPP_FILES=$(wildcard *.cpp)
HPP_FILES=$(wildcard *.hpp)
SRC_FILES:=$(CPP_FILES) $(HPP_FILES)
ANALYZE_FILES:=$(filter-out athena_arrays.hpp, $(SRC_FILES))
OBJ_FILES:=$(CPP_FILES:.cpp=.o)

STATIC_ANALYZER = cppcheck
STATIC_ANALYZER_FLAGS = --enable=all --suppress=missingIncludeSystem --std=c++17

CXX ?= clang++
CXXFLAGS ?= -g -O0 --std=c++17 -I/usr/include/python3.6 -DFMT_HEADER_ONLY -I$(FMT_PATH)

LINK = $(CXX)
LINKFLAGS = 
LIB = -lpython3.6m


default: driver.exe


## Kokkos static library (built in-tree)

KOKKOS_CXX_STANDARD=c++17
include $(KOKKOS_PATH)/Makefile.kokkos


## code formatting

format: $(SRC_FILES)
	clang-format-10 -i $^


## compiling

%.o: %.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) -c $<

driver.exe: $(OBJ_FILES) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(OBJ_FILES) $(KOKKOS_LIBS) $(LIB) -o driver.exe


## static analysis / code linting

tidy: $(ANALYZE_FILES)
	@clang-tidy -checks='*,-fuchsia-*,-misc-non-private-member-variables-in-classes,-cppcoreguidelines-non-private-member-variables-in-classes,-google-runtime-references,-readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers' $^ -- $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) | sed 's/\x1b\[[0-9;]*m//g'

cpplint: $(ANALYZE_FILES)
	cpplint $^

cppcheck: $(ANALYZE_FILES)
	$(STATIC_ANALYZER) $(STATIC_ANALYZER_FLAGS) $^

analyze: $(ANALYZE_FILES)
	scan-build-10 -v make driver.exe



