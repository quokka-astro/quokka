## Makefile for TwoMomentRad library

KOKKOS_PATH = ./kokkos
KOKKOS_DEVICES = OpenMP

CPP_FILES=$(wildcard *.cpp)
HPP_FILES=$(wildcard *.hpp)
SRC_FILES:=$(CPP_FILES) $(HPP_FILES)
ANALYZE_FILES:=$(filter-out athena_arrays.hpp, $(SRC_FILES))
OBJ_FILES:=$(CPP_FILES:.cpp=.o)

STATIC_ANALYZER = cppcheck
STATIC_ANALYZER_FLAGS = --enable=all --std=c++17 --suppress=missingIncludeSystem

CXX ?= g++
CXXFLAGS ?= -g -O0 --std=c++17 -I/usr/include/python3.6

LINK = $(CXX)
LINKFLAGS = 
LIB = -lpython3.6m


include $(KOKKOS_PATH)/Makefile.kokkos

default: format clang-tidy driver.exe


## code formatting

format: $(SRC_FILES)
	clang-format-9 -i $^


## compiling

%.o: %.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

driver.exe: $(OBJ_FILES) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(OBJ_FILES) $(KOKKOS_LIBS) $(LIB) -o driver.exe


## static analysis / code linting

clang-tidy: $(ANALYZE_FILES)
	clang-tidy -extra-arg=-std=c++17 -checks='*,-fuchsia-*,-misc-non-private-member-variables-in-classes,-cppcoreguidelines-non-private-member-variables-in-classes,-google-runtime-references' $^ -- | sed 's/\x1b\[[0-9;]*m//g'

cpplint: $(ANALYZE_FILES)
	cpplint $^

cppcheck: $(ANALYZE_FILES)
	$(STATIC_ANALYZER) $(STATIC_ANALYZER_FLAGS) $^

clang-analyzer: $(ANALYZE_FILES)
	scan-build -v make driver.exe



