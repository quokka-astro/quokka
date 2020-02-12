## Makefile for TwoMomentRad library

CPP_FILES=$(wildcard *.cpp)
HPP_FILES=$(wildcard *.hpp)
SRC_FILES:=$(CPP_FILES) $(HPP_FILES)

# exclude athena_arrays.hpp from static analysis
ANALYZE_FILES:=$(filter-out athena_arrays.hpp, $(SRC_FILES))
#ANALYZE_FILES:=$(CPP_FILES)

CXX ?= g++
CXXFLAGS ?= -g -O0

STATIC_ANALYZER = cppcheck
STATIC_ANALYZER_FLAGS = --enable=all --std=c++17 --suppress=missingIncludeSystem


default: format clang-tidy driver.exe


## code formatting

format: $(SRC_FILES)
	clang-format-9 -i $^


## compiling

driver.exe: $(CPP_FILES)
	$(CXX) $(CXXFLAGS) $^ -o driver.exe


## static analysis / code linting

clang-tidy: $(ANALYZE_FILES)
	clang-tidy -checks='*,-fuchsia-*,-misc-non-private-member-variables-in-classes,-cppcoreguidelines-non-private-member-variables-in-classes,-google-runtime-references' $^ --

cpplint: $(ANALYZE_FILES)
	cpplint $^

cppcheck: $(ANALYZE_FILES)
	$(STATIC_ANALYZER) $(STATIC_ANALYZER_FLAGS) $^

flawfinder: $(ANALYZE_FILES)
	flawfinder $^


