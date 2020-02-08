## Makefile for TwoMomentRad library

CPP_FILES=$(wildcard *.cpp)
HPP_FILES=$(wildcard *.hpp)

CXX = g++
CXXFLAGS = -g -O0


default:

format: $(CPP_FILES) $(HPP_FILES)
	clang-format -i $^

driver: main.cpp
	$(CXX) $(CXXFLAGS) $^ -o driver.exe

