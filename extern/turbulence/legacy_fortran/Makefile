#
#  Makefile for turbulence forcing generator.
#
#  Written by
#
#          Christoph Federrath
#

# Fortran compiler (e.g., gfortran, ifort)
FCOMP = mpif90

# For gnu compiler:
FLAGS = -fdefault-real-8 -O3
# For Intel compiler:
#FLAGS = -r8 -i4 -O3

# binary target
BIN = forcing_generator

$(BIN) : $(BIN).o
	$(FCOMP) $(FLAGS) -o $@ $(BIN).o

.SUFFIXES: .F90

.F90.o:
	$(FCOMP) $(FLAGS) -c $*.F90

clean :
	rm -f *.o *.mod *~ $(BIN)
