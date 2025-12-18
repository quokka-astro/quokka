#
#  Makefile for turbulence forcing generator.
#
#  Written by
#
#          Christoph Federrath
#

# C compiler (e.g., gcc, icpc), flags, and HDF5 library path
CCOMP = mpicc
CFLAGS = -O3
HDF5_PATH = /opt/local

# binary target
BIN = turbulence_generator

$(BIN) : $(BIN).o
	$(CCOMP) $(CFLAGS) -o $@ $(BIN).o -L$(HDF5_PATH)/lib -lhdf5

.SUFFIXES: .c .h

.c.o:
	$(CCOMP) $(CFLAGS) -c $*.c -I$(HDF5_PATH)/include

clean :
	rm -f *.o *~ $(BIN)

# dependencies
$(BIN).o : $(BIN).h
