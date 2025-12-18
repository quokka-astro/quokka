#
#  Makefile for turbulence generator
#
#  Written by
#
#          Christoph Federrath
#

# general complile switches
HAVE_MPI = yes
HAVE_HDF5 = yes

# C++ compiler (e.g., g++), flags, and HDF5 library path
CCOMP = mpicxx
CFLAGS = -O3
HDF5_PATH = /opt/local

ifeq ($(strip $(HAVE_MPI)), yes)
CFLAGS += -DHAVE_MPI
endif
ifeq ($(strip $(HAVE_HDF5)), yes)
CFLAGS += -DHAVE_HDF5 -I$(HDF5_PATH)/include -L$(HDF5_PATH)/lib -lhdf5
endif

BINS = TurbGen TurbGenDemo

all: $(BINS)

clean:
	rm -f *.o *~ $(BINS)

$(BINS): %: %.cpp
	$(CCOMP) $(CFLAGS) -o $@ $<

# dependencies
$(BINS).o: TurbGen.h HDFIO.h
