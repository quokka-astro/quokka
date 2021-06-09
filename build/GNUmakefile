# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../extern/amrex

TINY_PROFILE = TRUE
BL_NO_FORT = TRUE
DEBUG        = FALSE
USE_MPI      = TRUE
MPI_THREAD_MULTIPLE = FALSE
USE_OMP      = FALSE
USE_CUDA 	 = TRUE
COMP         = gnu
CXXSTD		 = c++17
DIM          = 2

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include ../src/Make.package
VPATH_LOCATIONS  += ../src
INCLUDE_LOCATIONS += ../src
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules