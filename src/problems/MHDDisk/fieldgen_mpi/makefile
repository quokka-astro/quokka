# Same build as makefile3, except this one also passes an explicit FFTW
# include path via $(FFTW_ROOT) (needed on systems where the FFTW module
# sets FFTW_ROOT but mpicc doesn't search it by default). Use makefile3
# instead if your mpicc already finds fftw3-mpi.h without that flag.
LDFLAGS		= -lm -lfftw3_mpi -lfftw3
DEFINES		= -DBINARY -DVERBOSE
#-DSOLENOIDAL
COPTF		= -O3
#COPTF		= -g
# fieldgen3.c includes mpi.h and links MPI-enabled FFTW, so it must be built
# with the MPI compiler wrapper. Plain `=` (not `?=`) is required here: Make's
# built-in CC default ("cc") has origin "default", not "undefined", so `?=`
# would silently never take effect. Override on the command line if needed
# (e.g. `make CC=cc` -- works regardless of `=` vs `?=` here).
CC		= mpicc
CFLAGS      = -I$(FFTW_ROOT)/include

fieldgen3: fieldgen3.o ran2.o
	${CC} ${COPTF} -o fieldgen3 fieldgen3.o ran2.o \
		${LDFLAGS}

fieldgen3.o: fieldgen3.c
	${CC} ${COPTF} ${CFLAGS} -c fieldgen3.c ${DEFINES}

ran2.o: ran2.c
	${CC} ${COPTF} -c ran2.c

clean:
	rm -f *.o fieldgen3