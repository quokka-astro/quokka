LDFLAGS		= -lm -lfftw3_mpi -lfftw3
DEFINES		= -DBINARY -DVERBOSE
#-DSOLENOIDAL
COPTF		= -O3
#COPTF		= -g
CC		= cc
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