#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <fftw3-mpi.h>

#define PI 3.141592654
#define MAXATTEMPTS 10000

/* FFTW3's fftw_complex is `double[2]`. We define a struct with the same
   memory layout so the field-generation math below can keep using
   the .re / .im notation from the original FFTW2 code with only a
   pointer cast at the FFTW3 API boundary. */
typedef struct { double re, im; } cplx_t;

int ngrid, ngridft;
long seed;
int nproc, myrank;
/* FFTW3-MPI reports sizes as ptrdiff_t (needed for >2^31-element
   transforms); locsize/alloc_local are in units of complex numbers
   per field component (the actual buffer FFTW3 requires you to
   allocate, which can be >= nxloc*ngrid*ngridft due to internal
   padding -- unlike FFTW2, where the local size was exactly
   nxloc*ngrid*ngridft). All per-component array offsets must use
   alloc_local/locsize, not nxloc*ngrid*ngridft, or you'll walk off
   the end of one component's buffer into the next. */
ptrdiff_t nxloc, xlocstart, alloc_local, locsize;

extern float ran2(long *idum);

void readSeedPar()
{
  /* Routine to initialize random seed for parallel runs. We need
     nproc different seeds. */
  FILE *fp;
  int n, val;
  long procseed;
  MPI_Status status;

  if (myrank==0) {
    if ((fp=fopen("seed.txt", "r"))) {
      val=fscanf(fp, "%ld", &seed);
      if (val==EOF) seed=nproc;
      for (n=1; n<nproc; n++) {
	val=fscanf(fp, "%ld", &procseed);
	if (val==EOF) procseed=n+nproc+1;
	MPI_Send(&procseed, 1, MPI_LONG, n, n, MPI_COMM_WORLD);
      }
      fclose(fp);
    } else {
      seed=nproc;
      for (n=1; n<nproc; n++) {
	procseed=n+nproc+1;
	MPI_Send(&procseed, 1, MPI_LONG, n, n, MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Recv(&seed, 1, MPI_LONG, 0, myrank, MPI_COMM_WORLD, &status);
  }
  seed *= -1;
  ran2(&seed);
}

void writeSeedPar()
{
  FILE *fp;
  int i, j=0;
  long procseed;
  MPI_Status status;
  char name[200];

  if (myrank==0) {
    fp=fopen("seed.txt", "w");
    fprintf(fp, "%ld\n", seed);
  }
  for (i=1; i<nproc; i++) {
    if (myrank==0) {
      MPI_Recv(&procseed, 1, MPI_LONG, i, i, MPI_COMM_WORLD, &status);
      fprintf(fp, "%ld\n", procseed);
    } else if (myrank==i) {
      MPI_Send(&seed, 1, MPI_LONG, 0, i, MPI_COMM_WORLD);
    }
  }
  if (myrank==0) fclose(fp);
}

void gridftGen(cplx_t *grid, float kmin, float kmax, float kidx)
{
  int i, j, k, ieff, jeff;
  long wavenumsqr; /* long, not int: ieff/jeff/k can be large enough at
                       high ngrid that ieff*ieff+jeff*jeff+k*k overflows
                       a 32-bit int */
  ptrdiff_t idx;
  float kminsqr, kmaxsqr;
  float r, th, rnum, sigma;

  kminsqr=kmin*kmin;
  kmaxsqr=kmax*kmax;
  /* Take away two powers to balance out the fact that the volume
     element dV \propto k^2 dk, meaning there are more gridpoints at
     higher k. */
  kidx-=2;
  for (i=0; i<nxloc; i++) {
    if (i+xlocstart<=ngridft) ieff=i+xlocstart;
    else ieff=ngrid-(i+xlocstart);
    for (j=0; j<ngrid; j++) {
      if (j<=ngridft) jeff=j; else jeff=ngrid-j;
      for (k=0; k<ngridft; k++) {
	wavenumsqr=(long)ieff*ieff+(long)jeff*jeff+(long)k*k;
	/* ptrdiff_t-safe flat index: at high ngrid, (i*ngrid+j)*ngridft+k
	   computed in 32-bit int silently overflows and corrupts the
	   offset, so force the arithmetic to 64-bit here. */
	idx=((ptrdiff_t)i*ngrid+j)*ngridft+k;
	if ((wavenumsqr>=kminsqr) && (wavenumsqr<=kmaxsqr)) {
	  rnum=ran2(&seed);
	  sigma=sqrt(pow((double)wavenumsqr, kidx/2.0));
	  r=sigma*sqrt(2*log(1.0/(1.0-rnum)));
	  th=ran2(&seed)*2.0*PI;
	  grid[idx].re=r*cos(th);
	  grid[idx].im=r*sin(th);
	} else {
	  grid[idx].re=0.0;
	  grid[idx].im=0.0;
	}
      }
    }
  }
}

void gridftProject(cplx_t *grid) {
  int i, j, k, l;
  ptrdiff_t idx;
  double kappa_mag, kappa[3], kappan[3];
  cplx_t vdotk;

  /* Project out non-solenoidal part of field. The transformation
     required to do this is v~ -> v~ - (v.kappa) kappa, where
     kappa = [ sin(2 pi i/N), sin(2 pi j/N), sin (2 pi k/N)] /
     sqrt(sin^2 (2 pi i/N) + sin^2 (2 pi j/N) + sin^2 (2 pi k/N)).

     NOTE: component stride is alloc_local (the FFTW3-required
     per-component complex buffer size), not nxloc*ngrid*ngridft as
     in the original FFTW2 code -- see comment on alloc_local above.
  */
  for (i=0; i<nxloc; i++) {
    kappa[2]=sin((2.0*PI*(i+xlocstart))/ngrid);
    for (j=0; j<ngrid; j++) {
      kappa[1]=sin((2.0*PI*j)/ngrid);
      for (k=0; k<ngridft; k++) {
	if ((i==0) && (j==0) && (k==0)) continue;
	kappa[0]=sin((2.0*PI*k)/ngrid);
	kappa_mag = sqrt(kappa[0]*kappa[0] + kappa[1]*kappa[1] +
			 kappa[2]*kappa[2]);
	for (l=0; l<3; l++) kappan[l] = kappa[l]/kappa_mag;
	idx=((ptrdiff_t)i*ngrid+j)*ngridft+k;
	vdotk.re = vdotk.im = 0.0;
	for (l=0; l<3; l++) {
	  vdotk.re += kappan[l] * grid[l*alloc_local + idx].re;
	  vdotk.im += kappan[l] * grid[l*alloc_local + idx].im;
	}
	for (l=0; l<3; l++) {
	  grid[l*alloc_local + idx].re -= vdotk.re * kappan[l];
	  grid[l*alloc_local + idx].im -= vdotk.im * kappan[l];
	}
      }
    }
  }
}


void normalize(double *grid, float stddev)
{
  double locpower, totpower, initdev, scale;
  int i, j, k, l;
  ptrdiff_t idx;

  /* Compute power on this processor. Component stride is locsize
     (real-element units, = 2*alloc_local), not
     nxloc*ngrid*2*ngridft as in the FFTW2 version. */
  locpower=0.0;
  for (l=0; l<3; l++) {
    for (i=0; i<nxloc; i++) {
      for (j=0; j<ngrid; j++) {
	for (k=0; k<ngrid; k++) {
	  idx=l*locsize + ((ptrdiff_t)i*ngrid+j)*2*ngridft+k;
	  locpower+=grid[idx]*grid[idx];
	}
      }
    }
  }

  /* Now sum over processors */
  MPI_Allreduce(&locpower, &totpower, 1, MPI_DOUBLE, MPI_SUM,
		MPI_COMM_WORLD);

  /* Normalize my part of the field */
  initdev=sqrt(totpower/((double)ngrid*ngrid*ngrid));
  scale=stddev/initdev;
  for (l=0; l<3; l++) {
    for (i=0; i<nxloc; i++) {
      for (j=0; j<ngrid; j++) {
	for (k=0; k<ngrid; k++) {
	  idx=l*locsize + ((ptrdiff_t)i*ngrid+j)*2*ngridft+k;
	  grid[idx]*=scale;
	}
      }
    }
  }
}


void err_exit(char *err)
{
  if (myrank==0) fprintf(stderr, "%s", err);
  MPI_Finalize();
  exit(1);
}

void fftinit(fftw_plan *plan, double **grid, double **work)
{
  alloc_local = fftw_mpi_local_size_3d(ngrid, ngrid, ngridft,
					MPI_COMM_WORLD,
					&nxloc, &xlocstart);
  locsize = 2*alloc_local;
  *grid = fftw_alloc_real(3*locsize);
  *work = NULL; /* FFTW3-MPI handles its own transpose scratch space */

  /* Single plan, reused for all three field components via
     new-array execute (fftw_mpi_execute_dft_c2r) below -- valid
     because each component's sub-array starts at an offset that is
     a whole number of fftw_alloc'd doubles from the base pointer,
     so all three retain the alignment FFTW assumed when planning. */
  *plan = fftw_mpi_plan_dft_c2r_3d(ngrid, ngrid, ngrid,
				    (fftw_complex *) (*grid), *grid,
				    MPI_COMM_WORLD, FFTW_ESTIMATE);
}


void writeData(char *outname, double *grid, int n)
{
  FILE *fp;
  char outname1[256];
  int xptr, owner, nxmax;
  int *slabsize, thisslab, zero=0;
  int i, j, k;
  ptrdiff_t idx;
  static double *gridtmp=NULL;
  MPI_Status status;
  int dummy;
  int nxloc_i = (int) nxloc, xlocstart_i = (int) xlocstart;
  MPI_Datatype rowtype;

  /* One row (ngrid*2*ngridft doubles) as a single MPI datatype, so the
     Send/Recv count below is a row count (<=ngrid) rather than a raw
     double count -- MPI_Send/Recv's count parameter is a 32-bit int,
     and nxloc*ngrid*2*ngridft raw doubles can exceed INT_MAX at large
     ngrid even though the actual per-rank byte count is fine. */
  MPI_Type_contiguous(ngrid*2*ngridft, MPI_DOUBLE, &rowtype);
  MPI_Type_commit(&rowtype);

  /* Proc 0 opens output file */
  if (myrank==0) {
    sprintf(outname1, "%s.%d", outname, n+1);
    fp=fopen(outname1, "w");
  }

  /* Figure out how much memory we need to store the largest block */
  MPI_Reduce(&nxloc_i, &nxmax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  if (myrank==0)
    if (gridtmp==NULL)
      gridtmp=(double*) calloc((size_t)nxmax*ngrid*2*ngridft, sizeof(double));

  /* Loop over decomposed direction */
  xptr=0;
  slabsize=calloc(nproc, sizeof(int));
  while (xptr<ngrid) {

    /* Get slab size from processor that has it */
    if (myrank==0) {
      if (xlocstart_i==xptr) slabsize[0]=nxloc_i;
      else slabsize[0]=0;
      for (n=1; n<nproc; n++) {
	MPI_Recv(slabsize+n, 1, MPI_INT, n, n, MPI_COMM_WORLD, &status);
      }
    } else {
      if (xlocstart_i==xptr)
	MPI_Send(&nxloc_i, 1, MPI_INT, 0, myrank, MPI_COMM_WORLD);
      else
	MPI_Send(&zero, 1, MPI_INT, 0, myrank, MPI_COMM_WORLD);
    }

    /* Now root figures out who owns this slab */
    if (myrank==0) {
      for (n=0; n<nproc; n++) {
	if (slabsize[n]!=0) {
	  owner=n;
	  thisslab=slabsize[owner];
	  break;
	}
      }

      if (owner==0) {

	/* If root already owns this data, just dump it to the file */
	for (i=0; i<thisslab; i++) {
	  for (j=0; j<ngrid; j++) {
	    idx=((ptrdiff_t)i*ngrid+j)*2*ngridft;
#ifndef BINARY
	    for (k=0; k<ngrid; k++) {
	      fprintf(fp, "%e\n", grid[idx+k]);
	    }
#else
	    fwrite(grid+idx, sizeof(double), ngrid, fp);
#endif
	  }
	}
      } else {

	/* Root doesn't own this data, so prepare to receive it from
	   the processor that does */
	MPI_Recv(gridtmp, thisslab, rowtype, owner,
		 1, MPI_COMM_WORLD, &status);
	/* Now write it out */
	for (i=0; i<thisslab; i++) {
	  for (j=0; j<ngrid; j++) {
	    idx=((ptrdiff_t)i*ngrid+j)*2*ngridft;
#ifndef BINARY
	    for (k=0; k<ngrid; k++) {
	      fprintf(fp, "%e\n", gridtmp[idx+k]);
	    }
#else
	    fwrite(gridtmp+idx, sizeof(double), ngrid, fp);
#endif
	  }
	}
      }

    } else if (xlocstart_i==xptr) {

      /* I'm not the root processor, but I own this data, so send it
	 to root */
      MPI_Send(grid, nxloc_i, rowtype, 0, 1,
	       MPI_COMM_WORLD);

    }

    /* Now root sends the size of the block just written to everyone,
       so that they know where to start looking for the start of the
       next block */
    MPI_Bcast(&thisslab, 1, MPI_INT, 0, MPI_COMM_WORLD);
    xptr+=thisslab;
  }

  /* Now all the data is written, so proc 0 closes the file */
  if (myrank==0) fclose(fp);
  free(slabsize);
  MPI_Type_free(&rowtype);
}

int main(int argc, char **argv)
{
  float kmin, kmax, kidx, stddev;
  char *outname;
  fftw_plan plan;
  double *grid, *work;
  int n;

  /* Start parallel */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  fftw_mpi_init();

  /* Parse inputs */
  if (argc!=7)
    err_exit("Usage: fieldgen outputname ngrid kmin kmax kidx stddev\n");
  outname=argv[1];
  ngrid=atoi(argv[2]);
  kmin=atof(argv[3]);
  kmax=atof(argv[4]);
  kidx=atof(argv[5]);
  stddev=atof(argv[6]);
  ngridft=ngrid/2+1;

  /* Set up random seed */
  readSeedPar();

  /* Initialize the plan and allocate necessary memory */
  fftinit(&plan, &grid, &work);

  /* Loop over components */
  for (n=0; n<3; n++) {

    /* Generate field in Fourier space */
#ifdef VERBOSE
    if (myrank==0) printf("Generating field %d...\n", n);
#endif
    gridftGen((cplx_t *) (grid+n*locsize),
	      kmin, kmax, kidx);
  }

#ifdef SOLENOIDAL
  /* Project out non-solenoidal component if requested */
#  ifdef VERBOSE
  if (myrank==0) printf("Projecting out non-solenoidal components...\n");
#  endif
  gridftProject((cplx_t *) grid);
#endif

  for (n=0; n<3; n++) {
    /* Transform to physical space */
#ifdef VERBOSE
    if (myrank==0) printf("Doing fft..\n");
#endif
    fftw_mpi_execute_dft_c2r(plan, (fftw_complex *) (grid+n*locsize),
			      grid+n*locsize);
  }

#if 1
  /* Normalize total power */
  if (stddev>0.0) {
#ifdef VERBOSE
    if (myrank==0)
      printf("Normalizing to desired standard deviation...\n");
#endif
      normalize(grid, stddev);
  }
#endif

    /* Dump to output file */
  for (n=0; n<3; n++) {
#ifdef VERBOSE
    if (myrank==0) printf("Writing output...\n");
#endif
    writeData(outname, grid+n*locsize, n);
  }

  /* Save seed and end */
  writeSeedPar();
  fftw_destroy_plan(plan);
  fftw_mpi_cleanup();
  MPI_Finalize();
}
