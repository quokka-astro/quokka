/**************************************************
 *** Generate turbulent field with TurbGen.h    ***
 *** Written by Christoph Federrath, 2022-2025. ***
***************************************************/

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "TurbGen.h"

// normally set via compiler defines: #define HAVE_HDF5
#ifdef HAVE_HDF5
#include "HDFIO.h"
#endif

// normally set via compiler defines: #define HAVE_MPI
#ifdef HAVE_MPI
#include "mpi.h"
#define MPI_COMM MPI_COMM_WORLD
#else
#ifndef MPI_COMM_NULL
#define MPI_COMM_NULL 0
#endif
#define MPI_COMM MPI_COMM_NULL
#endif

using namespace std;

// global variables
enum {X, Y, Z};
static const string ProgSign = "TurbGen.cpp: ";
int verbose = 1; // 0: all standard output off (quiet mode)
double ndim = 3; // dimensionality (must be 1 or 1.5 or 2 or 2.5 or 3)
int N[3] = {64, 64, 64}; // number of cells in turbulent output field in x, y, z
double L[3] = {1.0, 1.0, 1.0}; // size of box in x, y, z
double k_min = 2.0;  // minimum wavenumber for turbulent field (in units of 2pi / L[X])
double k_max = 20.0; // minimum wavenumber for turbulent field (in units of 2pi / L[X])
double k_mid = 1e38; // middle  wavenumber in case of optional 2nd PL section in [k_mid, k_max]
int spect_form = 2; // 0: band/rectangle/constant, 1: paraboloid, 2: power law
double power_law_exp = -2.0; // if spect_form == 2: power-law exponent (e.g., -2 would be Burgers,
                             // or -5/3 would be Kolmogorov, or 1.5 would Kazantsev)
double power_law_exp_2 = -2.0; // exponent for optional 2nd PL section in [k_mid, k_max]
double angles_exp = 1.0; // if spect_form == 2: spectral sampling of angles;
                         // number of modes (angles) in k-shell surface increases as k^angles_exp.
                         // For full sampling, angles_exp = 2.0; for healpix-type sampling, angles_exp = 0.0.
double sol_weight = 0.5; // solenoidal weight: 1.0: solenoidal driving, 0.0: compressive driving, 0.5: natural mixture
int random_seed = 140281; // random seed for this turbulent realisation
string outfilename = "TurbGen_output.h5"; // HDF5 output filename
bool write_modes = false; // switch to write Fourier modes and amplitudes to output file

// MPI stuff
int MyPE = 0, NPE = 1;

// forward functions
int ParseInputs(const vector<string> Argument);
void HelpMe(void);


int main(int argc, char * argv[])
{
    /// initialise MPI
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NPE);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
#endif

    /// Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) == -1)
    {
        if (MyPE==0 && verbose>0) cout<<endl<<ProgSign+"Error in ParseInputs(). Exiting."<<endl;
        HelpMe();
#ifdef HAVE_MPI
        MPI_Finalize();
#endif
        return -1;
    }

    if (MyPE==0 && verbose>1) cout<<ProgSign+"started..."<<endl;

#ifdef HAVE_MPI
    // stop on error in domain decomposition if we run with MPI and number cores is > number of grid cells in X
    if (NPE > N[X]) {
        if (MyPE==0 && verbose>0) cout<<ProgSign+"Error in domain decomposition: NPE must be <= N[X]"<<endl;
        MPI_Finalize();
        return -1;
    }
#endif

    long starttime = time(NULL);
    cout<<setprecision(9);

    // create TurbGen class object
    TurbGen tg = TurbGen(MyPE);
    tg.set_verbose(verbose);

    // initialise generator to return a single turbulent realisation based on input parameters
    tg.init_single_realisation(ndim, L, k_min, k_mid, k_max, spect_form, power_law_exp, power_law_exp_2, angles_exp, sol_weight, random_seed);

    // get the number of vector field components
    int ncmp = tg.get_number_of_components();

    // set dimensionality dependencies
    if ((int)ndim < 3) { N[Z] = 1; L[Z] = 1.0; } // 2D
    if ((int)ndim < 2) { N[Y] = 1; L[Y] = 1.0; } // 1D

    // define cell size of uniform grid
    double del[3];
    for (int d = 0; d < 3; d++) del[d] = L[d] / N[d];

    // always generate within [0,L], cell-centered; user can shift output to target physical location, if needed
    double pos_beg[3] = {0.0, 0.0, 0.0}, pos_end[3] = {0.0, 0.0, 0.0}; // start and end coordinates of output grid
    // parallelise along X (generalises to non-MPI case, where NPE=1 and MyPE=0)
    int divN_PE = ceil((double)N[X]/(double)NPE);
    int NPE_main = N[X] / divN_PE;
    int modN_PE = N[X] - NPE_main * divN_PE;
    double divL_PE = L[X] * (double)divN_PE / (double)N[X];
    double modL_PE = L[X] * (double)modN_PE / (double)N[X];
    int NPE_in_use = NPE_main; if (modN_PE > 0) NPE_in_use += 1;
    int NX = 0; // number of cells in X for MyPE (idle PEs get nothing, i.e., NX=0)
    if (MyPE==0 && verbose>1)
        cout<<ProgSign+"NPE_main, divN_PE, modN_PE, divL_PE, modL_PE = "<<NPE_main<<" "<<divN_PE<<" "<<modN_PE<<" "<<divL_PE<<" "<<modL_PE<<endl;
    // set pos_beg and pos_end
    for (int d = 0; d < 3; d++) {
        if (d==X) { // parallelised direction - x
            if (MyPE < NPE_main) { // equally distribute to NPE_main cores
                pos_beg[d] = MyPE*divL_PE + del[d]/2.0; // first cell coordinate (cell center)
                pos_end[d] = (MyPE+1)*divL_PE - del[d]/2.0; // last cell coordinate (cell center)
                NX = divN_PE;
            } else if (MyPE < NPE_in_use) { // last PE gets the rest (idle PEs get nothing, i.e., NX=0)
                pos_beg[d] = NPE_main*divL_PE + del[d]/2.0; // first cell coordinate (cell center)
                pos_end[d] = L[X] - del[d]/2.0; // last cell coordinate (cell center)
                NX = modN_PE;
            }
            if (verbose>1)
                cout<<ProgSign+"MyPE, pos_beg[X], pos_beg[X], NX = "<<MyPE<<" "<<pos_beg[X]<<" "<<pos_end[X]<<" "<<NX<<endl;
        } else { // non-parallelised direction(s)
            pos_beg[d] = del[d]/2.0; // first cell coordinate (cell center)
            pos_end[d] = L[d] - del[d]/2.0; // last cell coordinate (cell center)
        }
    }

#ifdef HAVE_MPI
    if (MyPE==0 && verbose>0) {
        cout<<ProgSign+"First "<<NPE_main<<" core(s) carry(ies) NX="<<divN_PE<<" cell(s) (each)."<<endl;
        if (modN_PE > 0) cout<<ProgSign+"Core #"<<NPE_main+1<<" carries NX="<<modN_PE<<" cell(s)."<<endl;
        if (NPE_in_use < NPE) cout<<ProgSign+"Warning: non-optimal load balancing; "<<NPE-NPE_in_use<<" core(s) remain(s) idle."<<endl;
    }
#endif

    // total number of output grid cells
    int N_out[3] = {NX, N[Y], N[Z]};
    long ntot = 1; for (int d = 0; d < 3; d++) ntot *= N_out[d];
    // output grid, which receives the turbulent field (up to ndim = 3)
    float * grid_out[3];
    // allocate
    for (int d = 0; d < ncmp; d++) grid_out[d] = new float[ntot];

    // call to return uniform grid(s) with ncmp components of the turbulent field at requested positions
    tg.get_turb_vector_unigrid(pos_beg, pos_end, N_out, grid_out);

    // compute mean and std of generated turbulent field and then re-normalise to mean=0 and std=1
    double mean [3] = {0.0, 0.0, 0.0};
    double mean2[3] = {0.0, 0.0, 0.0};
    double std[3] = {0.0, 0.0, 0.0};
    for (int d = 0; d < ncmp; d++) {
        for (long ni = 0; ni < ntot; ni++) {
            mean [d] += grid_out[d][ni];
            mean2[d] += pow(grid_out[d][ni],2.0);
        }
    }
#ifdef HAVE_MPI
    MPI_Allreduce(MPI_IN_PLACE, mean , 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, mean2, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    for (int d = 0; d < ncmp; d++) {
        mean [d] /= N[X]*N[Y]*N[Z]; // mean
        mean2[d] /= N[X]*N[Y]*N[Z]; // mean squared
        std[d] = sqrt(mean2[d] - mean[d]*mean[d]); // standard deviation
        for (long ni = 0; ni < ntot; ni++) {
            grid_out[d][ni] -= mean[d];
            grid_out[d][ni] /= std[d];
        }
    }

    // re-compute mean and std
    for (int d = 0; d < ncmp; d++) {
        mean[d] = 0.0; mean2[d] = 0.0;
        for (long ni = 0; ni < ntot; ni++) {
            mean [d] += grid_out[d][ni];
            mean2[d] += pow(grid_out[d][ni],2.0);
        }
    }
#ifdef HAVE_MPI
    MPI_Allreduce(MPI_IN_PLACE, mean , 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, mean2, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    for (int d = 0; d < ncmp; d++) {
        mean [d] /= N[X]*N[Y]*N[Z]; // mean
        mean2[d] /= N[X]*N[Y]*N[Z]; // mean squared
        std[d] = sqrt(mean2[d] - mean[d]*mean[d]); // standard deviation
    }

    if (MyPE==0 && verbose>0) {
        cout<<ProgSign+"Generated "<<ndim<<"D turbulent field with"<<endl;
        cout<<ProgSign+" number of grid cells N ="; for (int d = 0; d < (int)ndim; d++) cout<<" "<<N[d]; cout<<endl;
        cout<<ProgSign+" physical size L ="; for (int d = 0; d < (int)ndim; d++) cout<<" "<<L[d]; cout<<endl;
        cout<<ProgSign+" mean (for each component) ="; for (int d = 0; d < ncmp; d++) cout<<" "<<mean[d]; cout<<endl;
        cout<<ProgSign+" 1D standard deviation (for each component) ="; for (int d = 0; d < ncmp; d++) cout<<" "<<std[d]; cout<<endl;
        string expected = "";
        if (ncmp==1) expected = "1";
        if (ncmp==2) expected = "sqrt(2)";
        if (ncmp==3) expected = "sqrt(3)";
        cout<<ProgSign+" total ("<<ndim<<"D) standard deviation (expected: "<<expected<<") = "<<sqrt(std[0]*std[0]+std[1]*std[1]+std[2]*std[2])<<endl;
    }

#ifdef HAVE_HDF5
    // write to HDF5 file
    if (MyPE==0 && verbose>0) {
        cout<<"-----------------------------------------------------"<<endl;
        cout<<ProgSign+"Creating '"<<outfilename<<"' for output..."<<endl;
    }
    HDFIO hdfio = HDFIO();
    hdfio.create(outfilename, MPI_COMM);
    // write scalars
    vector<int> hdf5dims(0);
    hdfio.write(&ndim, "ndim", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
    hdfio.write(&ncmp, "ncmp", hdf5dims, H5T_NATIVE_INT, MPI_COMM);
    hdfio.write(&k_min, "kmin", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
    hdfio.write(&k_mid, "kmid", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
    hdfio.write(&k_max, "kmax", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
    hdfio.write(&spect_form, "spect_form", hdf5dims, H5T_NATIVE_INT, MPI_COMM);
    if (spect_form == 2) {
        hdfio.write(&power_law_exp,   "power_law_exp",   hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
        hdfio.write(&power_law_exp_2, "power_law_exp_2", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
        hdfio.write(&angles_exp, "angles_exp", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
    }
    hdfio.write(&sol_weight, "sol_weight", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
    hdfio.write(&random_seed, "random_seed", hdf5dims, H5T_NATIVE_INT, MPI_COMM);
    // write N and L vectors
    hdf5dims.resize(1); hdf5dims[0] = (int)ndim;
    int No[(int)ndim]; double Lo[(int)ndim]; // order Z,Y,X
     for (int d = 0; d < (int)ndim; d++) {
        No[(int)ndim-1-d] = N[d];
        Lo[(int)ndim-1-d] = L[d];
    }
    hdfio.write(No, "N", hdf5dims, H5T_NATIVE_INT, MPI_COMM);
    hdfio.write(Lo, "L", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
    // write turbulent field (components)
    hdf5dims.resize((int)ndim); for (int d = 0; d < (int)ndim; d++) hdf5dims[(int)ndim-1-d] = N[d]; // order Z,Y,X
    if (MyPE==0 && verbose>1) { cout<<ProgSign+"hdf5dims ="; for (int d = 0; d < (int)ndim; d++) cout<<" "<<hdf5dims[d]; cout<<endl; }
    for (int dc = 0; dc < ncmp; dc++) { // loop over component(s)
        string dsetname = "turb_field";
        if (dc == 0) dsetname += "_x";
        if (dc == 1) dsetname += "_y";
        if (dc == 2) dsetname += "_z";
        hdfio.create_dataset(dsetname, hdf5dims, H5T_NATIVE_FLOAT, MPI_COMM); // create HDF5 dataset
        // specify dimensions and offset for slab operation
        hsize_t offset[(int)ndim], count[(int)ndim], out_offset[(int)ndim], out_count[(int)ndim];
        for (int d = 0; d < (int)ndim; d++) {
            int dd = (int)ndim-1-d; // order Z,Y,X
            // collective HDF5 IO only works if inactive cores participate, but with counts=0 and offsets=0
            offset[dd] = 0; count[dd] = 0; out_offset[dd] = 0; out_count[dd] = 0;
            if (MyPE < NPE_in_use) {
                if (d==0) offset[dd] = MyPE*divN_PE;
                count[dd] = N_out[d];
                out_offset[dd] = 0;
                out_count[dd] = N_out[d];
            }
        }
        if (verbose>1) {
            for (int d = 0; d < (int)ndim; d++)
                cout<<"MyPE, d, offset, count, out_offset, out_count = "<<
                        MyPE<<" "<<d<<" "<<offset[d]<<" "<<count[d]<<" "<<out_offset[d]<<" "<<out_count[d]<<endl;
        }
        // write slab to file (in parallel, if we have MPI)
        hdfio.overwrite_slab(grid_out[dc], dsetname, H5T_NATIVE_FLOAT, offset, count, (int)ndim, out_offset, out_count, MPI_COMM);
        if (MyPE==0 && verbose>0) cout<<ProgSign+"Dataset '"<<dsetname<<"' in '"<<outfilename<<"' written."<<endl;
    }
    // output generating modes and their amplitudes
    if (write_modes) {
        // modes
        vector< vector<double> > modes = tg.get_modes();
        int nmodes = modes[0].size(); // all dims have the same number of modes
        hdf5dims.resize(2); hdf5dims[0] = (int)ndim; hdf5dims[1] = nmodes;
        double * ptmp = new double[((int)ndim)*nmodes];
        for (int d = 0; d < (int)ndim; d++)
            for (int m = 0; m < nmodes; m++)
                ptmp[d*nmodes+m] = modes[d][m];
        hdfio.write(ptmp, "Fourier_modes", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
        delete [] ptmp;
        // amplitudes
        vector<double> amplitudes = tg.get_amplitudes();
        ptmp = new double[amplitudes.size()];
        for (int i = 0; i < amplitudes.size(); i++) ptmp[i] = amplitudes[i];
        hdf5dims.resize(1); hdf5dims[0] = amplitudes.size();
        hdfio.write(ptmp, "Fourier_amplitudes", hdf5dims, H5T_NATIVE_DOUBLE, MPI_COMM);
        delete [] ptmp;
    }
    hdfio.close();
    if (MyPE==0 && verbose>0) cout<<ProgSign+"Finished writing '"<<outfilename<<"'."<<endl;
#endif

    // clean up
    for (int d = 0; d < ncmp; d++) {
      if (grid_out[d]) delete [] grid_out[d]; grid_out[d] = NULL;
    }

    /// print out wallclock time used
    long endtime = time(NULL);
    long duration = endtime-starttime;
    if (verbose>1) cout<<ProgSign+"["<<MyPE<<"] Local runtime: "<<duration<<"s"<<endl;
#ifdef HAVE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (MyPE==0 && verbose>0) {
        cout<<"-----------------------------------------------------"<<endl;
        cout<<ProgSign+"Total runtime: "<<duration<<"s"<<endl;
    }

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}


/** ------------------------- ParseInputs ----------------------------
 **  Parses the command line Arguments
 ** ------------------------------------------------------------------ */
int ParseInputs(const vector<string> Argument)
{
    static const string FuncSign = ProgSign+"ParseInputs: ";
    stringstream dummystream;

    /// read tool specific options
    if (Argument.size() < 2)
    {
        if (MyPE==0) cout << endl << FuncSign+"Specify at least 1 argument." << endl;
        return -1;
    }

    for (unsigned int i = 1; i < Argument.size(); i++)
    {
        if (Argument[i] != "" && Argument[i] == "-h")
        {
            HelpMe(); exit(0);
        }
        if (Argument[i] != "" && Argument[i] == "-verbose")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> verbose; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-ndim")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> ndim; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-N")
        {
            if (Argument.size()>i+(int)ndim) {
                dummystream << Argument[i+1]; dummystream >> N[X]; dummystream.clear();
                if ((int)ndim > 1) { dummystream << Argument[i+2]; dummystream >> N[Y]; dummystream.clear(); }
                if ((int)ndim > 2) { dummystream << Argument[i+3]; dummystream >> N[Z]; dummystream.clear(); }
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-L")
        {
            if (Argument.size()>i+(int)ndim) {
                dummystream << Argument[i+1]; dummystream >> L[X]; dummystream.clear();
                if ((int)ndim > 1) { dummystream << Argument[i+2]; dummystream >> L[Y]; dummystream.clear(); }
                if ((int)ndim > 2) { dummystream << Argument[i+3]; dummystream >> L[Z]; dummystream.clear(); }
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-kmin")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> k_min; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-kmid")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> k_mid; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-kmax")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> k_max; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-spect_form")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> spect_form; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-power_law_exp")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> power_law_exp; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-power_law_exp_2")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> power_law_exp_2; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-angles_exp")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> angles_exp; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-sol_weight")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> sol_weight; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-random_seed")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> random_seed; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-o")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> outfilename; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-write_modes")
        {
            write_modes = true;
        }
    } // loop over all args

    /// print out parsed values
    if (MyPE==0 && verbose>1) {
        cout << FuncSign+"Command line arguments: ";
        for (unsigned int i = 0; i < Argument.size(); i++) cout << Argument[i] << " ";
        cout << endl;
    }
    return 0;

} // end: ParseInputs()


/** --------------------------- HelpMe -------------------------------
 **  Prints out usage information
 ** ------------------------------------------------------------------ */
void HelpMe(void)
{
    if (MyPE==0 && verbose>0) {
        cout << endl
        << ProgSign+"Generates a turbulent field of mean=0 and std=1, with specified parameters (see OPTIONS)." << endl << endl
        << "Syntax:" << endl
        << " TurbGen [<OPTIONS>]" << endl << endl
        << "   <OPTIONS>:" << endl
        << "     -ndim <1, 1.5, 2, 2.5, 3> : number of spatial dimensions; (default: 3)" << endl
        << "     -N <Nx [Ny [Nz]]>         : number of grid cells for generated turbulent field; (default: 64 64 64)" << endl
        << "     -L <Lx [Ly [Lz]]>         : physical size of box in which to generate turbulent field; (default: 1.0 1.0 1.0)" << endl
        << "     -kmin <val>               : minimum wavenumber of generated field in units of 2pi/L[X]; (default: 2.0)" << endl
        << "     -kmid <val>               : middle  wavenumber for optional 2nd power-law section (only for spect_form=2); (default: 1e38; i.e., not active)" << endl
        << "     -kmax <val>               : maximum wavenumber of generated field in units of 2pi/L[X]; (default: 20.0)" << endl
        << "     -spect_form <0, 1, 2>     : spectral form: 0 (band/rectangle/constant), 1 (paraboloid), 2 (power law); (default: 2)" << endl
        << "     -power_law_exp <val>      : if spect_form 2: power-law exponent of power-law spectrum (e.g. Kolmogorov: -5/3, Burgers: -2, Kazantsev: 1.5); (default: -2.0)" << endl
        << "     -power_law_exp_2 <val>    : if spect_form 2: power-law exponent of 2nd power-law section (default: -2.0)" << endl
        << "     -angles_exp <val>         : if spect_form 2: angles exponent for sparse sampling (e.g., 2.0: full sampling, 0.0: healpix-like sampling); (default: 1.0)" << endl
        << "     -sol_weight <val>         : solenoidal weight: 1.0 (divergence-free field), 0.5 (natural mix), 0.0 (curl-free field); (default: 0.5)" << endl
        << "     -random_seed <val>        : random seed for turbulent field; (default: 140281)" << endl
        << "     -verbose <0, 1, 2>        : 0 (no shell output), 1 (standard shell output), 2 (more shell output); (default: 1)" << endl
        << "     -o <filename>             : output filename (for HDF5 output); (default: TurbGen_output.h5)" << endl
        << "     -write_modes              : write generating Fourier modes and amplitudes to output file" << endl
        << "     -h                        : print this help message" << endl
        << endl
        << "Example: TurbGen -ndim 2 -L 1.0 1.0"
        << endl << endl;
    }
}
