// *******************************************************************************
// *************************** Turbulence generator ******************************
// *******************************************************************************
//
// This header file contains functions and data structures used by the turbulence
// generator. The main input (parameter) file is 'turbulence_generator.inp'.
//
// Please see Federrath et al. (2010, A&A 512, A81) for details and cite :)
//
// DESCRIPTION
//
//  Contains functions to compute the time-dependent physical turbulent vector
//  field used to drive or initialise turbulence in hydro codes such as AREPO,
//  FLASH, GADGET, PHANTOM, PLUTO, QUOKKA, etc.
//  The driving sequence follows an Ornstein-Uhlenbeck (OU) process.
//
//  For example applications see Federrath et al. (2008, ApJ 688, L79);
//  Federrath et al. (2010, A&A 512, A81); Federrath (2013, MNRAS 436, 1245);
//  Federrath et al. (2021, Nature Astronomy 5, 365)
//
// AUTHOR: Christoph Federrath, 2008-2025
//
// *******************************************************************************

#ifndef TURBULENCE_GENERATOR_H
#define TURBULENCE_GENERATOR_H

#include <iostream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cfloat>
#include <cmath>

namespace NameSpaceTurbGen {
    // constants
    static const int tgd_max_nmodes = 100000;
}

/*********************************************************************************
 *
 * TurbGen class
 *   Generates turbulent vector fields for setting turbulent initial conditions,
 *   or continuous driving of turbulence based on an Ornstein-Uhlenbeck process.
 *
 * @author Christoph Federrath (christoph.federrath@anu.edu.au)
 * @version 2023
 *
 *********************************************************************************/

class TurbGen
{
    protected:
        enum {X, Y, Z};
        int verbose; // shell output level (0: no output, 1: standard output, 2: more output)
        std::string ClassSignature; // class signature
        std::string parameter_file; // parameter file for controlling turbulence driving
        std::vector<double> mode[3], aka[3], akb[3], OUphases, ampl; // modes arrays, phases, amplitudes
        int PE; // MPI task for printf purposes, if provided
        int nmodes; // number of modes
        double ndim; // number of spatial dimensions
        int ncmp; // number of components (1, 2, 3, depending on whether we create vx, vy, vz)
        int random_seed, seed; // 'random seed' is the original starting seed, then 'seed' gets updated by call to random number generator
        int spect_form; // spectral form (Band, Parabola, Power Law)
        int nsteps_per_t_turb; // number of driving patterns per turnover time
        int step; // internal OU step number
        double L[3]; // domain physical length L[dim] with dim = X, Y, Z
        double velocity; // velocity dispersion
        double t_decay; // turbulent turnover time (auto-correlation timescale)
        double dt; // time step for OU update and for generating driving patterns
        double energy; // driving energy injection rate (~ v^3 / L)
        double OUvar; // OU variance corresponding to turbulent turnover (decay) time and energy input rate (OUvar = sqrt(energy/t_decay))
        double kmin, kmax; // min and max wavenumber for driving or single realisation
        double kmid; // kmid is optional and in between kmin and kmax, which can be used for an optional 2nd PL section from kmid to kmax
        double sol_weight, sol_weight_norm; // weight for decomposition into solenoidal and compressive modes
        double power_law_exp, power_law_exp_2; // for power-law spectrum (spect_form = 2): exponent (power_law_exp_2 is for an optional 2nd PL section from kmid to kmax)
        double angles_exp; // for power-law spectrum (spect_form = 2): angles exponent for sparse sampling
        double ampl_factor[3]; // scale amplitude by this factor (default: 1.0, 1.0, 1.0)
        int ampl_auto_adjust; // switch (0,1) to turn off/on automatic amplitude adjustment
        std::string evolfile;

    /// Constructors
    public: TurbGen(void)
    {
        Constructor(0); // default non-MPI program constructor
    };
    public: TurbGen(const int PE) // MPI-programs should supply MPI rank in PE
    {
        Constructor(PE);
    };
    /// Destructor
    public: ~TurbGen()
    {
      if (verbose > 1) std::cout<<ClassSignature<<"destructor called."<<std::endl;
    };
    // General constructur
    private: void Constructor(const int PE)
    {
        ClassSignature = "TurbGen: ";
        this->PE = PE;
        verbose = 1; // default verbose level
        evolfile = "TurbGen.dat";
    };

    // get function signature for printing to stdout
    protected: std::string FuncSig(const std::string func_name)
    { return func_name+": "; };

    // ******************************************************
    // set functions
    // ******************************************************
    public: void set_verbose(const int verbose) {
        this->verbose = verbose;
    };
    // ******************************************************
    // get functions
    // ******************************************************
    public: double get_turnover_time(void) {
        return t_decay;
    };
    public: int get_nsteps_per_turnover_time(void) {
        return nsteps_per_t_turb;
    };
    // ******************************************************
    public: int get_number_of_components(void) {
        return ncmp;
    };
    // ******************************************************
    public: std::vector< std::vector<double> > get_modes(void) {
        std::vector< std::vector<double> > ret;
        ret.resize((int)ndim);
        for (int d = 0; d < (int)ndim; d++) ret[d] = mode[d];
        return ret;
    };
    // ******************************************************
    public: std::vector<double> get_amplitudes(void) {
        return ampl;
    };

    // ******************************************************
    private: void set_number_of_components(void) {
        // error check on user setting of ndim
        if ((ndim != 1) && (ndim != 1.5) && (ndim != 2) && (ndim != 2.5) && (ndim != 3)) {
            TurbGen_printf("ERROR: number of dimensions must be in [1, 1.5, 2, 2.5, 3].\n");
            exit(-1);
        }
        // set the number of vector field components
        if (ndim == 1) ncmp = 1;
        if (ndim == 2) ncmp = 2;
        if ((ndim == 1.5) || (ndim == 2.5) || (ndim == 3)) ncmp = 3;
    };

    // ******************************************************
    private: void set_solenoidal_weight_normalisation(void) {
        // this makes the rms of the turbulent field independent of the solenoidal weight (see Eq. 9 in Federrath et al. 2010)
        sol_weight_norm = sqrt(3.0/ncmp)*sqrt(3.0)*1.0/sqrt(1.0-2.0*sol_weight+ncmp*pow(sol_weight,2.0));
    };

    // ******************************************************
    public: int init_single_realisation(
        const double ndim, const double L[3], const double k_min, const double k_max,
        const int spect_form, const double power_law_exp, const double angles_exp,
        const double sol_weight, const int random_seed) {
        return init_single_realisation( ndim, L, k_min, k_max, k_max,
                                        spect_form, power_law_exp, power_law_exp, angles_exp,
                                        sol_weight, random_seed );
    }; // init_single_realisation (overloaded)

    // ******************************************************
    public: int init_single_realisation(
        const double ndim, const double L[3], const double k_min, const double k_mid, const double k_max,
        const int spect_form, const double power_law_exp, const double power_law_exp_2, const double angles_exp,
        const double sol_weight, const int random_seed) {
        // ******************************************************
        // Initialise the turbulence generator for a single turbulent realisation, e.g., for producing
        // turbulent initial conditions, with parameters specified as inputs to the function; see descriptions below.
        // ******************************************************
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");
        // set internal parameters
        this->ndim = ndim;
        this->L[X] = L[X]; // Length of box in x; used for wavenumber conversion below
        this->L[Y] = L[Y]; // Length of box in y
        this->L[Z] = L[Z]; // Length of box in z
        this->kmin = (k_min-DBL_EPSILON) * 2*M_PI / L[X]; // Minimum wavenumber <~  k_min * 2pi / Lx
        this->kmax = (k_max+DBL_EPSILON) * 2*M_PI / L[X]; // Maximum wavenumber >~  k_max * 2pi / Lx
        this->kmid = (k_mid+DBL_EPSILON) * 2*M_PI / L[X]; // Middle  wavenumber >~  k_mid * 2pi / Lx (only if spect_form = 2, i.e., power law; for 2nd PL section in [kmid, kmax])
        this->spect_form = spect_form; // (0: band/rectangle/constant, 1: paraboloid, 2: power law)
        this->power_law_exp = power_law_exp; // power-law amplitude exponent (only if spect_form = 2, i.e., power law)
        this->power_law_exp_2 = power_law_exp_2; // power-law amplitude exponent 2 (only if spect_form = 2, i.e., power law; for 2nd PL section in [kmid, kmax])
        this->angles_exp = angles_exp; // angles exponent (only if spect_form = 2, i.e., power law)
        this->sol_weight = sol_weight; // solenoidal weight (0: compressive, 0.5: natural mix, 1.0: solenoidal)
        this->random_seed = random_seed; // set random seed for this realisation
        seed = this->random_seed; // set random seed for this realisation
        OUvar = 1.0; // Ornstein-Uhlenbeck variance
        // the returned turbulent field needs to be re-normalised outside this call, so we just set these to 1
        for (unsigned int d = 0; d < 3; d++) ampl_factor[d] = 1.0; // amplitude factor
        if (verbose) TurbGen_printf("===============================================================================\n");
        // set derived parameters
        set_number_of_components();
        set_solenoidal_weight_normalisation();
        // initialise modes
        init_modes();
        // initialise random phases
        OU_noise_init();
        // calculate solenoidal and compressive coefficients (aka, akb) from OUphases
        get_decomposition_coeffs();
        // print info
        if (verbose) print_info("single_realisation");
        if (verbose) TurbGen_printf("===============================================================================\n");
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
        return 0;
    }; // init_single_realisation

    // ******************************************************
    public: virtual int init_driving(std::string parameter_file) {
        return init_driving(parameter_file, 0.0); // call with time = 0.0
    }; // init_driving (overloaded)

    // ******************************************************
    public: virtual int init_driving(const std::map<std::string, std::string> &params) {
        return init_driving(params, 0.0); // call with time = 0.0
    }; // init_driving (overloaded)

    // ******************************************************
    public: virtual int init_driving(std::string parameter_file, const double time) {
        // ******************************************************
        // Initialise the turbulence generator and all relevant internal data structures by reading from 'parameter_file'.
        // This is used for driving turbulence (as opposed to init_single_realisation, which is for creating a single pattern).
        // ******************************************************
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");
        // set parameter file
        this->parameter_file = parameter_file;
        // check if parameter file is present
        FILE * fp = fopen(parameter_file.c_str(), "r");
        if (fp == NULL) {
            std::string msg = ClassSignature+"ERROR: cannot access parameter file '%s'.\n";
            printf(msg.c_str(), parameter_file.c_str()); exit(-1);
        } else {
            fclose(fp);
        }
        // read parameter file
        std::vector<double> ret;
        double k_driv, k_min, k_max;
        // get the number of dimensions, and based on that, set the number of vector field components
        ret = read_from_parameter_file("ndim", "d"); ndim = ret[0]; // Number of spatial dimensions (1, 1.5, 2, 2.5, 3)
        set_number_of_components(); // set number of components
        std::string ncmp_str; // number of component (ncmp) as string type
        std::stringstream dummystream;
        dummystream << ncmp; dummystream >> ncmp_str; dummystream.clear();
        // read remaining parameters from file
        ret = read_from_parameter_file("L", ncmp_str+"d"); // Length of domain (box) x[y[z]]
        for (int d = 0; d < ncmp; d++) L[d] = ret[d];
        ret = read_from_parameter_file("velocity", "d"); velocity = ret[0]; // Target turbulent velocity dispersion
        ret = read_from_parameter_file("k_driv", "d"); k_driv = ret[0]; // driving wavenumber in 2pi/L; sets t_decay below
        ret = read_from_parameter_file("k_min", "d"); k_min = ret[0]; // min wavenumber in 2pi/L
        ret = read_from_parameter_file("k_max", "d"); k_max = ret[0]; // max wavenumber in 2pi/L
        ret = read_from_parameter_file("sol_weight", "d"); sol_weight = ret[0]; // solenoildal weight
        ret = read_from_parameter_file("spect_form", "i"); spect_form = (int)ret[0]; // spectral form
        ret = read_from_parameter_file("power_law_exp", "d"); power_law_exp = ret[0]; // power-law exponent (if spect_form=2)
        power_law_exp_2 = power_law_exp; // driving does not support a 2nd PL section (yet)
        ret = read_from_parameter_file("angles_exp", "d"); angles_exp = ret[0]; // angles sampling exponent (if spect_form=2)
        ret = read_from_parameter_file("ampl_factor", ncmp_str+"d"); // adjust driving amplitude by factor in x[y[z]] (to adjust to target velocity)
        for (int d = 0; d < ncmp; d++) ampl_factor[d] = ret[d];
        ret = read_from_parameter_file("ampl_auto_adjust", "i"); ampl_auto_adjust = (int)ret[0]; // automatic amplitude adjustment switch
        ret = read_from_parameter_file("random_seed", "i"); random_seed = (int)ret[0]; // random seed
        ret = read_from_parameter_file("nsteps_per_t_turb", "i"); nsteps_per_t_turb = (int)ret[0]; // number of pattern updates per t_decay
        return finalise_init_driving(k_driv, k_min, k_max, time);
    }; // init_driving

    // ******************************************************
    public: virtual int init_driving(const std::map<std::string, std::string> &params, const double &time) {
       // ******************************************************
       // Initialize turbulence generator using parameters supplied through an unordered map
       // These parameters are used to drive the turbulence.
       // Supported map keys are as follows:
       //
       // ndim:                            Number of spatial dimensions (1, 1.5, 2, 2.5 or 3)
       // ampl_factor:                     Automatic amplitude adjustment factor. Can be one number
       //                                  or comma sepatated list of numbers
       // length:                          Length of domain box. Can be one number or comma
       //                                  separated list of numbers
       // target_vdisp:                    Target turbulent velocity dispersion
       // ampl_auto_adjust:                Automatic amplitude adjustment switch
       // k_driv:                          Driving wavenumber in units of 2pi/L; Sets t_decay
       // k_min:                           Min wavenumber in units of 2pi/L
       // k_max:                           Max wavenumber in units of 2pi/L
       // sol_weight:                      Solenoidal weight
       // spect_form:                      Spectral form
       // power_law_exp:                   Power-law exponent (if spectral form = 2)
       // angles_exp:                      Angles sampling exponent if (spectral form = 2)
       // random_seed:                     Random seed value
       // nsteps_per_t_turb:               Number of pattern updates per t_decay
       //
       // Other variables used are the same as the above overload of init_driving
       // ******************************************************
       if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");
       double k_driv, k_min, k_max;
       ndim = parse_param<double>(read_from_map(params, "ndim"), "ndim");
       set_number_of_components();
       split(read_from_map(params, "ampl_factor"), ampl_factor, ',', "ampl_factor");
       split(read_from_map(params, "length"), L, ',', "length");
       velocity = parse_param<double>(read_from_map(params, "target_vdisp"), "target_vdisp");
       k_driv = parse_param<double>(read_from_map(params, "k_driv"), "k_driv");
       k_min = parse_param<double>(read_from_map(params, "k_min"), "k_min");
       k_max = parse_param<double>(read_from_map(params, "k_max"), "k_max");
       sol_weight = parse_param<double>(read_from_map(params, "sol_weight"), "sol_weight");
       ampl_auto_adjust = parse_param<int>(read_from_map(params, "ampl_auto_adjust"), "ampl_auto_adjust");
       spect_form = parse_param<int>(read_from_map(params, "spect_form"), "spect_form");
       random_seed = parse_param<int>(read_from_map(params, "random_seed"), "random_seed");
       nsteps_per_t_turb = parse_param<int>(read_from_map(params, "nsteps_per_t_turb"), "nsteps_per_t_turb");
       if (spect_form == 2) {
         power_law_exp = parse_param<double>(read_from_map(params, "power_law_exp"), "power_law_exp");
         angles_exp = parse_param<double>(read_from_map(params, "angles_exp"), "angles_exp");
       }
       power_law_exp_2 = power_law_exp;
       return finalise_init_driving(k_driv, k_min, k_max, time);
    }

    // ******************************************************
    protected: int finalise_init_driving(const double &k_driv, const double &k_min, const double &k_max, const double &time) {
       // define derived physical quantities
       kmin = (k_min-DBL_EPSILON) * 2*M_PI / L[X]; // Minimum driving wavenumber <~  k_min * 2pi / Lx
       kmax = (k_max+DBL_EPSILON) * 2*M_PI / L[X]; // Maximum driving wavenumber >~  k_max * 2pi / Lx
       kmid = kmax; // driving does not support a 2nd PL section (yet)
       t_decay = L[X] / k_driv / velocity;             // Auto-correlation time, t_turb = Lx / k_driv / velocity;
                                                       // i.e., turbulent turnover (crossing) time; with k_driv in units of 2pi/Lx
       dt = t_decay / nsteps_per_t_turb;               // time step in OU process and for creating new driving pattern
       step = -1;                                      // set internal OU step to -1 for start-up
       seed = random_seed;                             // copy original seed into local seed;
                                                       // local seeds gets updated everytime the random number generator is called
       const double ampl_coeff = 0.15;                 // This is the default amplitude coefficient ('ampl_coeff') that often
                                                       // (based on Mach ~ 1, naturally-mixed driving) leads to a good match of the
                                                       // user-to-target velocity dispersion.
                                                       // However, the amplitudes can be adjusted via ampl_factor_in[X,Y,Z].
       energy = pow(ampl_coeff*velocity, 3.0) / L[X];  // Energy input rate => driving amplitude ~ sqrt(energy/t_decay).
                                                       // Note that energy input rate ~ velocity^3 / L_box.
       OUvar = sqrt(energy/t_decay);                   // set Ornstein-Uhlenbeck (OU) variance
       // if we auto-adjust the amplitude, we need to read the ampl_factor from the evolution file, on restart (time > 0)
       if ((ampl_auto_adjust == 1) && (time > 0.0))
           if (!read_ampl_factor_from_evol_file(time)) return -1;
       // We raise the user-set ampl_factor to the 1.5th power, so the user can more easily adjust this as a pure factor
       // of target-to-measured velocity disperion (this is because the amplitude is actually ~ velocity^1.5; see just above).
       for (int d = 0; d < ncmp; d++) ampl_factor[d] = pow(ampl_factor[d], 1.5);
       if (verbose) TurbGen_printf("===============================================================================\n");
       // set solenoidal weight normalisation
       set_solenoidal_weight_normalisation();
       // initialise modes
       init_modes();
       // initialise Ornstein-Uhlenbeck sequence
       OU_noise_init();
       // calculate solenoidal and compressive coefficients (aka, akb) from OUphases
       get_decomposition_coeffs();
       // print info
       if (verbose) print_info("driving");
       // write header of time evolution file
       if (verbose) TurbGen_printf("Writing time evolution information to file '%s'.\n", evolfile.c_str());
       if (PE == 0) {
           std::ofstream outfilestream(evolfile.c_str(), std::ios::app);
           outfilestream   << std::setw(24) << "#01_time" << std::setw(24) << "#02_time_in_t_turb"
                           << std::setw(24) << "#03_ampl_factor_x" << std::setw(24) << "#04_ampl_factor_y" << std::setw(24) << "#05_ampl_factor_z"
                           << std::setw(24) << "#06_v_turb_prev_x" << std::setw(24) << "#07_v_turb_prev_y" << std::setw(24) << "#08_v_turb_prev_z"
                           << std::endl;
           outfilestream.close();
           outfilestream.clear();
       }
       if (verbose) TurbGen_printf("===============================================================================\n");
       if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
       return 0;
    }

    // ******************************************************
    public: bool check_for_update(const double time) {
        // call overloaded check_for_update() without automatic amplitude adjustment (v_turb < 0)
        double v_turb[3]; for (int d = 0; d < 3; d++) v_turb[d] = -1.0;
        return check_for_update(time, v_turb);
    }; // check_for_update(time)

    // ******************************************************
    public: virtual bool check_for_update(const double time, const double v_turb[]) {
        // ******************************************************
        // Update driving pattern based on input 'time'.
        // If it is 'time' to update the pattern, call OU noise update
        // and update the decomposition coefficients; otherwise, simply return.
        // ******************************************************
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");
        // check if we need to update the OU pattern
        int step_requested = floor(time / dt); // requested OU step number based on input 'time'
        if (verbose > 1) TurbGen_printf("step_requested = %i\n", step_requested);
        if (step_requested <= step) {
            if (verbose > 1) TurbGen_printf("no update of pattern...returning.\n");
            if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
            return false; // no update (yet) -> return false, i.e., no change of driving pattern
        }
        // check to see if we do automatic adjustment of the driving amplitude to reach user-defined target turbulent velocity dispersion
        if ((ampl_auto_adjust == 1) && (v_turb[X] > 0)) {
            double v_turb_for_ampl_adjust[3];
            // copy v_turb into v_turb_for_ampl_adjust (default, if we do not use time_averaging)
            for (int d = 0; d < ncmp; d++) v_turb_for_ampl_adjust[d] = v_turb[d]; // current v_turb
            // if the time is right, adjust the amplitude
            if (time > 0.1*t_decay) { // auto-adjustment start time (start at t > 0.1 turnover times; could start later)
                double ampl_factor_adjusted[3];
                double ampl_adjust_timescale[3];
                for (int d = 0; d < ncmp; d++) {
                    // ampl_factor_new would be the perfect match at this moment (based on target velocity and current v_turb)
                    double ampl_factor_new = ampl_factor[d] * pow(velocity / v_turb_for_ampl_adjust[d] / sqrt(ncmp), 1.5);
                    // approach ampl_factor_new based on timecale 'ampl_adjust_timescale' to avoid strong over- and under-shooting
                    ampl_adjust_timescale[d] = pow(ampl_factor_new/ampl_factor[d], 0.5) * t_decay; // experimental
                    double damping_factor = exp(-dt/ampl_adjust_timescale[d]);
                    ampl_factor_adjusted[d] = ampl_factor[d] * damping_factor + (1.0-damping_factor) * ampl_factor_new;
                }
                TurbGen_printf(" ===> auto-adjusting driving amplitude: ampl_factor (auto-adjusted) =");
                for (int d = 0; d < ncmp; d++) TurbGen_printf_raw(" %f", pow(ampl_factor_adjusted[d], 1.0/1.5));
                TurbGen_printf_raw("\n");
                TurbGen_printf(" ===> amplitude adjustment timescale (in units of t_turb) =");
                for (int d = 0; d < ncmp; d++) TurbGen_printf_raw(" %f", ampl_adjust_timescale[d]/t_decay);
                TurbGen_printf_raw("\n");
                for (int d = 0; d < ncmp; d++) ampl_factor[d] = ampl_factor_adjusted[d]; // set new (adjusted) ampl_factor
            }
        } // if (auto_adjust_amplitude)
        // if we are here: update OU vector
        for (int is = step; is < step_requested; is++) {
            OU_noise_update(); // this seeks to the requested OU state (updates OUphases)
            step++; // update internal OU step number
            if (verbose > 1) TurbGen_printf("step = %i, time = %f\n", step, step*dt);
        }
        get_decomposition_coeffs(); // calculate solenoidal and compressive coefficients (aka, akb) from OUphases
        double time_gen = step * dt;
        if (verbose) TurbGen_printf("Generated new turbulence driving pattern: #%6i, time = %e, time/t_turb = %-7.2f\n", step, time_gen, time_gen/t_decay);
        if (PE == 0) write_to_evol_file(time, ampl_factor, v_turb); // write evolution file
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
        return true; // we just updated the driving pattern
    }; // check_for_update(time, v_turb)

    // ******************************************************
    public: bool write_to_evol_file(const double time, const double ampl_factor[], const double v_turb[]) {
        if (verbose > 1) TurbGen_printf("Writing to evolution file: time = %e, time/t_turb = %-7.2f\n", time, time/t_decay);
        std::ofstream outfilestream(evolfile.c_str(), std::ios::app);
        outfilestream.precision(16);
        outfilestream   << std::scientific << std::setw(24) << time << std::setw(24) << time/t_decay
                        << std::setw(24) << pow(ampl_factor[X],1.0/1.5) << std::setw(24) << pow(ampl_factor[Y],1.0/1.5) << std::setw(24) << pow(ampl_factor[Z],1.0/1.5)
                        << std::setw(24) << v_turb[X] << std::setw(24) << v_turb[Y] << std::setw(24) << v_turb[Z]
                        << std::endl;
        outfilestream.close();
        outfilestream.clear();
        return true;
    }; // write_to_evol_file

    // ******************************************************
    public: bool read_ampl_factor_from_evol_file(const double time) {
        if (verbose > 1) TurbGen_printf("Reading from evolution file: time = %e, time/t_turb = %-7.2f\n", time, time/t_decay);
        std::ifstream infilestream(evolfile.c_str());
        if (infilestream.is_open()) {
            // read each line from file into a vector
            std::vector<std::string> lines;
            std::string line;
            while (std::getline(infilestream, line)) lines.push_back(line);
            infilestream.close();
            TurbGen_printf("Restoring ampl_factor from '%s' for current simulation time = %e\n", evolfile.c_str(), time);
            bool initial_copy_done = false;
            // loop backwards through the vector to find the right time entry
            for (int i = lines.size() - 1; i >= 0; i--) {
                if (verbose > 1) TurbGen_printf((lines[i]+'\n').c_str());
                // skip header line(s)
                if (lines[i].find("time") != std::string::npos) continue;
                // split line
                std::istringstream buffer(lines[i]);
                std::vector<std::string> line_split((std::istream_iterator<std::string>(buffer)), std::istream_iterator<std::string>());
                double time_in_file = double(atof(line_split[0].c_str())); // time
                double ampl_factor_in_file[3];
                for (int d = 0; d < ncmp; d++) ampl_factor_in_file[d] = double(atof(line_split[d+2].c_str()));
                if (verbose > 1) TurbGen_printf("time_in_file, ampl_factor_in_file = %e %e %e %e\n", time_in_file, ampl_factor_in_file[X], ampl_factor_in_file[Y], ampl_factor_in_file[Z]);
                if (!initial_copy_done) { // copy the last valid ampl_factor from the evolution file
                    for (int d = 0; d < ncmp; d++) ampl_factor[d] = ampl_factor_in_file[d];
                    initial_copy_done = true;
                }
                if (time_in_file >= time-1.1*dt) { // search backwards and update ampl_factor with last valid time
                    for (int d = 0; d < ncmp; d++) ampl_factor[d] = ampl_factor_in_file[d];
                }
                else break;
            }
            TurbGen_printf("restored ampl_factor = %e %e %e \n", ampl_factor[X], ampl_factor[Y], ampl_factor[Z]);
        }
        else {
            TurbGen_printf("This is a restart with ampl_auto_adjust = 1, but unable to open file '%s' to restore ampl_factor.\n", evolfile.c_str());
            return false;
        }
        return true;
    }; // read_ampl_factor_from_evol_file

    // ******************************************************
    public: void get_turb_vector_unigrid(const double pos_beg[], const double pos_end[], const int n[], float * return_grid[]) {
        // ******************************************************
        // Compute physical turbulent vector field on a uniform grid, provided
        // start coordinate pos_beg[ndim] and end coordinate pos_end[ndim]
        // of the grid and number of points in grid n[ndim].
        // Return into turbulent vector field into float * return_grid[ndim].
        // Note that index in return_grid[X][index] is looped with x (index i)
        // as the inner loop and with z (index k) as the outer loop.
        // ******************************************************
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");
        if (verbose > 1) TurbGen_printf("pos_beg = %f %f %f, pos_end = %f %f %f, n = %i %i %i\n",
                pos_beg[X], pos_beg[Y], pos_beg[Z], pos_end[X], pos_end[Y], pos_end[Z], n[X], n[Y], n[Z]);

        // compute output grid cell width (dx, dy, dz)
        double del[3] = {1.0, 1.0, 1.0};
        for (int d = 0; d < (int)ndim; d++) if (n[d] > 1) del[d] = (pos_end[d] - pos_beg[d]) / (n[d]-1);
        // pre-compute amplitude including normalisation factors
        std::vector<double> ampl(nmodes);
        for (int m = 0; m < nmodes; m++) ampl[m] = 2.0 * sol_weight_norm * this->ampl[m];
        // pre-compute grid position geometry, and trigonometry, to speed-up loops over modes below
        std::vector< std::vector<double> > sinxi(n[X], std::vector<double>(nmodes));
        std::vector< std::vector<double> > cosxi(n[X], std::vector<double>(nmodes));
        std::vector< std::vector<double> > sinyj(n[Y], std::vector<double>(nmodes));
        std::vector< std::vector<double> > cosyj(n[Y], std::vector<double>(nmodes));
        std::vector< std::vector<double> > sinzk(n[Z], std::vector<double>(nmodes));
        std::vector< std::vector<double> > coszk(n[Z], std::vector<double>(nmodes));
        for (int m = 0; m < nmodes; m++) {
            for (int i = 0; i < n[X]; i++) {
                sinxi[i][m] = sin(mode[X][m]*(pos_beg[X]+i*del[X]));
                cosxi[i][m] = cos(mode[X][m]*(pos_beg[X]+i*del[X]));
            }
            for (int j = 0; j < n[Y]; j++) {
                if ((int)ndim > 1) {
                    sinyj[j][m] = sin(mode[Y][m]*(pos_beg[Y]+j*del[Y]));
                    cosyj[j][m] = cos(mode[Y][m]*(pos_beg[Y]+j*del[Y]));
                } else {
                    sinyj[j][m] = 0.0;
                    cosyj[j][m] = 1.0;
                }
            }
            for (int k = 0; k < n[Z]; k++) {
                if ((int)ndim > 2) {
                    sinzk[k][m] = sin(mode[Z][m]*(pos_beg[Z]+k*del[Z]));
                    coszk[k][m] = cos(mode[Z][m]*(pos_beg[Z]+k*del[Z]));
                } else {
                    sinzk[k][m] = 0.0;
                    coszk[k][m] = 1.0;
                }
            }
        }
        // scratch variables
        double v[3];
        double real, imag;
        // loop over cells in return_grid
        for (int k = 0; k < n[Z]; k++) {
            for (int j = 0; j < n[Y]; j++) {
                for (int i = 0; i < n[X]; i++) {
                    // clear
                    v[X] = 0.0; v[Y] = 0.0; v[Z] = 0.0;
                    // loop over modes
                    for (int m = 0; m < nmodes; m++) {
                        // these are the real and imaginary parts, respectively, of
                        //  e^{ i \vec{k} \cdot \vec{x} } = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)
                        real =  ( cosxi[i][m]*cosyj[j][m] - sinxi[i][m]*sinyj[j][m] ) * coszk[k][m] -
                                ( sinxi[i][m]*cosyj[j][m] + cosxi[i][m]*sinyj[j][m] ) * sinzk[k][m];
                        imag =  ( cosyj[j][m]*sinzk[k][m] + sinyj[j][m]*coszk[k][m] ) * cosxi[i][m] +
                                ( cosyj[j][m]*coszk[k][m] - sinyj[j][m]*sinzk[k][m] ) * sinxi[i][m];
                        // accumulate total v as sum over modes
                        v[X] += ampl[m] * (aka[X][m]*real - akb[X][m]*imag);
                        if (ncmp > 1) v[Y] += ampl[m] * (aka[Y][m]*real - akb[Y][m]*imag);
                        if (ncmp > 2) v[Z] += ampl[m] * (aka[Z][m]*real - akb[Z][m]*imag);
                    }
                    // copy into return grid
                    long index = k*n[X]*n[Y] + j*n[X] + i;
                    return_grid[X][index] = v[X] * ampl_factor[X];
                    if (ncmp > 1) return_grid[Y][index] = v[Y] * ampl_factor[Y];
                    if (ncmp > 2) return_grid[Z][index] = v[Z] * ampl_factor[Z];
                } // i
            } // j
        } // k
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
    } // get_turb_vector_unigrid


    // ******************************************************
    public: void get_turb_vector(const double pos[], double v[]) {
        // ******************************************************
        // Compute physical turbulent vector v[ndim]=(vx,vy,vz) at position pos[ndim]=(x,y,z)
        // from loop over all turbulent modes; return into double v[ndim]
        // ******************************************************
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");
        // containers for speeding-up calculations below
        std::vector<double> ampl(nmodes);
        std::vector<double> sinx(nmodes); std::vector<double> cosx(nmodes);
        std::vector<double> siny(nmodes); std::vector<double> cosy(nmodes);
        std::vector<double> sinz(nmodes); std::vector<double> cosz(nmodes);
        // pre-compute some trigonometry
        for (int m = 0; m < nmodes; m++) {
            ampl[m] = 2.0 * sol_weight_norm * this->ampl[m]; // pre-compute amplitude including normalisation factors
            sinx[m] = sin(mode[X][m]*pos[X]);
            cosx[m] = cos(mode[X][m]*pos[X]);
            if ((int)ndim > 1) {
                siny[m] = sin(mode[Y][m]*pos[Y]);
                cosy[m] = cos(mode[Y][m]*pos[Y]);
            } else {
                siny[m] = 0.0;
                cosy[m] = 1.0;
            }
            if ((int)ndim > 2) {
                sinz[m] = sin(mode[Z][m]*pos[Z]);
                cosz[m] = cos(mode[Z][m]*pos[Z]);
            } else {
                sinz[m] = 0.0;
                cosz[m] = 1.0;
            }
        }
        // scratch variables
        double real, imag;
        // init return vector with zero
        v[X] = 0.0; if (ncmp > 1) v[Y] = 0.0; if (ncmp > 2) v[Z] = 0.0;
        // loop over modes
        for (int m = 0; m < nmodes; m++) {
            // these are the real and imaginary parts, respectively, of
            //  e^{ i \vec{k} \cdot \vec{x} } = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)
            real = ( cosx[m]*cosy[m] - sinx[m]*siny[m] ) * cosz[m] - ( sinx[m]*cosy[m] + cosx[m]*siny[m] ) * sinz[m];
            imag = cosx[m] * ( cosy[m]*sinz[m] + siny[m]*cosz[m] ) + sinx[m] * ( cosy[m]*cosz[m] - siny[m]*sinz[m] );
            // return vector for this position x, y, z
            v[X] += ampl[m] * (aka[X][m]*real - akb[X][m]*imag) * ampl_factor[X];
            if (ncmp > 1) v[Y] += ampl[m] * (aka[Y][m]*real - akb[Y][m]*imag) * ampl_factor[Y];
            if (ncmp > 2) v[Z] += ampl[m] * (aka[Z][m]*real - akb[Z][m]*imag) * ampl_factor[Z];
        }
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
    }; // get_turb_vector


    // ******************************************************
    private: void print_info(std::string print_mode) {
        // ******************************************************
        if (print_mode == "driving") {
            TurbGen_printf("Initialized %i modes for turbulence driving based on parameter file '%s'.\n", nmodes, parameter_file.c_str());
        }
        if (print_mode == "single_realisation") {
            TurbGen_printf("Initialized %i modes for generating a single turbulent realisation.\n", nmodes);
        }
        if (spect_form == 0) TurbGen_printf(" spectral form                                       = %i (Band)\n", spect_form);
        if (spect_form == 1) TurbGen_printf(" spectral form                                       = %i (Parabola)\n", spect_form);
        if (spect_form == 2) TurbGen_printf(" spectral form                                       = %i (Power Law)\n", spect_form);
        if (spect_form == 2) TurbGen_printf(" power-law exponent                                  = %e\n", power_law_exp);
        if (spect_form == 2 && kmid != kmax) TurbGen_printf(" power-law exponent 2                                = %e\n", power_law_exp_2);
        if (spect_form == 2) TurbGen_printf(" power-law angles sampling exponent                  = %e\n", angles_exp);
        TurbGen_printf(" box size Lx                                         = %e\n", L[X]);
        if (print_mode == "driving") {
            TurbGen_printf(" turbulent velocity dispersion                       = %e\n", velocity);
            TurbGen_printf(" turbulent turnover (auto-correlation) time          = %e\n", t_decay);
            TurbGen_printf("  -> characteristic turbulent wavenumber (in 2pi/Lx) = %e\n", L[X] / velocity / t_decay);
        }
        TurbGen_printf(" minimum wavenumber (in 2pi/Lx)                      = %e\n", kmin / (2*M_PI) * L[X]);
        if (spect_form == 2 && kmid != kmax) TurbGen_printf(" middle  wavenumber (in 2pi/Lx)                      = %e\n", kmid / (2*M_PI) * L[X]);
        TurbGen_printf(" maximum wavenumber (in 2pi/Lx)                      = %e\n", kmax / (2*M_PI) * L[X]);
        if (print_mode == "driving") {
            TurbGen_printf(" amplitude coefficient                               = %e\n", pow(energy*L[X],1.0/3.0) / velocity);
            TurbGen_printf("  -> driving energy (injection rate)                 = %e\n", energy);
        }
        TurbGen_printf(" solenoidal weight (0.0: comp, 0.5: mix, 1.0: sol)   = %e\n", sol_weight);
        TurbGen_printf("  -> solenoidal weight norm (based on Ndim = %3.1f)    = %e\n", ndim, sol_weight_norm);
        TurbGen_printf(" random seed                                         = %i\n", random_seed);
    }; // print_info


    // ******************************************************
    private: std::vector<double> read_from_parameter_file(std::string search, std::string type) {
        // ******************************************************
        // parse each line in turbulence generator 'parameter_file' and search for 'search'
        // at the beginning of each line; if 'search' is found return double of value after '='
        // type: 'i' for int, 'd' for double return type void *ret
        // ******************************************************
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");
        std::vector<double> ret; // return vector object
        FILE * fp;
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        fp = fopen(parameter_file.c_str(), "r");
        std::string msg = ClassSignature+"ERROR: could not open parameter file '%s'\n";
        if (fp == NULL) { printf(msg.c_str(), parameter_file.c_str()); exit(-1); }
        bool found = false;
        while ((read = getline(&line, &len, fp)) != -1) {
            if (strncmp(line, search.c_str(), strlen(search.c_str())) == 0) {
                if (verbose > 1) TurbGen_printf("line = '%s'\n", line);
                char * substr1 = strstr(line, "="); // extract everything after (and including) '='
                if (verbose > 1) TurbGen_printf("substr1 = '%s'\n", substr1);
                char * substr2 = strstr(substr1, "!"); // deal with comment '! ...'
                char * substr3 = strstr(substr1, "#"); // deal with comment '# ...'
                int end_index = strlen(substr1);
                if ((substr2 != NULL) && (substr3 != NULL)) { // if comment is present, reduce end_index
                    end_index -= std::max(strlen(substr2),strlen(substr3));
                } else { // if comment is present, reduce end_index
                    if (substr2 != NULL) end_index -= strlen(substr2);
                    if (substr3 != NULL) end_index -= strlen(substr3);
                }
                char dest[100]; memset(dest, '\0', sizeof(dest));
                strncpy(dest, substr1+1, end_index-1);
                if (verbose > 1) TurbGen_printf("dest = '%s'\n", dest);
                if ((type == "i") || (type == "1i")) {
                    ret.resize(1); ret[0] = double(atoi(dest));
                }
                if ((type == "d") || (type == "1d")) {
                    ret.resize(1); ret[0] = double(atof(dest));
                }
                if ((type == "2d") || (type == "3d")) {
                    int ncmp;
                    std::stringstream dummystream;
                    dummystream << type[0]; dummystream >> ncmp; dummystream.clear();
                    ret.resize(ncmp);
                    if (verbose > 1) TurbGen_printf("n = %i\n", ncmp);
                    std::string cmpstr = dest;
                    int cnt = 0; // count the number of ','
                    for (size_t offset = cmpstr.find(","); offset != std::string::npos; offset = cmpstr.find(",", offset+1)) ++cnt;
                    if (verbose > 1) TurbGen_printf("cnt = %i\n", cnt);
                    if ((cnt != 0) && (cnt != ncmp-1)) { // error check
                        std::string msg = ClassSignature+"ERROR: parameter '%s' specifies %i components, which does not match actual number of components: %i\n";
                        printf(msg.c_str(), search.c_str(), cnt+1, ncmp);
                        exit(-1);
                    }
                    for (int d=0; d<ncmp; d++) {
                        std::string substr = cmpstr; // copy into substr in case no commas provided
                        if (cnt == ncmp-1) { // if user has specified ncmp components; else copy the same value into each component
                            substr = cmpstr.substr(0, cmpstr.find(",")); // extract substring before first ','
                            cmpstr = cmpstr.substr(cmpstr.find(",")+1, cmpstr.size()-cmpstr.find(",")); // reduce cmpstr to remainder for next loop iter
                        }
                        if (verbose > 1)TurbGen_printf("substr = '%s'\n", substr.c_str());
                        dummystream << substr; dummystream >> ret[d]; dummystream.clear(); // copy into ret
                    }
                }
                found = true;
            }
            if (found) break;
        }
        fclose(fp);
        if (line) free(line);
        if (!found) {
            std::string msg = ClassSignature+"ERROR: requested parameter '%s' not found in file '%s'\n";
            printf(msg.c_str(), search.c_str(), parameter_file.c_str());
            exit(-1);
        }
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
        return ret;
    }; // read_from_parameter_file


    // ******************************************************
    private: int init_modes(void) {
        // ******************************************************
        // initialise all turbulent modes information
        // ******************************************************

        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");

        int ikmin[3], ikmax[3], ik[3], tot_nmodes_full_sampling;
        double k[3], ka, kc, amplitude, parab_prefact;

        // applies in case of power law (spect_form == 2)
        int iang, nang;
        double rand, phi, theta;

        // this is for spect_form = 1 (paraboloid) only
        // prefactor for amplitude normalistion to 1 at kc = 0.5*(kmin+kmax)
        parab_prefact = -4.0 / pow(kmax-kmin,2.0);

        // characteristic k for scaling the amplitude below
        kc = kmin;
        if (spect_form == 1) kc = 0.5*(kmin+kmax);

        ikmin[X] = 0;
        ikmin[Y] = 0;
        ikmin[Z] = 0;

        ikmax[X] = 256;
        ikmax[Y] = 0;
        ikmax[Z] = 0;
        if ((int)ndim > 1) ikmax[Y] = 256;
        if ((int)ndim > 2) ikmax[Z] = 256;

        // determine the number of required modes (in case of full sampling)
        nmodes = 0;
        for (ik[X] = -ikmax[X]; ik[X] <= ikmax[X]; ik[X]++) {
            k[X] = 2*M_PI * ik[X] / L[X];
            for (ik[Y] = -ikmax[Y]; ik[Y] <= ikmax[Y]; ik[Y]++) {
                k[Y] = 2*M_PI * ik[Y] / L[Y];
                for (ik[Z] = -ikmax[Z]; ik[Z] <= ikmax[Z]; ik[Z]++) {
                    k[Z] = 2*M_PI * ik[Z] / L[Z];
                    ka = sqrt( k[X]*k[X] + k[Y]*k[Y] + k[Z]*k[Z] );
                    if ((ka >= kmin) && (ka <= kmax)) nmodes++;
                }
            }
        }
        tot_nmodes_full_sampling = nmodes;
        if (spect_form != 2) { // for Band (spect_form=0) and Parabola (spect_form=1)
            if (tot_nmodes_full_sampling > NameSpaceTurbGen::tgd_max_nmodes) {
                TurbGen_printf(" nmodes = %i, maxmodes = %i", nmodes, NameSpaceTurbGen::tgd_max_nmodes);
                TurbGen_printf("Too many stirring modes"); exit(-1);
            }
            if (verbose) TurbGen_printf("Generating %i turbulent modes...\n", tot_nmodes_full_sampling);
        }
        nmodes = 0; // reset

        // ===================================================================
        // === for band and parabolic spectrum, use the standard full sampling
        if (spect_form != 2) {

            // loop over all kx, ky, kz to generate turbulent modes
            for (ik[X] = -ikmax[X]; ik[X] <= ikmax[X]; ik[X]++) {
                k[X] = 2*M_PI * ik[X] / L[X];
                for (ik[Y] = -ikmax[Y]; ik[Y] <= ikmax[Y]; ik[Y]++) {
                    k[Y] = 2*M_PI * ik[Y] / L[Y];
                        for (ik[Z] = -ikmax[Z]; ik[Z] <= ikmax[Z]; ik[Z]++) {
                        k[Z] = 2*M_PI * ik[Z] / L[Z];

                        ka = sqrt( k[X]*k[X] + k[Y]*k[Y] + k[Z]*k[Z] );

                        if ((ka >= kmin) && (ka <= kmax)) {

                            if (spect_form == 0) amplitude = 1.0;                                    // Band
                            if (spect_form == 1) amplitude = fabs(parab_prefact*pow(ka-kc,2.0)+1.0); // Parabola

                            // note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
                            amplitude = sqrt(amplitude) * pow(kc/ka,((int)ndim-1)/2.0);
                            ampl.push_back(amplitude);

                            mode[X].push_back(k[X]);
                            if ((int)ndim > 1) mode[Y].push_back(k[Y]);
                            if ((int)ndim > 2) mode[Z].push_back(k[Z]);
                            nmodes++;

                            if ((nmodes) % 1000 == 0) if (verbose) TurbGen_printf(" ... %i of total %i modes generated...\n", nmodes, tot_nmodes_full_sampling);

                        } // in k range
                    } // ikz
                } // iky
            } // ikx

        } // spect_form != 2

        // ===============================================================================
        // === for power law (spect_form=2), generate modes that are distributed randomly on the k-sphere
        // === with the number of angles growing ~ k^angles_exp
        if (spect_form == 2) {

            if (verbose) TurbGen_printf("There would be %i turbulent modes, if k-space were fully sampled (angles_exp = 2.0)...\n", tot_nmodes_full_sampling);
            if (verbose) TurbGen_printf("Here we are using angles_exp = %f\n", angles_exp);

            // initialize additional random numbers (uniformly distributed) to randomise angles
            int seed_init = -seed; // initialise Numerical Recipes rand gen (call with negative integer)
            rand = ran2(&seed_init);

            // loop between smallest and largest k
            ikmin[0] = std::max(1, (int)round(kmin*L[X]/(2*M_PI)));
            ikmax[0] =             (int)round(kmax*L[X]/(2*M_PI));

            if (verbose) TurbGen_printf("Generating turbulent modes within k = [%i, %i]\n", ikmin[0], ikmax[0]);

            for (ik[0] = ikmin[0]; ik[0] <= ikmax[0]; ik[0]++) {

                nang = pow(2.0,(int)ndim) * ceil(pow((double)ik[0],angles_exp));
                if (verbose) TurbGen_printf("ik, number of angles = %i, %i\n", ik[0], nang);

                for (iang = 1; iang <= nang; iang++) {

                    phi = 2*M_PI * ran2(&seed); // phi = [0,2pi] sample the whole sphere
                    if ((int)ndim == 1) {
                        if (phi <  M_PI) phi = 0.0; // left
                        if (phi >= M_PI) phi = M_PI; // right
                    }
                    theta = M_PI/2.0;
                    if ((int)ndim > 2) theta = acos(1.0 - 2.0*ran2(&seed)); // theta = [0,pi] sample the whole sphere

                    if (verbose > 1) TurbGen_printf("theta = %f, phi = %f\n", theta, phi);

                    rand = ik[0] + ran2(&seed) - 0.5;
                    k[X] = 2*M_PI * round(rand*sin(theta)*cos(phi)) / L[X];
                    if ((int)ndim > 1)
                        k[Y] = 2*M_PI * round(rand*sin(theta)*sin(phi)) / L[Y];
                    else
                        k[Y] = 0.0;
                    if ((int)ndim > 2)
                        k[Z] = 2*M_PI * round(rand*cos(theta)) / L[Z];
                    else
                        k[Z] = 0.0;

                    ka = sqrt( k[X]*k[X] + k[Y]*k[Y] + k[Z]*k[Z] );

                    if ((ka >= kmin) && (ka <= kmax)) {

                        if (nmodes > NameSpaceTurbGen::tgd_max_nmodes) {
                            TurbGen_printf(" nmodes = %i, maxmodes = %i", nmodes, NameSpaceTurbGen::tgd_max_nmodes);
                            TurbGen_printf("Too many stirring modes"); exit(-1);
                        }

                        amplitude = pow(ka/kc,power_law_exp); // Power law

                        if (ka >= kmid) // optional power-law section 2 (between kmid and kmax)
                            amplitude = pow(kmid/kmin,power_law_exp) * pow(ka/kmid,power_law_exp_2);

                        // note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
                        // ...and correct for the number of angles sampled relative to the full sampling (k^2 per k-shell in 3D)
                        amplitude = sqrt( amplitude * pow((double)ik[0],(int)ndim-1) / (double)(nang) * 4.0*sqrt(3.0) ) * pow(kc/ka,((int)ndim-1)/2.0);

                        nmodes++;
                        ampl.push_back(amplitude);
                        mode[X].push_back(k[X]);
                        if ((int)ndim > 1) mode[Y].push_back(k[Y]);
                        if ((int)ndim > 2) mode[Z].push_back(k[Z]);

                        if (verbose > 1) TurbGen_printf(" ... %i mode(s) generated...\n", nmodes);
                        if ((nmodes) % 1000 == 0) if (verbose) TurbGen_printf(" ... %i modes generated...\n", nmodes);

                    } // in k range

                } // loop over angles
            } // loop over k
        } // spect_form == 2

        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
        return 0;

    }; // init_modes


    // ******************************************************
    private: void OU_noise_init(void) {
        // ******************************************************
        // initialize pseudo random sequence for the Ornstein-Uhlenbeck (OU) process
        // ******************************************************
        OUphases.resize(nmodes*ncmp*2);
        for (int m = 0; m < nmodes; m++) {
            for (int d = 0; d < ncmp; d++) {
                for (int ir = 0; ir < 2; ir++) {
                    OUphases[2*ncmp*m+2*d+ir] = OUvar * get_random_number();
                }
            }
        }
    }; // OU_noise_init


    // ******************************************************
    private: void OU_noise_update(void) {
        // ******************************************************
        // update Ornstein-Uhlenbeck sequence
        //
        // The sequence x_n is a Markov process that takes the previous value,
        // weights by an exponential damping factor with a given correlation
        // time 'ts', and drives by adding a Gaussian random variable with
        // variance 'variance', weighted by a second damping factor, also
        // with correlation time 'ts'. For a timestep of dt, this sequence
        // can be written as
        //
        //     x_n+1 = f x_n + sigma * sqrt (1 - f**2) z_n,
        //
        // where f = exp (-dt / ts), z_n is a Gaussian random variable drawn
        // from a Gaussian distribution with unit variance, and sigma is the
        // desired variance of the OU sequence.
        //
        // The resulting sequence should satisfy the properties of zero mean,
        // and stationarity (independent of portion of sequence) RMS equal to
        // 'variance'. Its power spectrum in the time domain can vary from
        // white noise to "brown" noise (P (f) = const. to 1 / f^2).
        //
        // References:
        //   Bartosch (2001)
        //   Eswaran & Pope (1988)
        //   Schmidt et al. (2009)
        //   Federrath et al. (2010, A&A 512, A81); Eq. (4)
        //
        // ******************************************************
        const double damping_factor = exp(-dt/t_decay);
        for (int m = 0; m < nmodes; m++) {
            for (int d = 0; d < ncmp; d++) {
                for (int ir = 0; ir < 2; ir++) {
                    OUphases[2*ncmp*m+2*d+ir] = OUphases[2*ncmp*m+2*d+ir] * damping_factor +
                        sqrt(1.0 - damping_factor*damping_factor) * OUvar * get_random_number();
                }
            }
        }
    }; // OU_noise_update


    // ******************************************************
    private: void get_decomposition_coeffs(void) {
        // ******************************************************
        // This routine applies the projection operator based on the OU phases.
        // See Eq. (6) in Federrath et al. (2010).
        // ******************************************************
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");
        // resize aka and akb
        for (int d = 0; d < 3; d++) {
            aka[d].resize(nmodes);
            akb[d].resize(nmodes);
        }
        double ka, kb, kk, diva, divb, curla, curlb;
        for (int m = 0; m < nmodes; m++) {
            ka = 0.0;
            kb = 0.0;
            kk = 0.0;
            for (int d = 0; d < (int)ndim; d++)
            for (int d = 0; d < ncmp; d++) {
                double this_mode = mode[std::min(d,(int)ndim-1)][m];
                kk = kk + this_mode*this_mode; // get k**2
                ka = ka + this_mode*OUphases[2*ncmp*m+2*d+1];
                kb = kb + this_mode*OUphases[2*ncmp*m+2*d+0];
            }
            for (int d = 0; d < ncmp; d++) {
                double this_mode = mode[std::min(d,(int)ndim-1)][m];
                diva  = this_mode*ka/kk;
                divb  = this_mode*kb/kk;
                curla = OUphases[2*ncmp*m+2*d+0] - divb;
                curlb = OUphases[2*ncmp*m+2*d+1] - diva;
                aka[d][m] = sol_weight*curla+(1.0-sol_weight)*divb;
                akb[d][m] = sol_weight*curlb+(1.0-sol_weight)*diva;
                // purely compressive
                // aka[d][m] = mode[d][m]*kb/kk;
                // akb[d][m] = mode[d][m]*ka/kk;
                // purely solenoidal
                // aka[d][m] = R - mode[d][m]*kb/kk;
                // akb[d][m] = I - mode[d][m]*ka/kk;
                if (verbose > 1) {
                    TurbGen_printf("mode(dim=%1i, mode=%3i) = %12.6f\n", d, m, this_mode);
                    TurbGen_printf("aka (dim=%1i, mode=%3i) = %12.6f\n", d, m, aka [d][m]);
                    TurbGen_printf("akb (dim=%1i, mode=%3i) = %12.6f\n", d, m, akb [d][m]);
                    TurbGen_printf("ampl(mode=%3i) = %12.6f\n", d, m, ampl[m]);
                }
            }
        }
        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
    }; // get_decomposition_coeffs


    // ******************************************************
    private: double get_random_number(void) {
        // ******************************************************
        //  get random number; draws a number randomly from a Gaussian distribution
        //  with the standard uniform distribution function "ran1s"
        //  using the Box-Muller transformation in polar coordinates. The
        //  resulting Gaussian has unit variance.
        // ******************************************************
        double r1 = ran1s(&seed);
        double r2 = ran1s(&seed);
        double g1 = sqrt(2.0*log(1.0/r1))*cos(2*M_PI*r2);
        return g1;
    }; // get_random_number


    // ************** Numerical recipes ran1s ***************
    private: double ran1s(int * idum) {
        // ******************************************************
        static const int IA=16807, IM=2147483647, IQ=127773, IR=2836;
        static const double AM=1.0/IM, RNMX=1.0-1.2e-7;
        if (*idum <= 0) *idum = std::max(-*idum, 1);
        int k = *idum/IQ;
        *idum = IA*(*idum-k*IQ)-IR*k;
        if (*idum < 0) *idum = *idum+IM;
        int iy = *idum;
        double ret = std::min(AM*iy, RNMX); // returns uniform random number in [0,1[
        return ret;
    }; // ran1s


    // ************** Numerical recipes ran2 ****************
    private: double ran2(int * idum) {
        // ******************************************************
        // Long period (> 2 x 10^18) random number generator of L'Ecuyer
        // with Bays-Durham shuffle and added safeguards.
        // Returns a uniform random deviate between 0.0 and 1.0 (exclusive of the endpoint values).
        // Call with idum a negative integer to initialize; thereafter,
        // do not alter idum between successive deviates in a sequence.
        // RNMX should approximate the largest floating value that is less than 1.
        // ******************************************************
        static const int IM1=2147483563, IM2=2147483399, IMM1=IM1-1, IA1=40014, IA2=40692, IQ1=53668,
                        IQ2=52774, IR1=12211, IR2=3791, NTAB=32, NDIV=1+IMM1/NTAB;
        static const double AM=1.0/IM1, RNMX=1.0-1.2e-7;
        static int idum2 = 123456789;
        static int iy = 0;
        static int iv[NTAB];
        int j, k;
        if (*idum <= 0) {
            *idum = std::max(-*idum, 1);
            idum2 = *idum;
            for (j = NTAB+7; j >= 0; j--) {
                k = *idum/IQ1;
                *idum = IA1*(*idum-k*IQ1)-k*IR1;
                if (*idum < 0) *idum += IM1;
                if (j < NTAB) iv[j] = *idum;
            }
            iy = iv[0];
        }
        k = *idum/IQ1;
        *idum = IA1*(*idum-k*IQ1)-k*IR1;
        if (*idum < 0) *idum += IM1;
        k = idum2/IQ2;
        idum2 = IA2*(idum2-k*IQ2)-k*IR2;
        if (idum2 < 0) idum2 += IM2;
        j = iy/NDIV;
        iy = iv[j]-idum2;
        iv[j] = *idum;
        if (iy < 1) iy += IMM1;
        double ret = std::min(AM*iy, RNMX);
        return ret;
    }; // ran2

    protected: void split(const std::string &line, double vals[3], char delimiter, const std::string &param) {
       // ******************************************************
       // Split string into array of doubles depending on delimiter
       // ******************************************************

       std::stringstream ss(line);
       std::string token;
       double val;
       std::vector<double> buffer;
       int param_len;

      while (std::getline(ss, token, delimiter)) { // Split string and store in array
         val = parse_param<double>(token,  param);
         buffer.push_back(val);
      }

       param_len = static_cast<int>(buffer.size());
       if (param_len != ncmp && param_len != 1) { // Raise exception if incorrect number of parameters specified
          if (PE == 0) {
             std::cerr << "Invalid number of parameters specified for " << param << ": " << param_len << std::endl;
          }
          exit(1);
       }

       // Set supplied array
       for (int i=0; i<ncmp; i++) vals[i] = param_len == 1 ? buffer[0] : buffer[i];

    }

    protected:
    template<typename T>
    T parse_param(const std::string &num, const std::string &param) {
       // ******************************************************
       // Templated function to parse integer and double parameter
       // ******************************************************

       T val;
       std::stringstream ss(num);
       if (!(ss >> val)) {
               // We manually throw an exception to trigger your catch block
               throw std::runtime_error("Conversion failed for parameter: " + param);
           }
       ss >> val;


      return val;
    }

    protected: std::string read_from_map(const std::map<std::string, std::string> &params, const std::string &param) {
       // ******************************************************
       // Read parameters form the supplied hash map in the form of strings
       // ******************************************************

       std::map<std::string, std::string>::const_iterator it = params.find(param);

       if (it == params.end()) {
           if (PE == 0) {
              std::cerr << "Parameter not found: " << param << std::endl;
           }
           exit(1);
       }

       std::string val = it->second;

       return val;
    };



    // ******************************************************
    protected: void TurbGen_printf(std::string format, ...) {
        // ******************************************************
        // special printf prepends string and only lets PE=0 print
        // ******************************************************
        va_list args;
        va_start(args, format);
        std::string new_format = ClassSignature+format;
        if (PE == 0) vfprintf(stdout, new_format.c_str(), args);
        va_end(args);
    }; // TurbGen_printf

    // ******************************************************
    protected: void TurbGen_printf_raw(std::string format, ...) {
        // ******************************************************
        // special printf prepends string and only lets PE=0 print
        // ******************************************************
        va_list args;
        va_start(args, format);
        std::string new_format = format;
        if (PE == 0) vfprintf(stdout, new_format.c_str(), args);
        va_end(args);
    }; // TurbGen_printf_raw

}; // end class TurbGen

#endif
// end of TURBULENCE_GENERATOR_H
