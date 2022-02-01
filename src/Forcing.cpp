#include "AMReX_Array4.H"
#include "AMReX_FArrayBox.H"

#include "Forcing.H"
#include "MersenneTwister.hpp"
#include "hydro_system.hpp"

using Real = amrex::Real;

int StochasticForcing::verbose = 0;
int StochasticForcing::SpectralRank = 3;

//
//  Default constructor
//
StochasticForcing::StochasticForcing() {
  i1 = i2 = 0;
  j1 = j2 = 0;
  k1 = k2 = 0;
  NumModes = 0;
  NumNonZeroModes = 0;
  decay = 0;
  seed = 27011974;

  SpectProfile = Parabolic;

  AmpltThresh = 1.051250e-1;
  SolenoidalWeight = 1.0;
  DecayInitTime = 0.0;

  for (int dim = 0; dim < MAX_DIMENSION; dim++) {
    alpha[dim] = 2;
    BandWidth[dim] = 1.0;
    IntgrVelocity[dim] = 0.0;
    IntgrLength[dim] = 0.0;
    WaveNumber[dim] = 0.0;
    IntgrTime[dim] = 0.0;
    AutoCorrlTime[dim] = 1.0;

    Amplitude[dim] = nullptr;
    InjectionEven[dim] = nullptr;
    InjectionOdd[dim] = nullptr;
    SpectrumEven[dim] = nullptr;
    SpectrumOdd[dim] = nullptr;
    wavevectors[dim] = nullptr;
    modes_even[dim] = nullptr;
    modes_odd[dim] = nullptr;
  }
  mask = nullptr;
}

/***********************************************************************
 *
 *  STOCHASTIC FORCING CLASS: evolve
 *
 *  written by: Wolfram Schmidt
 *  date:       May, 2005
 *  modified1:  Oct, 2014: updated to support Enzo 2.4 // P. Grete
 *  modified2:  May, 2017: ported to Nyx
 *  modified3:  Feb, 2022: ported to Quokka by Ben Wibking
 *
 *  PURPOSE: evolves the random forcing spectrum in the fashion of
 *           a multi-dimensional Ornstein-Uhlenbeck process
 *
 *           Parameters:
 *           dt -- time step (small compared to AutoCorrlTime)
 *
 *  AUXILIARIES: inject, gauss_deviate, distribute, rms
 *
 ***********************************************************************/

void StochasticForcing::evolve(Real dt) {
  if (amrex::ParallelDescriptor::IOProcessor()) {

    Real DriftCoeff[MAX_DIMENSION];
    Real DiffCoeff[MAX_DIMENSION];

    if (decay == 0) {

      inject();

      /* Increment forcing spectrum (drift and random diffusion)
       * For general properties of Ornstein-Uhlenbeck process, see e.g.
       * Turbulent Flows by Pope (2000) Appendix J with
       * drift and diffusion coefficients given eq (J.41)
       */

      for (int dim = 0; dim < SpectralRank; dim++) {
        DriftCoeff[dim] = exp(-dt / AutoCorrlTime[dim]);
        DiffCoeff[dim] = sqrt(1 - DriftCoeff[dim] * DriftCoeff[dim]);
        for (int n = 0, m = 0; n < NumModes; n++) {
          if (mask[n] != 0) {
            SpectrumEven[dim][m] = DriftCoeff[dim] * SpectrumEven[dim][m] +
                                   DiffCoeff[dim] * InjectionEven[dim][n];
            SpectrumOdd[dim][m] = DriftCoeff[dim] * SpectrumOdd[dim][m] +
                                  DiffCoeff[dim] * InjectionOdd[dim][n];
            ++m;
          }
        }
      }

    } else {

      /* increment forcing spectrum (drift only) */

      for (int dim = 0; dim < SpectralRank; dim++) {
        DriftCoeff[dim] = exp(-dt / AutoCorrlTime[dim]);
        for (int m = 0; m < NumNonZeroModes; m++) {
          SpectrumEven[dim][m] = DriftCoeff[dim] * SpectrumEven[dim][m];
          SpectrumOdd[dim][m] = DriftCoeff[dim] * SpectrumOdd[dim][m];
        }
      }
    }
  }

  /* communicate spectrum among processors */

  distribute();
}

//
// Compute new random injection
//
void StochasticForcing::inject() {
  if (amrex::ParallelDescriptor::IOProcessor()) {

    int i;
    int j;
    int k;
    int n;
    int dim;
    Real a;
    Real b;
    Real contr;

    /* compute Gaussian deviates */

    for (dim = 0; dim < SpectralRank; dim++) {
      for (n = 0; n < NumModes; n++) {
        if (mask[n] != 0) {
          gauss_deviate(Amplitude[dim][n], &a, &b);
        } else {
          a = 0.0;
          b = 0.0;
        }
        InjectionEven[dim][n] = a;
        InjectionOdd[dim][n] = b;
      }
    }

    /* project modes
     * see eq (8) in Schmidt et al., A&A (2009)
     * http://dx.doi.org/10.1051/0004-6361:200809967 */

    for (i = 0; i < i2; i++) { // wave amrex::Vectors in positive x-direction
      InjectionEven[0][i] = (1.0 - SolenoidalWeight) * InjectionEven[0][i];
      InjectionOdd[0][i] = (1.0 - SolenoidalWeight) * InjectionOdd[0][i];
    }

    if (SpectralRank > 1) {

      for (n = 0; n < i2; n++) { // wave amrex::Vectors in positive x-direction
        InjectionEven[1][n] = SolenoidalWeight * InjectionEven[1][n];
        InjectionOdd[1][n] = SolenoidalWeight * InjectionOdd[1][n];
      }

      n = i2;
      for (j = 1; j <= j2; j++) { // wave amrex::Vectors in xy-plane
        for (i = i1; i <= i2; i++) {
          contr = (1.0 - 2.0 * SolenoidalWeight) *
                  (i * InjectionEven[0][n] + j * InjectionEven[1][n]) /
                  Real(i * i + j * j);
          InjectionEven[0][n] =
              SolenoidalWeight * InjectionEven[0][n] + i * contr;
          InjectionEven[1][n] =
              SolenoidalWeight * InjectionEven[1][n] + j * contr;
          contr = (1.0 - 2.0 * SolenoidalWeight) *
                  (i * InjectionOdd[0][n] + j * InjectionOdd[1][n]) /
                  Real(i * i + j * j);
          InjectionOdd[0][n] =
              SolenoidalWeight * InjectionOdd[0][n] + i * contr;
          InjectionOdd[1][n] =
              SolenoidalWeight * InjectionOdd[1][n] + j * contr;
          ++n;
        }
      }

      if (SpectralRank > 2) {

        for (n = 0; n < i2 + j2 * (i2 - i1 + 1);
             n++) { // wave amrex::Vectors in xy-plane
          InjectionEven[2][n] = SolenoidalWeight * InjectionEven[2][n];
          InjectionOdd[2][n] = SolenoidalWeight * InjectionOdd[2][n];
        }

        for (k = 1; k <= k2;
             k++) { // wave amrex::Vectors not aligned to xy-plane
          for (j = j1; j <= j2; j++) {
            for (i = i1; i <= i2; i++) {
              contr = (1.0 - 2.0 * SolenoidalWeight) *
                      (i * InjectionEven[0][n] + j * InjectionEven[1][n] +
                       k * InjectionEven[2][n]) /
                      Real(i * i + j * j + k * k);
              InjectionEven[0][n] =
                  SolenoidalWeight * InjectionEven[0][n] + i * contr;
              InjectionEven[1][n] =
                  SolenoidalWeight * InjectionEven[1][n] + j * contr;
              InjectionEven[2][n] =
                  SolenoidalWeight * InjectionEven[2][n] + k * contr;
              contr = (1.0 - 2.0 * SolenoidalWeight) *
                      (i * InjectionOdd[0][n] + j * InjectionOdd[1][n] +
                       k * InjectionOdd[2][n]) /
                      Real(i * i + j * j + k * k);
              InjectionOdd[0][n] =
                  SolenoidalWeight * InjectionOdd[0][n] + i * contr;
              InjectionOdd[1][n] =
                  SolenoidalWeight * InjectionOdd[1][n] + j * contr;
              InjectionOdd[2][n] =
                  SolenoidalWeight * InjectionOdd[2][n] + k * contr;
              ++n;
            }
          }
        }
      }
    }
  }
}

//
// Generate couple of normally distributed random deviates
// (Box-Muller-Algorithm)
//
void StochasticForcing::gauss_deviate(Real amplt, Real *x, Real *y) {
  Real v_sqr;
  Real v1;
  Real v2;
  Real norm;

  do {
    v1 = 2.0 * static_cast<Real>(mt_random() % 2147483563) / (2147483563.0) -
         1.0;
    v2 = 2.0 * static_cast<Real>(mt_random() % 2147483563) / (2147483563.0) -
         1.0;
    v_sqr = v1 * v1 + v2 * v2;
  } while (v_sqr >= 1.0 || v_sqr == 0.0);

  norm = amplt * sqrt(-2.0 * log(v_sqr) / v_sqr);

  *x = norm * v1;
  *y = norm * v2;
}

//
// Distribute the spectrum
//
void StochasticForcing::distribute(void) {
  /* communicate spectrum among processors */

  for (int dim = 0; dim < SpectralRank; dim++) {
    amrex::ParallelDescriptor::Bcast(
        SpectrumEven[dim], NumNonZeroModes,
        amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::Bcast(
        SpectrumOdd[dim], NumNonZeroModes,
        amrex::ParallelDescriptor::IOProcessorNumber());
  }

  /* copy sepctrum to forcing_spect_module */

  for (int dim = 0; dim < SpectralRank; dim++) {
    for (int l = 0; l < NumNonZeroModes; l++) {
      modes_even[dim][l] = SpectrumEven[dim][l];
      modes_odd[dim][l] = SpectrumOdd[dim][l];
    }
  }
}

//
// Compute RMS magnitude
//
auto StochasticForcing::rms() -> Real {
  int m;
  Real sum_even = 0.0;
  Real sum_odd = 0.0;

  for (int dim = 0; dim < SpectralRank; dim++) {
    for (m = 0; m < NumNonZeroModes; m++) {
      sum_even += SpectrumEven[dim][m] * SpectrumEven[dim][m];
    }
    for (m = 0; m < NumNonZeroModes; m++) {
      sum_odd += SpectrumOdd[dim][m] * SpectrumOdd[dim][m];
    }
  }

  return sqrt(sum_even + sum_odd);
}

void StochasticForcing::integrate_state_force(
    amrex::Box const &bx, amrex::Array4<Real> const &state,
    amrex::Array4<Real> const & /*diag_eos*/,
    amrex::GeometryData const &geomdata, Real /*a*/, Real dt,
    Real /*small_eint*/, Real /*small_temp*/) {
  int mi = 0;
  int mj = 0;
  int mk = 0;
  int num_phases[3];
  Real delta_phase[3];
  Real phase_lo[3];
  Real accel[3];

  int num_modes = NumNonZeroModes;

  amrex::Vector<Real> buf(num_modes);
  amrex::Vector<Real> phasefct_init_even(num_modes);
  amrex::Vector<Real> phasefct_init_odd(num_modes);

  amrex::Vector<Real> phasefct_mult_even_x(num_modes);
  amrex::Vector<Real> phasefct_mult_even_y(num_modes);
  amrex::Vector<Real> phasefct_mult_even_z(num_modes);

  amrex::Vector<Real> phasefct_mult_odd_x(num_modes);
  amrex::Vector<Real> phasefct_mult_odd_y(num_modes);
  amrex::Vector<Real> phasefct_mult_odd_z(num_modes);
  amrex::Vector<Real> phasefct_yz0(num_modes);
  amrex::Vector<Real> phasefct_yz1(num_modes);

  Real *phasefct_even_x = nullptr;
  Real *phasefct_even_y = nullptr;
  Real *phasefct_even_z = nullptr;
  Real *phasefct_odd_x = nullptr;
  Real *phasefct_odd_y = nullptr;
  Real *phasefct_odd_z = nullptr;

  Real alpha_const = 100.0;
  Real temp0_const = 10.0;

  const auto prob_hi = geomdata.ProbHi();
  const auto prob_lo = geomdata.ProbLo();
  const auto dx = geomdata.CellSize();
  for (int dim = 0; dim < SpectralRank; dim++) {
    delta_phase[dim] =
        2.0 * M_PI * dx[dim] /
        (prob_hi[dim] - prob_lo[dim]); // phase increment per cell
    phase_lo[dim] = (double(bx.smallEnd(dim)) + 0.5) *
                    delta_phase[dim]; // phase of low corner
    num_phases[dim] = (bx.bigEnd(dim) - bx.smallEnd(dim) + 1) * num_modes;
  }

  phasefct_even_x = new Real[num_phases[0]];
  phasefct_even_y = new Real[num_phases[1]];
  phasefct_even_z = new Real[num_phases[2]];
  phasefct_odd_x = new Real[num_phases[0]];
  phasefct_odd_y = new Real[num_phases[1]];
  phasefct_odd_z = new Real[num_phases[2]];

  for (int m = 0; m < num_modes; m++) {
    int i = wavevectors[0][m];
    int j = wavevectors[1][m];
    int k = wavevectors[2][m];
    phasefct_init_even[m] = (cos(i * phase_lo[0]) * cos(j * phase_lo[1]) -
                             sin(i * phase_lo[0]) * sin(j * phase_lo[1])) *
                                cos(k * phase_lo[2]) -
                            (cos(i * phase_lo[0]) * sin(j * phase_lo[1]) +
                             sin(i * phase_lo[0]) * cos(j * phase_lo[1])) *
                                sin(k * phase_lo[2]);

    phasefct_init_odd[m] = (cos(i * phase_lo[0]) * cos(j * phase_lo[1]) -
                            sin(i * phase_lo[0]) * sin(j * phase_lo[1])) *
                               sin(k * phase_lo[2]) +
                           (cos(i * phase_lo[0]) * sin(j * phase_lo[1]) +
                            sin(i * phase_lo[0]) * cos(j * phase_lo[1])) *
                               cos(k * phase_lo[2]);

    phasefct_mult_even_x[m] = cos(i * delta_phase[0]);
    phasefct_mult_odd_x[m] = sin(i * delta_phase[0]);

    phasefct_mult_even_y[m] = cos(j * delta_phase[1]);
    phasefct_mult_odd_y[m] = sin(j * delta_phase[1]);

    phasefct_mult_even_z[m] = cos(k * delta_phase[2]);
    phasefct_mult_odd_z[m] = sin(k * delta_phase[2]);
  }

  // initialize phase factors for each coordinate axis:
  // since phase factors for inverse FT are given by
  // exp(i*(k1*x + k2*y + k3*z)) = exp(i*k1*x) * exp(i*k2*y)*...,
  // we iteratively multiply with exp(i*k1*delta_x), etc.
  for (int m = 0; m < num_modes; m++) {
    phasefct_even_x[m] = 1.0;
    phasefct_odd_x[m] = 0.0;
  }

  for (int i = bx.smallEnd(0) + 1; i <= bx.bigEnd(0); i++) {
    mi = (i - bx.smallEnd(0)) * num_modes;
    for (int m = 0; m < num_modes; m++) {
      buf[m] = phasefct_even_x[mi - num_modes];
      phasefct_even_x[mi] =
          phasefct_mult_even_x[m] * phasefct_even_x[mi - num_modes] -
          phasefct_mult_odd_x[m] * phasefct_odd_x[mi - num_modes];
      phasefct_odd_x[mi] =
          phasefct_mult_even_x[m] * phasefct_odd_x[mi - num_modes] +
          phasefct_mult_odd_x[m] * buf[m];
      mi = mi + 1;
    }
  }

  for (int m = 0; m < num_modes; m++) {
    phasefct_even_y[m] = 1.0;
    phasefct_odd_y[m] = 0.0;
  }

  for (int j = bx.smallEnd(1) + 1; j <= bx.bigEnd(1); j++) {
    mj = (j - bx.smallEnd(1)) * num_modes;
    for (int m = 0; m < num_modes; m++) {
      buf[m] = phasefct_even_y[mj - num_modes];
      phasefct_even_y[mj] =
          phasefct_mult_even_y[m] * phasefct_even_y[mj - num_modes] -
          phasefct_mult_odd_y[m] * phasefct_odd_y[mj - num_modes];
      phasefct_odd_y[mj] =
          phasefct_mult_even_y[m] * phasefct_odd_y[mj - num_modes] +
          phasefct_mult_odd_y[m] * buf[m];
      mj = mj + 1;
    }
  }

  for (int m = 0; m < num_modes; m++) {
    phasefct_even_z[m] = phasefct_init_even[m];
    phasefct_odd_z[m] = phasefct_init_odd[m];
  }

  for (int k = bx.smallEnd(2) + 1; k <= bx.bigEnd(2); k++) {
    mk = (k - bx.smallEnd(2)) * num_modes;
    for (int m = 0; m < num_modes; m++) {
      buf[m] = phasefct_even_z[mk - num_modes];
      phasefct_even_z[mk] =
          phasefct_mult_even_z[m] * phasefct_even_z[mk - num_modes] -
          phasefct_mult_odd_z[m] * phasefct_odd_z[mk - num_modes];
      phasefct_odd_z[mk] =
          phasefct_mult_even_z[m] * phasefct_odd_z[mk - num_modes] +
          phasefct_mult_odd_z[m] * buf[m];
      mk = mk + 1;
    }
  }

  // apply forcing in physical space
  // TODO(bwibking): fix this to run on GPU

  for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); k++) {
    for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); j++) {
      mj = (j - bx.smallEnd(1)) * num_modes;
      mk = (k - bx.smallEnd(2)) * num_modes;

      // pre-compute products of phase factors depending on y- and z-coordinates
      for (int m = 0; m < num_modes; m++) {
        phasefct_yz0[m] = phasefct_even_y[mj] * phasefct_even_z[mk] -
                          phasefct_odd_y[mj] * phasefct_odd_z[mk];
        phasefct_yz1[m] = phasefct_odd_y[mj] * phasefct_even_z[mk] +
                          phasefct_even_y[mj] * phasefct_odd_z[mk];
        mj = mj + 1;
        mk = mk + 1;
      }

      for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); i++) {

        accel[0] = 0.0;
        accel[1] = 0.0;
        accel[2] = 0.0;

        // compute components of acceleration via inverse FT
        for (int n = 0; n < SpectralRank; n++) {
          mi = (i - bx.smallEnd(0)) * num_modes;

          for (int m = 0; m < num_modes; m++) {
            // sum up even modes
            accel[n] = accel[n] + (phasefct_even_x[mi] * phasefct_yz0[m] -
                                   phasefct_odd_x[mi] * phasefct_yz1[m]) *
                                      modes_even[n][m];
            // sum up odd modes
            accel[n] = accel[n] - (phasefct_even_x[mi] * phasefct_yz1[m] +
                                   phasefct_odd_x[mi] * phasefct_yz0[m]) *
                                      modes_odd[n][m];
            mi = mi + 1;
          }

          accel[n] = M_SQRT2 * accel[n];
        }

        auto compute_KE = [=](int i, int j, int k) {
          const amrex::Real rho = state(i, j, k, density_index);
          const amrex::Real px = state(i, j, k, x1Momentum_index);
          const amrex::Real py = state(i, j, k, x2Momentum_index);
          const amrex::Real pz = state(i, j, k, x3Momentum_index);
          return (px * px + py * py + pz * pz) / (2.0 * rho);
        };

        // compute old kinetic energy
        amrex::Real old_KE = compute_KE(i, j, k);

        // add forcing
        const amrex::Real rho = state(i, j, k, density_index);
        state(i, j, k, x1Momentum_index) += dt * rho * accel[0];
        state(i, j, k, x2Momentum_index) += dt * rho * accel[1];
        state(i, j, k, x3Momentum_index) += dt * rho * accel[2];

        // compute new kinetic energy
        amrex::Real new_KE = compute_KE(i, j, k);

        // update total energy
        state(i, j, k, energy_index) += (new_KE - old_KE);
      }
    }
  }
}
