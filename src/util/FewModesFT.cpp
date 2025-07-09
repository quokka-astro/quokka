//==============================================================================
// ABOUTME: Gaussian random vector field generator using few-modes inverse FFT
// ABOUTME: Based on AthenaPK implementation for turbulent driving fields
//==============================================================================
/// \file FewModesFT.cpp
/// \brief Implementation of Gaussian random vector field generator

#include "FewModesFT.hpp"
#include <cmath>
#include <iostream>

namespace quokka::util {

FewModesFT::FewModesFT(const std::string &prefix, int num_modes, const std::vector<std::vector<amrex::Real>> &k_vec,
                       amrex::Real k_peak, amrex::Real sol_weight, amrex::Real t_corr, uint32_t rseed,
                       const amrex::BoxArray &ba, const amrex::DistributionMapping &dm, bool fill_ghosts)
    : num_modes_(num_modes), prefix_(prefix), k_vec_(k_vec), k_peak_(k_peak), sol_weight_(sol_weight), t_corr_(t_corr),
      fill_ghosts_(fill_ghosts) {

    if (num_modes > 100) {
        amrex::Print() << "### WARNING: Using more than 100 explicit modes will significantly increase runtime.\n";
        amrex::Print() << "If many modes are required, consider using a full FFT-based driving mechanism.\n";
    }

    AMREX_ALWAYS_ASSERT((sol_weight == -1.0) || (sol_weight >= 0.0 && sol_weight <= 1.0));

    // Initialize arrays
    var_hat_.resize(3);
    var_hat_new_.resize(3);
    random_num_.resize(3);

    for (int n = 0; n < 3; ++n) {
        var_hat_[n].resize(num_modes);
        var_hat_new_[n].resize(num_modes);
        random_num_[n].resize(num_modes);
        for (int m = 0; m < num_modes; ++m) {
            var_hat_[n][m] = Complex(0.0, 0.0);
            var_hat_new_[n][m] = Complex(0.0, 0.0);
            random_num_[n][m].resize(2); // real and imaginary parts
        }
    }

    // Initialize phase arrays
    const int nghost = fill_ghosts ? 1 : 0;
    phases_i_.define(ba, dm, num_modes * 2, nghost); // 2 components for real/imag
    phases_j_.define(ba, dm, num_modes * 2, nghost);
    phases_k_.define(ba, dm, num_modes * 2, nghost);

    // Initialize random number generator
    rng_.seed(rseed);
    dist_ = std::uniform_real_distribution<>(-1.0, 1.0);
}

void FewModesFT::SetPhases(const amrex::Geometry &geom) {
    const auto *prob_lo = geom.ProbLo();
    const auto *prob_hi = geom.ProbHi();
    const auto *dx = geom.CellSize();

    const amrex::Real Lx = prob_hi[0] - prob_lo[0];
    const amrex::Real Ly = prob_hi[1] - prob_lo[1];
    const amrex::Real Lz = prob_hi[2] - prob_lo[2];

    const auto domain = geom.Domain();
    const int gnx1 = domain.length(0);
    const int gnx2 = domain.length(1);
    const int gnx3 = domain.length(2);

    // Check that the domain is cubic and has uniform spacing
    AMREX_ALWAYS_ASSERT(gnx1 == gnx2 && gnx2 == gnx3);
    AMREX_ALWAYS_ASSERT(std::abs(Lx - Ly) < 1e-12 && std::abs(Ly - Lz) < 1e-12);

    const Complex I(0.0, 1.0);

    // Set phases for each direction
    for (amrex::MFIter mfi(phases_i_); mfi.isValid(); ++mfi) {
        const amrex::Box &bx = mfi.validbox();
        const amrex::Box &gbx = mfi.growntilebox();
        
        auto &phases_i_fab = phases_i_[mfi];
        auto &phases_j_fab = phases_j_[mfi];
        auto &phases_k_fab = phases_k_[mfi];

        // Calculate phases for i-direction
        for (int i = gbx.loVect()[0]; i <= gbx.hiVect()[0]; ++i) {
            const amrex::Real gi = static_cast<amrex::Real>(i % gnx1);
            
            for (int m = 0; m < num_modes_; ++m) {
                const amrex::Real w_kx = k_vec_[0][m] * 2.0 * M_PI / static_cast<amrex::Real>(gnx1);
                Complex phase;
                
                // Adjust phase factor for Complex->Real IFT: u_hat*(k) = u_hat(-k)
                if (k_vec_[0][m] == 0.0) {
                    phase = 0.5 * std::exp(I * w_kx * gi);
                } else {
                    phase = std::exp(I * w_kx * gi);
                }
                
                phases_i_fab(amrex::IntVect(i, 0, 0), m * 2) = phase.real();
                phases_i_fab(amrex::IntVect(i, 0, 0), m * 2 + 1) = phase.imag();
            }
        }

        // Calculate phases for j-direction
        for (int j = gbx.loVect()[1]; j <= gbx.hiVect()[1]; ++j) {
            const amrex::Real gj = static_cast<amrex::Real>(j % gnx2);
            
            for (int m = 0; m < num_modes_; ++m) {
                const amrex::Real w_ky = k_vec_[1][m] * 2.0 * M_PI / static_cast<amrex::Real>(gnx2);
                const Complex phase = std::exp(I * w_ky * gj);
                
                phases_j_fab(amrex::IntVect(0, j, 0), m * 2) = phase.real();
                phases_j_fab(amrex::IntVect(0, j, 0), m * 2 + 1) = phase.imag();
            }
        }

        // Calculate phases for k-direction
        for (int k = gbx.loVect()[2]; k <= gbx.hiVect()[2]; ++k) {
            const amrex::Real gk = static_cast<amrex::Real>(k % gnx3);
            
            for (int m = 0; m < num_modes_; ++m) {
                const amrex::Real w_kz = k_vec_[2][m] * 2.0 * M_PI / static_cast<amrex::Real>(gnx3);
                const Complex phase = std::exp(I * w_kz * gk);
                
                phases_k_fab(amrex::IntVect(0, 0, k), m * 2) = phase.real();
                phases_k_fab(amrex::IntVect(0, 0, k), m * 2 + 1) = phase.imag();
            }
        }
    }
}

void FewModesFT::Generate(amrex::MultiFab &mf, amrex::Real dt) {
    const Complex I(0.0, 1.0);

    // Generate random numbers on host to ensure deterministic behavior
    amrex::Real v1 = 0.0;
    amrex::Real v2 = 0.0;
    amrex::Real v_sqr = 0.0;
    for (int n = 0; n < 3; ++n) {
        for (int m = 0; m < num_modes_; ++m) {
            do {
                v1 = dist_(rng_);
                v2 = dist_(rng_);
                v_sqr = v1 * v1 + v2 * v2;
            } while (v_sqr >= 1.0 || v_sqr == 0.0);

            random_num_[n][m][0] = v1;
            random_num_[n][m][1] = v2;
        }
    }

    // Generate new power spectrum (injection)
    for (int n = 0; n < 3; ++n) {
        for (int m = 0; m < num_modes_; ++m) {
            const amrex::Real kx = k_vec_[0][m];
            const amrex::Real ky = k_vec_[1][m];
            const amrex::Real kz = k_vec_[2][m];

            const amrex::Real kmag = std::sqrt(kx * kx + ky * ky + kz * kz);

            var_hat_new_[n][m] = Complex(0.0, 0.0);

            amrex::Real tmp = std::pow(kmag / k_peak_, 2.0) * (2.0 - std::pow(kmag / k_peak_, 2.0));
            if (tmp < 0.0) {
                tmp = 0.0;
            }
            
            const amrex::Real v_sqr_local = random_num_[n][m][0] * random_num_[n][m][0] + 
                                           random_num_[n][m][1] * random_num_[n][m][1];
            const amrex::Real norm = std::sqrt(-2.0 * std::log(v_sqr_local) / v_sqr_local);

            var_hat_new_[n][m] = Complex(tmp * norm * random_num_[n][m][0], tmp * norm * random_num_[n][m][1]);
        }
    }

    // Enforce symmetry for complex to real transform
    for (int n = 0; n < 3; ++n) {
        for (int m = 0; m < num_modes_; ++m) {
            if (k_vec_[0][m] == 0.0) {
                for (int m2 = 0; m2 < m; ++m2) {
                    if (k_vec_[1][m] == -k_vec_[1][m2] && k_vec_[2][m] == -k_vec_[2][m2]) {
                        var_hat_new_[n][m] = Complex(var_hat_new_[n][m2].real(), -var_hat_new_[n][m2].imag());
                    }
                }
            }
        }
    }

    // Apply projection if requested
    if (sol_weight_ >= 0.0) {
        for (int m = 0; m < num_modes_; ++m) {
            const amrex::Real kx = k_vec_[0][m];
            const amrex::Real ky = k_vec_[1][m];
            const amrex::Real kz = k_vec_[2][m];

            amrex::Real kmag = std::sqrt(kx * kx + ky * ky + kz * kz);

            // Avoid division by zero
            if (kmag == 0.0) {
                kmag = 1.0;
            }

            // Make unit vector
            const amrex::Real kx_unit = kx / kmag;
            const amrex::Real ky_unit = ky / kmag;
            const amrex::Real kz_unit = kz / kmag;

            const Complex dot(var_hat_new_[0][m].real() * kx_unit + var_hat_new_[1][m].real() * ky_unit +
                                  var_hat_new_[2][m].real() * kz_unit,
                              var_hat_new_[0][m].imag() * kx_unit + var_hat_new_[1][m].imag() * ky_unit +
                                  var_hat_new_[2][m].imag() * kz_unit);

            var_hat_new_[0][m] = Complex(var_hat_new_[0][m].real() * sol_weight_ + (1.0 - 2.0 * sol_weight_) * dot.real() * kx_unit,
                                         var_hat_new_[0][m].imag() * sol_weight_ + (1.0 - 2.0 * sol_weight_) * dot.imag() * kx_unit);
            var_hat_new_[1][m] = Complex(var_hat_new_[1][m].real() * sol_weight_ + (1.0 - 2.0 * sol_weight_) * dot.real() * ky_unit,
                                         var_hat_new_[1][m].imag() * sol_weight_ + (1.0 - 2.0 * sol_weight_) * dot.imag() * ky_unit);
            var_hat_new_[2][m] = Complex(var_hat_new_[2][m].real() * sol_weight_ + (1.0 - 2.0 * sol_weight_) * dot.real() * kz_unit,
                                         var_hat_new_[2][m].imag() * sol_weight_ + (1.0 - 2.0 * sol_weight_) * dot.imag() * kz_unit);
        }
    }

    // Evolve (Ornstein-Uhlenbeck process)
    const amrex::Real c_drift = std::exp(-dt / t_corr_);
    const amrex::Real c_diff = std::sqrt(1.0 - c_drift * c_drift);

    for (int n = 0; n < 3; ++n) {
        for (int m = 0; m < num_modes_; ++m) {
            var_hat_[n][m] = Complex(var_hat_[n][m].real() * c_drift + var_hat_new_[n][m].real() * c_diff,
                                     var_hat_[n][m].imag() * c_drift + var_hat_new_[n][m].imag() * c_diff);
        }
    }

    // Perform inverse Fourier transform
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box &bx = mfi.validbox();
        auto &mf_fab = mf[mfi];
        const auto &phases_i_fab = phases_i_[mfi];
        const auto &phases_j_fab = phases_j_[mfi];
        const auto &phases_k_fab = phases_k_[mfi];

        for (int k = bx.loVect()[2]; k <= bx.hiVect()[2]; ++k) {
            for (int j = bx.loVect()[1]; j <= bx.hiVect()[1]; ++j) {
                for (int i = bx.loVect()[0]; i <= bx.hiVect()[0]; ++i) {
                    const amrex::IntVect iv(i, j, k);
                    
                    for (int n = 0; n < 3; ++n) {
                        mf_fab(iv, n) = 0.0;
                        
                        for (int m = 0; m < num_modes_; ++m) {
                            const Complex phase_i(phases_i_fab(amrex::IntVect(i, 0, 0), m * 2),
                                                   phases_i_fab(amrex::IntVect(i, 0, 0), m * 2 + 1));
                            const Complex phase_j(phases_j_fab(amrex::IntVect(0, j, 0), m * 2),
                                                   phases_j_fab(amrex::IntVect(0, j, 0), m * 2 + 1));
                            const Complex phase_k(phases_k_fab(amrex::IntVect(0, 0, k), m * 2),
                                                   phases_k_fab(amrex::IntVect(0, 0, k), m * 2 + 1));
                            
                            const Complex phase = phase_i * phase_j * phase_k;
                            mf_fab(iv, n) += 2.0 * (var_hat_[n][m].real() * phase.real() - var_hat_[n][m].imag() * phase.imag());
                        }
                    }
                }
            }
        }
    }
}

auto MakeRandomModes(int num_modes, amrex::Real k_peak, uint32_t rseed) -> std::vector<std::vector<amrex::Real>> {
    std::vector<std::vector<amrex::Real>> k_vec(3);
    for (int i = 0; i < 3; ++i) {
        k_vec[i].resize(num_modes);
    }

    const int k_low = std::floor(k_peak / 2.0);
    const int k_high = std::ceil(2.0 * k_peak);

    std::mt19937 rng;
    rng.seed(rseed);
    std::uniform_int_distribution<> dist(-k_high, k_high);

    int n_mode = 0;
    int n_attempt = 0;
    constexpr int max_attempts = 1000000;
    amrex::Real kx1 = 0.0;
    amrex::Real kx2 = 0.0;
    amrex::Real kx3 = 0.0;
    amrex::Real k_mag = 0.0;
    amrex::Real ampl = 0.0;
    bool mode_exists = false;

    while (n_mode < num_modes && n_attempt < max_attempts) {
        n_attempt++;

        kx1 = dist(rng);
        kx2 = dist(rng);
        kx3 = dist(rng);
        k_mag = std::sqrt(kx1 * kx1 + kx2 * kx2 + kx3 * kx3);

        // Expected amplitude of the spectral function
        ampl = (k_mag / k_peak) * (k_mag / k_peak) * (2.0 - (k_mag / k_peak) * (k_mag / k_peak));

        // Check if mode was already picked
        mode_exists = false;
        for (int n_mode_exist = 0; n_mode_exist < n_mode; ++n_mode_exist) {
            if (k_vec[0][n_mode_exist] == kx1 && k_vec[1][n_mode_exist] == kx2 && k_vec[2][n_mode_exist] == kx3) {
                mode_exists = true;
                break;
            }
        }

        // kx1 < 0.0 because we use an explicit symmetric Complex to Real transform
        if (ampl < 0.0 || k_mag < k_low || k_mag > k_high || mode_exists || kx1 < 0.0) {
            continue;
        }

        k_vec[0][n_mode] = kx1;
        k_vec[1][n_mode] = kx2;
        k_vec[2][n_mode] = kx3;
        n_mode++;
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_attempt < max_attempts,
                                     "MakeRandomModes did not succeed in calculating perturbation modes.");

    return k_vec;
}

} // namespace quokka::util