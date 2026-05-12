//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDDisk.cpp
/// \brief Defines a simulation using disk galaxy initial conditions.
///

#include <cmath>
#include <cstring>
#include <fstream>
#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArrayBase.H"

#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

// Binary file read in for initial potential field (Aphi_2d). This field is R*Aphi in cylindrical coordinates,
//  but stored as a 2D array in (R,z) with dimensions (nR, nz). The potential is initialized from this field using 
// finite differences, and then the gravitational acceleration is applied as a source term in the momentum and 
// energy equations. See init_seed_pot_field.py for details on how this file is generated.


namespace
{
	constexpr double keV_in_ergs = 1000.0 * C::ev2erg; 	// ergs == 1 keV
	constexpr double seconds_per_year = 3.15576e7;
	constexpr double Rd_kpc = 3.0; 						// disk scale length in kpc
	constexpr double Rc_kpc = 2.0; 						// rotation curve core radius in kpc
	constexpr double Rd = Rd_kpc * 1.0e3 * C::parsec;
	constexpr double Rc = Rc_kpc * 1.0e3 * C::parsec;
	constexpr double alpha_profile = 2.0;               // Eq. 1 shape parameter
	constexpr double beta_profile  = 0.5;               // Eq. 1 shape parameter
	constexpr double q_flatten     = 0.7;               // Binney & Tremaine flattening (Arora+25 Eq. A.3)
	constexpr double rho_transition = 1.0e-28; 			// g/cm^3, disc-CGM interface density
	constexpr double Rmax_kpc = 8.0;
	constexpr double Rmax = Rmax_kpc * 1.0e3 * C::parsec;
	constexpr double refine_Rcyl_kpc = 6.0;
	constexpr double refine_Hcyl_pc  = 300.0;
	constexpr double refine_Rcyl     = refine_Rcyl_kpc * 1.0e3 * C::parsec;
	constexpr double refine_Hcyl     = refine_Hcyl_pc  * C::parsec;
} // namespace

struct MHDGalaxy {
};

static_assert(AMREX_SPACEDIM == 3, "MHD disk galaxy problem requires AMREX_SPACEDIM == 3.");

template <> struct quokka::EOS_Traits<MHDGalaxy> {
	static constexpr double gamma = 1.0001;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double T_cgm =  1.0e7;// K, already defined in anonymous namespace
	static constexpr double cs_cgm = gcem::sqrt(gamma * C::k_B * T_cgm / mean_molecular_weight);
	static constexpr double cs_disk = 7.0e5; // disk sound speed [cm/s]
};

template <> struct HydroSystem_Traits<MHDGalaxy> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Particle_Traits<MHDGalaxy> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
};

template <> struct Physics_Traits<MHDGalaxy> {
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 0;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr int nGroups = 1;
};

template <> struct SimulationData<MHDGalaxy> {
	// Primary dimensionless parameters (Table 1 of Arora+25)
	amrex::Real Q_mean{};
	amrex::Real Mc{};
	amrex::Real vc{};       // saturated circular velocity [cm/s]
	amrex::Real Sigma0{};   // surface density normalisation [g/cm^2]
	amrex::Real rho_cgm{};
	amrex::Real n_cell{};
	amrex::Real max_level{};

	amrex::Gpu::PinnedVector<amrex::Real> RAphi_2d; // (nR, nz)
	amrex::Gpu::DeviceVector<amrex::Real> RA_device;

	int seed_nR{}, seed_nz{};
	amrex::Real seed_Rmax{}; 
	amrex::Real seed_Lz{};
	amrex::Real seed_B0_gauss{};
	amrex::Real seed_B0_HL{};
	amrex::Real seed{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto surfaceDensityProfile(double R, double Sigma0) -> double
{
	const double x = R / Rd;
	return Sigma0 * std::exp(-x - beta_profile * std::exp(-alpha_profile * x));
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto diskDensityAnalytic(double R, double z,
                           double Sigma0,
                           double vc,
                           double cs) -> double
{
    const double Sigma = surfaceDensityProfile(R, Sigma0);
    if (Sigma <= 0.0) {
        return 0.0;
    }

    // Isothermal disk scale height
    const double H = cs*cs / (M_PI * C::Gconst * Sigma);

    // Midplane density from exact normalization: Sigma = 2 rho0 H
    const double rho0 =
        (M_PI * C::Gconst * Sigma * Sigma) / (2.0 * cs*cs);

	// Disk self-gravity (sech^2 profile)
	const double sech = 1.0 / std::cosh(z / H);
	const double disk_factor = sech * sech;

    // Vertical confinement from flattened halo (exact ΔΦ)
    const double denom = R*R + Rc*Rc;
    const double halo_factor =
        pow(
            1.0 + (z*z) / (q_flatten*q_flatten * denom),
            -vc*vc / (2.0 * cs*cs)
        );

    return rho0 * disk_factor * halo_factor;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto sample_RA(
    const amrex::Real* RA,
    int nR,
    int nz,
    double Rmax,
    double Lz,
    double R,
    double z)-> double	
{
    const double dR = Rmax / static_cast<double>(nR);
    const double dz = Lz   / static_cast<double>(nz);

    const double zmin = -0.5 * Lz;

    // clamp
    R = amrex::max(0.0,
        amrex::min(R, Rmax - 1e-12));

    z = amrex::max(zmin,
        amrex::min(z, zmin + Lz - 1e-12));

    const double u = R / dR - 0.5;
    const double v = (z - zmin) / dz - 0.5;

    int iR = static_cast<int>(amrex::Math::floor(u));
    int iz = static_cast<int>(amrex::Math::floor(v));

    const double fu = u - iR;
    const double fv = v - iz;

    iR = amrex::max(0, amrex::min(iR, nR - 2));
    iz = amrex::max(0, amrex::min(iz, nz - 2));

    auto idx = [=](int ir, int izz)
    {
        return ir * nz + izz;
    };

    const double f00 = RA[idx(iR  , iz  )];
    const double f10 = RA[idx(iR+1, iz  )];
    const double f01 = RA[idx(iR  , iz+1)];
    const double f11 = RA[idx(iR+1, iz+1)];

    return
        (1.0-fu)*(1.0-fv)*f00 +
        fu      *(1.0-fv)*f10 +
        (1.0-fu)*fv      *f01 +
        fu      *fv      *f11;
}

template <> void QuokkaSimulation<MHDGalaxy>::preCalculateInitialConditions()
{
	amrex::ParmParse const pp("mhd_galaxy");
    pp.get("Mc",     userData_.Mc);
    pp.get("Q_mean", userData_.Q_mean);

	amrex::ParmParse const pp_amr("amr");
    amrex::Vector<int> n_cell_vec(3);
	pp_amr.getarr("n_cell", n_cell_vec);
	// Take the maximum across dimensions to be conservative
	const int n_cell_max = *std::max_element(n_cell_vec.begin(), n_cell_vec.end());
	userData_.n_cell = static_cast<amrex::Real>(n_cell_max);
    pp_amr.get("max_level", userData_.max_level);
	
	constexpr double cs_disk = quokka::EOS_Traits<MHDGalaxy>::cs_disk;
    constexpr double cs_cgm  = quokka::EOS_Traits<MHDGalaxy>::cs_cgm;

    userData_.vc = userData_.Mc * cs_disk;
    const double vc = userData_.vc;

    // Toomre Q integral uses cs_disk
    auto integrand = [=](double R) -> double {
        const double D = R * R + Rc * Rc;
		const double sqrtD = std::sqrt(D);
		const double Omega    = vc / sqrtD;
		const double dOmegadR = -vc * R / (D * sqrtD);
        const double kappa    = std::sqrt(std::max(4.0 * Omega * Omega + 2.0 * R * Omega * dOmegadR, 0.0));
        const double fR       = surfaceDensityProfile(R, 1.0);
        return kappa * cs_disk / (M_PI * C::Gconst * fR);
    };

	constexpr int N = 1000;
	static_assert(N % 2 == 0, "Simpson's rule requires even N");
	const double a = 0.0;
	const double b = Rmax;
	const double h = (b - a) / N;

	double integral = integrand(a) + integrand(b);
	for (int i = 1; i < N; ++i) {
		const double R = a + i * h;
		integral += (i % 2 == 0 ? 2.0 : 4.0) * integrand(R);
	}
	integral *= h / 3.0;
	userData_.Sigma0 = integral / (userData_.Q_mean * Rmax);
    // Pressure matching: P = cs_disk^2 * rho_transition = cs_cgm^2 * rho_cgm
    userData_.rho_cgm = rho_transition * (cs_disk * cs_disk) / (cs_cgm * cs_cgm);

	if (userData_.RAphi_2d.empty()) {
		std::ifstream meta("Aphi_2d_meta.txt");
		std::string line;
		while (std::getline(meta, line)) {
			const auto eq = line.find('=');
			if (eq == std::string::npos) {
				continue;
			}
			std::string key = line.substr(0, eq);
			std::string val = line.substr(eq + 1);
			auto trim = [](std::string &s) {
				s.erase(0, s.find_first_not_of(" \t"));
				s.erase(s.find_last_not_of(" \t") + 1);
			};
			trim(key);
			trim(val);
			try {
				if (key == "nR") {
					userData_.seed_nR = std::stoi(val);
				}
				else if (key == "nz") {
					userData_.seed_nz = std::stoi(val);
				}
				else if (key == "Rmax_cm") {
					userData_.seed_Rmax = std::stod(val);
				}
				else if (key == "Lz_cm") {
					userData_.seed_Lz = std::stod(val);
				}
				else if (key == "B0_gauss") {
					userData_.seed_B0_gauss = std::stod(val);
				}
				else if (key == "B0_HL") {
					userData_.seed_B0_HL = std::stod(val);
				}
				else if (key == "seed") {
					userData_.seed = std::stod(val);
				}
			} catch (const std::exception &e) {
				amrex::Print()
					<< "Warning: failed to parse line:\n"
					<< line << "\n";
			}
		}
		const std::size_t n_tot =
			static_cast<std::size_t>(userData_.seed_nR) *
			static_cast<std::size_t>(userData_.seed_nz);
		userData_.RAphi_2d.resize(n_tot);
		const std::size_t bytes = n_tot * sizeof(amrex::Real);
		std::ifstream bin("Aphi_2d.bin", std::ios::binary);
		if (!bin) {
			amrex::Abort("Failed to open Aphi_2d.bin");
		}
		std::vector<char> raw(bytes);
		bin.read(raw.data(), static_cast<std::streamsize>(bytes));
		if (!bin) {
			amrex::Abort("Error reading Aphi_2d.bin");
		}
		std::memcpy(userData_.RAphi_2d.data(), raw.data(), bytes);
		userData_.RA_device.resize(n_tot);
		amrex::Gpu::copy(
			amrex::Gpu::hostToDevice,
			userData_.RAphi_2d.begin(),
			userData_.RAphi_2d.end(),
			userData_.RA_device.begin()
		);
		amrex::Gpu::synchronize();
		amrex::Print()
			<< "Loaded Aphi_2d.bin with "
			<< n_tot << " values.\n";
		amrex::Print() << "MHDGalaxy:"
					<< " Q_mean = "    << userData_.Q_mean
					<< ", Mc = "       << userData_.Mc
					<< ", vc = "       << vc / 1.0e5        << " km/s"
					<< ", cs_disk = "  << cs_disk / 1.0e5   << " km/s"
					<< ", cs_cgm = "   << cs_cgm  / 1.0e5   << " km/s"
					<< ", Sigma0 = "   << userData_.Sigma0   << " g/cm^2"
					<< ", rho_cgm = "  << userData_.rho_cgm  << " g/cm^3"
					<< ", B0 = " << userData_.seed_B0_gauss * 1.0e6 << " μG"
					<< ", seed = "     << userData_.seed     << "\n";
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.seed_nR > 0 && userData_.seed_nz > 0,
	"Aphi_2d_meta.txt was not parsed correctly — nR/nz are zero");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.seed_B0_HL > 0.0,
	"B0_gauss not loaded from Aphi_2d_meta.txt");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.seed_nR > 0 && userData_.seed_nz > 0,
	"Aphi_2d_meta.txt was not parsed correctly — nR/nz are zero");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.seed_B0_HL > 0.0,
		"B0_HL not loaded from Aphi_2d_meta.txt");
	const double n_fine = userData_.n_cell * std::pow(2.0, userData_.max_level);
	const double min_required_nR = n_fine * std::numbers::sqrt2 / 2.0; // 2*nR*sqrt(2) >= n_fine
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		static_cast<double>(userData_.seed_nR) >= min_required_nR,
		"Aphi_2d table radial resolution (nR) is coarser than the finest AMR level — "
		"regenerate Aphi_2d.bin with higher resolution");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		static_cast<double>(userData_.seed_nz) >= n_fine,
		"Aphi_2d table vertical resolution (nz) is coarser than the finest AMR level — "
		"regenerate Aphi_2d.bin with higher resolution");
}

template <> void QuokkaSimulation<MHDGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const double vc      = userData_.vc;
	const double Sigma0  = userData_.Sigma0;
	const double cs_disk = quokka::EOS_Traits<MHDGalaxy>::cs_disk;
	const double cs_cgm  = quokka::EOS_Traits<MHDGalaxy>::cs_cgm;
	const double rho_cgm = userData_.rho_cgm;
	constexpr double gamma = quokka::EOS_Traits<MHDGalaxy>::gamma;
	const double B0 = userData_.seed_B0_HL;

	const amrex::Box &indexRange                                = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx       = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc                        = grid_elem.array_;


	const amrex::Real* RA_ptr = userData_.RA_device.dataPtr();
	const int nR   = userData_.seed_nR;
	const int nz   = userData_.seed_nz;
	const double Rmax = userData_.seed_Rmax;
	const double Lz   = userData_.seed_Lz;


	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		const double y = prob_lo[1] + (j + 0.5) * dx[1];
		const double z = prob_lo[2] + (k + 0.5) * dx[2];
		const double R = std::sqrt(x * x + y * y);
		const double rho_disc_raw = diskDensityAnalytic(R, z, Sigma0, vc, cs_disk);
		const double Sigma_R = surfaceDensityProfile(R, Sigma0);

		// Two-phase assignment (Disk vs CGM)
		const bool in_disk = (rho_disc_raw > rho_transition);
		const double rho =
			in_disk ? amrex::max(rho_disc_raw, rho_transition*1e-6)
					: rho_cgm;
		const double cs  = in_disk ? cs_disk : cs_cgm;

		// Rotation velocity (Arora+25 Eq. A.4)
		double vrot = 0.0;
		if (R > 0.0) {
			vrot = vc * R / std::sqrt(R*R + Rc*Rc);
		}

		// Velocity components
		double vx = 0.0;
		double vy = 0.0;
		if (in_disk && R > 0.0) {
			vx = -vrot * y / R;
			vy =  vrot * x / R;
		}

		// 8. Conserved variables
		const double pressure = rho * cs * cs;
		const double Eint     = pressure / (gamma - 1.0);
		const double Ekin     = 0.5 * rho * (vx * vx + vy * vy);

		// --- reconstruct B at cell center ---

		const double eps = 1e-12;

		// finite differences
		const double dz_fd = dx[2];
		const double dR    = dx[0];

		// RA samples
		const double RA_zp = sample_RA(RA_ptr, nR, nz, Rmax, Lz, R, z + 0.5*dz_fd);
		const double RA_zm = sample_RA(RA_ptr, nR, nz, Rmax, Lz, R, z - 0.5*dz_fd);

		double BR = 0.0;
		if (R > eps) {
			BR = -(RA_zp - RA_zm) / (dz_fd * R);
		}

		double BZ = 0.0;
		if (R > eps) {
			const double ex = x / R;
			const double ey = y / R;

			const double Rp = std::sqrt((x + 0.5*dR*ex)*(x + 0.5*dR*ex) +
										(y + 0.5*dR*ey)*(y + 0.5*dR*ey));

			const double Rm = std::sqrt((x - 0.5*dR*ex)*(x - 0.5*dR*ex) +
										(y - 0.5*dR*ey)*(y - 0.5*dR*ey));

			const double RA_Rp = sample_RA(RA_ptr, nR, nz, Rmax, Lz, Rp, z);
			const double RA_Rm = sample_RA(RA_ptr, nR, nz, Rmax, Lz, Rm, z);

			BZ = (RA_Rp - RA_Rm) / (dR * R);
		}

		// convert cylindrical → Cartesian
		double Bx = 0.0;
		double By = 0.0;

		if (R > eps) {
			Bx = BR * x / R;
			By = BR * y / R;
		}
		Bx = Bx * B0;
		By = By * B0;
		BZ = BZ * B0;
		
		const double Emag = 0.5 * (Bx*Bx + By*By + BZ*BZ);

		state_cc(i, j, k, HydroSystem<MHDGalaxy>::density_index)        = rho;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index)     = rho * vx;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index)     = rho * vy;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index)     = 0.0;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::energy_index)         = Ekin + Eint + Emag;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::internalEnergy_index) = Eint;
	});
}


template <> void QuokkaSimulation<MHDGalaxy>::setInitialConditionsOnGridFaceVars(
    quokka::grid const &grid_elem)
{
    const amrex::Array4<double> &state_fc = grid_elem.array_;
    const amrex::Box &indexRange = grid_elem.indexRange_;
    const quokka::direction dir = grid_elem.dir_;
	const double B0 = userData_.seed_B0_HL;

    const auto dx = grid_elem.dx_;
    const auto prob_lo = grid_elem.prob_lo_;

    constexpr int mhd_index =
        Physics_Indices<MHDGalaxy>::mhdFirstIndex;

    // ---- device pointer to RA ----
    const amrex::Real* RA_ptr = userData_.RA_device.dataPtr();
    const int nR = userData_.seed_nR;
    const int nz = userData_.seed_nz;
    const double Rmax = userData_.seed_Rmax;
    const double Lz   = userData_.seed_Lz;

    constexpr double eps = 1e-12;

    amrex::ParallelFor(indexRange,
    [=] AMREX_GPU_DEVICE(int i, int j, int k)
    {
        const double x =
            prob_lo[0] + i * dx[0] +
            (dir == quokka::direction::x ? 0.0 : 0.5 * dx[0]);

        const double y =
            prob_lo[1] + j * dx[1] +
            (dir == quokka::direction::y ? 0.0 : 0.5 * dx[1]);

        const double z =
            prob_lo[2] + k * dx[2] +
            (dir == quokka::direction::z ? 0.0 : 0.5 * dx[2]);

        const double R = std::sqrt(x*x + y*y);
        const double dR = dx[0];   // characteristic radial step
        const double dz_fd = dx[2];
        const double RA_zp = sample_RA(RA_ptr, nR, nz, Rmax, Lz,
                                      R, z + 0.5*dz_fd);
        const double RA_zm = sample_RA(RA_ptr, nR, nz, Rmax, Lz,
                                      R, z - 0.5*dz_fd);
        double BR = 0.0;
        if (R > eps) {
            BR = -(RA_zp - RA_zm) / (dz_fd * R);
        }
        double BZ = 0.0;
        if (R > eps) {
            const double ex = x / R;
            const double ey = y / R;

            const double xp = x + 0.5 * dR * ex;
            const double yp = y + 0.5 * dR * ey;

            const double xm = x - 0.5 * dR * ex;
            const double ym = y - 0.5 * dR * ey;

            const double Rp = std::sqrt(xp*xp + yp*yp);
            const double Rm = std::sqrt(xm*xm + ym*ym);

            const double RA_Rp =
                sample_RA(RA_ptr, nR, nz, Rmax, Lz, Rp, z);

            const double RA_Rm =
                sample_RA(RA_ptr, nR, nz, Rmax, Lz, Rm, z);
            BZ = (RA_Rp - RA_Rm) / (dR * R);
        }

        double Bx = 0.0;
        double By = 0.0;

        if (R > eps) {
            Bx = BR * x / R;
            By = BR * y / R;
        }

        if (dir == quokka::direction::x) {
            state_fc(i,j,k,mhd_index) = Bx * B0;
        } else if (dir == quokka::direction::y) {
            state_fc(i,j,k,mhd_index) = By * B0;
        } else if (dir == quokka::direction::z) {
            state_fc(i,j,k,mhd_index) = BZ * B0;
        }
    });
}



template <> void QuokkaSimulation<MHDGalaxy>::addStrangSplitSources(
    amrex::MultiFab &mf, int lev, amrex::Real /*time*/, amrex::Real dt_lev)
{
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx      = geom[lev].CellSizeArray();
    const double vc = userData_.vc;

    for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
        const amrex::Box &indexRange = iter.validbox();
        auto const &state = mf.array(iter);

        amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const double x  = prob_lo[0] + (i + 0.5) * dx[0];
            const double y  = prob_lo[1] + (j + 0.5) * dx[1];
            const double z  = prob_lo[2] + (k + 0.5) * dx[2];
            const double R2 = x*x + y*y;
            const double R  = std::sqrt(R2);

            const double rho   = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
            const double px    = state(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index);
            const double py    = state(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index);
            const double pz    = state(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index);
            const double Eint  = state(i, j, k, HydroSystem<MHDGalaxy>::internalEnergy_index);
            const double Etot_old = state(i, j, k, HydroSystem<MHDGalaxy>::energy_index);
            const double Ekin_old = 0.5 * (px*px + py*py + pz*pz) / rho;
            const double Emag     = Etot_old - Ekin_old - Eint;  // Etot = Ekin + Eint + Emag

            // Arora+25 Eq. A.3).  Gas self-gravity is handled by the Poisson solver
            const double D = R2 + Rc*Rc + (z/q_flatten)*(z/q_flatten);

            // g_R and g_z are the cylindrical acceleration components.
            // They are zero at R=0 / z=0 by symmetry.
            const double g_R = (R > 0.0) ? -(vc*vc * R / D) : 0.0;
            const double g_z =             -(vc*vc * z / (q_flatten*q_flatten * D));

            const double gx = (R > 0.0) ? g_R * x / R : 0.0;
            const double gy = (R > 0.0) ? g_R * y / R : 0.0;

            // Momentum kick  (p_new = p_old + dt * rho * g)
            const double px_new = px + dt_lev * rho * gx;
            const double py_new = py + dt_lev * rho * gy;
            const double pz_new = pz + dt_lev * rho * g_z;

            const double Ekin_new = 0.5 * (px_new*px_new + py_new*py_new + pz_new*pz_new) / rho;
            const double Etot_new = Ekin_new + Eint + Emag;

            state(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) = px_new;
            state(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) = py_new;
            state(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index) = pz_new;
            state(i, j, k, HydroSystem<MHDGalaxy>::energy_index)     = Etot_new;
        });
    }
}


template <> void QuokkaSimulation<MHDGalaxy>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real x0 = prob_lo[0] + (i * dx[0]);
		const amrex::Real y0 = prob_lo[1] + (j * dx[1]);
		const amrex::Real z0 = prob_lo[2] + (k * dx[2]);
		const amrex::Real x1 = prob_lo[0] + ((i + 1) * dx[0]);
		const amrex::Real y1 = prob_lo[1] + ((j + 1) * dx[1]);
		const amrex::Real z1 = prob_lo[2] + ((k + 1) * dx[2]);

		auto tagIfPointInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			const amrex::Real R = std::sqrt(x * x + y * y);
			if ((R < refine_Rcyl) && (std::abs(z) < refine_Hcyl)) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};
		for (auto const &x : {x0, x1}) {
			for (auto const &y : {y0, y1}) {
				for (auto const &z : {z0, z1}) {
					tagIfPointInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<MHDGalaxy>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	constexpr double cs_disk = quokka::EOS_Traits<MHDGalaxy>::cs_disk;
	constexpr double cs_cgm  = quokka::EOS_Traits<MHDGalaxy>::cs_cgm;
	const int ncomp          = ncomp_cc_in;
	const auto prob_lo       = geom[lev].ProbLoArray();
	const auto dx            = geom[lev].CellSizeArray();

	if (dname == "gpot") {
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k);  //cm^2/s^2
		});
		amrex::Gpu::streamSynchronize();
	}

	if (dname == "pressure") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state  = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
				const double cs  = (rho > rho_transition) ? cs_disk : cs_cgm;
				output(i, j, k, ncomp) = rho * cs * cs;  //dyne/cm^2
			});
		}
	}

	if (dname == "radius_sph") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double x    = prob_lo[0] + (static_cast<double>(i) + 0.5) * dx[0];
				const double y    = prob_lo[1] + (static_cast<double>(j) + 0.5) * dx[1];
				const double z    = prob_lo[2] + (static_cast<double>(k) + 0.5) * dx[2];
				const double r_cm = std::sqrt(x * x + y * y + z * z);
				output(i, j, k, ncomp) = r_cm / C::parsec / 1.0e3;  // kpc
			});
		}
	}

	if (dname == "radial_velocity") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state  = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double rho  = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
				const double vx   = state(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) / rho;
				const double vy   = state(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) / rho;
				const double vz   = state(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index) / rho;
				const double x    = prob_lo[0] + (static_cast<double>(i) + 0.5) * dx[0];
				const double y    = prob_lo[1] + (static_cast<double>(j) + 0.5) * dx[1];
				const double z    = prob_lo[2] + (static_cast<double>(k) + 0.5) * dx[2];
				const double r_cm = std::sqrt(x * x + y * y + z * z);
				output(i, j, k, ncomp) = (r_cm > 0.0) ? ((x * vx + y * vy + z * vz) / r_cm) / 1.0e5 : 0.0;  // km/s
			});
		}
	}

	if (dname == "circular_velocity") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state  = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double rho   = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
				const double vx    = state(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) / rho;
				const double vy    = state(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) / rho;
				const double x     = prob_lo[0] + (static_cast<double>(i) + 0.5) * dx[0];
				const double y     = prob_lo[1] + (static_cast<double>(j) + 0.5) * dx[1];
				const double r_cyl = std::sqrt(x * x + y * y);
				output(i, j, k, ncomp) = (r_cyl > 0.0) ? ((x * vy - y * vx) / r_cyl) / 1.0e5 : 0.0; // km/s
			});
		}
	}

	if (dname == "mach") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state  = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double rho  = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
				const double momx = state(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index);
				const double momy = state(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index);
				const double momz = state(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index);
				const double cs   = (rho > rho_transition) ? cs_disk : cs_cgm;
				const double v2   = (momx * momx + momy * momy + momz * momz) / (rho * rho);
				output(i, j, k, ncomp) = std::sqrt(v2) / cs;  // Mach number (unitless)
			});
		}
	}
}

template <> auto QuokkaSimulation<MHDGalaxy>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	std::map<std::string, amrex::Real> stats;

	// Volume-averaged mean density over whole box
	const amrex::Real mean_density = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		return state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
	});
	stats["mean_density"] = mean_density / geom[0].ProbSize();  // g/cm³;

	// Disk mass (rho integrated over disk cells)
	const amrex::Real disk_mass = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
		return (rho > rho_transition) ? rho : static_cast<amrex::Real>(0.0); //M☉
	});
	stats["disk_mass"] = disk_mass / C::M_solar;

	// Disk volume (cm^3) — needed to convert rho integral to mean density
	const amrex::Real disk_volume = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
		return (rho > rho_transition) ? static_cast<amrex::Real>(1.0) : static_cast<amrex::Real>(0.0);  //M☉
	});

	// Mass-weighted mean disk density: <rho> = disk_mass / disk_volume
	// This is a host-side scalar — safe to capture by value into the next kernel
	const amrex::Real mean_disk_density = (disk_volume > 0.0) ? (disk_mass / disk_volume) : 1.0;  // g/cm³
	stats["mean_disk_density"] = mean_disk_density; // g/cm³;
	stats["disk_mass"] = disk_mass / C::M_solar;  // convert to solar masses after

	// Volume-weighted log-density variance over disk cells:
	// sigma_eta^2 = (1/V_disk) * int_{disk} [ln(rho/<rho>)]^2 dV
	const amrex::Real sigma_eta_sq_times_vol = computeVolumeIntegral(
		[=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
		if (rho <= rho_transition) { return static_cast<amrex::Real>(0.0); }
		const amrex::Real eta = std::log(rho / mean_disk_density);
		return eta * eta;
	});

	stats["sigma_eta"] = (disk_volume > 0.0) ? std::sqrt(sigma_eta_sq_times_vol / disk_volume) : static_cast<amrex::Real>(0.0);

	return stats;
}

auto problem_main() -> int
{
    auto BCs_cc = quokka::BC<MHDGalaxy>(quokka::BCType::reflecting);

	const int nvars_fc = Physics_Indices<MHDGalaxy>::nvarTotal_fc;
    const int nvars_per_dim_fc = Physics_Indices<MHDGalaxy>::nvarPerDim_fc;
    amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
    for (int icomp = 0; icomp < nvars_fc; ++icomp) {
        int const component_dir = (nvars_per_dim_fc > 0) ? (icomp / nvars_per_dim_fc) : 0;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {  //reflecting with divB=0 for face-centered B
            int const bc_type = (component_dir == idim) ? amrex::BCType::reflect_even
                                                        : amrex::BCType::reflect_odd;
            BCs_fc[icomp].setLo(idim, bc_type);
            BCs_fc[icomp].setHi(idim, bc_type);
        }
    }
    QuokkaSimulation<MHDGalaxy> sim(BCs_cc, BCs_fc);
    sim.preCalculateInitialConditions();
    sim.setInitialConditions();
    sim.evolve();

    return 0;
}
