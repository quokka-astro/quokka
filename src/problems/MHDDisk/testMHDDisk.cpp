//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDDisk.cpp
/// \brief Defines a simulation using disk galaxy initial conditions.
///

#include <math.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArrayBase.H"

#include "AMReX_GpuContainers.H"
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

namespace
{
	constexpr double Rd_kpc = 3.0;
	constexpr double Rc_kpc = 2.0;
	constexpr double Rd = Rd_kpc * 1.0e3 * C::parsec;
	constexpr double Rc = Rc_kpc * 1.0e3 * C::parsec;
	constexpr double alpha_profile = 2.0;
	constexpr double beta_profile  = 0.5;
	constexpr double q_flatten     = 0.7;
	constexpr double rho_transition = 1.0e-28;
	constexpr double target_beta_seed = 1.0e3;
	constexpr double Rmax_kpc = 8.0;
	constexpr double Rmax = Rmax_kpc * 1.0e3 * C::parsec;
	constexpr double refine_Rcyl_kpc = 8.0;
	constexpr double refine_Hcyl_pc  = 600.0;
	constexpr double refine_Rcyl     = refine_Rcyl_kpc * 1.0e3 * C::parsec;
	constexpr double refine_Hcyl     = refine_Hcyl_pc  * C::parsec;
	constexpr double axis_fallback_cells = 1.0;
} // namespace

struct MHDGalaxy {
};

static_assert(AMREX_SPACEDIM == 3, "MHD disk galaxy problem requires AMREX_SPACEDIM == 3.");

template <> struct quokka::EOS_Traits<MHDGalaxy> {
	static constexpr double gamma = 1.0001;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double T_cgm =  1.0e7;
	static constexpr double cs_cgm = gcem::sqrt(gamma * C::k_B * T_cgm / mean_molecular_weight);
	static constexpr double cs_disk = 7.0e5;
};

template <> struct HydroSystem_Traits<MHDGalaxy> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Particle_Traits<MHDGalaxy> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
};

template <> struct Physics_Traits<MHDGalaxy> : DefaultPhysicsTraits {
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
	amrex::Real Q_mean{};
	amrex::Real Mc{};
	amrex::Real vc{};
	amrex::Real Sigma0{};
	amrex::Real rho_cgm{};
	amrex::Real n_cell{};
	amrex::Real max_level{};
	
	// 2D Cylindrical potential field variables, read from metadata file 
	std::size_t seed_nR{};
	std::size_t seed_nz{};
	amrex::Real seed_Rmax{};
	amrex::Real seed_Lz{};
	amrex::Real seed_B0_HL{};
	amrex::Real seed{};

	// Vector allocation on the GPU
	amrex::Gpu::DeviceVector<amrex::Real> Aphi_device;
	amrex::Real sn_jeans_J;
	amrex::Real sn_momentum;
	amrex::Real sn_remnant_fraction;
	amrex::Real sn_ejecta_mass{};   // Mej, grams
};

// for Initializing Gas Densities & Surface Densities


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
	if (Sigma <= 0.0) { return 0.0; }

	const double H    = cs*cs / (M_PI * C::Gconst * Sigma);
	const double rho0 = (M_PI * C::Gconst * Sigma * Sigma) / (2.0 * cs*cs);

	const double sech        = 1.0 / std::cosh(z / H);
	const double disk_factor = sech * sech;

	const double denom       = R*R + Rc*Rc;
	const double halo_factor = pow(
		1.0 + (z*z) / (q_flatten*q_flatten * denom),
		-vc*vc / (2.0 * cs*cs));

	return rho0 * disk_factor * halo_factor;
}

// 1D and 2D Interpolation Operators for Cylindrical A_phi Evaluation

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto cubic_interp(double p0, double p1, double p2, double p3, double t) -> double
{
	const double a0 = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
	const double a1 =  p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
	const double a2 = -0.5*p0 + 0.5*p2;
	const double a3 =  p1;
	return ((a0*t + a1)*t + a2)*t + a3;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto sample_bicubic(
	const amrex::Real* table,
	int nR, int nz,
	double Rmax, double Lz,
	double R, double z) -> double
{
	const double zmin = -0.5 * Lz;
	
	// Matches Python cell-centered spacing delta definition
	const double dR   = Rmax / static_cast<double>(nR);
	const double dz   = Lz   / static_cast<double>(nz);

	// Exact Boundary Zeroing out 
	if (R < 0.0 || R >= Rmax || z <= zmin || z >= 0.5 * Lz) {
		return 0.0;
	}
	if (R < 1e-12 * Rmax) {
		return 0.0;
	}

	// Map to cell-centered coordinates
	const double fR = (R / dR) - 0.5;
	const double fz = ((z - zmin) / dz) - 0.5;

	int i = static_cast<int>(std::floor(fR));
	int j = static_cast<int>(std::floor(fz));

	// Fallback to bilinear interpolation near the axis to prevent numerical overshoot
	if (i < 2) {
		i = amrex::max(0, amrex::min(i, nR - 2));
		j = amrex::max(0, amrex::min(j, nz - 2));
		const double tR = fR - static_cast<double>(i);
		const double tZ = fz - static_cast<double>(j);
		auto idx = [&](int ii, int jj) -> double { return table[ii * nz + jj]; };
		return (1.0 - tR) * (1.0 - tZ) * idx(i,   j  )
		     +        tR  * (1.0 - tZ) * idx(i+1, j  )
		     + (1.0 - tR) * tZ  * idx(i,   j+1)
		     +        tR  * tZ  * idx(i+1, j+1);
	}

	const double tR = fR - static_cast<double>(i);
	const double tZ = fz - static_cast<double>(j);

	auto idx = [&](int ii, int jj) -> double {
		ii = amrex::max(0, amrex::min(ii, nR - 1));
		jj = amrex::max(0, amrex::min(jj, nz - 1));
		return table[ii * nz + jj];
	};

	std::array<double, 4> col{};
	for (int m = -1; m <= 2; ++m) {
		col[m + 1] = cubic_interp(idx(i-1, j+m), idx(i, j+m),
		                          idx(i+1, j+m), idx(i+2, j+m), tR);
	}
	return cubic_interp(col[0], col[1], col[2], col[3], tZ);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
auto get_taper_factor(double x, double y, double z, 
                        double Rmax, double Lz, 
                        double dx, double dy, double dz) -> double
{
    const double R = std::sqrt(x*x + y*y);
    const double absZ = std::abs(z);
    
    // Taper parameters: adjust n_taper to change the steepness of the fall-off
    const double n_taper = 4.0;
    const double R_taper_start = Rmax - n_taper * amrex::max(dx, dy);
    const double Z_taper_start = 0.5 * Lz - n_taper * dz;
    
    double taper = 1.0;
    
    // Smooth taper for outer R and Z boundaries
    if (R > R_taper_start) {
        taper *= 0.5 * (1.0 - std::cos(M_PI * (Rmax - R) / (Rmax - R_taper_start)));
    }
    if (absZ > Z_taper_start) {
        taper *= 0.5 * (1.0 - std::cos(M_PI * (0.5 * Lz - absZ) / (0.5 * Lz - Z_taper_start)));
    }
    
    // Hard mask for extreme domain overflow to prevent NaN/Inf
    if (R >= Rmax || absZ >= 0.5 * Lz) { return 0.0;}
    
    // Taper near axis to avoid singularity
    const double R_axis_thresh = 1e-6 * dx;
    if (R < R_axis_thresh) { return 0.0;}

    return taper;
}

inline auto
load_bin_to_device(const std::string &path, std::size_t n_expect) -> amrex::Gpu::DeviceVector<amrex::Real>
{
    // Use amrex::Real so this remains compatible if you change precision
    std::vector<amrex::Real> host(n_expect);
    std::ifstream f(path, std::ios::binary);
    
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f, ("Cannot open " + path).c_str());
    const std::size_t total_bytes = n_expect * sizeof(amrex::Real);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast, cppcoreguidelines-narrowing-conversions)
    f.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(total_bytes));
    
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f, ("Error reading " + path).c_str());

    // Allocate on device and copy
    amrex::Gpu::DeviceVector<amrex::Real> dev(n_expect);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), dev.begin());
    
    // Synchronize to ensure data is ready before proceeding
    amrex::Gpu::synchronize();
    
    amrex::Print() << "Loaded " << path << " (" << n_expect << " elements)\n";
    return dev;
}

// preCalculateInitialConditions: loads parameters and the 2D cylindrical A_phi potential table from disk, 
// and calculates Sigma0 via Simpson integration of the Toomre Q condition.

template <> void QuokkaSimulation<MHDGalaxy>::preCalculateInitialConditions()
{
	amrex::ParmParse const pp("mhd_galaxy");
	pp.get("Mc",     userData_.Mc);
	pp.get("Q_mean", userData_.Q_mean);
	pp.query("sn_jeans_J", userData_.sn_jeans_J);
	pp.query("sn_momentum", userData_.sn_momentum);	
	pp.query("sn_remnant_fraction", userData_.sn_remnant_fraction);
	double Mej_msun = 10.0; // default, matches Kim & Ostriker test value
	pp.query("sn_ejecta_mass_msun", Mej_msun);
	userData_.sn_ejecta_mass = Mej_msun * C::M_solar;

	amrex::ParmParse const pp_amr("amr");
	amrex::Vector<int> n_cell_vec(3);
	pp_amr.getarr("n_cell", n_cell_vec);
	userData_.n_cell = static_cast<amrex::Real>(
		*std::max_element(n_cell_vec.begin(), n_cell_vec.end()));
	pp_amr.get("max_level", userData_.max_level);

	constexpr double cs_disk = quokka::EOS_Traits<MHDGalaxy>::cs_disk;
	constexpr double cs_cgm  = quokka::EOS_Traits<MHDGalaxy>::cs_cgm;

	userData_.vc = userData_.Mc * cs_disk;
	const double vc = userData_.vc;

	// Sigma0 via Simpson integration of Toomre Q condition
	auto integrand = [=](double R) -> double {
		const double D    = R * R + Rc * Rc;
		const double sqrtD  = std::sqrt(D);
		const double Omega  = vc / sqrtD;
		const double dOdR   = -vc * R / (D * sqrtD);
		const double kappa  = std::sqrt(std::max(
			4.0 * Omega * Omega + 2.0 * R * Omega * dOdR, 0.0));
		return kappa * cs_disk / (M_PI * C::Gconst * surfaceDensityProfile(R, 1.0));
	};
	constexpr int N = 1000;
	static_assert(N % 2 == 0);
	const double h = Rmax / N;
	double integral = integrand(0.0) + integrand(Rmax);
	for (int i = 1; i < N; ++i) {
		integral += (i % 2 == 0 ? 2.0 : 4.0) * integrand(i * h);
	}
	integral *= h / 3.0;
	userData_.Sigma0  = integral / (userData_.Q_mean * Rmax);
	userData_.rho_cgm = rho_transition * (cs_disk * cs_disk) / (cs_cgm * cs_cgm);

    // Load 2D Cylindrical A_phi Potential Table first time only
	if (userData_.Aphi_device.empty()) {
		std::string meta_filename = "Aphi_2d_meta.txt";
		std::ifstream meta_file(meta_filename);
		if (!meta_file.is_open()) {
			amrex::Abort("Could not open 2D seed field metadata file: " + meta_filename);
		}

		std::string line;
		while (std::getline(meta_file, line)) {
			if (line.empty() || line[0] == '#') {
				continue;
			}
			std::size_t eq_pos = line.find('=');
			if (eq_pos == std::string::npos) {
				continue;
			}
			std::string key = line.substr(0, eq_pos);
			while (!key.empty() && (std::isspace(key.back()) != 0)) {
				key.pop_back();
			}
			std::size_t start = key.find_first_not_of(" \t");
			if (start != std::string::npos) {
				key = key.substr(start);
			}
			std::string val_str = line.substr(eq_pos + 1);
			std::size_t first_num = val_str.find_first_not_of(" \t");
			if (first_num != std::string::npos) {
				val_str = val_str.substr(first_num);
			}
			std::size_t end_num = val_str.find_first_of(" \t#[]");
			if (end_num != std::string::npos) {
				val_str = val_str.substr(0, end_num);
			}

			try {
				// Strictly mapping keys generated by init_seed_pot_field.py
				if (key == "seed_nR") {
					userData_.seed_nR = std::stoul(val_str);
				} else if (key == "seed_nz") {
					userData_.seed_nz = std::stoul(val_str);
				} else if (key == "seed_Rmax") {
					userData_.seed_Rmax = std::stod(val_str);
				} else if (key == "seed_Lz") {
					userData_.seed_Lz = std::stod(val_str);
				} else if (key == "seed_B0_HL") {
					userData_.seed_B0_HL = std::stod(val_str);
				} else if (key == "seed") {
    				userData_.seed = std::stod(val_str);
				}
			} catch (const std::exception& e) {
				continue;
			}
		}
		meta_file.close();

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.seed_nR > 0 && userData_.seed_nz > 0,
			"Error parsing cylindrical vector potential meta variables from init_seed_pot_field.");

		std::size_t total_elements = userData_.seed_nR * userData_.seed_nz;
		std::string data_filename = "Aphi_2d_Aphi.bin";
		userData_.Aphi_device = load_bin_to_device(data_filename, total_elements);

		amrex::Print() << "Loaded 2D Cylindrical Aphi Table cleanly. Map Size: " 
		               << userData_.seed_nR << " x " << userData_.seed_nz << "\n";

		// Derive B0 from target plasma beta at the disk midplane (R=Rd, z=0).
		// The stored Aphi table is dimensionless with rms(curl_nd) = 1 in units of 1/Rmax.
		// Physical B = aphi_nd * B0_scale / Rmax * curl_nd, so B_rms = B0_scale / Rmax.
		// beta = (rho cs^2 / gamma) / (B_rms^2 / 2)
		// => B_rms = cs * sqrt(2 rho_mid / (gamma * beta))
		// => B0_scale = B_rms * Rmax
		{
			constexpr double cs  = quokka::EOS_Traits<MHDGalaxy>::cs_disk;
			constexpr double gam = quokka::EOS_Traits<MHDGalaxy>::gamma;
			const double Sigma_Rd = surfaceDensityProfile(Rd, userData_.Sigma0);
			const double rho_mid  = (M_PI * C::Gconst * Sigma_Rd * Sigma_Rd) / (2.0 * cs * cs);
			const double B_rms_HL = cs * std::sqrt(2.0 * rho_mid / (gam * target_beta_seed));
			userData_.seed_B0_HL  = B_rms_HL * userData_.seed_Rmax;
			amrex::Print() << "Seed field: target_beta=" << target_beta_seed
			               << "  rho_mid=" << rho_mid
			               << "  B_rms_HL=" << B_rms_HL
			               << "  B0_scale=" << userData_.seed_B0_HL << " G*cm (HL)\n";
		}
	}

	amrex::Print()
		<< "MHDGalaxy init complete\n"
		<< "Mc=" << userData_.Mc
		<< " Q=" << userData_.Q_mean
		<< " Sigma0=" << userData_.Sigma0 
		<< " Seed=" << userData_.seed << "\n"
		<< "M_solar=" <<  C::M_solar << "\n";
}

// Set initial conditions on the grid by evaluating the analytic disk density and velocity profiles at cell centers, 
// and calculating the local magnetic energy at cell centers by taking the curl of the analytically 
// sampled vector potential A_phi at the surrounding staggered Yee mesh nodes.

template <>
void QuokkaSimulation<MHDGalaxy>::setInitialConditionsOnGrid(
	quokka::grid const &grid_elem)
{
	const double vc      = userData_.vc;
	const double Sigma0  = userData_.Sigma0;
	const double cs_disk = quokka::EOS_Traits<MHDGalaxy>::cs_disk;
	const double cs_cgm  = quokka::EOS_Traits<MHDGalaxy>::cs_cgm;
	const double rho_cgm = userData_.rho_cgm;
	constexpr double gamma = quokka::EOS_Traits<MHDGalaxy>::gamma;
	
	const double B0_scale = userData_.seed_B0_HL;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(B0_scale > 0.0, "Seed field strength must be positive.");

	const amrex::Box            &indexRange = grid_elem.indexRange_;
	const auto                  dx        = grid_elem.dx_;
	const auto                  prob_lo   = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc  = grid_elem.array_;

	// Cylindrical Potential Table Pointers & Parameters for GPU Lambdas
	const amrex::Real* aphi_ptr = userData_.Aphi_device.data();
	const int nR_table = static_cast<int>(userData_.seed_nR);
	const int nz_table = static_cast<int>(userData_.seed_nz);
	const double Rmax_table = userData_.seed_Rmax;
	const double Lz_table = userData_.seed_Lz;

    const double dR_table = Rmax_table / static_cast<double>(nR_table);
    const double axis_dead_zone = axis_fallback_cells * dR_table;

    // physical potential with hard boundary guard and smooth tapering, sampled directly in the node lambdas 
	// to ensure consistency with the interpolated values used for curl calculation.
	auto get_Aphi_physical = [=] AMREX_GPU_DEVICE(double x_val, double y_val, double z_val) -> double {
		const double R_val = std::sqrt(x_val * x_val + y_val * y_val);
		double aphi_nd = sample_bicubic(aphi_ptr, nR_table, nz_table, Rmax_table, Lz_table, R_val, z_val);
		return aphi_nd * B0_scale;
	};

    auto get_Ax = [=] AMREX_GPU_DEVICE(double x_e, double y_e, double z_e) -> double {
        const double R_e = std::sqrt(x_e * x_e + y_e * y_e);
        
        // Force zero within 2.0*dx to prevent 1/R amplification of interpolation noise near the axis.
        if (R_e < axis_dead_zone) { return 0.0; }
        
        const double taper = get_taper_factor(x_e, y_e, z_e, Rmax_table, Lz_table, dx[0], dx[1], dx[2]);
		
        const double Aphi  = get_Aphi_physical(x_e, y_e, z_e);
        return -Aphi * (y_e / R_e) * taper;
    };

    auto get_Ay = [=] AMREX_GPU_DEVICE(double x_e, double y_e, double z_e) -> double {
        const double R_e = std::sqrt(x_e * x_e + y_e * y_e);
        
        // Force zero within 2.0*dx to prevent 1/R amplification of interpolation noise near the axis.
        if (R_e < axis_dead_zone) { return 0.0; }
        
        const double taper = get_taper_factor(x_e, y_e, z_e, Rmax_table, Lz_table, dx[0], dx[1], dx[2]);
        const double Aphi  = get_Aphi_physical(x_e, y_e, z_e);
        return Aphi * (x_e / R_e) * taper;
    };

	amrex::ParallelFor(indexRange,
	[=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	{
		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		const double y = prob_lo[1] + (j + 0.5) * dx[1];
		const double z = prob_lo[2] + (k + 0.5) * dx[2];
		const double R = std::sqrt(x*x + y*y);

		// Hydrodynamic quantities: density, velocity, pressure, internal energy
		const double rho_disc_raw = diskDensityAnalytic(R, z, Sigma0, vc, cs_disk);
		const bool   in_disk      = (rho_disc_raw > rho_transition);
		const double rho          = in_disk
			? amrex::max(rho_disc_raw, rho_transition * 1e-6)
			: rho_cgm;
		const double cs = in_disk ? cs_disk : cs_cgm;

		const double vrot = (R > 0.0) ? vc * R / std::sqrt(R*R + Rc*Rc) : 0.0;
		double vx = 0.0;
		double vy = 0.0;
		if (in_disk && R > 0.0) {
			vx = -vrot * y / R;
			vy =  vrot * x / R;
		}

		const double pressure = rho * cs * cs;
		const double Eint     = pressure / (gamma - 1.0);
		const double Ekin     = 0.5 * rho * (vx*vx + vy*vy);

		// Cell-Centered Magnetic Energy Calculation 
		// define local total energy at cell centers by calculating the analytic differences
		// of the surrounding staggered Yee mesh nodes directly at the cell center location.
		const double x_node_lo = prob_lo[0] + i * dx[0];
		const double x_node_hi = prob_lo[0] + (i + 1) * dx[0];
		const double y_node_lo = prob_lo[1] + j * dx[1];
		const double y_node_hi = prob_lo[1] + (j + 1) * dx[1];
		const double z_node_lo = prob_lo[2] + k * dx[2];
		const double z_node_hi = prob_lo[2] + (k + 1) * dx[2];

		// Analytical Bx at cell center via averaged face stencils
		double Ay_hi_left  = get_Ay(x_node_lo, y, z_node_hi);
		double Ay_lo_left  = get_Ay(x_node_lo, y, z_node_lo);
		double Ay_hi_right = get_Ay(x_node_hi, y, z_node_hi);
		double Ay_lo_right = get_Ay(x_node_hi, y, z_node_lo);
		double Bx_cc = -0.5 * ((Ay_hi_left - Ay_lo_left) + (Ay_hi_right - Ay_lo_right)) / dx[2];

		// Analytical By at cell center via averaged face stencils
		double Ax_hi_bot = get_Ax(x, y_node_lo, z_node_hi);
		double Ax_lo_bot = get_Ax(x, y_node_lo, z_node_lo);
		double Ax_hi_top = get_Ax(x, y_node_hi, z_node_hi);
		double Ax_lo_top = get_Ax(x, y_node_hi, z_node_lo);
		double By_cc = 0.5 * ((Ax_hi_bot - Ax_lo_bot) + (Ax_hi_top - Ax_lo_top)) / dx[2];

		// Analytical Bz at cell center via averaged face stencils
		double Ay_r_cc = get_Ay(x_node_hi, y, z);
		double Ay_l_cc = get_Ay(x_node_lo, y, z);
		double Ax_t_cc = get_Ax(x, y_node_hi, z);
		double Ax_b_cc = get_Ax(x, y_node_lo, z);
		double Bz_cc = ((Ay_r_cc - Ay_l_cc) / dx[0]) - ((Ax_t_cc - Ax_b_cc) / dx[1]);

		const double Emag = 0.5 * (Bx_cc * Bx_cc + By_cc * By_cc + Bz_cc * Bz_cc);

		// State vector update 
		state_cc(i,j,k,HydroSystem<MHDGalaxy>::density_index)        = rho;
		state_cc(i,j,k,HydroSystem<MHDGalaxy>::x1Momentum_index)     = rho * vx;
		state_cc(i,j,k,HydroSystem<MHDGalaxy>::x2Momentum_index)     = rho * vy;
		state_cc(i,j,k,HydroSystem<MHDGalaxy>::x3Momentum_index)     = 0.0;
		state_cc(i,j,k,HydroSystem<MHDGalaxy>::energy_index)         = Ekin + Eint + Emag;
		state_cc(i,j,k,HydroSystem<MHDGalaxy>::internalEnergy_index) = Eint;
	});
}

template <>
void QuokkaSimulation<MHDGalaxy>::setInitialConditionsOnGridFaceVars(
	quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc   = grid_elem.array_;
	const amrex::Box            &indexRange = grid_elem.indexRange_;
	const quokka::direction      dir        = grid_elem.dir_;
	const auto                   dx         = grid_elem.dx_;
	const auto                   prob_lo    = grid_elem.prob_lo_;

	const double B0_scale = userData_.seed_B0_HL;

	// Cylindrical Potential Table Pointers & Parameters 
	const amrex::Real* aphi_ptr = userData_.Aphi_device.data();
	const int nR_table = static_cast<int>(userData_.seed_nR);
	const int nz_table = static_cast<int>(userData_.seed_nz);
	const double Rmax_table = userData_.seed_Rmax;
	const double Lz_table = userData_.seed_Lz;
    const double dR_table     = Rmax_table / static_cast<double>(nR_table);
	const double axis_dead_zone = axis_fallback_cells * dR_table;

	// True physical domain extents (level-independent), used to force exact B=0
	// at the real simulation boundary regardless of whether seed_Rmax/seed_Lz
	// (the table's taper calibration) are matched to the actual domain size.
	const auto prob_lo_dom = geom[0].ProbLoArray();
	const auto prob_hi_dom = geom[0].ProbHiArray();
	constexpr double boundary_tol = 1.0e-6; // relative tolerance, scaled by dx below

	// magnetic potential with hard boundary guard
	auto get_Aphi_physical = [=] AMREX_GPU_DEVICE(double x_val, double y_val, double z_val) -> double {
		const double R_val = std::sqrt(x_val * x_val + y_val * y_val);
		double aphi_nd = sample_bicubic(aphi_ptr, nR_table, nz_table, Rmax_table, Lz_table, R_val, z_val);
		return aphi_nd * B0_scale;
	};
    // cartesian mapping with deadzone
    auto get_Ax_node = [=] AMREX_GPU_DEVICE(double x_n, double y_n, double z_n) -> double {
        const double R = std::sqrt(x_n * x_n + y_n * y_n);
        
        // axis deadzone matched to sample_bicubic bilinear fallback (i < 2 → R < 2*dR_table)
        if (R < axis_dead_zone) { return 0.0; }
        
        const double taper = get_taper_factor(x_n, y_n, z_n, Rmax_table, Lz_table, dx[0], dx[1], dx[2]);
        const double Aphi  = get_Aphi_physical(x_n, y_n, z_n);
        return -Aphi * (y_n / R) * taper;
    };

    auto get_Ay_node = [=] AMREX_GPU_DEVICE(double x_n, double y_n, double z_n) -> double {
        const double R = std::sqrt(x_n * x_n + y_n * y_n);
        
        // axis deadzone matched to sample_bicubic bilinear fallback (i < 2 → R < 2*dR_table)
        if (R < axis_dead_zone) { return 0.0; }
        
        const double taper = get_taper_factor(x_n, y_n, z_n, Rmax_table, Lz_table, dx[0], dx[1], dx[2]);
        const double Aphi  = get_Aphi_physical(x_n, y_n, z_n);
        return Aphi * (x_n / R) * taper;
    };

	amrex::ParallelFor(indexRange,
	[=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	{
		double B_face = 0.0;

		if (dir == quokka::direction::x) {
			// Bx lives at x-faces: index (i) represents the x-node plane.
			// Python samples Ay_node at the x-node, averages over y, and differentiates over z.
			const double xf  = prob_lo[0] + i * dx[0];
			const double yj  = prob_lo[1] + j * dx[1];
			const double yjp = prob_lo[1] + (j + 1) * dx[1];
			const double zk  = prob_lo[2] + k * dx[2];
			const double zkp = prob_lo[2] + (k + 1) * dx[2];

			double Ay_j_kp  = get_Ay_node(xf, yj,  zkp);
			double Ay_jp_kp = get_Ay_node(xf, yjp, zkp);
			double Ay_j_k   = get_Ay_node(xf, yj,  zk);
			double Ay_jp_k  = get_Ay_node(xf, yjp, zk);

			double Ay_xface_kp = 0.5 * (Ay_j_kp + Ay_jp_kp);
			double Ay_xface_k  = 0.5 * (Ay_j_k + Ay_jp_k);

			// Bx = -dAy/dz
			B_face = -(Ay_xface_kp - Ay_xface_k) / dx[2];

			// Hard-zero enforcement at the true x-domain boundary (normal component
			// on x-faces), independent of seed-table taper calibration.
			if (std::abs(xf - prob_lo_dom[0]) < boundary_tol * dx[0] ||
			    std::abs(xf - prob_hi_dom[0]) < boundary_tol * dx[0]) {
				B_face = 0.0;
			}

		} else if (dir == quokka::direction::y) {
			// By lives at y-faces: index (j) represents the y-node plane.
			// Python samples Ax_node at the y-node, averages over x, and differentiates over z.
			const double xi  = prob_lo[0] + i * dx[0];
			const double xip = prob_lo[0] + (i + 1) * dx[0];
			const double yf  = prob_lo[1] + j * dx[1];
			const double zk  = prob_lo[2] + k * dx[2];
			const double zkp = prob_lo[2] + (k + 1) * dx[2];

			double Ax_i_kp  = get_Ax_node(xi,  yf, zkp);
			double Ax_ip_kp = get_Ax_node(xip, yf, zkp);
			double Ax_i_k   = get_Ax_node(xi,  yf, zk);
			double Ax_ip_k  = get_Ax_node(xip, yf, zk);

			double Ax_yface_kp = 0.5 * (Ax_i_kp + Ax_ip_kp);
			double Ax_yface_k  = 0.5 * (Ax_i_k + Ax_ip_k);

			// By = dAx/dz
			B_face = (Ax_yface_kp - Ax_yface_k) / dx[2];

			// Hard-zero enforcement at the true y-domain boundary (normal component
			// on y-faces), independent of seed-table taper calibration.
			if (std::abs(yf - prob_lo_dom[1]) < boundary_tol * dx[1] ||
			    std::abs(yf - prob_hi_dom[1]) < boundary_tol * dx[1]) {
				B_face = 0.0;
			}

		} else {
			// Bz lives at z-faces: index (k) represents the z-node plane.
			// Bz = dAy/dx - dAx/dy using the exact node-level cross-averages.
			const double xi  = prob_lo[0] + i * dx[0];
			const double xip = prob_lo[0] + (i + 1) * dx[0];
			const double yj  = prob_lo[1] + j * dx[1];
			const double yjp = prob_lo[1] + (j + 1) * dx[1];
			const double zf  = prob_lo[2] + k * dx[2];

			// dAy/dx term: difference over x, then average over y
			double Ay_ip_j  = get_Ay_node(xip, yj,  zf);
			double Ay_i_j   = get_Ay_node(xi,  yj,  zf);
			double Ay_ip_jp = get_Ay_node(xip, yjp, zf);
			double Ay_i_jp  = get_Ay_node(xi,  yjp, zf);

			double dAy_dx_j  = (Ay_ip_j - Ay_i_j) / dx[0];
			double dAy_dx_jp = (Ay_ip_jp - Ay_i_jp) / dx[0];
			double dAy_dx_cc = 0.5 * (dAy_dx_j + dAy_dx_jp);

			// dAx/dy term: difference over y, then average over x
			double Ax_i_jp  = get_Ax_node(xi,  yjp, zf);
			double Ax_i_j   = get_Ax_node(xi,  yj,  zf);
			double Ax_ip_jp = get_Ax_node(xip, yjp, zf);
			double Ax_ip_j  = get_Ax_node(xip, yj,  zf);

			double dAx_dy_i  = (Ax_i_jp - Ax_i_j) / dx[1];
			double dAx_dy_ip = (Ax_ip_jp - Ax_ip_j) / dx[1];
			double dAx_dy_cc = 0.5 * (dAx_dy_i + dAx_dy_ip);

			B_face = dAy_dx_cc - dAx_dy_cc;

			// Hard-zero enforcement at the true z-domain boundary (normal component
			// on z-faces), independent of seed-table taper calibration.
			if (std::abs(zf - prob_lo_dom[2]) < boundary_tol * dx[2] ||
			    std::abs(zf - prob_hi_dom[2]) < boundary_tol * dx[2]) {
				B_face = 0.0;
			}
		}
		state_fc(i, j, k, 0) = B_face;
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

			const double rho      = state(i,j,k,HydroSystem<MHDGalaxy>::density_index);
			const double px       = state(i,j,k,HydroSystem<MHDGalaxy>::x1Momentum_index);
			const double py       = state(i,j,k,HydroSystem<MHDGalaxy>::x2Momentum_index);
			const double pz       = state(i,j,k,HydroSystem<MHDGalaxy>::x3Momentum_index);
			const double Eint     = state(i,j,k,HydroSystem<MHDGalaxy>::internalEnergy_index);
			const double Etot_old = state(i,j,k,HydroSystem<MHDGalaxy>::energy_index);
			const double Ekin_old = 0.5 * (px*px + py*py + pz*pz) / rho;
			const double Emag     = Etot_old - Ekin_old - Eint;

			const double D   = R2 + Rc*Rc + (z/q_flatten)*(z/q_flatten);
			const double g_R = (R > 0.0) ? -(vc*vc * R / D) : 0.0;
			const double g_z =             -(vc*vc * z / (q_flatten*q_flatten * D));
			const double gx  = (R > 0.0) ? g_R * x / R : 0.0;
			const double gy  = (R > 0.0) ? g_R * y / R : 0.0;

			const double px_new   = px + dt_lev * rho * gx;
			const double py_new   = py + dt_lev * rho * gy;
			const double pz_new   = pz + dt_lev * rho * g_z;
			const double Ekin_new = 0.5 * (px_new*px_new + py_new*py_new + pz_new*pz_new) / rho;

			state(i,j,k,HydroSystem<MHDGalaxy>::x1Momentum_index) = px_new;
			state(i,j,k,HydroSystem<MHDGalaxy>::x2Momentum_index) = py_new;
			state(i,j,k,HydroSystem<MHDGalaxy>::x3Momentum_index) = pz_new;
			state(i,j,k,HydroSystem<MHDGalaxy>::energy_index)     = Ekin_new + Eint + Emag;
		});
	}

}




template <> void QuokkaSimulation<MHDGalaxy>::refineGrid(
	int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx      = geom[lev].CellSizeArray();
	const auto tag     = tags.arrays();

	amrex::ParmParse pp("mhd_galaxy");
	amrex::Real shrink_kpc = 1.0;
	amrex::Real shrink_pc  = 50.0;
	pp.query("refine_shrink_per_level_kpc", shrink_kpc);
	pp.query("refine_shrink_per_level_pc",  shrink_pc);

	const amrex::Real margin_R = static_cast<amrex::Real>(lev) * shrink_kpc * 1.0e3 * C::parsec;
	const amrex::Real margin_H = static_cast<amrex::Real>(lev) * shrink_pc  * C::parsec;

	const amrex::Real Rcyl_lev = amrex::max(static_cast<amrex::Real>(refine_Rcyl) - margin_R,
	                                         static_cast<amrex::Real>(0.3 * refine_Rcyl));
	const amrex::Real Hcyl_lev = amrex::max(static_cast<amrex::Real>(refine_Hcyl) - margin_H,
	                                         static_cast<amrex::Real>(0.3 * refine_Hcyl));

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real x0 = prob_lo[0] + i * dx[0];
		const amrex::Real y0 = prob_lo[1] + j * dx[1];
		const amrex::Real z0 = prob_lo[2] + k * dx[2];
		const amrex::Real x1 = x0 + dx[0];
		const amrex::Real y1 = y0 + dx[1];
		const amrex::Real z1 = z0 + dx[2];

		auto tagIfInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			if (std::sqrt(x*x + y*y) < Rcyl_lev && std::abs(z) < Hcyl_lev) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};
		for (auto const &x : {x0, x1}) {
			for (auto const &y : {y0, y1}) {
				for (auto const &z : {z0, z1}) {
					tagIfInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<MHDGalaxy>::computeAfterTimestep()
{
	if (!(userData_.sn_jeans_J > 0.0)) {
		return;
	}

	constexpr double MSUN = C::M_solar;
	constexpr double KM_S = 1.0e5;
	constexpr double Esn = 1.0e51;          // erg, fixed SN energy (Sec 3.2.1)
	constexpr double mu_H = 1.4 * C::m_u;   // mass per H nucleus, standard He abundance
	constexpr int stencil_radius = 2;       // rK in units of dx (paper default is 3dx; adjust as needed)

	const double Mej = userData_.sn_ejecta_mass;              // ejecta mass, Eq 17
	const double sn_jeans_J = userData_.sn_jeans_J;
	const double sn_momentum_ref = userData_.sn_momentum;     // calibration coefficient for Eq 20



	for (int lev = 0; lev <= finest_level; ++lev) {
		auto &state          = state_new_cc_[lev];
		auto const &state_fc = state_new_fc_[lev];

		// Fill ghost zones before reading neighbor data below. 
		const auto time = tNew_[lev];
		fillBoundaryConditions(state, state, lev, time, quokka::centering::cc, quokka::direction::na,
		                        InterpHookNone, InterpHookNone, FillPatchType::fillpatch_function);
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			fillBoundaryConditions(state_new_fc_[lev][idim], state_new_fc_[lev][idim], lev, time, quokka::centering::fc,
			                        static_cast<quokka::direction>(idim), InterpHookNone, InterpHookNone,
			                        FillPatchType::fillpatch_function);
		}

		const auto dx        = geom[lev].CellSizeArray();
		const double dx_max  = amrex::max(dx[0], amrex::max(dx[1], dx[2]));
		const double vol     = dx[0] * dx[1] * dx[2];

		// Fine mask: covered cells -> 0, uncovered (real, finest-resolution) cells -> 1.
		amrex::iMultiFab mask(state.boxArray(), state.DistributionMap(), 1, 0);
		if (lev < finest_level) {
			mask = amrex::makeFineMask(state.boxArray(), state.DistributionMap(),
			                            state_new_cc_[lev + 1].boxArray(), refRatio(lev), 1, 0);
		} else {
			mask.setVal(1);
		}


		// delta components: 0=drho, 1=dpx, 2=dpy, 3=dpz, 4=dE.
		// Everything here is the UNLIMITED deposit (paper Sec 2.2, Steps 1-2); the
		// analytic limiter (Eq 21, Step 3) is applied once, after SumBoundary, in the
		// apply pass below -- exactly matching the paper's separation of concerns.
		amrex::MultiFab delta(state.boxArray(), state.DistributionMap(), 5, stencil_radius);
		delta.setVal(0.0);

		// --- Pass 1: deposit (unlimited) mass/momentum/energy contributions ---
		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();
			auto const &s = state.const_array(mfi);
			auto d = delta.array(mfi);
			auto const &mask_arr = mask.const_array(mfi);

			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fab_fc{
				state_fc[0].const_array(mfi), state_fc[1].const_array(mfi), state_fc[2].const_array(mfi)};
			auto const *fab_fc_ptr = &fab_fc;

			const auto slo = state[mfi].box().smallEnd();
			const auto shi = state[mfi].box().bigEnd();
			const auto dlo = delta[mfi].box().smallEnd();
			const auto dhi = delta[mfi].box().bigEnd();

			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				// Skip cells covered by a finer level
				if (mask_arr(i, j, k) == 0) { return; }

				const double rho = s(i, j, k, HydroSystem<MHDGalaxy>::density_index);

				const double cs          = HydroSystem<MHDGalaxy>::ComputeIsothermalSoundSpeed(s, i, j, k, fab_fc_ptr);
				const double plasma_beta = HydroSystem<MHDGalaxy>::ComputePlasmaBeta(s, i, j, k, fab_fc_ptr);
				const double beta_safe   = amrex::max(plasma_beta, 1.0e-10);
				const double cs_eff_sq   = cs * cs * (1.0 + 0.74 / beta_safe);
				const double rho_J = M_PI * cs_eff_sq / (C::Gconst * sn_jeans_J * sn_jeans_J * (dx_max * dx_max));
				if (rho <= rho_J) { return; } // Jeans trigger, unchanged from before

				const double px = s(i,j,k,HydroSystem<MHDGalaxy>::x1Momentum_index);
				const double py = s(i,j,k,HydroSystem<MHDGalaxy>::x2Momentum_index);
				const double pz = s(i,j,k,HydroSystem<MHDGalaxy>::x3Momentum_index);
				const double vx_c = px / rho;
				const double vy_c = py / rho;
				const double vz_c = pz / rho;

				// --- Gather kernel sums: cell count, gas mass, momentum (for vCOM), Eq 17-18 ---
				int    N_kernel   = 0;
				double mass_sum   = 0.0; // Sum rho_ijk over kernel (incl. source cell)
				double momx_sum   = 0.0;
				double momy_sum   = 0.0;
				double momz_sum   = 0.0;
				double neighbor_mass_sum = 0.0; // excludes center; used for mass-weighted w_mom

				for (int di = -stencil_radius; di <= stencil_radius; ++di) {
					for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
						for (int dk = -stencil_radius; dk <= stencil_radius; ++dk) {
							if (di*di + dj*dj + dk*dk > stencil_radius * stencil_radius) { continue; }
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] ||
								kk < slo[2] || kk > shi[2]) { continue; }
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] ||
								kk < dlo[2] || kk > dhi[2]) { continue; }

							const double rho_nb = s(ii, jj, kk, HydroSystem<MHDGalaxy>::density_index);
							++N_kernel;
							mass_sum += rho_nb;
							momx_sum += s(ii, jj, kk, HydroSystem<MHDGalaxy>::x1Momentum_index);
							momy_sum += s(ii, jj, kk, HydroSystem<MHDGalaxy>::x2Momentum_index);
							momz_sum += s(ii, jj, kk, HydroSystem<MHDGalaxy>::x3Momentum_index);
							if (di != 0 || dj != 0 || dk != 0) { neighbor_mass_sum += rho_nb; }
						}
					}
				}
				if (N_kernel == 0) { return; }
				const double Vsnr_local = static_cast<double>(N_kernel) * vol;

				// Eq 17-18: total swept-up + ejecta mass, and mass-weighted vCOM.
				// Simplification: ejecta velocity v_ej taken equal to the source cell's
				// own velocity (we do not track an actual star particle here).
				const double Msnr = mass_sum * vol + Mej;
				const double vCOMx = (momx_sum * vol + Mej * vx_c) / Msnr;
				const double vCOMy = (momy_sum * vol + Mej * vy_c) / Msnr;
				const double vCOMz = (momz_sum * vol + Mej * vz_c) / Msnr;

				// nH_amb, shell-formation mass (Sec 3.2.1). M_sf is rescaled by
				// (p_terminal_ref / p_terminal_canonical)^2 so that the kinetic energy
				// p_terminal^2/(2 M_sf) stays invariant under changes to the runtime
				// sn_momentum_ref parameter -- matches the real reference implementation
				// (SNFeedbackUtils::depositThermalKineticMomentumSNR)
				const double nH_amb = Msnr / (mu_H * Vsnr_local);
				constexpr double M_sf_canonical = 1679.0 * MSUN;   // at nH=1 cm^-3, calibrated to p_ref_canonical
				constexpr double p_ref_canonical = 2.8e5 * MSUN * KM_S; // Kim & Ostriker (2015) canonical terminal momentum
				const double p_ratio = (sn_momentum_ref * MSUN * KM_S) / p_ref_canonical;
				const double M_sf = M_sf_canonical * std::pow(amrex::max(nH_amb, 1.0e-8), -0.26) * p_ratio * p_ratio;
				const double R_M  = Msnr / M_sf;

				// Eq 20: terminal momentum (extensive, g*cm/s)
				const double p_terminal = sn_momentum_ref * MSUN * KM_S * std::pow(amrex::max(nH_amb, 1.0e-8), -0.17);

				// MC regime only (Eq 19, R_M > 1): full terminal momentum, no thermal-only
				// or Sedov-Taylor branching. R_M is retained purely as a diagnostic.
				const double p_radial_mag = p_terminal / vol;

				// Full weight omega_ijk = 1 applied to every included kernel cell; the
				// 1/N_kernel normalization is already folded into Vsnr_local = N_kernel*vol,
				// so summing this over all N_kernel cells gives exactly Mej (mass conserved).
				const double drho_ej_cell = Mej / Vsnr_local;
				const bool have_mass_weight = (neighbor_mass_sum > 0.0);

				// --- Pass A: raw radial kicks, accumulate net vector S for re-centering.
				// Even though p_radial is isotropic in the continuum limit (paper: "the
				// cross term cancels when summing over all cells in the stencil"), a
				// discrete/domain-clipped stencil is not exactly isotropic, so we still
				// need the same re-centering trick as before to guarantee zero net thrust. ---
				double Sx = 0.0;
				double Sy = 0.0;
				double Sz = 0.0;
				double Lraw = 0.0;
				for (int di = -stencil_radius; di <= stencil_radius; ++di) {
					for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
						for (int dk = -stencil_radius; dk <= stencil_radius; ++dk) {
							if (di == 0 && dj == 0 && dk == 0) { continue; }
							if (di*di + dj*dj + dk*dk > stencil_radius * stencil_radius) { continue; }
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] ||
								kk < slo[2] || kk > shi[2]) { continue; }
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] ||
								kk < dlo[2] || kk > dhi[2]) { continue; }

							const double rx = di * dx[0];
							const double ry = dj * dx[1];
							const double rz = dk * dx[2];
							const double r = std::sqrt(rx*rx + ry*ry + rz*rz);
							if (r == 0.0) { continue; }
							const double rho_nb = s(ii, jj, kk, HydroSystem<MHDGalaxy>::density_index);
							const double w_mom = have_mass_weight ? (rho_nb / neighbor_mass_sum)
							                                       : (1.0 / static_cast<double>(N_kernel - 1));
							const double ex = rx / r;
							const double ey = ry / r;
							const double ez = rz / r;
							const double dp = p_radial_mag * w_mom;
							Sx += dp * ex; Sy += dp * ey; Sz += dp * ez;
							Lraw += dp;
						}
					}
				}

				double Lprime = 0.0;
				for (int di = -stencil_radius; di <= stencil_radius; ++di) {
					for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
						for (int dk = -stencil_radius; dk <= stencil_radius; ++dk) {
							if (di == 0 && dj == 0 && dk == 0) { continue; }
							if (di*di + dj*dj + dk*dk > stencil_radius * stencil_radius) { continue; }
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] ||
								kk < slo[2] || kk > shi[2]) { continue; }
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] ||
								kk < dlo[2] || kk > dhi[2]) { continue; }

							const double rx = di * dx[0];
							const double ry = dj * dx[1];
							const double rz = dk * dx[2];
							const double r = std::sqrt(rx*rx + ry*ry + rz*rz);
							if (r == 0.0) { continue; }
							const double rho_nb = s(ii, jj, kk, HydroSystem<MHDGalaxy>::density_index);
							const double w_mom = have_mass_weight ? (rho_nb / neighbor_mass_sum)
							                                       : (1.0 / static_cast<double>(N_kernel - 1));
							const double ex = rx / r;
							const double ey = ry / r;
							const double ez = rz / r;
							const double dp = p_radial_mag * w_mom;
							const double cx = dp * ex - Sx * w_mom;
							const double cy = dp * ey - Sy * w_mom;
							const double cz = dp * ez - Sz * w_mom;
							Lprime += std::sqrt(cx*cx + cy*cy + cz*cz);
						}
					}
				}
				const double rescale = (Lprime > 0.0) ? (Lraw / Lprime) : 0.0;

				// --- Pass B: deposit Eq 17 into every kernel cell (mass, base momentum
				// from velocity-smoothing, corrected radial kick, and energy including
				// the vCOM . p_radial cross term). All UNLIMITED -- limiter applied later. ---
				for (int di = -stencil_radius; di <= stencil_radius; ++di) {
					for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
						for (int dk = -stencil_radius; dk <= stencil_radius; ++dk) {
							if (di*di + dj*dj + dk*dk > stencil_radius * stencil_radius) { continue; }
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] ||
								kk < slo[2] || kk > shi[2]) { continue; }
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] ||
								kk < dlo[2] || kk > dhi[2]) { continue; }

							const double rho_ijk = s(ii, jj, kk, HydroSystem<MHDGalaxy>::density_index);
							const double px_ijk  = s(ii, jj, kk, HydroSystem<MHDGalaxy>::x1Momentum_index);
							const double py_ijk  = s(ii, jj, kk, HydroSystem<MHDGalaxy>::x2Momentum_index);
							const double pz_ijk  = s(ii, jj, kk, HydroSystem<MHDGalaxy>::x3Momentum_index);
							const double rho_new_ijk = rho_ijk + drho_ej_cell;

							// Eq 17, momentum row: rho_new*vCOM - rho_old*v_old, plus the
							// (re-centered) radial kick for non-center cells.
							double pradx = 0.0;
							double prady = 0.0;
							double pradz = 0.0;
							if (di != 0 || dj != 0 || dk != 0) {
								const double rx = di * dx[0];
								const double ry = dj * dx[1];
								const double rz = dk * dx[2];
								const double r = std::sqrt(rx*rx + ry*ry + rz*rz);
								const double rho_nb = rho_ijk;
								const double w_mom = have_mass_weight ? (rho_nb / neighbor_mass_sum)
								                                       : (1.0 / static_cast<double>(N_kernel - 1));
								const double ex = rx / r;
								const double ey = ry / r;
								const double ez = rz / r;
								const double dp = p_radial_mag * w_mom;
								pradx = (dp * ex - Sx * w_mom) * rescale;
								prady = (dp * ey - Sy * w_mom) * rescale;
								pradz = (dp * ez - Sz * w_mom) * rescale;
							}

							const double dpx = rho_new_ijk * vCOMx - px_ijk + pradx;
							const double dpy = rho_new_ijk * vCOMy - py_ijk + prady;
							const double dpz = rho_new_ijk * vCOMz - pz_ijk + pradz;

							// Eq 17, energy row, with omega_ijk = 1 (see mass-deposit comment
							// above): (Esn+Ekin_ej)/Vsnr_local + vCOM . p_radial. Summed over
							// all N_kernel cells this adds exactly (Esn+Ekin_ej) total, plus
							// the vCOM.p_radial cross term which nets to ~0 by construction.
							// Ekin_ej (ejecta kinetic energy) is set to 0 here: we do not track
							// a separate ejecta launch velocity distinct from the source cell.
							constexpr double Ekin_ej = 0.0;
							const double dE_ijk = (Esn + Ekin_ej) / Vsnr_local
								+ (vCOMx * pradx + vCOMy * prady + vCOMz * pradz);

							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 0), drho_ej_cell);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 1), dpx);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 2), dpy);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 3), dpz);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 4), dE_ijk);
						}
					}
				}
			});
		}
		amrex::Gpu::streamSynchronize();
		delta.SumBoundary(geom[lev].periodicity()); // Step 2: inter-rank buffer summation

		// --- Pass 2 / Step 3-5: apply the analytic limiter (Eq 21) once per cell using
		// the fully-summed delta, then commit to state. ---
		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();
			auto s = state.array(mfi);
			auto const &d = delta.const_array(mfi);

			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double drho = d(i, j, k, 0);
				const double dE    = d(i, j, k, 4);
				if (drho == 0.0 && dE == 0.0) { return; }

				const double rho_old  = s(i, j, k, HydroSystem<MHDGalaxy>::density_index);
				const double px_old   = s(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index);
				const double py_old   = s(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index);
				const double pz_old   = s(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index);
				const double Eint_old = s(i, j, k, HydroSystem<MHDGalaxy>::internalEnergy_index);
				const double Etot_old = s(i, j, k, HydroSystem<MHDGalaxy>::energy_index);
				const double Ekin_old = 0.5 * (px_old*px_old + py_old*py_old + pz_old*pz_old) / rho_old;
				const double Emag     = Etot_old - Ekin_old - Eint_old;

				const double rho_new = rho_old + drho;

				// Internal-energy floor: hold SPECIFIC internal energy fixed and scale by
				// the new density, i.e. the added ejecta mass carries its proportional
				// share of internal energy too. Matches the real reference implementation
				// (SNFeedbackUtils::addCompositeBufferToState's e_int_new_tmp), rather than
				// holding the absolute Eint_old fixed as we had been doing, which implicitly
				// assumed the added mass carries zero extra internal energy.
				const double Eint_floor = (Eint_old / rho_old) * rho_new;

				const double dpx = d(i, j, k, 1);
				const double dpy = d(i, j, k, 2);
				const double dpz = d(i, j, k, 3);

				// Eq 21 limiter: find largest lambda in [0,1] s.t.
				//   Eint_floor + |p_old + lambda*dp|^2 / (2 rho_new) <= Etot_old + dE
				// Mass and energy are always added in full (matches paper: Delta e is fixed);
				// only the momentum *vector* is rescaled.
				const double a = dpx*dpx + dpy*dpy + dpz*dpz;
				const double b = 2.0 * (px_old*dpx + py_old*dpy + pz_old*dpz);
				const double c = (px_old*px_old + py_old*py_old + pz_old*pz_old)
					- 2.0 * rho_new * (Etot_old + dE - Eint_floor);

				double lambda = 1.0;
				if (a > 0.0) {
					// Feasible at lambda=1 iff a+b+c <= 0; else solve a*l^2+b*l+c=0 for
					// the largest root in [0,1].
					if (a + b + c > 0.0) {
						const double disc = b*b - 4.0*a*c;
						if (disc <= 0.0) {
							lambda = 0.0; // no physically valid nonzero kick this step
						} else {
							const double sq = std::sqrt(disc);
							const double l1 = (-b + sq) / (2.0*a);
							const double l2 = (-b - sq) / (2.0*a);
							const double lmax = amrex::max(l1, l2);
							lambda = amrex::max(0.0, amrex::min(1.0, lmax));
						}
					}
				}

				const double px_new = px_old + lambda * dpx;
				const double py_new = py_old + lambda * dpy;
				const double pz_new = pz_old + lambda * dpz;

				const double Etot_new = Etot_old + dE; // full energy always added (unscaled)
				const double Ekin_new = 0.5 * (px_new*px_new + py_new*py_new + pz_new*pz_new) / rho_new;
				const double Eint_new = Etot_new - Ekin_new - Emag; // absorbs any un-kicked energy as heat

				s(i, j, k, HydroSystem<MHDGalaxy>::density_index)        = rho_new;
				s(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index)     = px_new;
				s(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index)     = py_new;
				s(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index)     = pz_new;
				s(i, j, k, HydroSystem<MHDGalaxy>::internalEnergy_index) = Eint_new;
				s(i, j, k, HydroSystem<MHDGalaxy>::energy_index)         = Etot_new;
			});
		}
		amrex::Gpu::streamSynchronize();
	}
	AverageDown(); 
}
template <>
void QuokkaSimulation<MHDGalaxy>::ComputeDerivedVar(
    int lev, std::string const &dname, amrex::MultiFab &mf,
    const int ncomp_cc_in,
    amrex::MultiFab const &state_cc,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
    constexpr double cs_disk = quokka::EOS_Traits<MHDGalaxy>::cs_disk;
    constexpr double cs_cgm  = quokka::EOS_Traits<MHDGalaxy>::cs_cgm;

    const int  ncomp   = ncomp_cc_in;
    const auto prob_lo = geom[lev].ProbLoArray();
    const auto dx      = geom[lev].CellSizeArray();

    if (dname == "gpot") {
        auto const &phi_arr = phi[lev].const_arrays();
        auto        output  = mf.arrays();
        amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
            output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k);
        });
        amrex::Gpu::streamSynchronize();
        return;
    }

    if (dname == "pressure") {
        auto const &state_arrs = state_cc.const_arrays();
        auto        out_arrs   = mf.arrays();
        amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
            const double rho = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
            const double cs  = (rho > rho_transition) ? cs_disk : cs_cgm;
            out_arrs[bx](i, j, k, ncomp) = rho * cs * cs;
        });
        amrex::Gpu::streamSynchronize();
        return;
    }

    if (dname == "radius_sph") {
        auto out_arrs = mf.arrays();
        amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
            const double x = prob_lo[0] + (i + 0.5) * dx[0];
            const double y = prob_lo[1] + (j + 0.5) * dx[1];
            const double z = prob_lo[2] + (k + 0.5) * dx[2];
            out_arrs[bx](i, j, k, ncomp) =
                std::sqrt(x * x + y * y + z * z) / C::parsec / 1.0e3;
        });
        amrex::Gpu::streamSynchronize();
        return;
    }

    if (dname == "radial_velocity") {
        auto const &state_arrs = state_cc.const_arrays();
        auto        out_arrs   = mf.arrays();
        amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
            const double rho = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
            const double vx  = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) / rho;
            const double vy  = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) / rho;
            const double vz  = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index) / rho;
            const double x   = prob_lo[0] + (i + 0.5) * dx[0];
            const double y   = prob_lo[1] + (j + 0.5) * dx[1];
            const double z   = prob_lo[2] + (k + 0.5) * dx[2];
            const double r   = std::sqrt(x * x + y * y + z * z);
            out_arrs[bx](i, j, k, ncomp) =
                (r > 0.0) ? ((x * vx + y * vy + z * vz) / r) / 1.0e5 : 0.0;
        });
        amrex::Gpu::streamSynchronize();
        return;
    }

    if (dname == "circular_velocity") {
        auto const &state_arrs = state_cc.const_arrays();
        auto        out_arrs   = mf.arrays();
        amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
            const double rho   = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
            const double vx    = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) / rho;
            const double vy    = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) / rho;
            const double x     = prob_lo[0] + (i + 0.5) * dx[0];
            const double y     = prob_lo[1] + (j + 0.5) * dx[1];
            const double r_cyl = std::sqrt(x * x + y * y);
            out_arrs[bx](i, j, k, ncomp) =
                (r_cyl > 0.0) ? ((x * vy - y * vx) / r_cyl) / 1.0e5 : 0.0;
        });
        amrex::Gpu::streamSynchronize();
        return;
    }

    if (dname == "mach") {
        auto const &state_arrs = state_cc.const_arrays();
        auto        out_arrs   = mf.arrays();
        amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
            const double rho  = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
            const double momx = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index);
            const double momy = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index);
            const double momz = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index);
            const double cs   = (rho > rho_transition) ? cs_disk : cs_cgm;
            const double v2   = (momx * momx + momy * momy + momz * momz) / (rho * rho);
            out_arrs[bx](i, j, k, ncomp) = std::sqrt(v2) / cs;
        });
        amrex::Gpu::streamSynchronize();
        return;
    }

    if (dname == "plasma_beta") {
        auto const &state_arrs = state_cc.const_arrays();
        auto const &Bx_arrs    = state_fc[0].const_arrays();
        auto const &By_arrs    = state_fc[1].const_arrays();
        auto const &Bz_arrs    = state_fc[2].const_arrays();
        auto        out_arrs   = mf.arrays();
        amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
            const double rho   = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
            const double cs    = (rho > rho_transition) ? cs_disk : cs_cgm;
            const double Pgas  = rho * cs * cs;
            const double Bx_cc = 0.5 * (Bx_arrs[bx](i, j, k) + Bx_arrs[bx](i + 1, j, k));
            const double By_cc = 0.5 * (By_arrs[bx](i, j, k) + By_arrs[bx](i, j + 1, k));
            const double Bz_cc = 0.5 * (Bz_arrs[bx](i, j, k) + Bz_arrs[bx](i, j, k + 1));
            const double Pmag  = amrex::max(
                0.5 * (Bx_cc * Bx_cc + By_cc * By_cc + Bz_cc * Bz_cc), 1.0e-30);
            out_arrs[bx](i, j, k, ncomp) = Pgas / Pmag;
        });
        amrex::Gpu::streamSynchronize();
        return;
    }

    if (dname == "divB") {
        const double idx     = 1.0 / dx[0];
        const double idy     = 1.0 / dx[1];
        const double idz     = 1.0 / dx[2];
        auto const &Bx_arrs  = state_fc[0].const_arrays();
        auto const &By_arrs  = state_fc[1].const_arrays();
        auto const &Bz_arrs  = state_fc[2].const_arrays();
        auto        out_arrs = mf.arrays();
        amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
            out_arrs[bx](i, j, k, ncomp) =
                (Bx_arrs[bx](i + 1, j, k) - Bx_arrs[bx](i, j, k)) * idx +
                (By_arrs[bx](i, j + 1, k) - By_arrs[bx](i, j, k)) * idy +
                (Bz_arrs[bx](i, j, k + 1) - Bz_arrs[bx](i, j, k)) * idz;
        });
        amrex::Gpu::streamSynchronize();
        return;
    }

    if (dname == "Bphi") {
        auto const &Bx_arrs  = state_fc[0].const_arrays();
        auto const &By_arrs  = state_fc[1].const_arrays();
        auto        out_arrs = mf.arrays();
        amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
            const double x  = prob_lo[0] + (i + 0.5) * dx[0];
            const double y  = prob_lo[1] + (j + 0.5) * dx[1];
            const double R2 = x * x + y * y;
            if (R2 < 1e-20) {
                out_arrs[bx](i, j, k, ncomp) = 0.0;
                return;
            }
            const double Bx_cc = 0.5 * (Bx_arrs[bx](i, j, k) + Bx_arrs[bx](i + 1, j, k));
            const double By_cc = 0.5 * (By_arrs[bx](i, j, k) + By_arrs[bx](i, j + 1, k));
            out_arrs[bx](i, j, k, ncomp) = (By_cc * x - Bx_cc * y) / std::sqrt(R2);
        });
        amrex::Gpu::streamSynchronize();
        return;
    }
}

template <>
auto QuokkaSimulation<MHDGalaxy>::ComputeStatistics()
    -> std::map<std::string, amrex::Real>
{
    std::map<std::string, amrex::Real> stats;
    const amrex::Real R_min = 2.0 * 1.0e3 * C::parsec;
    const amrex::Real R_max = 8.0 * 1.0e3 * C::parsec;
    const amrex::Real z_max = 0.5 * 1.0e3 * C::parsec;

    // Threshold for "field-bearing" cells: well below seed B_rms (~1e-6 G), well above
    // the CGM's essentially-zero field, so the normalized divB ratio isn't dominated
    // by near-zero-B halo cells dividing tiny residuals into huge ratios.
    constexpr amrex::Real Bmag_threshold = 1.0e-9;

    amrex::Real total_E_tot = 0.0;
    amrex::Real total_E_tor = 0.0;
    amrex::Real total_E_pol = 0.0;

    amrex::Real divB_max_global = 0.0;
    amrex::Real divB_sumsq_global = 0.0;
    amrex::Real divB_norm_sumsq_global = 0.0;
    amrex::Real divB_norm_ncells_global = 0.0;

    using FaceStateArray = std::array<amrex::Array4<const Real>, AMREX_SPACEDIM>;

    const amrex::Real mean_density = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
            return state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
        });
    stats["mean_density"] = mean_density / geom[0].ProbSize();

    const amrex::Real disk_mass = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
            const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
            return (rho > rho_transition) ? rho : static_cast<amrex::Real>(0.0);
        });

    const amrex::Real disk_volume = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
            const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
            return (rho > rho_transition) ? static_cast<amrex::Real>(1.0) : static_cast<amrex::Real>(0.0);
        });

    const amrex::Real mean_disk_density =
        (disk_volume > 0.0) ? (disk_mass / disk_volume) : static_cast<amrex::Real>(1.0);

    stats["mean_disk_density"] = mean_disk_density;
    stats["disk_mass"]         = disk_mass / C::M_solar;

    const amrex::Real sigma_vol = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
            const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
            if (rho <= rho_transition) { return static_cast<amrex::Real>(0.0); }
            const amrex::Real eta = std::log(rho / mean_disk_density);
            return eta * eta;
        });

    stats["sigma_eta"] = (disk_volume > 0.0)
                             ? std::sqrt(sigma_vol / disk_volume)
                             : static_cast<amrex::Real>(0.0);

    for (int lev = 0; lev <= finest_level; ++lev) {
        const auto& geom_lev = geom[lev];
        auto const &state_fc = state_new_fc_[lev];
        auto const &state_fc_x = state_fc[0].const_arrays();
        auto const &state_fc_y = state_fc[1].const_arrays();
        auto const &state_fc_z = state_fc[2].const_arrays();
        const auto prob_lo = geom_lev.ProbLoArray();
        const auto dx = geom_lev.CellSizeArray();
        const amrex::Real vol = dx[0] * dx[1] * dx[2];
        const amrex::Real idx = 1.0 / dx[0];
        const amrex::Real idy = 1.0 / dx[1];
        const amrex::Real idz = 1.0 / dx[2];
        const amrex::Real dx_min = amrex::min(dx[0], amrex::min(dx[1], dx[2]));

        amrex::iMultiFab mask(state_new_cc_[lev].boxArray(), state_new_cc_[lev].DistributionMap(), 1, 0);
        if (lev < finest_level) {
            // covered cells -> 0, uncovered -> 1
            mask = amrex::makeFineMask(state_new_cc_[lev].boxArray(), state_new_cc_[lev].DistributionMap(),
                            state_new_cc_[lev + 1].boxArray(), refRatio(lev), 1, 0);
        } else {
            mask.setVal(1);
        }
        auto const &mask_arrs = mask.const_arrays();

        // Combined reduction: [0]=E_tot, [1]=E_tor, [2]=E_pol (annulus-masked),
        //                      [3]=max|divB| (whole domain), [4]=sum(divB^2 * vol) (whole domain),
        //                      [5]=sum(divB_norm^2) (field-bearing cells only),
        //                      [6]=field-bearing cell count
        auto level_result = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpMax, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>{},
            amrex::TypeList<amrex::Real, amrex::Real, amrex::Real,
                             amrex::Real, amrex::Real, amrex::Real, amrex::Real>{},
            state_new_cc_[lev],
            amrex::IntVect(0),
            1,
            [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int /*n*/) noexcept
                -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real,
                                    amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
                if (mask_arrs[bx](i, j, k) == 0) { return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; }

                const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
                const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
                const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
                const amrex::Real R = std::sqrt(x * x + y * y);

                const amrex::Real Bx_cc = 0.5 * (state_fc_x[bx](i, j, k) + state_fc_x[bx](i + 1, j, k));
                const amrex::Real By_cc = 0.5 * (state_fc_y[bx](i, j, k) + state_fc_y[bx](i, j + 1, k));
                const amrex::Real Bz_cc = 0.5 * (state_fc_z[bx](i, j, k) + state_fc_z[bx](i, j, k + 1));

                // --- divB diagnostics: raw max/RMS computed everywhere (whole domain), so
                //     boundary/grid-decomposition artifacts outside the disk are still caught ---
                const amrex::Real divB =
                    (state_fc_x[bx](i + 1, j, k) - state_fc_x[bx](i, j, k)) * idx +
                    (state_fc_y[bx](i, j + 1, k) - state_fc_y[bx](i, j, k)) * idy +
                    (state_fc_z[bx](i, j, k + 1) - state_fc_z[bx](i, j, k)) * idz;

                const amrex::Real divB_abs = std::abs(divB);
                const amrex::Real divB_sq_vol = divB * divB * vol;

                // --- normalized divB: restricted to field-bearing cells (Option A), so
                //     near-zero-B CGM cells don't blow up the ratio with tiny residuals ---
                const amrex::Real Bmag = std::sqrt(Bx_cc*Bx_cc + By_cc*By_cc + Bz_cc*Bz_cc);
                const bool has_field = (Bmag > Bmag_threshold);

                const amrex::Real divB_norm = has_field
                    ? (divB_abs * dx_min / Bmag)
                    : static_cast<amrex::Real>(0.0);
                const amrex::Real divB_norm_sq = has_field ? (divB_norm * divB_norm) : static_cast<amrex::Real>(0.0);
                const amrex::Real norm_cell_count = has_field ? static_cast<amrex::Real>(1.0) : static_cast<amrex::Real>(0.0);

                // --- annulus-restricted toroidal/poloidal energy stats (unchanged) ---
                amrex::Real e_tot = 0.0;
                amrex::Real e_tor = 0.0;
                amrex::Real e_pol = 0.0;
                if (R >= R_min && R <= R_max && std::abs(z) <= z_max) {
                    const amrex::Real invR = (R > 0.0) ? 1.0 / R : 0.0;
                    const amrex::Real cos_phi = x * invR;
                    const amrex::Real sin_phi = y * invR;

                    const amrex::Real Bphi = -Bx_cc * sin_phi + By_cc * cos_phi;
                    const amrex::Real Br   =  Bx_cc * cos_phi + By_cc * sin_phi;

                    e_tot = 0.5 * (Bx_cc * Bx_cc + By_cc * By_cc + Bz_cc * Bz_cc) * vol;
                    e_tor = 0.5 * (Bphi * Bphi) * vol;
                    e_pol = 0.5 * (Br * Br + Bz_cc * Bz_cc) * vol;
                }

                return {e_tot, e_tor, e_pol, divB_abs, divB_sq_vol, divB_norm_sq, norm_cell_count};
            });

        total_E_tot += amrex::get<0>(level_result);
        total_E_tor += amrex::get<1>(level_result);
        total_E_pol += amrex::get<2>(level_result);

        divB_max_global         = amrex::max(divB_max_global, amrex::get<3>(level_result));
        divB_sumsq_global       += amrex::get<4>(level_result);
        divB_norm_sumsq_global  += amrex::get<5>(level_result);
        divB_norm_ncells_global += amrex::get<6>(level_result);
    }

    // Combine energy sums across all MPI ranks
    std::array<amrex::Real, 3> global_sums = {total_E_tot, total_E_tor, total_E_pol};
    amrex::ParallelAllReduce::Sum(global_sums.data(), 3, amrex::ParallelContext::CommunicatorSub());

    stats["energy_Btot_annulus"] = global_sums[0];
    stats["energy_Btor_annulus"] = global_sums[1];
    stats["energy_Bpol_annulus"] = global_sums[2];

    // Combine divB diagnostics across all MPI ranks
    amrex::Real divB_max_reduced = divB_max_global;
    amrex::ParallelAllReduce::Max(divB_max_reduced, amrex::ParallelContext::CommunicatorSub());

    std::array<amrex::Real, 3> divB_sums = {divB_sumsq_global, divB_norm_sumsq_global, divB_norm_ncells_global};
    amrex::ParallelAllReduce::Sum(divB_sums.data(), 3, amrex::ParallelContext::CommunicatorSub());

    const amrex::Real total_volume = geom[0].ProbSize();
    stats["divB_max"]            = divB_max_reduced;
    stats["divB_rms"]            = std::sqrt(divB_sums[0] / total_volume);
    stats["divB_rms_normalized"] = (divB_sums[2] > 0.0)
        ? std::sqrt(divB_sums[1] / divB_sums[2])
        : static_cast<amrex::Real>(0.0);

    return stats;
}

auto problem_main() -> int
{
    auto BCs_cc = quokka::BC<MHDGalaxy>(quokka::BCType::reflecting);

    const int nvars_fc         = Physics_Indices<MHDGalaxy>::nvarTotal_fc;
    const int nvars_per_dim_fc = Physics_Indices<MHDGalaxy>::nvarPerDim_fc;
    amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
    for (int icomp = 0; icomp < nvars_fc; ++icomp) {
        const int component_dir =
            (nvars_per_dim_fc > 0) ? (icomp / nvars_per_dim_fc) : 0;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const int bc_type = (component_dir == idim)
                                    ? amrex::BCType::reflect_even
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
