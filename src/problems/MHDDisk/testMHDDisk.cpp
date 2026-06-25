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
	constexpr double keV_in_ergs = 1000.0 * C::ev2erg;
	constexpr double seconds_per_year = 3.15576e7;
	constexpr double Rd_kpc = 3.0;
	constexpr double Rc_kpc = 2.0;
	constexpr double Rd = Rd_kpc * 1.0e3 * C::parsec;
	constexpr double Rc = Rc_kpc * 1.0e3 * C::parsec;
	constexpr double alpha_profile = 2.0;
	constexpr double beta_profile  = 0.5;
	constexpr double q_flatten     = 0.7;
	constexpr double rho_transition = 1.0e-28;
	constexpr double target_beta_seed = 1.0e4;
	constexpr double Rmax_kpc = 8.0;
	constexpr double Rmax = Rmax_kpc * 1.0e3 * C::parsec;
	constexpr double refine_Rcyl_kpc = 9.0;
	constexpr double refine_Hcyl_pc  = 600.0;
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
	static constexpr double T_cgm =  1.0e7;
	static constexpr double cs_cgm = gcem::sqrt(gamma * C::k_B * T_cgm / mean_molecular_weight);
	static constexpr double cs_disk = 7.0e5;
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
	amrex::Real seed_B0_gauss{};
	amrex::Real seed_B0_HL{};
	amrex::Real seed{};

	// Vector allocation on the GPU
	amrex::Gpu::DeviceVector<amrex::Real> Aphi_device;
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
		<< " Seed=" << userData_.seed << "\n";
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
	//const double B0_scale = 1.0; 
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
    const double axis_dead_zone = 1.0 * dR_table;

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
	const double axis_dead_zone = 1.0 * dR_table;


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

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real x0 = prob_lo[0] + i * dx[0];
		const amrex::Real y0 = prob_lo[1] + j * dx[1];
		const amrex::Real z0 = prob_lo[2] + k * dx[2];
		const amrex::Real x1 = x0 + dx[0];
		const amrex::Real y1 = y0 + dx[1];
		const amrex::Real z1 = z0 + dx[2];

		auto tagIfInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			if (std::sqrt(x*x + y*y) < refine_Rcyl && std::abs(z) < refine_Hcyl) {
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


template <>
void QuokkaSimulation<MHDGalaxy>::ComputeDerivedVar(
    int lev, std::string const &dname, amrex::MultiFab &mf,
    const int ncomp_cc_in) const
{
    constexpr double cs_disk = quokka::EOS_Traits<MHDGalaxy>::cs_disk;
    constexpr double cs_cgm  = quokka::EOS_Traits<MHDGalaxy>::cs_cgm;

    const int  ncomp   = ncomp_cc_in;
    const auto prob_lo = geom[lev].ProbLoArray();
    const auto dx      = geom[lev].CellSizeArray();

    for (int l = finest_level - 1; l >= 0; --l) {
        amrex::Array<const amrex::MultiFab *, AMREX_SPACEDIM> fine_ptrs = {
            &(state_new_fc_[l + 1][0]),
            &(state_new_fc_[l + 1][1]),
            &(state_new_fc_[l + 1][2])};
		amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> crse_ptrs = {
			const_cast<amrex::MultiFab *>(&(state_new_fc_[l][0])),
			const_cast<amrex::MultiFab *>(&(state_new_fc_[l][1])),
			const_cast<amrex::MultiFab *>(&(state_new_fc_[l][2]))};
        amrex::average_down_faces(fine_ptrs, crse_ptrs, refRatio(l), geom[l]);
    }
    for (int l = 0; l <= finest_level; ++l) {
        for (int dir = 0; dir < 3; ++dir) {
            const_cast<amrex::MultiFab &>(state_new_fc_[l][dir]).FillBoundary(geom[l].periodicity());
        }
    }

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
        auto const &state_arrs = state_new_cc_[lev].const_arrays();
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
        auto const &state_arrs = state_new_cc_[lev].const_arrays();
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
        auto const &state_arrs = state_new_cc_[lev].const_arrays();
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
        auto const &state_arrs = state_new_cc_[lev].const_arrays();
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
        auto const &state_arrs = state_new_cc_[lev].const_arrays();
        auto const &Bx_arrs    = state_new_fc_[lev][0].const_arrays();
        auto const &By_arrs    = state_new_fc_[lev][1].const_arrays();
        auto const &Bz_arrs    = state_new_fc_[lev][2].const_arrays();
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
        auto const &Bx_arrs  = state_new_fc_[lev][0].const_arrays();
        auto const &By_arrs  = state_new_fc_[lev][1].const_arrays();
        auto const &Bz_arrs  = state_new_fc_[lev][2].const_arrays();
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
        auto const &Bx_arrs  = state_new_fc_[lev][0].const_arrays();
        auto const &By_arrs  = state_new_fc_[lev][1].const_arrays();
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

    const amrex::Real mean_density = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state) noexcept {
            return state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
        });
    stats["mean_density"] = mean_density / geom[0].ProbSize();

    const amrex::Real disk_mass = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state) noexcept {
            const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
            return (rho > rho_transition) ? rho : amrex::Real(0.0);
        });

    const amrex::Real disk_volume = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state) noexcept {
            const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
            return (rho > rho_transition) ? amrex::Real(1.0) : amrex::Real(0.0);
        });

    const amrex::Real mean_disk_density =
        (disk_volume > 0.0) ? (disk_mass / disk_volume) : amrex::Real(1.0);

    stats["mean_disk_density"] = mean_disk_density;
    stats["disk_mass"]         = disk_mass / C::M_solar;

    const amrex::Real sigma_vol = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state) noexcept {
            const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
            if (rho <= rho_transition) { return amrex::Real(0.0); }
            const amrex::Real eta = std::log(rho / mean_disk_density);
            return eta * eta;
        });

    stats["sigma_eta"] = (disk_volume > 0.0)
                             ? std::sqrt(sigma_vol / disk_volume)
                             : amrex::Real(0.0);

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
