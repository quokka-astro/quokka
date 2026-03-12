/// \file testGravBonnorEbertSphere.cpp
/// \brief Defines a test problem for a self-gravitating isothermal Bonnor-Ebert sphere.
///
/// The Bonnor-Ebert sphere is an isothermal self-gravitating gas sphere in
/// hydrostatic equilibrium, described by the isothermal Lane-Emden equation:
///   (1/ξ²) d/dξ(ξ² dψ/dξ) = e^(-ψ)
/// where ψ = ln(ρ_c/ρ) and ξ = r/r_0 with r_0 = c_s/sqrt(4πGρ_c).
///
/// The critical Bonnor-Ebert sphere has ξ_max ≈ 6.451 and density contrast
/// ρ_c/ρ_edge ≈ 14.04. At the critical state, the sphere is marginally stable.
///
/// This test validates:
///   1. Stability: with the exact critical density, the sphere remains in
///      approximate hydrostatic equilibrium.
///   2. Collapse: with an overdensity factor > 1, the sphere collapses
///      (central density increases).

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "util/BC.hpp"

#include <cmath>
#include <vector>

struct BESphereProblem {
};

// ============================================================
// Physical parameters (typical star-forming molecular cloud core)
// ============================================================
constexpr double mu = 2.33 * C::m_p;	  // mean molecular weight (molecular H2 + He)
constexpr double gamma_ = 1.001;	  // nearly isothermal (gamma -> 1 limit)
constexpr double T0 = 10.0;		  // temperature [K]
constexpr double xi_crit = 6.451;	  // critical dimensionless radius
constexpr int n_profile = 10000;	  // number of points in Lane-Emden profile
constexpr double rho_floor = 1.0e-25;	  // density floor [g cm^-3]
constexpr double pressure_contrast = 1.; // P_ext / P_edge (set to 1 for pressure equilibrium)

// Runtime parameters (set from input file)
static double rho_c = 3.0e-18;		// central density [g cm^-3] - NOLINT
static double overdensity_factor = 1.0; // 1.0 = critical, >1.0 = collapse - NOLINT

// ============================================================
// Template specializations
// ============================================================
template <> struct quokka::EOS_Traits<BESphereProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<BESphereProblem> {
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<BESphereProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// ============================================================
// Lane-Emden solver for isothermal sphere
// ============================================================
// Solves: (1/ξ²) d/dξ(ξ² dψ/dξ) = e^(-ψ)
// Rewritten as system:
//   dψ/dξ = u
//   du/dξ = e^(-ψ) - 2u/ξ
// BCs: ψ(0) = 0, u(0) = 0
struct LaneEmdenSolution {
	std::vector<double> xi;  // dimensionless radius
	std::vector<double> psi; // ψ = ln(ρ_c/ρ)
	double xi_max{};	  // outer dimensionless radius
	double r0{};		  // length scale c_s / sqrt(4πGρ_c) [cm]
	double cs{};		  // isothermal sound speed [cm/s]
	double R_sphere{};	  // physical outer radius [cm]
};

auto solveLaneEmden(double rho_central, double xi_outer, int npts) -> LaneEmdenSolution
{
	LaneEmdenSolution sol;
	sol.xi.resize(npts);
	sol.psi.resize(npts);
	sol.xi_max = xi_outer;

	// Sound speed
	sol.cs = std::sqrt(C::k_B * T0 / mu);
	// Length scale
	sol.r0 = sol.cs / std::sqrt(4.0 * M_PI * C::Gconst * rho_central);
	sol.R_sphere = xi_outer * sol.r0;

	const double dxi = xi_outer / static_cast<double>(npts - 1);

	// Initial conditions at ξ = 0
	// Use Taylor expansion near origin: ψ ≈ ξ²/6 - ξ⁴/120 + ...
	sol.xi[0] = 0.0;
	sol.psi[0] = 0.0;
	double u = 0.0; // dψ/dξ

	// RK4 integration
	for (int i = 1; i < npts; ++i) {
		const double xi_cur = static_cast<double>(i - 1) * dxi;
		const double psi_cur = sol.psi[i - 1];
		const double u_cur = u;

		// Handle ξ = 0 singularity using L'Hôpital: 2u/ξ -> 2/3 * e^(-ψ) at ξ=0
		auto dudt = [](double xi_val, double psi_val, double u_val) -> double {
			if (xi_val < 1.0e-10) {
				return std::exp(-psi_val) / 3.0; // L'Hôpital limit
			}
			return std::exp(-psi_val) - 2.0 * u_val / xi_val;
		};

		// k1
		const double k1_psi = u_cur;
		const double k1_u = dudt(xi_cur, psi_cur, u_cur);

		// k2
		const double xi_half = xi_cur + 0.5 * dxi;
		const double psi_k2 = psi_cur + 0.5 * dxi * k1_psi;
		const double u_k2 = u_cur + 0.5 * dxi * k1_u;
		const double k2_psi = u_k2;
		const double k2_u = dudt(xi_half, psi_k2, u_k2);

		// k3
		const double psi_k3 = psi_cur + 0.5 * dxi * k2_psi;
		const double u_k3 = u_cur + 0.5 * dxi * k2_u;
		const double k3_psi = u_k3;
		const double k3_u = dudt(xi_half, psi_k3, u_k3);

		// k4
		const double xi_next = xi_cur + dxi;
		const double psi_k4 = psi_cur + dxi * k3_psi;
		const double u_k4 = u_cur + dxi * k3_u;
		const double k4_psi = u_k4;
		const double k4_u = dudt(xi_next, psi_k4, u_k4);

		sol.xi[i] = xi_next;
		sol.psi[i] = psi_cur + (dxi / 6.0) * (k1_psi + 2.0 * k2_psi + 2.0 * k3_psi + k4_psi);
		u = u_cur + (dxi / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u);
	}

	return sol;
}

// ============================================================
// Store profile in device-accessible memory
// ============================================================
static amrex::Gpu::DeviceVector<double> d_xi_arr;   // NOLINT
static amrex::Gpu::DeviceVector<double> d_psi_arr;  // NOLINT
static double R_sphere_global = 0.0;		     // NOLINT
static double r0_global = 0.0;			     // NOLINT
static double rho_c_global = 0.0;		     // NOLINT
static double cs_global = 0.0;			     // NOLINT
static double rho_edge_global = 0.0;		     // NOLINT
static double P_ext_global = 0.0;		     // NOLINT

// ============================================================
// Initial conditions
// ============================================================
template <> void QuokkaSimulation<BESphereProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;

	// Domain center
	const double x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	const double y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	const double z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// Capture profile data for GPU
	const double *xi_ptr = d_xi_arr.data();
	const double *psi_ptr = d_psi_arr.data();
	const int npts = static_cast<int>(d_xi_arr.size());
	const double R_sph = R_sphere_global;
	const double r0_val = r0_global;
	const double rho_central = rho_c_global;
	const double rho_edge = rho_edge_global;
	const double P_ext = P_ext_global;
	const double cs_val = cs_global;
	const double rho_floor_val = rho_floor;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		const double y = prob_lo[1] + (j + 0.5) * dx[1];
		const double z = prob_lo[2] + (k + 0.5) * dx[2];
		const double r = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

		double rho = 0.0;
		double P = 0.0;

		if (r <= R_sph) {
			// Inside the sphere: interpolate Lane-Emden profile
			const double xi_val = r / r0_val;
			const double dxi = xi_ptr[npts - 1] / static_cast<double>(npts - 1);
			const int idx = static_cast<int>(xi_val / dxi);

			double psi_val = 0.0;
			if (idx >= npts - 1) {
				psi_val = psi_ptr[npts - 1];
			} else {
				// Linear interpolation
				const double frac = (xi_val - xi_ptr[idx]) / (xi_ptr[idx + 1] - xi_ptr[idx]);
				psi_val = psi_ptr[idx] + frac * (psi_ptr[idx + 1] - psi_ptr[idx]);
			}

			rho = rho_central * std::exp(-psi_val);
			P = rho * cs_val * cs_val; // isothermal: P = ρ c_s²
		} else {
			// Outside the sphere: uniform ambient medium at edge pressure
			rho = rho_edge;
			P = P_ext;
		}

		rho = amrex::max(rho, rho_floor_val);

		state_cc(i, j, k, HydroSystem<BESphereProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<BESphereProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<BESphereProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<BESphereProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<BESphereProblem>::energy_index) = quokka::EOS<BESphereProblem>::ComputeEintFromPres(rho, P);
		state_cc(i, j, k, HydroSystem<BESphereProblem>::internalEnergy_index) = quokka::EOS<BESphereProblem>::ComputeEintFromPres(rho, P);
	});
}

// ============================================================
// AMR refinement: refine on density
// ============================================================
template <> void QuokkaSimulation<BESphereProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const Real q_min = 2.0 * rho_edge_global; // refine where density > 2x edge density

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<BESphereProblem>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real q = state(i, j, k, nidx);
			if (q > q_min) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

// ============================================================
// Derived variable: gravitational potential
// ============================================================
template <>
void QuokkaSimulation<BESphereProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	if (dname == "gpot") {
		const int ncomp = ncomp_cc_in;
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
	}
}

// ============================================================
// Main
// ============================================================
auto problem_main() -> int
{
	// Read runtime parameters
	amrex::ParmParse const pp("problem");
	pp.query("rho_c", rho_c);
	pp.query("overdensity_factor", overdensity_factor);

	// Apply overdensity factor to central density
	const double rho_central = rho_c * overdensity_factor;

	amrex::Print() << "\n=== Bonnor-Ebert Sphere Test ===\n";
	amrex::Print() << "Central density rho_c = " << rho_c << " g/cm^3\n";
	amrex::Print() << "Overdensity factor = " << overdensity_factor << "\n";
	amrex::Print() << "Effective central density = " << rho_central << " g/cm^3\n";
	amrex::Print() << "Temperature = " << T0 << " K\n";
	amrex::Print() << "Mean molecular weight = " << mu / C::m_p << " m_p\n";

	// Solve Lane-Emden equation
	const LaneEmdenSolution sol = solveLaneEmden(rho_central, xi_crit, n_profile);

	amrex::Print() << "Sound speed c_s = " << sol.cs << " cm/s\n";
	amrex::Print() << "Length scale r_0 = " << sol.r0 << " cm (" << sol.r0 / C::parsec << " pc)\n";
	amrex::Print() << "Sphere radius R = " << sol.R_sphere << " cm (" << sol.R_sphere / C::parsec << " pc)\n";

	// Density at the edge
	const double psi_edge = sol.psi[n_profile - 1];
	const double rho_edge = rho_central * std::exp(-psi_edge);
	amrex::Print() << "Edge density rho_edge = " << rho_edge << " g/cm^3\n";
	amrex::Print() << "Density contrast rho_c/rho_edge = " << rho_central / rho_edge << "\n";

	// Store globals for GPU
	R_sphere_global = sol.R_sphere;
	r0_global = sol.r0;
	rho_c_global = rho_central;
	cs_global = sol.cs;
	rho_edge_global = rho_edge;
	P_ext_global = pressure_contrast * rho_edge * sol.cs * sol.cs;

	// Copy profile to device
	d_xi_arr.resize(n_profile);
	d_psi_arr.resize(n_profile);
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, sol.xi.begin(), sol.xi.end(), d_xi_arr.begin());
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, sol.psi.begin(), sol.psi.end(), d_psi_arr.begin());

	// Compute free-fall time for reference
	const double t_ff = std::sqrt(3.0 * M_PI / (32.0 * C::Gconst * rho_central));
	amrex::Print() << "Free-fall time t_ff = " << t_ff << " s (" << t_ff / (3.15576e7) << " yr)\n";

	// Compute sound crossing time
	const double t_sc = sol.R_sphere / sol.cs;
	amrex::Print() << "Sound crossing time t_sc = " << t_sc << " s (" << t_sc / (3.15576e7) << " yr)\n";

	// ============================================================
	// Run the simulation
	// ============================================================
	QuokkaSimulation<BESphereProblem> sim;

	sim.reconstructionOrder_ = 3; // PPM
	sim.cflNumber_ = 0.3;

	// Initialize
	sim.setInitialConditions();

	// Record initial central density (max density on grid)
	const amrex::Real rho_max_init = sim.state_new_cc_[0].max(HydroSystem<BESphereProblem>::density_index);
	amrex::Print() << "\nInitial max density on grid = " << rho_max_init << " g/cm^3\n";

	// Evolve
	sim.evolve();

	// Record final central density
	const amrex::Real rho_max_final = sim.state_new_cc_[0].max(HydroSystem<BESphereProblem>::density_index);
	amrex::Print() << "\nFinal max density on grid = " << rho_max_final << " g/cm^3\n";

	const double rho_change_frac = (rho_max_final - rho_max_init) / rho_max_init;
	amrex::Print() << "Fractional change in max density = " << rho_change_frac << "\n";

	int status = 0;

	if (overdensity_factor <= 1.0) {
		// Stability test: density should not change significantly
		// Allow up to 10% change (numerical diffusion causes some drift)
		const double stability_tol = 0.10;
		if (std::abs(rho_change_frac) > stability_tol) {
			amrex::Print() << "FAIL: Sphere is not stable (density changed by " << rho_change_frac * 100.0 << "%)\n";
			status = 1;
		} else {
			amrex::Print() << "PASS: Sphere remains approximately stable (density changed by " << rho_change_frac * 100.0 << "%)\n";
		}
	} else {
		// Collapse test: central density should increase
		if (rho_change_frac <= 0.0) {
			amrex::Print() << "FAIL: Sphere did not collapse (density did not increase)\n";
			status = 1;
		} else {
			amrex::Print() << "PASS: Sphere is collapsing (density increased by " << rho_change_frac * 100.0 << "%)\n";
		}
	}

	return status;
}
