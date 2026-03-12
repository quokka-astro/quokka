/// \file testGravBonnorEbertSphere.cpp
/// \brief Defines a test problem for a self-gravitating isothermal Bonnor-Ebert sphere
///        with dust that contributes to the gravitational potential.
///
/// ## Physics
/// The Bonnor-Ebert sphere is an isothermal self-gravitating gas sphere in
/// hydrostatic equilibrium, described by the isothermal Lane-Emden equation:
///   (1/ξ²) d/dξ(ξ² dψ/dξ) = e^(-ψ)
/// where ψ = ln(ρ_c/ρ) and ξ = r/r_0 with r_0 = c_s/sqrt(4πGρ_c_total).
///
/// ## Dust setup
/// Dust with total density ρ_dust = dust_fraction × ρ_gas is added, distributed
/// evenly into two groups. Dust is perfectly coupled to gas (short stopping time).
/// With dust contributing to the Poisson source, the equilibrium condition requires
/// the Lane-Emden length scale r_0 to use the TOTAL central density ρ_c_total:
///   r_0 = c_s / sqrt(4πG ρ_c_total)  where  ρ_c_total = (1 + dust_fraction) × ρ_c_gas
///
/// The equilibrium density distribution is:
///   ρ_total(r) = ρ_c_total × e^{-ψ(ξ)}
///   ρ_gas(r)   = ρ_c_total / (1 + dust_fraction) × e^{-ψ(ξ)}
///   ρ_dust_i(r) = ρ_c_total × dust_fraction/(2(1+dust_fraction)) × e^{-ψ(ξ)}
///
/// Gas provides 1/(1+dust_fraction) = 2/3 of total gravity needed;
/// dust provides dust_fraction/(1+dust_fraction) = 1/3.
///
/// ## Validation
///   1. Stability: at critical total density the sphere stays in near-hydrostatic
///      equilibrium (gas+dust gravity together balances the gas pressure gradient).
///   2. Collapse: with overdensity_factor > 1 the sphere collapses.

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
constexpr double mu = 2.33 * C::m_p;	    // mean molecular weight (molecular H2 + He)
constexpr double gamma_ = 1.001;	    // nearly isothermal (gamma -> 1 limit)
constexpr double T0 = 10.0;		    // temperature [K]
constexpr double xi_crit = 6.451;	    // critical dimensionless radius of BE sphere
constexpr int n_profile = 10000;	    // number of points in Lane-Emden profile
constexpr double rho_floor = 1.0e-25;	    // density floor [g cm^-3]
constexpr double dust_fraction = 0.5;	    // ρ_dust_total / ρ_gas (= f)
constexpr int n_dust_groups = 2;	    // number of dust groups (each gets dust_fraction/2)
constexpr double t_stop_s = 1.0e8;	    // stopping time [s], << t_ff to enforce tight coupling

// Runtime parameters (set from input file)
static double rho_c = 3.0e-18;		// total central density ρ_c_total [g cm^-3] - NOLINT
static double overdensity_factor = 1.0; // 1.0 = critical (stable), >1.0 = collapse - NOLINT
static double overdensity_global = 1.0; // NOLINT

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
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = n_dust_groups;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// ============================================================
// Dust drag: perfectly coupled (t_stop << t_ff)
// ============================================================
template <>
AMREX_GPU_HOST_DEVICE auto
DustDrag<BESphereProblem>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
							 amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/, double /*cs*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> alpha{};
	for (int g = 0; g < nDustGroups_; ++g) {
		alpha[g] = 1.0 / t_stop_s;
	}
	return alpha;
}

// ============================================================
// Lane-Emden solver for isothermal sphere (RK4)
// ============================================================
// Solves: (1/ξ²) d/dξ(ξ² dψ/dξ) = e^(-ψ)
// with BCs ψ(0) = 0, u(0) = 0
// rho_central is the TOTAL central density (gas + all dust) used for r_0.
struct LaneEmdenSolution {
	std::vector<double> xi;  // dimensionless radius
	std::vector<double> psi; // ψ = ln(ρ_c_total/ρ_total)
	double xi_max{};	  // outer dimensionless radius
	double r0{};		  // length scale c_s / sqrt(4πGρ_c_total) [cm]
	double cs{};		  // isothermal sound speed [cm/s]
	double R_sphere{};	  // physical outer radius [cm]
};

auto solveLaneEmden(double rho_central_total, double xi_outer, int npts) -> LaneEmdenSolution
{
	LaneEmdenSolution sol;
	sol.xi.resize(npts);
	sol.psi.resize(npts);
	sol.xi_max = xi_outer;

	sol.cs = std::sqrt(C::k_B * T0 / mu);
	sol.r0 = sol.cs / std::sqrt(4.0 * M_PI * C::Gconst * rho_central_total);
	sol.R_sphere = xi_outer * sol.r0;

	const double dxi = xi_outer / static_cast<double>(npts - 1);

	sol.xi[0] = 0.0;
	sol.psi[0] = 0.0;
	double u = 0.0;

	auto dudt = [](double xi_val, double psi_val, double u_val) -> double {
		if (xi_val < 1.0e-10) {
			return std::exp(-psi_val) / 3.0; // L'Hôpital at ξ=0
		}
		return std::exp(-psi_val) - 2.0 * u_val / xi_val;
	};

	for (int i = 1; i < npts; ++i) {
		const double xi_cur = static_cast<double>(i - 1) * dxi;
		const double psi_cur = sol.psi[i - 1];
		const double u_cur = u;

		const double k1_psi = u_cur;
		const double k1_u = dudt(xi_cur, psi_cur, u_cur);

		const double xi_half = xi_cur + 0.5 * dxi;
		const double k2_psi = u_cur + 0.5 * dxi * k1_u;
		const double k2_u = dudt(xi_half, psi_cur + 0.5 * dxi * k1_psi, k2_psi);

		const double k3_psi = u_cur + 0.5 * dxi * k2_u;
		const double k3_u = dudt(xi_half, psi_cur + 0.5 * dxi * k2_psi, k3_psi);

		const double xi_next = xi_cur + dxi;
		const double k4_psi = u_cur + dxi * k3_u;
		const double k4_u = dudt(xi_next, psi_cur + dxi * k3_psi, k4_psi);

		sol.xi[i] = xi_next;
		sol.psi[i] = psi_cur + (dxi / 6.0) * (k1_psi + 2.0 * k2_psi + 2.0 * k3_psi + k4_psi);
		u = u_cur + (dxi / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u);
	}

	return sol;
}

// ============================================================
// Profile stored in device-accessible memory
// ============================================================
static amrex::Gpu::DeviceVector<double> d_xi_arr;  // NOLINT
static amrex::Gpu::DeviceVector<double> d_psi_arr; // NOLINT
static double R_sphere_global = 0.0;		    // NOLINT
static double r0_global = 0.0;			    // NOLINT
static double rho_c_total_global = 0.0;		    // total central density [g/cm³] - NOLINT
static double cs_global = 0.0;			    // NOLINT
static double rho_gas_edge_global = 0.0;	    // NOLINT
static double P_ext_global = 0.0;		    // NOLINT

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

	const double x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	const double y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	const double z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	const double *xi_ptr = d_xi_arr.data();
	const double *psi_ptr = d_psi_arr.data();
	const int npts = static_cast<int>(d_xi_arr.size());
	const double R_sph = R_sphere_global;
	const double r0_val = r0_global;
	const double rho_c_tot = rho_c_total_global;
	const double rho_gas_edge = rho_gas_edge_global;
	const double P_ext = P_ext_global;
	const double cs_val = cs_global;
	const double rho_floor_val = rho_floor;
	const double od_factor = overdensity_global;

	// Fractions of total density per species
	// gas:         1 / (1 + dust_fraction)
	// each dust group: (dust_fraction / n_dust_groups) / (1 + dust_fraction)
	constexpr double gas_frac = 1.0 / (1.0 + dust_fraction);
	constexpr double dust_frac_per_group = (dust_fraction / n_dust_groups) / (1.0 + dust_fraction);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		const double y = prob_lo[1] + (j + 0.5) * dx[1];
		const double z = prob_lo[2] + (k + 0.5) * dx[2];
		const double r = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

		double rho_gas = 0.0;
		double P = 0.0;
		double rho_dust_per_group = 0.0;

		if (r <= R_sph) {
			// Interpolate Lane-Emden equilibrium profile
			const double xi_val = r / r0_val;
			const double dxi_val = xi_ptr[npts - 1] / static_cast<double>(npts - 1);
			const int idx = static_cast<int>(xi_val / dxi_val);

			double psi_val = 0.0;
			if (idx >= npts - 1) {
				psi_val = psi_ptr[npts - 1];
			} else {
				const double frac = (xi_val - xi_ptr[idx]) / (xi_ptr[idx + 1] - xi_ptr[idx]);
				psi_val = psi_ptr[idx] + frac * (psi_ptr[idx + 1] - psi_ptr[idx]);
			}

			const double rho_total_eq = rho_c_tot * std::exp(-psi_val); // equilibrium total density
			rho_gas = rho_total_eq * gas_frac * od_factor;		     // overdensity applied to gas
			rho_dust_per_group = rho_total_eq * dust_frac_per_group * od_factor; // overdensity applied to dust
			P = rho_total_eq * gas_frac * cs_val * cs_val;			     // pressure from EQUILIBRIUM gas density
		} else {
			rho_gas = rho_gas_edge;
			rho_dust_per_group = rho_gas_edge * (dust_fraction / n_dust_groups);
			P = P_ext;
		}

		rho_gas = amrex::max(rho_gas, rho_floor_val);
		rho_dust_per_group = amrex::max(rho_dust_per_group, rho_floor_val);

		state_cc(i, j, k, HydroSystem<BESphereProblem>::density_index) = rho_gas;
		state_cc(i, j, k, HydroSystem<BESphereProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<BESphereProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<BESphereProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<BESphereProblem>::energy_index) = quokka::EOS<BESphereProblem>::ComputeEintFromPres(rho_gas, P);
		state_cc(i, j, k, HydroSystem<BESphereProblem>::internalEnergy_index) = quokka::EOS<BESphereProblem>::ComputeEintFromPres(rho_gas, P);

		// dust groups: each with rho_dust_per_group, zero velocity
		for (int g = 0; g < n_dust_groups; ++g) {
			const int stride = g * HydroSystem<BESphereProblem>::numDustVars_;
			state_cc(i, j, k, HydroSystem<BESphereProblem>::dustDensity_index + stride) = rho_dust_per_group;
			state_cc(i, j, k, HydroSystem<BESphereProblem>::x1DustMomentum_index + stride) = 0.0;
			state_cc(i, j, k, HydroSystem<BESphereProblem>::x2DustMomentum_index + stride) = 0.0;
			state_cc(i, j, k, HydroSystem<BESphereProblem>::x3DustMomentum_index + stride) = 0.0;
		}
	});
}

// ============================================================
// AMR refinement: tag on total density
// ============================================================
template <> void QuokkaSimulation<BESphereProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const Real q_min = 2.0 * rho_gas_edge_global;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<BESphereProblem>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			if (state(i, j, k, nidx) > q_min) {
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
	amrex::ParmParse const pp("problem");
	pp.query("rho_c", rho_c);
	pp.query("overdensity_factor", overdensity_factor);

	amrex::Print() << "\n=== Bonnor-Ebert Sphere + Dust Test ===\n";
	amrex::Print() << "Total central density rho_c = " << rho_c << " g/cm^3\n";
	amrex::Print() << "  Gas central density = " << rho_c / (1.0 + dust_fraction) << " g/cm^3\n";
	amrex::Print() << "  Dust central density (per group) = " << rho_c * dust_fraction / (n_dust_groups * (1.0 + dust_fraction)) << " g/cm^3\n";
	amrex::Print() << "Dust fraction (rho_dust/rho_gas) = " << dust_fraction << "\n";
	amrex::Print() << "  Gas provides " << 1.0 / (1.0 + dust_fraction) * 100.0 << "% of total gravity\n";
	amrex::Print() << "  Dust provides " << dust_fraction / (1.0 + dust_fraction) * 100.0 << "% of total gravity\n";
	amrex::Print() << "Overdensity factor = " << overdensity_factor << "\n";
	amrex::Print() << "Temperature = " << T0 << " K\n";

	// Lane-Emden solution. The correct length scale for gas+dust equilibrium is:
	//   r_0 = c_s / sqrt(4*pi*G * (1+f)^2 * rho_c_gas)
	//       = c_s / sqrt(4*pi*G * (1+f) * rho_c_total)
	// so we pass (1+f)*rho_c_total as the effective density to solveLaneEmden.
	// The profile ψ(ξ) satisfies the standard isothermal Lane-Emden equation and
	// maps to total density via ρ_total(r) = rho_c_total * exp(-ψ(r/r_0)).
	const LaneEmdenSolution sol = solveLaneEmden(rho_c * (1.0 + dust_fraction), xi_crit, n_profile);

	amrex::Print() << "Sound speed c_s = " << sol.cs << " cm/s\n";
	amrex::Print() << "Length scale r_0 = " << sol.r0 << " cm (" << sol.r0 / C::parsec << " pc)\n";
	amrex::Print() << "Sphere radius R = " << sol.R_sphere << " cm (" << sol.R_sphere / C::parsec << " pc)\n";

	const double psi_edge = sol.psi[n_profile - 1];
	const double rho_total_edge = rho_c * std::exp(-psi_edge);
	const double rho_gas_edge = rho_total_edge / (1.0 + dust_fraction);
	amrex::Print() << "Total edge density = " << rho_total_edge << " g/cm^3\n";
	amrex::Print() << "Gas edge density = " << rho_gas_edge << " g/cm^3\n";
	amrex::Print() << "Total density contrast = " << rho_c / rho_total_edge << "\n";

	R_sphere_global = sol.R_sphere;
	r0_global = sol.r0;
	rho_c_total_global = rho_c;
	cs_global = sol.cs;
	rho_gas_edge_global = rho_gas_edge;
	P_ext_global = rho_gas_edge * sol.cs * sol.cs;
	overdensity_global = overdensity_factor;

	d_xi_arr.resize(n_profile);
	d_psi_arr.resize(n_profile);
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, sol.xi.begin(), sol.xi.end(), d_xi_arr.begin());
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, sol.psi.begin(), sol.psi.end(), d_psi_arr.begin());

	const double rho_eff = rho_c * overdensity_factor;
	const double t_ff = std::sqrt(3.0 * M_PI / (32.0 * C::Gconst * rho_eff));
	const double t_sc = sol.R_sphere / sol.cs;
	amrex::Print() << "Free-fall time t_ff = " << t_ff << " s (" << t_ff / (3.15576e7) << " yr)\n";
	amrex::Print() << "Sound crossing time t_sc = " << t_sc << " s (" << t_sc / (3.15576e7) << " yr)\n";
	amrex::Print() << "Stopping time t_stop = " << t_stop_s << " s (" << t_stop_s / t_ff << " t_ff) — tight coupling\n";

	QuokkaSimulation<BESphereProblem> sim;
	sim.reconstructionOrder_ = 3;
	sim.cflNumber_ = 0.3;

	sim.setInitialConditions();

	const amrex::Real rho_max_init = sim.state_new_cc_[0].max(HydroSystem<BESphereProblem>::density_index);
	amrex::Print() << "\nInitial max gas density = " << rho_max_init << " g/cm^3\n";

	sim.evolve();

	const amrex::Real rho_max_final = sim.state_new_cc_[0].max(HydroSystem<BESphereProblem>::density_index);
	amrex::Print() << "\nFinal max gas density = " << rho_max_final << " g/cm^3\n";

	const double rho_change_frac = (rho_max_final - rho_max_init) / rho_max_init;
	amrex::Print() << "Fractional change in max gas density = " << rho_change_frac << "\n";

	int status = 0;

	const double stability_tol = 0.10;
	if (overdensity_factor == 1.0) {
		if (std::abs(rho_change_frac) > stability_tol) {
			amrex::Print() << "FAIL: Sphere is not stable (density changed by " << rho_change_frac * 100.0 << "%)\n";
			status = 1;
		} else {
			amrex::Print() << "PASS: Sphere remains approximately stable (density changed by " << rho_change_frac * 100.0 << "%)\n";
		}
	} else if (overdensity_factor > 1.0) {
		if (rho_change_frac < stability_tol) {
			amrex::Print() << "FAIL: Sphere did not collapse (density did not increase by more than " << stability_tol * 100.0 << "%)\n";
			status = 1;
		} else {
			amrex::Print() << "PASS: Sphere is collapsing (density changed by " << rho_change_frac * 100.0 << "%)\n";
		}
	} else {
		if (rho_change_frac > -stability_tol) {
			amrex::Print() << "FAIL: Sphere did not expand (density did not decrease by more than " << stability_tol * 100.0 << "%)\n";
			status = 1;
		} else {
			amrex::Print() << "PASS: Sphere is expanding (density changed by " << rho_change_frac * 100.0 << "%)\n";
		}
	}

	return status;
}
