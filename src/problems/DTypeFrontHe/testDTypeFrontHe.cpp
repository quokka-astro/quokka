//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDTypeFrontHe.cpp
/// \brief Defines a test problem for a D Type front with an H + He network.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <cmath>
#include <map>
#include <math/quadrature.hpp>
#include <string>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct DTypeFrontHe {
};

constexpr double c_hat = C::c_light / 10;

template <> struct quokka::EOS_Traits<DTypeFrontHe> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DTypeFrontHe> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = NumSpec;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// the helium network defines 3 chemistry bands (H-ionizing, H+He-ionizing, H+He+He+-ionizing);
	// there are no separate thermal bands in this problem, so nGroups == NUM_CHEM_BANDS.
	static constexpr int nGroups = 3;
};

template <> struct RadSystem_Traits<DTypeFrontHe> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	// Erad_floor sets the M1 radiation energy density floor (erg cm^-3), defined here as a
	// blackbody at T=0.01 K.  The corresponding photon number density floor is
	//   N_gamma_floor = Erad_floor / E_photon ~ 1.25e-10 cm^-3.
	//
	// SetAtolFromPhysics() derives atol_rad_num = 1e-6 * a_rad * T_min^4 / E_photon, where
	// T_min = typical_minimal_radiation_T.  With T_min = 10 K, atol_rad_num ~ 1.25e-6 cm^-3.
	// The ratio atol_rad_num / N_gamma_floor ~ 1e4 ensures that VODE returns in one BDF step
	// even in the darkest cells.
	//
	// The 1e-6 prefactor means radiation at T_min becomes numerically negligible after
	// ~1e6 VODE steps (accumulated local error stays below the physically meaningful level).
	static constexpr double Erad_floor = C::a_rad * 1.0e-8;
	// photochemistry momentum deposition is gated on beta_order == 1, so this test validates pure
	// thermal-pressure D-type front expansion with radiation pressure.
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg;
	// All 3 groups are chemistry bands (no separate thermal groups), so radBoundaries is exactly the
	// helium network's chemistry band edges (H-, H+He-, H+He+He+-ionizing), and nGroupsThermal_ == 0.
	static constexpr amrex::GpuArray<double, Physics_Traits<DTypeFrontHe>::nGroups + 1> radBoundaries = {13.6, 24.59, 54.42, 61.0};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr auto ChemBands() { return ChemBandsHeader_; }
	static constexpr auto ChemBandsPowerLawIndex() { return ChemBandsPowerLawIndex_; }
};

template <> struct SimulationData<DTypeFrontHe> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real n_e_init{};
	amrex::Real n_HI_init{};
	amrex::Real n_HII_init{};
	amrex::Real n_HeI_init{};
	amrex::Real n_HeII_init{};
	amrex::Real n_HeIII_init{};
	amrex::Real Q{};
	int recombination_switch{};
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> r_HII_vec_;
	amrex::Vector<amrex::Real> r_HeI_vec_;
	amrex::Vector<amrex::Real> r_HeII_vec_;
	amrex::Vector<amrex::Real> r_HeIII_vec_;
	std::ofstream output_file_;
};

namespace
{

struct EffectiveRadii {
	amrex::Real r_HII;
	amrex::Real r_HeI;
	amrex::Real r_HeII;
	amrex::Real r_HeIII;
};

auto compute_effective_radii(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> EffectiveRadii
{
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data,
		       [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
			       const amrex::Real n_HI =
				   state[box_no](i, j, k, HydroSystem<DTypeFrontHe>::scalar0_index + static_cast<int>(Species::H)) / spmasses[Species::H];
			       const amrex::Real n_HII =
				   state[box_no](i, j, k, HydroSystem<DTypeFrontHe>::scalar0_index + static_cast<int>(Species::Hp)) / spmasses[Species::Hp];
			       const amrex::Real n_HeI =
				   state[box_no](i, j, k, HydroSystem<DTypeFrontHe>::scalar0_index + static_cast<int>(Species::He)) / spmasses[Species::He];
			       const amrex::Real n_HeII =
				   state[box_no](i, j, k, HydroSystem<DTypeFrontHe>::scalar0_index + static_cast<int>(Species::Hep)) / spmasses[Species::Hep];
			       const amrex::Real n_HeIII =
				   state[box_no](i, j, k, HydroSystem<DTypeFrontHe>::scalar0_index + static_cast<int>(Species::He2p)) / spmasses[Species::He2p];

			       const amrex::Real denom_H = n_HI + n_HII;
			       const amrex::Real x_HII = (denom_H > 0.0_rt) ? (n_HII / denom_H) : 0.0_rt;

			       const amrex::Real denom_He = n_HeI + n_HeII + n_HeIII;
			       const amrex::Real x_HeI = (denom_He > 0.0_rt) ? (n_HeI / denom_He) : 0.0_rt;
			       const amrex::Real x_HeII = (denom_He > 0.0_rt) ? (n_HeII / denom_He) : 0.0_rt;
			       const amrex::Real x_HeIII = (denom_He > 0.0_rt) ? (n_HeIII / denom_He) : 0.0_rt;

			       return {cell_volume * x_HII, cell_volume * x_HeI, cell_volume * x_HeII, cell_volume * x_HeIII};
		       });

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real volume_HII = amrex::get<0>(hv);
	amrex::Real volume_HeI = amrex::get<1>(hv);
	amrex::Real volume_HeII = amrex::get<2>(hv);
	amrex::Real volume_HeIII = amrex::get<3>(hv);
	amrex::ParallelAllReduce::Sum(volume_HII, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(volume_HeI, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(volume_HeII, amrex::ParallelContext::CommunicatorSub());
	amrex::ParallelAllReduce::Sum(volume_HeIII, amrex::ParallelContext::CommunicatorSub());

	auto volume_to_radius = [](amrex::Real volume) -> amrex::Real { return std::cbrt((3.0_rt * 8.0_rt * volume) / (4.0_rt * M_PI)); };
	return {volume_to_radius(volume_HII), volume_to_radius(volume_HeI), volume_to_radius(volume_HeII), volume_to_radius(volume_HeIII)};
}

#ifdef DTYPEFRONT_USE_ROSENBROCK
auto rosenbrock_tableau_name(int tableau) -> char const *
{
	switch (tableau) {
		case 0:
			return "Rodas5P";
		case 1:
			return "Rodas4P";
		case 2:
			return "Rodas3P";
		case 3:
			return "ROS2S";
		default:
			return "unknown";
	}
}
#endif

void print_microphysics_integrator()
{
#ifdef DTYPEFRONT_USE_ROSENBROCK
	amrex::Print() << "DTypeFrontHe microphysics integrator: Rosenbrock (Rosenbrock tableau " << integrator_rp::rosenbrock_tableau << ": "
		       << rosenbrock_tableau_name(integrator_rp::rosenbrock_tableau) << ")\n";
#else
	amrex::Print() << "DTypeFrontHe microphysics integrator: VODE\n";
#endif
}

} // namespace

AMREX_GPU_HOST_DEVICE auto wendland_c2(amrex::Real r) -> amrex::Real
{
	if (r > 1.0) {
		return 0.0;
	}
	return (21. / (2. * M_PI)) * std::pow((1.0 - r), 4) * (4.0 * r + 1.0);
}

template <>
void RadSystem<DTypeFrontHe>::AddRadSource(array_t &radEnergy, array_t & /*reducedFluxSource*/, const amrex::Box &indexRange,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	amrex::ParmParse const pp("stromgen");
	amrex::Real Q = 1.0e49_rt;
	pp.query("Q", Q);

	constexpr int N = 2;
	constexpr amrex::Real inv_N = 1.0 / static_cast<amrex::Real>(N);
	constexpr auto cutoff_r2 = static_cast<amrex::Real>(N * N);

	// Q is the source's total ionizing-photon rate (photons/s); split it across the 3 helium-network
	// chemistry bands (]13.6,24.59], ]24.59,54.42], ]54.42,61.0] eV) in the ratio 0.6:0.35:0.05. The
	// fractions are photon-number fractions and sum to 1, so the photon rate deposited into band g is
	// Q * band_photon_fraction[g]. radEnergy is an energy-density source (erg cm^-3 s^-1), so each
	// band's photon rate must be converted to an energy rate via that band's mean photon energy
	// (GetChemBandQuanta), not deposited directly as a photon-number source.
	constexpr amrex::GpuArray<amrex::Real, 3> band_photon_fraction = {0.6_rt, 0.35_rt, 0.05_rt};
	amrex::GpuArray<amrex::Real, 3> L_star{};
	for (int g = 0; g < 3; ++g) {
		L_star[g] = Q * band_photon_fraction[g] * RadSystem<DTypeFrontHe>::GetChemBandQuanta(g);
	}
	const amrex::Real x0 = 0.0_rt;
	const amrex::Real y0 = 0.0_rt;
	const amrex::Real z0 = 0.0_rt;
	const amrex::Real volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	const amrex::Real inv_volume = 1.0 / volume;

	const int src_i = static_cast<int>(amrex::Math::floor((x0 - prob_lo[0]) / dx[0]));
	const int src_j = static_cast<int>(amrex::Math::floor((y0 - prob_lo[1]) / dx[1]));
	const int src_k = static_cast<int>(amrex::Math::floor((z0 - prob_lo[2]) / dx[2]));
	const amrex::Real frac_x = (x0 - prob_lo[0]) / dx[0] - static_cast<amrex::Real>(src_i);
	const amrex::Real frac_y = (y0 - prob_lo[1]) / dx[1] - static_cast<amrex::Real>(src_j);
	const amrex::Real frac_z = (z0 - prob_lo[2]) / dx[2] - static_cast<amrex::Real>(src_k);

	constexpr int stencil_width = 2 * N + 1;
	const int nz_loop = (AMREX_SPACEDIM >= 3) ? stencil_width : 1;
	const int ny_loop = (AMREX_SPACEDIM >= 2) ? stencil_width : 1;
	amrex::Real norm_sum = 0.0_rt;
	for (int kk = 0; kk < nz_loop; ++kk) {
		const amrex::Real dz = (AMREX_SPACEDIM >= 3) ? static_cast<amrex::Real>(kk - N) + 0.5 - frac_z : 0.0;
		for (int jj = 0; jj < ny_loop; ++jj) {
			const amrex::Real dy = (AMREX_SPACEDIM >= 2) ? static_cast<amrex::Real>(jj - N) + 0.5 - frac_y : 0.0;
			for (int ii = 0; ii < stencil_width; ++ii) {
				const amrex::Real di = static_cast<amrex::Real>(ii - N) + 0.5 - frac_x;
				const amrex::Real r2 = AMREX_D_TERM(di * di, +dy * dy, +dz * dz);
				if (r2 <= cutoff_r2) {
					norm_sum += wendland_c2(std::sqrt(r2) * inv_N);
				}
			}
		}
	}

	const amrex::Real inv_norm = 1.0_rt / norm_sum;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real di = static_cast<amrex::Real>(i - src_i) + 0.5 - frac_x;
		const amrex::Real dj = (AMREX_SPACEDIM >= 2) ? static_cast<amrex::Real>(j - src_j) + 0.5 - frac_y : 0.0;
		const amrex::Real dk = (AMREX_SPACEDIM >= 3) ? static_cast<amrex::Real>(k - src_k) + 0.5 - frac_z : 0.0;
		const amrex::Real r2 = AMREX_D_TERM(di * di, +dj * dj, +dk * dk);
		const amrex::Real weight = (r2 <= cutoff_r2) ? wendland_c2(std::sqrt(r2) * inv_N) * inv_norm * inv_volume : 0.0_rt;
		for (int g = 0; g < 3; ++g) {
			radEnergy(i, j, k, g) = L_star[g] * weight;
		}
	});
}

template <> void QuokkaSimulation<DTypeFrontHe>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species and temperature
	amrex::ParmParse const pp("stromgen");
	userData_.small_temp = 1e-2;
	userData_.small_dens = 1e-60;
	userData_.temperature = 1.0e4;
	userData_.n_e_init = 0.0e0_rt;
	userData_.n_HI_init = 1.0e2_rt;
	userData_.n_HII_init = 0.0e0_rt;
	userData_.n_HeI_init = 0.0e0_rt;
	userData_.n_HeII_init = 0.0e0_rt;
	userData_.n_HeIII_init = 0.0e0_rt;
	userData_.Q = 1.0e49_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("n_e_init", userData_.n_e_init);
	pp.query("n_HI_init", userData_.n_HI_init);
	pp.query("n_HII_init", userData_.n_HII_init);
	pp.query("n_HeI_init", userData_.n_HeI_init);
	pp.query("n_HeII_init", userData_.n_HeII_init);
	pp.query("n_HeIII_init", userData_.n_HeIII_init);
	pp.query("Q", userData_.Q);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::string const filename = "dtype_front_radii.csv";
		userData_.output_file_.open(filename);
		userData_.output_file_ << "time,r_HII,r_HeI,r_HeII,r_HeIII\n";
	}
}

// ComputePlanckOpacity / ComputeFluxMeanOpacity are only consulted by the single-group solver. This problem
// runs with nGroups = 3 (all 3 groups are chemistry bands), so group opacities come from
// DefineOpacityExponentsAndLowerValues below instead.
template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DTypeFrontHe>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							      const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	// Every chemistry band is left fully transparent, matching the original DTypeFront's single-group opacity.
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = 0.0;
	}
	return exponents_and_values;
}

template <> void QuokkaSimulation<DTypeFrontHe>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	numdens[Species::e] = userData_.n_e_init;
	numdens[Species::H] = userData_.n_HI_init;
	numdens[Species::Hp] = userData_.n_HII_init;
	numdens[Species::He] = userData_.n_HeI_init;
	numdens[Species::Hep] = userData_.n_HeII_init;
	numdens[Species::He2p] = userData_.n_HeIII_init;

	state.T = userData_.temperature;
	// find the density in g/cm^3
	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n]; // spmasses contains the masses of all species, defined in EOS
	}
	state.rho = rhotot;

	// call the EOS to set initial internal energy e
	eos(eos_input_rt, state);
	const auto Egas0 = state.e * rhotot;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<DTypeFrontHe>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DTypeFrontHe>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = 1.e-99_rt;
			state_cc(i, j, k, RadSystem<DTypeFrontHe>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFrontHe>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFrontHe>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<DTypeFrontHe>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFrontHe>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<DTypeFrontHe>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFrontHe>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFrontHe>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFrontHe>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<DTypeFrontHe>::scalar0_index + nn) =
			    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<DTypeFrontHe>::computeAfterTimestep()
{
	const int lev = 0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const EffectiveRadii radii = compute_effective_radii(state_new_cc_[lev], dx);
	const amrex::Real t = tNew_[lev];
	userData_.r_HII_vec_.push_back(radii.r_HII);
	userData_.r_HeI_vec_.push_back(radii.r_HeI);
	userData_.r_HeII_vec_.push_back(radii.r_HeII);
	userData_.r_HeIII_vec_.push_back(radii.r_HeIII);
	userData_.t_vec_.push_back(t);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << t << ',' << radii.r_HII << ',' << radii.r_HeI << ',' << radii.r_HeII << ',' << radii.r_HeIII << '\n';
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.3;
	const double dt_max = 1e99;

	// Problem initialization
	QuokkaSimulation<DTypeFrontHe> sim;
	print_microphysics_integrator();

	// initialize
	sim.setInitialConditions();
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.plotfileInterval_ = -1;

	sim.evolve();

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr amrex::Real seconds_per_Myr = 3.15576e13;
		constexpr amrex::Real cm_per_pc = 3.085677581491367e18;

		std::vector<amrex::Real> t_Myr(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> r_HII_pc(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> r_HeI_pc(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> r_HeII_pc(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> r_HeIII_pc(sim.userData_.t_vec_.size());
		for (int i = 0; i < static_cast<int>(sim.userData_.t_vec_.size()); ++i) {
			t_Myr[i] = sim.userData_.t_vec_[i] / seconds_per_Myr;
			r_HII_pc[i] = sim.userData_.r_HII_vec_[i] / cm_per_pc;
			r_HeI_pc[i] = sim.userData_.r_HeI_vec_[i] / cm_per_pc;
			r_HeII_pc[i] = sim.userData_.r_HeII_vec_[i] / cm_per_pc;
			r_HeIII_pc[i] = sim.userData_.r_HeIII_vec_[i] / cm_per_pc;
		}
		// Plot radii vs time
		matplotlibcpp::clf();
		std::map<std::string, std::string> HII_args;
		HII_args["label"] = "HII";
		HII_args["color"] = "C0";
		std::map<std::string, std::string> HeI_args;
		HeI_args["label"] = "HeI";
		HeI_args["color"] = "C1";
		std::map<std::string, std::string> HeII_args;
		HeII_args["label"] = "HeII";
		HeII_args["color"] = "C2";
		std::map<std::string, std::string> HeIII_args;
		HeIII_args["label"] = "HeIII";
		HeIII_args["color"] = "C3";

		matplotlibcpp::plot(t_Myr, r_HII_pc, HII_args);
		matplotlibcpp::plot(t_Myr, r_HeI_pc, HeI_args);
		matplotlibcpp::plot(t_Myr, r_HeII_pc, HeII_args);
		matplotlibcpp::plot(t_Myr, r_HeIII_pc, HeIII_args);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("radius (pc)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dtype_front_radii.pdf");
	}
#endif

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
