/// \file ProdGalaxyTallBox.cpp
/// \brief Defines a problem for a tall box in a Milky way-mass galaxy.
///

#include <cmath>
#include <iostream>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_Random.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "util/BC.hpp"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "io/projection.hpp"
#include "math/interpolate.hpp"
#include "radiation/radiation_dust_system.hpp"
#include "turbulence/TurbDataReader.hpp"
#include "util/DataTable.hpp"

static constexpr int BC_TYPE = 1; // 1: Periodic, 2: foextrap
static constexpr bool is_rad_on = false;
static constexpr int nGroups_ = 1;
static constexpr amrex::GpuArray<double, nGroups_ + 1> radBoundaries_{ 0.0, 100.0 };
static constexpr amrex::GpuArray<double, nGroups_ + 1> dust_opacity_{0.0, 0.0};
// static constexpr int nGroups_ = 4;
// static constexpr amrex::GpuArray<double, nGroups_ + 1> radBoundaries_ = { 1.e-04, 1.00778140e-01, 1.00778140e+00, 5.53817071e+00, 1.e+2 };
// static constexpr amrex::GpuArray<double, nGroups_ + 1> dust_opacity_{6e2, 1e3, 2e4, 1e5, 2e5}; // dust opacity, cm2/g. last element not used
static constexpr bool enable_dust_ = false;
static constexpr bool enable_PE_ = false;

static constexpr bool enable_self_gravity = true;

constexpr double pc = C::parsec;
constexpr double mu = 1.0 * C::m_p;
constexpr double gamma_ = 5. / 3.;
constexpr double arad = C::a_rad;
constexpr double TCMB = 2.7;		 // K, CMB temperature
constexpr double initial_Erad = 1e-40 * arad * TCMB * TCMB * TCMB * TCMB;
constexpr double chat_over_c = 2000.0 * 1e5 / C::c_light; // chat = 2000 km/s

struct TheProblem {
};

// GPU-friendly const table access for initial conditions
// 3 outputs: rho, g_z, Phi
struct ICGpuConstTables {
	quokka::DataTableGpuConst<1, 3, quokka::OutOfBounds::clamp> ic_table; // 1D table: z -> (rho, g_z, Phi)
};

template <> struct SimulationData<TheProblem> {
	// turbulent velocity fields
	amrex::TableData<Real, 3> dvx;
	amrex::TableData<Real, 3> dvy;
	amrex::TableData<Real, 3> dvz;
	Real dv_rms_generated{};
	Real turbulent_amplitude = 1500.0; // cm/s,  0.05 * cs at 10K (~0.3 km/s)
	int turbulent_size = 128;

	Real refine_parameter = 1.0; // placeholder for refinement control
	std::string stars_file = "none"; // default: no stars
	std::string IC_file = "none"; // Initial disk vertical structure

	// Initial conditions table: z -> (rho, g_z, Phi)
	quokka::DataTable<1, 3, quokka::OutOfBounds::clamp> ic_table;

	// Galaxy parameters (default is solar neighborhood)
	Real rho01 = 4.320441e-24; // 2.58 m_p/cm^3
	Real sigma1 = 700000.0;
};

template <> struct Particle_Traits<TheProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<TheProblem> {
	static constexpr double gamma = gamma_;
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct Physics_Traits<TheProblem> {
	static constexpr bool is_self_gravity_enabled = enable_self_gravity;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = is_rad_on;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = is_rad_on ? nGroups_ : 1;			     // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg; // set boundary unit to eV
	static constexpr amrex::GpuArray<double, Physics_Traits<TheProblem>::nGroups + 1> radBoundaries = radBoundaries_;
	static constexpr OpacityModel opacity_model = is_rad_on ? OpacityModel::piecewise_constant_opacity : OpacityModel::single_group;
};

template <> struct ISM_Traits<TheProblem> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = enable_dust_;
	static constexpr bool enable_photoelectric_heating = enable_PE_;
	static constexpr double gas_dust_coupling_threshold = 1.0e-5;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<TheProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							    const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	constexpr double gas_to_dust_ratio = 1.0e-3;
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0; // power-law slopes
	}
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[1][i] = dust_opacity_[i] * gas_to_dust_ratio;
	}
	return exponents_and_values;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/,
											     amrex::Real const num_density) -> amrex::Real
{
	// Values in cgs units from Bate & Keto (2015), Eq. 26.
	const double epsilon = 0.05; // default efficiency factor for cold molecular clouds
	const double ref_J_ISR = 5.29e-14; // reference value for the ISR in erg cm^3
	const double coeff = 1.33e-24;
	return coeff * epsilon * num_density / ref_J_ISR; // s^-1
}

template <> void QuokkaSimulation<TheProblem>::createInitialStochasticStellarPopParticles()
{
	if (userData_.stars_file == "none") {
		return;
	}

	// Read particles from ASCII file. Note that this only read real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.stars_file, nreal_extra, nullptr);

	// Using a for loop from lev = 0 to StochasticStellarPopParticles->maxLevel() won't work because not all levels necessarily have particles, and when
	// some levels do not have particles, StochasticStellarPopParticles->GetParticles(lev) will result in a Segfault. Therefore, we loop over the actual
	// particle container.
	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();

			if (np == 0) {
				continue;
			}

			auto *pdata = particle_array().data();

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<TheProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// geometrical refinement
	// tag cells within the cylinder defined by R < Rmax and abs(z) < zmax
	amrex::ParmParse const pp("problem");
	std::vector<amrex::Real> refine_zmax_list;
	pp.queryarr("refine_zmax", refine_zmax_list);
	
	// If no list is provided or level exceeds list size, skip refinement
	if (refine_zmax_list.empty() || lev >= static_cast<int>(refine_zmax_list.size())) {
		return;
	}
	
	const amrex::Real refine_zmax = refine_zmax_list[lev];

	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// NOTE: must check all nodes of the cell!
		// Otherwise, cells that are too big can completely prevent refinement.
		// amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		// amrex::Real const y0 = prob_lo[1] + (j * dx[1]);
		// amrex::Real const z0 = prob_lo[2] + (k * dx[2]);

		// amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		// amrex::Real const y1 = prob_lo[1] + ((j + 1) * dx[1]);
		// amrex::Real const z1 = prob_lo[2] + ((k + 1) * dx[2]);

		amrex::Real const z = prob_lo[2] + ((k + 0.5) * dx[2]);

		if (std::abs(z) < refine_zmax) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<TheProblem>::preCalculateInitialConditions()
{
	static bool isSamplingDone = false;
	if (!isSamplingDone) {
		// read perturbations from file
		turb_data turbData;
		amrex::ParmParse const pp("perturb");
		std::string turbdata_filename = "zdrv.hdf5";
		pp.query("filename", turbdata_filename);
		initialize_turbdata(turbData, turbdata_filename);

		pp.query("amplitude", userData_.turbulent_amplitude); // amplitude in cm/s, default is 0.05 * 0.3 km/s = 1,500 cm/s

		// copy to pinned memory
		auto pinned_dvx = get_tabledata(turbData.dvx);
		auto pinned_dvy = get_tabledata(turbData.dvy);
		auto pinned_dvz = get_tabledata(turbData.dvz);

		// compute normalisation
		userData_.dv_rms_generated = computeRms(pinned_dvx, pinned_dvy, pinned_dvz);
		amrex::Print() << "rms dv = " << userData_.dv_rms_generated << "\n";

		amrex::Print() << "turbulent amplitude = " << userData_.turbulent_amplitude << " cm/s\n";

		userData_.turbulent_size = turbData.dvx.end[0] - turbData.dvx.begin[0];
		amrex::Print() << "turbulence data size is: " << userData_.turbulent_size << "^3\n";

		// copy to GPU
		userData_.dvx.resize(pinned_dvx.lo(), pinned_dvx.hi());
		userData_.dvx.copy(pinned_dvx);

		userData_.dvy.resize(pinned_dvy.lo(), pinned_dvy.hi());
		userData_.dvy.copy(pinned_dvy);

		userData_.dvz.resize(pinned_dvz.lo(), pinned_dvz.hi());
		userData_.dvz.copy(pinned_dvz);

		// Read initial conditions from file if specified
		if (userData_.IC_file != "none") {
			amrex::Print() << "Reading initial conditions from: " << userData_.IC_file << "\n";
			// Read CSV file with linear spacing for outputs
			userData_.ic_table = quokka::DataTable<1, 3, quokka::OutOfBounds::clamp>::CSVReader(userData_.IC_file, quokka::SpacingType::linear);
			amrex::Print() << "Initial conditions table loaded successfully.\n";
		} else {
			amrex::Print() << "No IC file specified. Using hardcoded initial conditions.\n";
		}

		isSamplingDone = true;
	}
}

template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// turbulence parameters
	const Real turb_amp = userData_.turbulent_amplitude;
	const Real dv_rms = userData_.dv_rms_generated;
	const Real renorm_factor = (dv_rms > 0.0) ? turb_amp / dv_rms : 0.0;

	auto const &dvx = userData_.dvx.const_table();
	auto const &dvy = userData_.dvy.const_table();
	auto const &dvz = userData_.dvz.const_table();

	// get turbulence data bounds
	amrex::Array<int, 3> turb_lo = userData_.dvx.lo();
	amrex::Array<int, 3> turb_hi = userData_.dvx.hi();

	// get simulation box x-dimension as reference
	const int nx = indexRange.length(0);
	const int nturb = turb_hi[0] - turb_lo[0] + 1;

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nx <= nturb, "nx must be less than or equal to turbulent_size (128)");
	
	// z-range limits: apply turbulence only from 1.5*nx to 2.5*nx
	const int k_start = nx + nx / 2;
	const int k_end = 2 * nx + nx / 2;

	// Capture galaxy parameters from userData_ for GPU kernel
	const Real sigma1_ic = userData_.sigma1;
	const Real sigma2_ic = 10.0 * sigma1_ic;
	const Real rho01_ic = userData_.rho01;
	const Real rho02_ic = 1.0e-5 * rho01_ic;

	// Create GPU const tables for initial conditions if available
	ICGpuConstTables gpu_ic_tables;
	gpu_ic_tables.ic_table = userData_.ic_table.const_tables();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const z = prob_lo[2] + ((k + static_cast<amrex::Real>(0.5)) * dx[2]);

		// Use DataTable to interpolate initial conditions from file
		// Table provides: [rho, g_z, Phi] as functions of z
		std::array<amrex::Real, 1> const point = {std::abs(z)};
		auto const ic_values = gpu_ic_tables.ic_table.interpolate(point);
		
		// Extract values: ic_values[0] = g_1, ic_values[1] = g_ext, ic_values[2] = phi_tot
		const double phi_tot = ic_values[2];
		const double rho1 = rho01_ic * std::exp(-phi_tot / (sigma1_ic * sigma1_ic));
		const double rho2 = rho02_ic * std::exp(-phi_tot / (sigma2_ic * sigma2_ic));
		const double rho = rho1 + rho2;

		const double P = rho1 * sigma1_ic * sigma1_ic + rho2 * sigma2_ic * sigma2_ic;

		AMREX_ASSERT(!std::isnan(rho));

		// const double Tgas = P / (rho / mu * C::k_B);

		const auto gamma = HydroSystem<TheProblem>::gamma_;

		// add turbulent velocities
		// Clamp indices to [turb_lo, turb_hi] range
		// const int turb_i = amrex::Math::max(turb_lo[0], amrex::Math::min(turb_hi[0], turb_lo[0] + (i % nturb)));
		// const int turb_j = amrex::Math::max(turb_lo[1], amrex::Math::min(turb_hi[1], turb_lo[1] + (j % nturb)));
		// const int turb_k = amrex::Math::max(turb_lo[2], amrex::Math::min(turb_hi[2], turb_lo[2] + (k % nturb)));
		const int turb_i = turb_lo[0] + (i % nturb);
		const int turb_j = turb_lo[1] + (j % nturb);
		const int turb_k = turb_lo[2] + (k % nturb);
		
		const double vx = dvx(turb_i, turb_j, turb_k) * renorm_factor;
		const double vy = dvy(turb_i, turb_j, turb_k) * renorm_factor;
		const double vz = dvz(turb_i, turb_j, turb_k) * renorm_factor;

		state_cc(i, j, k, HydroSystem<TheProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = P / (gamma - 1.);
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx*vx + vy*vy + vz*vz);

		// Set radiation variables
		if constexpr (is_rad_on) {
			// compute energy fractions
			const auto Erad_g = RadSystem<TheProblem>::ComputeThermalRadiationMultiGroup(TCMB, RadSystem<TheProblem>::radBoundaries_);
			for (int g = 0; g < Physics_Traits<TheProblem>::nGroups; ++g) {
				state_cc(i, j, k, RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad_g[g];
				state_cc(i, j, k, RadSystem<TheProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
				state_cc(i, j, k, RadSystem<TheProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
				state_cc(i, j, k, RadSystem<TheProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			}
		}
	});
}

template <> void QuokkaSimulation<TheProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in) const
{
	// compute derived variables and save in 'mf'

	if (dname == "temperature") {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coolingTableType_ == "resampled", "diagnostics require resampled cooling tables.");
		const int ncomp = ncomp_in;
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();
		auto tables = resampledTables_.const_tables();
		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<TheProblem>::density_index);
			Real const x1Mom = state[bx](i, j, k, HydroSystem<TheProblem>::x1Momentum_index);
			Real const x2Mom = state[bx](i, j, k, HydroSystem<TheProblem>::x2Momentum_index);
			Real const x3Mom = state[bx](i, j, k, HydroSystem<TheProblem>::x3Momentum_index);
			Real const Egas = state[bx](i, j, k, HydroSystem<TheProblem>::energy_index);
			Real const Eint = RadSystem<TheProblem>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
			Real const Tgas = quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint, tables);
			output[bx](i, j, k, ncomp) = Tgas;
		});
	} else if (dname == "c_s") {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coolingTableType_ == "resampled", "diagnostics require resampled cooling tables.");
		const int ncomp = ncomp_in;
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();
		auto tables = resampledTables_.const_tables();
		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<TheProblem>::density_index);
			Real const x1Mom = state[bx](i, j, k, HydroSystem<TheProblem>::x1Momentum_index);
			Real const x2Mom = state[bx](i, j, k, HydroSystem<TheProblem>::x2Momentum_index);
			Real const x3Mom = state[bx](i, j, k, HydroSystem<TheProblem>::x3Momentum_index);
			Real const Egas = state[bx](i, j, k, HydroSystem<TheProblem>::energy_index);
			Real const Eint = RadSystem<TheProblem>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
			Real const cs = quokka::ResampledCooling::ComputeSoundSpeedFromRhoEint(rho, Eint, tables);
			output[bx](i, j, k, ncomp) = cs / 1.0e5; // km/s
		});
	}
	amrex::Gpu::streamSynchronizeAll();
}

// Add Strang Split Source Term for External Fixed Potential Here
template <> void QuokkaSimulation<TheProblem>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev) // NOLINT
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx = geom[lev].CellSizeArray();
	const Real dt = dt_lev;

	// Create GPU const tables for initial conditions if available
	ICGpuConstTables gpu_ic_tables;
	gpu_ic_tables.ic_table = userData_.ic_table.const_tables();

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec;  // NOLINT
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> GradPhi; // NOLINT
			double x1mom_new = NAN;
			double x2mom_new = NAN;
			double x3mom_new = NAN;

			const Real rho = state(i, j, k, HydroSystem<TheProblem>::density_index);
			const Real x1mom = state(i, j, k, HydroSystem<TheProblem>::x1Momentum_index);
			const Real x2mom = state(i, j, k, HydroSystem<TheProblem>::x2Momentum_index);
			const Real x3mom = state(i, j, k, HydroSystem<TheProblem>::x3Momentum_index);
			const Real Egas = state(i, j, k, HydroSystem<TheProblem>::energy_index);

			const Real Eint = RadSystem<TheProblem>::ComputeEintFromEgas(rho, x1mom, x2mom, x3mom, Egas);

			posvec[0] = prob_lo[0] + (i + 0.5) * dx[0];
			posvec[1] = prob_lo[1] + (j + 0.5) * dx[1];
			posvec[2] = prob_lo[2] + (k + 0.5) * dx[2];

			// Calculate gradient of fixed potential using captured parameters
			double const z = posvec[2];

			std::array<amrex::Real, 1> const point = {std::abs(z)};
			auto const ic_values = gpu_ic_tables.ic_table.interpolate(point);
			// ic_values[0] = g_1, ic_values[1] = g_ext, ic_values[2] = phi_tot
			const double g_ext = ic_values[1];

			GradPhi[0] = 0.0;
			GradPhi[1] = 0.0;
			// g_ext is the gravitational acceleration (g_z) at z > 0 (it is negative)
			// GradPhi = -g (vector)
			// For z > 0, GradPhi_z = -g_ext (positive)
			// For z < 0, g_z = -g_ext (positive), GradPhi_z = -g_z = g_ext (negative)
			// This is equivalent to GradPhi_z = -g_ext * sign(z)
			GradPhi[2] = -g_ext * std::copysign(1.0, z);
			AMREX_ASSERT(!std::isnan(GradPhi[2]));

			x1mom_new = x1mom + dt * (-rho * GradPhi[0]);
			x2mom_new = x2mom + dt * (-rho * GradPhi[1]);
			x3mom_new = x3mom + dt * (-rho * GradPhi[2]);

			AMREX_ASSERT(!std::isnan(x1mom_new));
			AMREX_ASSERT(!std::isnan(x2mom_new));
			AMREX_ASSERT(!std::isnan(x3mom_new));

			// State momentum values need to be updated this way.
			state(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = x2mom_new;
			state(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = x3mom_new;

			const Real Egas_new = RadSystem<TheProblem>::ComputeEgasFromEint(rho, x1mom_new, x2mom_new, x3mom_new, Eint);
			AMREX_ASSERT(!std::isnan(Egas_new));

			state(i, j, k, HydroSystem<TheProblem>::energy_index) = Egas_new;
		});
	}
}

// Code for producing in-situ Projection plots
template <>
auto QuokkaSimulation<TheProblem>::ComputeProjections(const amrex::Direction dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	// compute density projection
	std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj;

	proj["rho"] = quokka::diagnostics::ComputePlaneProjection<amrex::ReduceOpSum>(
	    state_new_cc_, finestLevel(), geom, ref_ratio, dir, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<TheProblem>::density_index);
		    return (rho);
	    });
	return proj;
}

// Implement User-defined diode BC
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<TheProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
							 amrex::GeometryData const &geom, const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
							 int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int klo = domain_lo[2];
	const int khi = domain_hi[2];
	int kedge = 0;
	int normal = 0;

	// if (k < klo) {
	// 	kedge = klo;
	// 	normal = -1;
	// } else if (k > khi) {
	// 	kedge = khi;
	// 	normal = 1.0;
	// }

	// This should be the correct way?
	if (k < klo) {
		kedge = klo;
		normal = -1;
	} else if (k >= khi) {
		kedge = khi - 1;
		normal = 1.0;
	}

	// Or, perhaps we also need this?
	// if (i < domain_lo[0]) {
	// 	ii = domain_lo[0];
	// } else if (i >= domain_hi[0]) {
	// 	ii = domain_hi[0] - 1;
	// }
	// if (j < domain_lo[1]) {
	// 	jj = domain_lo[1];
	// } else if (j >= domain_hi[1]) {
	// 	jj = domain_hi[1] - 1;
	// }

	const double rho_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::density_index);
	const double x1Mom_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::x1Momentum_index);
	const double x2Mom_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::x2Momentum_index);
	double x3Mom_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::x3Momentum_index);
	const double etot_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::energy_index);
	const double eint_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::internalEnergy_index);

	if ((x3Mom_edge * normal) < 0) { // gas is inflowing
		x3Mom_edge *= -1.;
	}

	consVar(i, j, k, HydroSystem<TheProblem>::density_index) = rho_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = x1Mom_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = x2Mom_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = x3Mom_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::energy_index) = etot_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = eint_edge;

	if constexpr (is_rad_on) {
		// copy radiation variables from edge to boundary cells
		const int ii = i;
		const int jj = j;
		const int kk = kedge;
		for (int g = 0; g < Physics_Traits<TheProblem>::nGroups; ++g) {
			const double radEnergy_edge = consVar(ii, jj, kk, RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g);
			const double x1RadFlux_edge = consVar(ii, jj, kk, RadSystem<TheProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g);
			const double x2RadFlux_edge = consVar(ii, jj, kk, RadSystem<TheProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g);
			double x3RadFlux_edge = consVar(ii, jj, kk, RadSystem<TheProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g);

			if ((x3RadFlux_edge * normal) < 0) { // radiation is inflowing
				x3RadFlux_edge *= -1.;
			}

			consVar(i, j, k, RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = radEnergy_edge;
			consVar(i, j, k, RadSystem<TheProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = x1RadFlux_edge;
			consVar(i, j, k, RadSystem<TheProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = x2RadFlux_edge;
			consVar(i, j, k, RadSystem<TheProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = x3RadFlux_edge;
		}
	}
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<TheProblem>(quokka::BCType::reflecting);
	if constexpr (BC_TYPE == 1) {
		BCs_cc = quokka::BC<TheProblem>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::ext_dir);
	} else if constexpr (BC_TYPE == 2) {
		BCs_cc = quokka::BC<TheProblem>(quokka::BCType::foextrap);
	}

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// Problem initialization
	QuokkaSimulation<TheProblem> sim(BCs_cc);
	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!

	amrex::ParmParse const pp("problem");
	pp.query("stars_file", sim.userData_.stars_file);
	pp.query("IC_file", sim.userData_.IC_file);
	pp.query("rho01", sim.userData_.rho01);
	pp.query("sigma1", sim.userData_.sigma1);

	// initialize (this will parse particle parameters and load luminosity table)
	sim.setInitialConditions();

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	// Total radiation energy in the field
	amrex::Real total_Erad_init = 0.0;
	if constexpr (is_rad_on) {
		for (int g = 0; g < Physics_Traits<TheProblem>::nGroups; ++g) {
			total_Erad_init += sim.state_new_cc_[0].sum(RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
		}
	}

	const amrex::Real total_gas_energy_init = sim.state_new_cc_[0].sum(RadSystem<TheProblem>::gasEnergy_index) * vol;
	const amrex::Real total_energy_init = total_Erad_init / chat_over_c + total_gas_energy_init;

	// set force finest level to true for test particles
	// sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->setForceFinestLevel(true);

	sim.evolve();

	amrex::Real total_Erad = 0.0;
	if constexpr (is_rad_on) {
		for (int g = 0; g < Physics_Traits<TheProblem>::nGroups; ++g) {
			total_Erad += sim.state_new_cc_[0].sum(RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) * vol;
		}
	}

	const amrex::Real total_gas_energy = sim.state_new_cc_[0].sum(RadSystem<TheProblem>::gasEnergy_index) * vol;
	const amrex::Real total_energy_final = total_Erad / chat_over_c + total_gas_energy;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Total gas energy (initial): " << total_gas_energy_init << "\n";
		amrex::Print() << "Total gas energy (final): " << total_gas_energy << "\n";
		if (is_rad_on) {
			amrex::Print() << "Total radiation energy (initial): " << total_Erad_init / chat_over_c << "\n";
			amrex::Print() << "Total radiation energy (final): " << total_Erad / chat_over_c << "\n";
		}
		amrex::Print() << "Change of total energy: " << total_energy_final - total_energy_init << "\n";
		amrex::Print() << "Relative change of total energy: " << (total_energy_final - total_energy_init) / total_energy_init << "\n";
	}

	return 0;
}
