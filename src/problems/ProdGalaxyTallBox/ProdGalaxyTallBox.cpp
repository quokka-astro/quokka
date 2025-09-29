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
#include "radiation/radiation_system.hpp"
#include "turbulence/TurbDataReader.hpp"

static constexpr int BC_TYPE = 1; // 1: Periodic, 2: foextrap, 3: symmetry
static constexpr bool enable_self_gravity = true;
// static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop | ParticleSwitch::CIC;
// static std::string stars_file = "../stars.txt";
// static std::string CIC_file = "../CICs.txt";
static std::string stars_file = "none";
static std::string CIC_file = "none";

constexpr double pc = C::parsec;
constexpr int turbdata_size = 128;

struct TheProblem {
};

template <> struct SimulationData<TheProblem> {
	// turbulent velocity fields
	amrex::TableData<Real, 3> dvx;
	amrex::TableData<Real, 3> dvy;
	amrex::TableData<Real, 3> dvz;
	Real dv_rms_generated{};
	Real turbulent_amplitude = 1500.0; // cm/s,  0.05 * cs at 10K (~0.3 km/s)

	Real refine_parameter = 1.0; // placeholder for refinement control
};

// global variables needed for Dirichlet boundary condition and initial conditions
// copy from data_sets.dat depending on galaxy environment
static constexpr int ARR_SIZE = 100;
// NOLINTBEGIN
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, ARR_SIZE> logphi_data{
    5.23749982, 5.83925514, 6.19098487, 6.44028658, 6.63341552, 6.79097415, 6.92395454, 7.03892608, 7.1401333,	7.23047697, 7.31202697, 7.38631194, 7.45449324,
    7.5174744,	7.57597231, 7.63056519, 7.68172614, 7.72984716, 7.77525663, 7.81823237, 7.85901159, 7.89779849, 7.93477008, 7.9700808,	8.00386622, 8.03624594,
    8.06732603, 8.09720099, 8.12595536, 8.1536651,  8.18039866, 8.206218,   8.23117933, 8.25533386, 8.2787285,	8.30140621, 8.32340645, 8.34476546, 8.36551667,
    8.38569095, 8.40531688, 8.42442095, 8.44302778, 8.46116029, 8.47883984, 8.49608638, 8.51291857, 8.52935389, 8.54540873, 8.5610985,	8.57643767, 8.59143987,
    8.60611797, 8.62048409, 8.6345497,	8.64832568, 8.66182248, 8.6750501,  8.68801804, 8.70073526, 8.71321029, 8.72545119, 8.73746565, 8.74926096, 8.76084405,
    8.77222155, 8.78339974, 8.79438465, 8.80518201, 8.81579732, 8.8262358,  8.83650248, 8.84660217, 8.85653946, 8.86631876, 8.87594432, 8.88542019, 8.89475028,
    8.90393833, 8.91298796, 8.92190263, 8.93068568, 8.93934034, 8.94786969, 8.95627673, 8.96456434, 8.97273531, 8.98079244, 8.98873851, 8.99657621, 9.00430812,
    9.01193675, 9.01946449, 9.02689367, 9.03422652, 9.0414652,	9.04861178, 9.0556683,	9.06263668, 9.06951882};
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, ARR_SIZE> logg_data{
    -9.85457856, -9.39107618, -9.19504351, -9.08451673, -9.01789453, -8.9773358,  -8.95292764, -8.93860797, -8.93051793, -8.92611324, -8.92379746, -8.92261559,
    -8.92202532, -8.92173815, -8.92160118, -8.92153673, -8.92150669, -8.9214927,  -8.92148604, -8.92148266, -8.92148075, -8.92147948, -8.92147848, -8.92147761,
    -8.9214768,	 -8.92147602, -8.92147525, -8.9214745,	-8.92147377, -8.92147304, -8.92147233, -8.92147163, -8.92147094, -8.92147026, -8.92146959, -8.92146894,
    -8.92146829, -8.92146765, -8.92146703, -8.92146641, -8.92146581, -8.92146521, -8.92146462, -8.92146405, -8.92146348, -8.92146293, -8.92146238, -8.92146185,
    -8.92146132, -8.9214608,  -8.92146029, -8.92145979, -8.9214593,  -8.92145882, -8.92145835, -8.92145789, -8.92145744, -8.92145699, -8.92145655, -8.92145612,
    -8.92145569, -8.92145528, -8.92145487, -8.92145447, -8.92145407, -8.92145369, -8.92145331, -8.92145294, -8.92145257, -8.92145221, -8.92145186, -8.92145152,
    -8.92145118, -8.92145085, -8.92145052, -8.9214502,	-8.92144989, -8.92144959, -8.92144929, -8.921449,   -8.92144871, -8.92144844, -8.92144816, -8.9214479,
    -8.92144764, -8.92144738, -8.92144713, -8.92144689, -8.92144665, -8.92144642, -8.92144619, -8.92144596, -8.92144574, -8.92144553, -8.92144531, -8.92144511,
    -8.9214449,	 -8.9214447,  -8.92144451, -8.92144431};
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, ARR_SIZE> z_data{
    6.08467742e+19, 1.82540323e+20, 3.04233871e+20, 4.25927419e+20, 5.47620968e+20, 6.69314516e+20, 7.91008065e+20, 9.12701613e+20, 1.03439516e+21,
    1.15608871e+21, 1.27778226e+21, 1.39947581e+21, 1.52116935e+21, 1.64286290e+21, 1.76455645e+21, 1.88625000e+21, 2.00794355e+21, 2.12963710e+21,
    2.25133065e+21, 2.37302419e+21, 2.49471774e+21, 2.61641129e+21, 2.73810484e+21, 2.85979839e+21, 2.98149194e+21, 3.10318548e+21, 3.22487903e+21,
    3.34657258e+21, 3.46826613e+21, 3.58995968e+21, 3.71165323e+21, 3.83334677e+21, 3.95504032e+21, 4.07673387e+21, 4.19842742e+21, 4.32012097e+21,
    4.44181452e+21, 4.56350806e+21, 4.68520161e+21, 4.80689516e+21, 4.92858871e+21, 5.05028226e+21, 5.17197581e+21, 5.29366935e+21, 5.41536290e+21,
    5.53705645e+21, 5.65875000e+21, 5.78044355e+21, 5.90213710e+21, 6.02383065e+21, 6.14552419e+21, 6.26721774e+21, 6.38891129e+21, 6.51060484e+21,
    6.63229839e+21, 6.75399194e+21, 6.87568548e+21, 6.99737903e+21, 7.11907258e+21, 7.24076613e+21, 7.36245968e+21, 7.48415323e+21, 7.60584677e+21,
    7.72754032e+21, 7.84923387e+21, 7.97092742e+21, 8.09262097e+21, 8.21431452e+21, 8.33600806e+21, 8.45770161e+21, 8.57939516e+21, 8.70108871e+21,
    8.82278226e+21, 8.94447581e+21, 9.06616935e+21, 9.18786290e+21, 9.30955645e+21, 9.43125000e+21, 9.55294355e+21, 9.67463710e+21, 9.79633065e+21,
    9.91802419e+21, 1.00397177e+22, 1.01614113e+22, 1.02831048e+22, 1.04047984e+22, 1.05264919e+22, 1.06481855e+22, 1.07698790e+22, 1.08915726e+22,
    1.10132661e+22, 1.11349597e+22, 1.12566532e+22, 1.13783468e+22, 1.15000403e+22, 1.16217339e+22, 1.17434274e+22, 1.18651210e+22, 1.19868145e+22,
    1.21085081e+22}; // NOLINTEND

static constexpr amrex::Real z_star = 245.0 * pc;
static constexpr amrex::Real Sigma_star = 29.0 * C::M_solar / pc / pc; // originally 42.0 when there is no self gravity
static constexpr amrex::Real rho_dm = 0.0064 * C::M_solar / pc / pc / pc;
static constexpr amrex::Real R0_Gal = 8.e3 * pc;
static constexpr amrex::Real ks_sigma_sfr = 2.088579882548443e-55;
static constexpr amrex::Real hscale = 150. * pc;
static constexpr amrex::Real sigma1 = 700000.0;
static constexpr amrex::Real sigma2 = 7000000.0;
static constexpr amrex::Real rho01 = 2.78556e-24;
static constexpr amrex::Real rho02 = 2.7855600000000006e-29;

template <> struct Particle_Traits<TheProblem> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<TheProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr bool reconstruct_eint = true; // Set to true - temperature
};

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<TheProblem> {
	static constexpr bool is_self_gravity_enabled = enable_self_gravity;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;    // number of mass scalars
	static constexpr int numPassiveScalars = 0; // number of passive scalars
	static constexpr int nGroups = 1;	    // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> void QuokkaSimulation<TheProblem>::createInitialStochasticStellarPopParticles()
{
	// if stars_file == "none", return 
	if (stars_file == "none") {
		return;
	}

	// read particles from ASCII file
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(stars_file, nreal_extra, nullptr);

	// Loop over all particle at all levels and set first integer component to SNProgenitor
	for (int lev = 0; lev <= StochasticStellarPopParticles->finestLevel(); ++lev) {
		auto &particles = StochasticStellarPopParticles->GetParticles(lev);

		for (auto &kv : particles) {
			auto &particle_array = kv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
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

template <> void QuokkaSimulation<TheProblem>::createInitialCICParticles()
{
	// if CIC_file == "none", return
	if (CIC_file == "none") {
		return;
	}

	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile(CIC_file, nreal_extra, nullptr);
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
		amrex::Print() << "turbulence data size assumed: " << turbdata_size << "^3\n";

		// copy to GPU
		userData_.dvx.resize(pinned_dvx.lo(), pinned_dvx.hi());
		userData_.dvx.copy(pinned_dvx);

		userData_.dvy.resize(pinned_dvy.lo(), pinned_dvy.hi());
		userData_.dvy.copy(pinned_dvy);

		userData_.dvz.resize(pinned_dvz.lo(), pinned_dvz.hi());
		userData_.dvz.copy(pinned_dvz);

		isSamplingDone = true;
	}
}

template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	// turbulence parameters
	const Real turb_amp = userData_.turbulent_amplitude;
	const Real dv_rms = userData_.dv_rms_generated;
	const Real renorm_factor = (dv_rms > 0.0) ? turb_amp / dv_rms : 0.0;

	auto const &dvx = userData_.dvx.const_table();
	auto const &dvy = userData_.dvy.const_table();
	auto const &dvz = userData_.dvz.const_table();

	// get turbulence data bounds
	amrex::Array<int, 3> turb_lo = userData_.dvx.lo();

	// get simulation box x-dimension as reference
	const int nx = indexRange.length(0);

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nx <= turbdata_size, "nx must be less than or equal to turbdata_size (128)");
	
	// z-range limits: apply turbulence only from 1.5*nx to 2.5*nx
	const int k_start = static_cast<int>(1.5 * nx);
	const int k_end = static_cast<int>(2.5 * nx);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const z = prob_lo[2] + ((k + static_cast<amrex::Real>(0.5)) * dx[2]);

		// Calculate DM Potential
		double prefac = NAN;
		prefac = 2. * M_PI * Gconst_ * rho_dm * std::pow(R0_Gal, 2);
		const double Phidm = (prefac * std::log(1. + std::pow(z / R0_Gal, 2)));

		// Calculate Stellar Disk Potential
		double prefac2 = NAN;
		prefac2 = 2. * M_PI * Gconst_ * Sigma_star * z_star;
		const double Phist = prefac2 * (std::pow(1. + (z * z / z_star / z_star), 0.5) - 1.);

		// Calculate Gas Disk Potential

		auto const &x_arr = z_data;
		auto const &y_arr = logphi_data;
		const double phi_interp = interpolate_value<BoundaryPolicy::Clamp>(std::abs(z), x_arr.data(), y_arr.data(), ARR_SIZE);
		const double Phigas = std::pow(10., phi_interp);

		const double Phitot = Phist + Phidm + Phigas;

		double rho, rho_disk, rho_halo; // NOLINT
		rho_disk = rho01 * std::exp(-Phitot / std::pow(sigma1, 2.0));
		rho_halo = rho02 * std::exp(-Phitot / std::pow(sigma2, 2.0)); // in g/cc
		rho = (rho_disk + rho_halo);

		const double P = (rho_disk * std::pow(sigma1, 2.0)) + rho_halo * std::pow(sigma2, 2.0);

		AMREX_ASSERT(!std::isnan(rho));

		const auto gamma = HydroSystem<TheProblem>::gamma_;

		// add turbulent velocities
		double vx = 0.0;
		double vy = 0.0;
		double vz = 0.0;

		// check if we're in the z-range where turbulence should be applied
		if (renorm_factor > 0.0 && k >= k_start && k < k_end) {
			// use first nx elements from turbdata directly
			const int turb_i = i;
			const int turb_j = j;
			const int turb_k = k - k_start;
			
			vx = dvx(turb_i, turb_j, turb_k) * renorm_factor;
			vy = dvy(turb_i, turb_j, turb_k) * renorm_factor;
			vz = dvz(turb_i, turb_j, turb_k) * renorm_factor;
		}

		state_cc(i, j, k, HydroSystem<TheProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = P / (gamma - 1.);
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx*vx + vy*vy + vz*vz);
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

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<TheProblem>::GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec)
    -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> grad_potential; // NOLINT
	grad_potential[0] = 0.0;
	grad_potential[1] = 0.0;

	double const z = posvec[2];

	// Interpolate to find the accurate g-value from array
	auto const &x_arr = z_data;
	auto const &y_arr = logg_data;
	const amrex::Real ginterp = interpolate_value<BoundaryPolicy::Clamp>(std::abs(z), x_arr.data(), y_arr.data(), ARR_SIZE);
	AMREX_ASSERT(!std::isnan(ginterp));

	grad_potential[2] = 2. * M_PI * C::Gconst * rho_dm * std::pow(R0_Gal, 2) * (2. * z / std::pow(R0_Gal, 2)) / (1. + std::pow(z, 2) / std::pow(R0_Gal, 2));
	grad_potential[2] += 2. * M_PI * C::Gconst * Sigma_star * (z / z_star) * (std::pow(1. + (z * z / (z_star * z_star)), -0.5));
	grad_potential[2] += (z / std::abs(z)) * std::pow(10., ginterp);
	AMREX_ASSERT(!std::isnan(grad_potential[2]));

	return grad_potential;
}

// Add Strang Split Source Term for External Fixed Potential Here
template <> void QuokkaSimulation<TheProblem>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev) // NOLINT
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx = geom[lev].CellSizeArray();
	const Real dt = dt_lev;

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

			GradPhi = HydroSystem<TheProblem>::GetGradFixedPotential(posvec);

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

	if (k < klo) {
		kedge = klo;
		normal = -1;
	} else if (k > khi) {
		kedge = khi;
		normal = 1.0;
	}

	const double rho_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::density_index);
	const double x1Mom_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::x1Momentum_index);
	const double x2Mom_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::x2Momentum_index);
	double x3Mom_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::x3Momentum_index);
	const double etot_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::energy_index);
	const double eint_edge = consVar(i, j, kedge, HydroSystem<TheProblem>::internalEnergy_index);

	if ((x3Mom_edge * normal) < 0) { // gas is inflowing
		x3Mom_edge = -1. * consVar(i, j, kedge, HydroSystem<TheProblem>::x3Momentum_index);
	}

	consVar(i, j, k, HydroSystem<TheProblem>::density_index) = rho_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = x1Mom_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = x2Mom_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = x3Mom_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::energy_index) = etot_edge;
	consVar(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = eint_edge;
}

auto problem_main() -> int
{

	const int ncomp_cc = Physics_Indices<TheProblem>::nvarTotal_cc;
	// amrex::Vector<quokka::BCRec> BCs_cc(ncomp_cc);

	auto BCs_cc = quokka::BC<TheProblem>(quokka::BCType::reflecting);
	if constexpr (BC_TYPE == 1) {
		BCs_cc = quokka::BC<TheProblem>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::ext_dir);
	} else if constexpr (BC_TYPE == 2) {
		BCs_cc = quokka::BC<TheProblem>(quokka::BCType::foextrap);
	}

	amrex::ParmParse const pp("problem");
	pp.query("stars_file", stars_file);
	pp.query("CIC_file", CIC_file);

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// Problem initialization
	QuokkaSimulation<TheProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!

	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	return 0;
}
