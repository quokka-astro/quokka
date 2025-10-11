/// \file random_blast_rad.cpp
/// \brief Implements the random blast problem with multigroup radiation transport and radiative cooling.
///
#include "AMReX.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include <fmt/format.h>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/quadrature.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

using amrex::Real;

constexpr Real chat_over_c = 1.0e-3;
constexpr Real mu = 1.0 * C::m_p;
constexpr Real gamma_ = 5. / 3.;
constexpr Real arad = C::a_rad;
constexpr Real TCMB = 2.7;		 // K, CMB temperature
constexpr Real floor_Erad = 1e-40 * arad * TCMB * TCMB * TCMB * TCMB;
constexpr Real Tgas0 = 1.0e4; // K
constexpr Real nH0 = 0.1;     // cm^-3
constexpr Real cloudy_H_mass_fraction = 1.0 / (1.0 + 0.1 * 3.971);
constexpr Real seconds_in_year = 3.1536e7;
constexpr Real parsec_in_cm = C::parsec; // cm == 1 pc
constexpr Real m_H = C::m_p + C::m_e;	   // mass of hydrogen atom
constexpr Real rho0 = nH0 * (m_H / cloudy_H_mass_fraction); // g cm^-3

struct TheProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct Particle_Traits<TheProblem> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<TheProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct Physics_Traits<TheProblem> {
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 1;
	static constexpr int nGroups = 4; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_hat_over_c = chat_over_c;
	static constexpr double Erad_floor = floor_Erad;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::ev2erg; // set boundary unit to eV
	// groups: FIR, NIR, Optical, FUV
	static constexpr amrex::GpuArray<double, Physics_Traits<TheProblem>::nGroups + 1> radBoundaries{
		1.e-04, 1.00778140e-01, 1.00778140e+00, 5.53817071e+00, 1.e+2
	};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
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
	const amrex::GpuArray<double, nGroups_ + 1> dust_opacity{6e2, 1e3, 2e4, 1e5, 2e5}; // dust opacity, cm2/g. last element not used
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[1][i] = dust_opacity[i] * gas_to_dust_ratio;
	}
	return exponents_and_values;
}

template <> struct SimulationData<TheProblem> {
	std::string stars_file = "../inputs/cluster_N500_r20.0_ng4.txt";

	Real refine_threshold = 1.0; // gradient refinement threshold
	int use_periodic_bc = 1;     // default is periodic
};

template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const rho = rho0;
		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = quokka::EOS<TheProblem>::ComputeEintFromTgas(rho, Tgas0);
		Real const Egas = Eint;
		Real const scalar_density = 0;

		state_cc(i, j, k, HydroSystem<TheProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<TheProblem>::scalar0_index) = scalar_density;
	});
}

template <> void QuokkaSimulation<TheProblem>::createInitialStochasticStellarPopParticles()
{
	// Read particles from ASCII file. Note that this only read real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	const int nreal_extra = 6 + Physics_Traits<TheProblem>::nGroups; // mass vx vy vz birth_time death_time lum
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.stars_file, nreal_extra, nullptr);

	constexpr Real SN_mass_threshold = 9.0 * C::M_solar;

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
				const Real mass = p.rdata(quokka::StochasticStellarPopParticleMassIdx);
				const Real death_time = p.rdata(quokka::StochasticStellarPopParticleDeathTimeIdx);
				if (mass < SN_mass_threshold) {
					p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::LowMassStar);
					return;
				}
				if (death_time < 0.0) {
					p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNRemnant);
					return;
				}
				p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<TheProblem>::computeAfterTimestep()
{
	// check conservation of mass
	static auto const &dx = geom[0].CellSizeArray();
	static Real const cvol = AMREX_D_TERM(dx[0], +dx[1], +dx[2]);
	static Real const initial_mass = cvol * state_new_cc_[0].sum(HydroSystem<TheProblem>::density_index);

	const Real mass = cvol * state_new_cc_[0].sum(HydroSystem<TheProblem>::density_index);
	const Real cons_err = (mass - initial_mass) / initial_mass;

	amrex::Print() << "Initial mass = " << initial_mass << "\n"
		       << "Final mass = " << mass << "\n"
		       << "Relative error = " << cons_err << "\n";

	// Will not abort mass nonconservation is expected -- particles will add mass to gas
}

template <> void QuokkaSimulation<TheProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coolingTableType_ == "resampled", "TheProblem diagnostics require resampled cooling tables.");
		const int ncomp = ncomp_cc_in;
		auto tables = resampledTables_.const_tables();

		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<TheProblem>::density_index);
				Real const x1Mom = state(i, j, k, HydroSystem<TheProblem>::x1Momentum_index);
				Real const x2Mom = state(i, j, k, HydroSystem<TheProblem>::x2Momentum_index);
				Real const x3Mom = state(i, j, k, HydroSystem<TheProblem>::x3Momentum_index);
				Real const Egas = state(i, j, k, HydroSystem<TheProblem>::energy_index);
				Real const Eint = RadSystem<TheProblem>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
				Real const Tgas = quokka::ResampledCooling::ComputeTgasFromEgas(rho, Eint, tables);

				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

// template <> void QuokkaSimulation<TheProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
// {
// 	// tag cells for refinement
// 	const Real q_min = 1e-5 * rho0; // minimum density for refinement
// 	const Real eta_threshold = userData_.refine_threshold;

// 	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
// 		const amrex::Box &box = mfi.validbox();
// 		const auto state = state_new_cc_[lev].const_array(mfi);
// 		const auto tag = tags.array(mfi);
// 		const int nidx = HydroSystem<TheProblem>::density_index;

// 		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
// 			Real const q = state(i, j, k, nidx);
// 			Real const q_xplus = state(i + 1, j, k, nidx);
// 			Real const q_xminus = state(i - 1, j, k, nidx);
// 			Real const q_yplus = state(i, j + 1, k, nidx);
// 			Real const q_yminus = state(i, j - 1, k, nidx);
// 			Real const q_zplus = state(i, j, k + 1, nidx);
// 			Real const q_zminus = state(i, j, k - 1, nidx);

// 			Real const del_x = 0.5 * (q_xplus - q_xminus);
// 			Real const del_y = 0.5 * (q_yplus - q_yminus);
// 			Real const del_z = 0.5 * (q_zplus - q_zminus);
// 			Real const gradient_indicator = std::sqrt(del_x * del_x + del_y * del_y + del_z * del_z) / q;

// 			if ((gradient_indicator > eta_threshold) && (q > q_min)) {
// 				tag(i, j, k) = amrex::TagBox::SET;
// 			}
// 		});
// 	}
// }

auto problem_main() -> int
{
	// This problem is only implemented in CGS units because the cooling tables are provided in CGS units.
	static_assert(Physics_Traits<TheProblem>::unit_system == UnitSystem::CGS);

	// read parameters
	amrex::ParmParse const pp;

	// // read in refinement threshold (relative gradient in density)
	// Real refine_threshold = 0.1;
	// pp.query("refine_threshold", refine_threshold); // dimensionless

	// use periodic boundary conditions or not
	int use_periodic_bc = 0;
	pp.query("use_periodic_bc", use_periodic_bc);

	// Problem initialization
	auto BCs_cc = (use_periodic_bc == 1) ? quokka::BC<TheProblem>(quokka::BCType::int_dir) : quokka::BC<TheProblem>(quokka::BCType::reflecting);

	QuokkaSimulation<TheProblem> sim(BCs_cc);
	sim.densityFloor_ = 1.0e-5 * rho0; // density floor (to prevent vacuum)
	sim.userData_.use_periodic_bc = use_periodic_bc;

	// Set initial conditions
	sim.setInitialConditions();

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// run simulation
	sim.evolve();

	// Cleanup and exit
	const int status = 0;
	return status;
}
