/// \file testTallBoxSf.cpp
/// \brief Defines a problem for a galactic patch (tall box) with self-consistent star formation and SN feedback.
///

#include <cmath>
#include <iostream>
#include <utility>

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
#include "turbulence/TurbDataReader.hpp"
#include "util/DataTable.hpp"

constexpr double mu = 1.0 * C::m_p;

struct TheProblem {
};

template <> struct SimulationData<TheProblem> {
	// turbulent velocity fields
	amrex::TableData<Real, 3> dvx;
	amrex::TableData<Real, 3> dvy;
	amrex::TableData<Real, 3> dvz;
	Real dv_rms_generated{};
	Real turbulent_amplitude = 1500.0; // cm/s,  0.05 * cs at 10K (~0.3 km/s)
	int turbulent_size = 128;
	Real initial_scalar_per_cell = 0.0; // the actual density is initial_scalar_per_cell / cell_volume

	Real refine_parameter = 1.0; // placeholder for refinement control
	std::string stars_file;	     // default: no stars
	std::string IC_file;	     // Initial disk vertical structure

	// Initial conditions table: z -> (g_1, g_ext, phi_tot)
	quokka::DataTable<1, 3, quokka::OutOfBounds::clamp> ic_table;

	// Galaxy parameters (default is solar neighborhood)
	Real rho01 = 4.320441e-24; // 2.58 m_p/cm^3
	Real sigma1 = 700000.0;

	// hot and warm gas definitions
	Real hot_T = 1.0e6;  // K
	Real warm_T = 2.0e4; // K
};

template <> struct Particle_Traits<TheProblem> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<TheProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = mu;
	using EOSBackend = quokka::EOSTabulated<TheProblem>;
};

template <> struct Physics_Traits<TheProblem> : DefaultPhysicsTraits {
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numPassiveScalars = numMassScalars + 1; // number of passive scalars
};

template <> void QuokkaSimulation<TheProblem>::createInitialStochasticStellarPopParticles()
{
	if (userData_.stars_file.empty()) {
		amrex::Print() << "No stars file specified. Skipping particle creation.\n";
		return;
	}

	// Read particles from ASCII file. Note that this only read real components and not integer components, therefore we need to use
	// InitSetPhyParticles to set the integer components
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.stars_file, nreal_extra, nullptr);

	// Using a for loop from lev = 0 to StochasticStellarPopParticles->maxLevel() won't work because not all levels necessarily have particles, and when
	// some levels do not have particles, StochasticStellarPopParticles->GetParticles(lev) will result in a Segfault. Therefore, we loop over the actual
	// particle container. See https://github.com/AMReX-Codes/amrex/issues/4896
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
	if (refine_zmax_list.empty() || std::cmp_greater_equal(lev, refine_zmax_list.size())) {
		return;
	}

	const amrex::Real refine_zmax = refine_zmax_list[lev];

	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
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
		const int nturb_y = turbData.dvx.end[1] - turbData.dvx.begin[1];
		const int nturb_z = turbData.dvx.end[2] - turbData.dvx.begin[2];
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.turbulent_size == nturb_y && nturb_y == nturb_z, "Turbulence data must be a cube");
		amrex::Print() << "turbulence data size is: " << userData_.turbulent_size << "^3\n";

		// copy to GPU
		userData_.dvx.resize(pinned_dvx.lo(), pinned_dvx.hi());
		userData_.dvx.copy(pinned_dvx);

		userData_.dvy.resize(pinned_dvy.lo(), pinned_dvy.hi());
		userData_.dvy.copy(pinned_dvy);

		userData_.dvz.resize(pinned_dvz.lo(), pinned_dvz.hi());
		userData_.dvz.copy(pinned_dvz);

		// Read initial conditions from file if specified
		if (!userData_.IC_file.empty()) {
			amrex::Print() << "Reading initial conditions from: " << userData_.IC_file << "\n";
			// Read CSV file with linear spacing for outputs
			userData_.ic_table = quokka::DataTable<1, 3, quokka::OutOfBounds::clamp>::CSVReader(userData_.IC_file, quokka::TransformType::linear);
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.ic_table.is_initialized(), "Initial conditions table failed to load.");
			amrex::Print() << "Initial conditions table loaded successfully.\n";
		} else {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, "No IC file specified. Please specify problem.IC_file in the input file.");
		}

		isSamplingDone = true;
	}
}

template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
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

	// get simulation box dimensions
	const int nx = indexRange.length(0);
	const int ny = indexRange.length(1);
	const int nz = indexRange.length(2);
	const int nturb = turb_hi[0] - turb_lo[0] + 1;

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nx <= nturb, "nx must be less than or equal to turbulent_size (128)");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ny >= nx && nz >= nx, "ny and nz must be greater than or equal to nx");

	// Capture galaxy parameters from userData_ for GPU kernel
	const Real sigma1_ic = userData_.sigma1;
	const Real sigma2_ic = 10.0 * sigma1_ic;
	const Real rho01_ic = userData_.rho01;
	const Real rho02_ic = 1.0e-5 * rho01_ic;

	// Create GPU const tables for initial conditions if available
	const auto &ic_table = userData_.ic_table.const_tables();

	amrex::Real initial_scalar_density = 0.0;
	if constexpr (Physics_Traits<TheProblem>::numPassiveScalars > 0) {
		const amrex::Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
		initial_scalar_density = userData_.initial_scalar_per_cell / cell_vol;
	}

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const z = prob_lo[2] + ((k + static_cast<amrex::Real>(0.5)) * dx[2]);

		// Use DataTable to interpolate initial conditions from file
		// Table provides: [rho, g_z, Phi] as functions of z
		std::array<amrex::Real, 1> const point = {std::abs(z)};
		auto const ic_values = ic_table.interpolate(point);

		// Extract values: ic_values[0] = g_1, ic_values[1] = g_ext, ic_values[2] = phi_tot
		const double phi_tot = ic_values[2];
		const double rho1 = rho01_ic * std::exp(-phi_tot / (sigma1_ic * sigma1_ic));
		const double rho2 = rho02_ic * std::exp(-phi_tot / (sigma2_ic * sigma2_ic));
		const double rho = rho1 + rho2;

		const double P = rho1 * sigma1_ic * sigma1_ic + rho2 * sigma2_ic * sigma2_ic;

		AMREX_ASSERT(!std::isnan(rho));

		// const double Tgas = P / (rho / mu * C::k_B);

		const auto gamma = quokka::EOS_Traits<TheProblem>::gamma;

		// add turbulent velocities
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
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx + vy * vy + vz * vz);

		const auto initial_scalar_density_d = initial_scalar_density;

		// Initialize passive scalar field
		if constexpr (Physics_Traits<TheProblem>::numPassiveScalars > 0) {
			state_cc(i, j, k, HydroSystem<TheProblem>::scalar0_index) = initial_scalar_density_d;
		}
	});
}

template <>
void QuokkaSimulation<TheProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in,
						     amrex::MultiFab const &state_cc, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & /*state_fc*/) const
{
	const int ncomp = ncomp_in;
	auto const &output = mf.arrays();
	auto const &state = state_cc.const_arrays();

	if (dname == "gpot") {
		auto const &phi_arr = phi[lev].const_arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
		amrex::Gpu::streamSynchronize();
	} else if (dname == "temperature") {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coolingTableType_ == "resampled", "diagnostics require resampled cooling tables.");
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<TheProblem>::density_index);
			Real const Eint = HydroSystem<TheProblem>::ComputeInternalEnergy(state[bx], i, j, k, nullptr);
			output[bx](i, j, k, ncomp) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Eint);
		});
	} else if (dname == "c_s") {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(coolingTableType_ == "resampled", "diagnostics require resampled cooling tables.");
		auto tables = resampledTables_.const_tables();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<TheProblem>::density_index);
			Real const Eint = HydroSystem<TheProblem>::ComputeInternalEnergy(state[bx], i, j, k, nullptr);
			output[bx](i, j, k, ncomp) = quokka::ResampledCooling::ComputeSoundSpeedFromRhoEint(rho, Eint, tables) / 1.0e5; // km/s
		});
	} else if (dname == "scalar0_z_outflow_rate") {
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<TheProblem>::density_index);
			Real const vz = state[bx](i, j, k, HydroSystem<TheProblem>::x3Momentum_index) / rho;
			Real const scalar0 = state[bx](i, j, k, HydroSystem<TheProblem>::scalar0_index);
			output[bx](i, j, k, ncomp) = scalar0 * vz;
		});
	} else {
		Real const hot_T = userData_.hot_T;
		Real const warm_T = userData_.warm_T;
		if (dname == "hot_gas_z_outflow_rate") {
			amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
				Real const rho = state[bx](i, j, k, HydroSystem<TheProblem>::density_index);
				Real const Eint = HydroSystem<TheProblem>::ComputeInternalEnergy(state[bx], i, j, k, nullptr);
				Real const Tgas = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Eint);
				output[bx](i, j, k, ncomp) = (Tgas > hot_T) ? state[bx](i, j, k, HydroSystem<TheProblem>::x3Momentum_index) : 0.0;
			});
		} else if (dname == "warm_gas_z_outflow_rate") {
			amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
				Real const rho = state[bx](i, j, k, HydroSystem<TheProblem>::density_index);
				Real const Eint = HydroSystem<TheProblem>::ComputeInternalEnergy(state[bx], i, j, k, nullptr);
				Real const Tgas = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Eint);
				output[bx](i, j, k, ncomp) = (Tgas < warm_T) ? state[bx](i, j, k, HydroSystem<TheProblem>::x3Momentum_index) : 0.0;
			});
		} else if (dname == "hot_scalar0_z_outflow_rate") {
			amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
				Real const rho = state[bx](i, j, k, HydroSystem<TheProblem>::density_index);
				Real const Eint = HydroSystem<TheProblem>::ComputeInternalEnergy(state[bx], i, j, k, nullptr);
				Real const Tgas = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Eint);
				Real const vz = state[bx](i, j, k, HydroSystem<TheProblem>::x3Momentum_index) / rho;
				Real const scalar0 = state[bx](i, j, k, HydroSystem<TheProblem>::scalar0_index);
				output[bx](i, j, k, ncomp) = (Tgas > hot_T) ? scalar0 * vz : 0.0;
			});
		} else if (dname == "warm_scalar0_z_outflow_rate") {
			amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
				Real const rho = state[bx](i, j, k, HydroSystem<TheProblem>::density_index);
				Real const Eint = HydroSystem<TheProblem>::ComputeInternalEnergy(state[bx], i, j, k, nullptr);
				Real const Tgas = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Eint);
				Real const vz = state[bx](i, j, k, HydroSystem<TheProblem>::x3Momentum_index) / rho;
				Real const scalar0 = state[bx](i, j, k, HydroSystem<TheProblem>::scalar0_index);
				output[bx](i, j, k, ncomp) = (Tgas < warm_T) ? scalar0 * vz : 0.0;
			});
		}
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
	const auto &ic_table = userData_.ic_table.const_tables();

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

			static_assert(!Physics_Traits<TheProblem>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
			const Real Eint = quokka::EOS<TheProblem>::ComputeEintFromEgas(rho, x1mom, x2mom, x3mom, Egas, 0.0);

			posvec[0] = prob_lo[0] + (i + 0.5) * dx[0];
			posvec[1] = prob_lo[1] + (j + 0.5) * dx[1];
			posvec[2] = prob_lo[2] + (k + 0.5) * dx[2];

			// Calculate gradient of fixed potential using captured parameters
			double const z = posvec[2];

			std::array<amrex::Real, 1> const point = {std::abs(z)};
			auto const ic_values = ic_table.interpolate(point);
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

			static_assert(!Physics_Traits<TheProblem>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
			const Real Egas_new = quokka::EOS<TheProblem>::ComputeEgasFromEint(rho, x1mom_new, x2mom_new, x3mom_new, Eint, 0.0);
			AMREX_ASSERT(!std::isnan(Egas_new));

			state(i, j, k, HydroSystem<TheProblem>::energy_index) = Egas_new;
		});
	}
}

// Implement User-defined diode BC
// Diode BC: allows outflow, prevents inflow by reflecting the z-momentum
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<TheProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
												int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
												const Real /*time*/, const amrex::BCRec * /*bcr*/,
												int /*bcomp*/, int /*orig_comp*/)
{
	// Apply diode boundary conditions in z-direction (direction 2)
	setDiodeBCLo<2>(iv, consVar, geom);
	setDiodeBCHi<2>(iv, consVar, geom);
}

auto problem_main() -> int
{
	// set random state
	const int rank = amrex::ParallelDescriptor::MyProc();
	const int seed = 42 + rank;
	amrex::InitRandom(seed, 1);

	// Problem initialization
	QuokkaSimulation<TheProblem> sim;

	amrex::ParmParse const pp("problem");
	pp.query("stars_file", sim.userData_.stars_file);
	pp.query("IC_file", sim.userData_.IC_file);
	pp.query("rho01", sim.userData_.rho01);
	pp.query("sigma1", sim.userData_.sigma1);
	pp.query("initial_scalar_per_cell", sim.userData_.initial_scalar_per_cell);
	pp.query("hot_T", sim.userData_.hot_T);
	pp.query("warm_T", sim.userData_.warm_T);

	// preCalculate must be explicitly called here to ensure
	// ic_table is initialized even when restarting from checkpoint
	sim.preCalculateInitialConditions();

	// initialize (this will parse particle parameters and load luminosity table)
	sim.setInitialConditions();

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	const amrex::Real total_gas_energy_init = sim.state_new_cc_[0].sum(HydroSystem<TheProblem>::energy_index) * vol;
	const amrex::Real total_energy_init = total_gas_energy_init;

	// set force finest level to true for test particles
	// sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StochasticStellarPop)->setForceFinestLevel(true);

	sim.evolve();

	const amrex::Real total_gas_energy = sim.state_new_cc_[0].sum(HydroSystem<TheProblem>::energy_index) * vol;
	const amrex::Real total_energy_final = total_gas_energy;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Total gas energy (initial): " << total_gas_energy_init << "\n";
		amrex::Print() << "Total gas energy (final): " << total_gas_energy << "\n";
		amrex::Print() << "Change of total energy: " << total_energy_final - total_energy_init << "\n";
		amrex::Print() << "Relative change of total energy: " << (total_energy_final - total_energy_init) / total_energy_init << "\n";
	}

	return 0;
}
