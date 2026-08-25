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
constexpr amrex::Real Binit_x = 5.0 ;
constexpr amrex::Real Binit_y = 0.0;
constexpr amrex::Real Binit_z = 0.0;
struct TheProblem {
};

template <> struct SimulationData<TheProblem> {
	// turbulent velocity fields
	Real initial_scalar_density = NAN; // scalar density in cgs units

	Real refine_parameter = 1.0; // placeholder for refinement control
	std::string stars_file;	     // default: no stars
	std::string IC_file;	     // Initial disk vertical structure

	// Initial conditions table: z -> (g_1, g_ext, phi_tot)
	quokka::DataTable<1, 3, quokka::OutOfBounds::clamp> ic_table;

	// Galaxy parameters (default is solar neighborhood)
	Real rho01 = 4.320441e-24; // 2.58 m_p/cm^3
	Real sigma1 = 700000.0;

	// Magnetic field: Bx = B_0 * exp(-|z|/z_d), By = 0, Bz = 0 (parsed once in problem_main)
	Real B_0 = NAN;	 // code units (Lorentz-Heaviside)
	Real z_d = NAN;	 // cm

	// hot and warm gas definitions
	Real hot_T = 1.0e6;  // K
	Real warm_T = 2.0e4; // K
	Real cold_T = 200.0; // K
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
	static constexpr bool is_mhd_enabled = true;
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

	// Capture galaxy parameters from userData_ for GPU kernel
	const Real sigma1_ic = userData_.sigma1;
	const Real sigma2_ic = 10.0 * sigma1_ic;
	const Real rho01_ic = userData_.rho01;
	const Real rho02_ic = 1.0e-5 * rho01_ic;

	// Magnetic field parameters (parsed once in problem_main; must match setInitialConditionsOnGridFaceVars)
	const Real B_0_ic = userData_.B_0;
	const Real z_d_ic = userData_.z_d;

	// Create GPU const tables for initial conditions if available
	const auto &ic_table = userData_.ic_table.const_tables();

	const amrex::Real initial_scalar_density = userData_.initial_scalar_density;

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

		// Magnetic field (must match setInitialConditionsOnGridFaceVars): Bx = B_0 * exp(-|z|/z_d), By = 0, Bz = 0
		const double Bx_ic = B_0_ic * std::exp(-std::abs(z) / z_d_ic);
		const double Emag = 0.5 * Bx_ic * Bx_ic;
		const double Eint = P / (gamma - 1.);

		state_cc(i, j, k, HydroSystem<TheProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = Eint + Emag;

		const auto initial_scalar_density_d = initial_scalar_density;

		// Initialize passive scalar field
		if constexpr (Physics_Traits<TheProblem>::numPassiveScalars > 0) {
			state_cc(i, j, k, HydroSystem<TheProblem>::scalar0_index) = initial_scalar_density_d;
		}
	});
}
//Add magnetic field initial conditions

template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
    // Magnetic field parameters (parsed once in problem_main; must match setInitialConditionsOnGrid)
    const Real B_0 = userData_.B_0;
    const Real z_d = userData_.z_d;

    const amrex::Array4<double> &state_fc = grid_elem.array_;
    const amrex::Box &indexRange = grid_elem.indexRange_;
    const quokka::direction dir = grid_elem.dir_;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        // Cartesian coordinates at this face
        amrex::Real const dx_cen = (dir == quokka::direction::x) ? 0.0 : 0.5 * dx[0];
        amrex::Real const dy_cen = (dir == quokka::direction::y) ? 0.0 : 0.5 * dx[1];
        amrex::Real const dz_cen = (dir == quokka::direction::z) ? 0.0 : 0.5 * dx[2];
        amrex::Real const x = prob_lo[0] + (i * dx[0]) + dx_cen;
        amrex::Real const y = prob_lo[1] + (j * dx[1]) + dy_cen;
        amrex::Real const z = prob_lo[2] + (k * dx[2]) + dz_cen;

        amrex::Real Bx = B_0 * std::exp(-std::abs(z) / z_d);
        amrex::Real By = 0.0;
        amrex::Real Bz = 0.0;

        constexpr int mhd_index = Physics_Indices<TheProblem>::mhdFirstIndex;
        if (dir == quokka::direction::x) {
            state_fc(i, j, k, mhd_index) = Bx;
        } else if (dir == quokka::direction::y) {
            state_fc(i, j, k, mhd_index) = By;
        } else if (dir == quokka::direction::z) {
            state_fc(i, j, k, mhd_index) = Bz;
        }
    });
}

template <>
void QuokkaSimulation<TheProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in,
						     amrex::MultiFab const &state_cc, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & state_fc) const
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
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
			    AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<TheProblem>::density_index);
				Real const Eint = HydroSystem<TheProblem>::ComputeInternalEnergy(state, i, j, k, &cons_fc);
				output(i, j, k, ncomp) = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Eint);
			});
		}
	} else if (dname == "bfield_strength") {
		static_assert(Physics_Traits<TheProblem>::is_mhd_enabled, "bfield_strength requires MHD to be enabled.");
		const int ncomp = ncomp_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &bx_fc = state_fc[0].const_array(iter);
			auto const &by_fc = state_fc[1].const_array(iter);
			auto const &bz_fc = state_fc[2].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real bx_cc = 0.5 * (bx_fc(i, j, k, 0) + bx_fc(i + 1, j, k, 0));
				const amrex::Real by_cc = 0.5 * (by_fc(i, j, k, 0) + by_fc(i, j + 1, k, 0));
				const amrex::Real bz_cc = 0.5 * (bz_fc(i, j, k, 0) + bz_fc(i, j, k + 1, 0));
				const amrex::Real b_code = std::sqrt(bx_cc * bx_cc + by_cc * by_cc + bz_cc * bz_cc);
				output(i, j, k, ncomp) = b_code * std::sqrt(4.0 * M_PI) * 1.0e6;
			});
		}
	} else if (dname == "Bx") {
		static_assert(Physics_Traits<TheProblem>::is_mhd_enabled, "Bx requires MHD to be enabled.");
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &bx_fc = state_fc[0].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real bx_cc = 0.5 * (bx_fc(i, j, k, 0) + bx_fc(i + 1, j, k, 0));
				output(i, j, k, ncomp) = bx_cc * std::sqrt(4.0 * M_PI) * 1.0e6;
			});
		}
	} else if (dname == "By") {
		static_assert(Physics_Traits<TheProblem>::is_mhd_enabled, "By requires MHD to be enabled.");
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &by_fc = state_fc[1].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real by_cc = 0.5 * (by_fc(i, j, k, 0) + by_fc(i, j + 1, k, 0));
				output(i, j, k, ncomp) = by_cc * std::sqrt(4.0 * M_PI) * 1.0e6;
			});
		}
	} else if (dname == "Bz") {
		static_assert(Physics_Traits<TheProblem>::is_mhd_enabled, "Bz requires MHD to be enabled.");
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &bz_fc = state_fc[2].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real bz_cc = 0.5 * (bz_fc(i, j, k, 0) + bz_fc(i, j, k + 1, 0));
				output(i, j, k, ncomp) = bz_cc * std::sqrt(4.0 * M_PI) * 1.0e6;
			});
		}
	}
	amrex::Gpu::streamSynchronizeAll();
}

// Add Strang Split Source Term for External Fixed Potential Here
template <> void QuokkaSimulation<TheProblem>::addStrangSplitSources(amrex::MultiFab &mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &mf_fc, int lev, amrex::Real time, amrex::Real dt_lev) // NOLINT
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx = geom[lev].CellSizeArray();
	const Real dt = dt_lev;

	// Create GPU const tables for initial conditions if available
	const auto &ic_table = userData_.ic_table.const_tables();

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> state_fc{};
		if constexpr (Physics_Traits<TheProblem>::is_mhd_enabled) {
						state_fc[0] = mf_fc[0].const_array(iter);
						state_fc[1] = mf_fc[1].const_array(iter);
						state_fc[2] = mf_fc[2].const_array(iter);
					}

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

			// static_assert(!Physics_Traits<TheProblem>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
			Real Emag = 0.0;
			if constexpr (Physics_Traits<TheProblem>::is_mhd_enabled) {
				Emag = ComputeCellCenteredMagneticEnergy<TheProblem>(i, j, k, state_fc);
			}
			const Real Eint = quokka::EOS<TheProblem>::ComputeEintFromEgas(rho, x1mom, x2mom, x3mom, Egas, Emag);

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

			// static_assert(!Physics_Traits<TheProblem>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
			const Real Egas_new = quokka::EOS<TheProblem>::ComputeEgasFromEint(rho, x1mom_new, x2mom_new, x3mom_new, Eint, Emag);
			AMREX_ASSERT(!std::isnan(Egas_new));

			state(i, j, k, HydroSystem<TheProblem>::energy_index) = Egas_new;
		});
	}
}

// Outflow boundary conditions in z (direction 2) are implemented via the native AMReX
// "foextrap" BCType (zero-gradient extrapolation), set at runtime through quokka.bc for both
// cell-centered and face-centered variables. No custom device function is required for foextrap,
// so no setCustomBoundaryConditions/setCustomBoundaryConditionsFaceVar specialization is needed here.

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
	pp.query("initial_scalar_density", sim.userData_.initial_scalar_density);
	pp.query("hot_T", sim.userData_.hot_T);
	pp.query("warm_T", sim.userData_.warm_T);
	if constexpr (Physics_Traits<TheProblem>::numPassiveScalars > 0) {
		AMREX_ALWAYS_ASSERT(!std::isnan(sim.userData_.initial_scalar_density));
	}
	// particles.scalar_yield_per_SN must be set as well, and it should be greater than (initial_scalar_density * (128 pc)^3),
	// so that the SN ejected metal density in SN remnant is greater than the background density.
	amrex::ParmParse const pp_particles("particles");
	double scalar_yield_per_SN = NAN;
	pp_particles.query("scalar_yield_per_SN", scalar_yield_per_SN);
	AMREX_ALWAYS_ASSERT(!std::isnan(scalar_yield_per_SN));
	const Real SNR_volume = std::pow(128.0 * C::parsec, 3);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(scalar_yield_per_SN > sim.userData_.initial_scalar_density * SNR_volume,
					 "particles.scalar_yield_per_SN must be greater than (initial_scalar_density * (128 pc)^3)");

	// Magnetic field: Bx = B_0 * exp(-|z|/z_d), By = 0, Bz = 0.
	// Parsed once here (rather than in each IC-setting function) and shared via userData_.
	double magnetic_field_microgauss = 5.0; // default value in microGauss
	double disk_zscale_kpc = 1.0;
	amrex::ParmParse const pp_bfield("magnetic_field");
	pp_bfield.query("magnetic_field_microgauss", magnetic_field_microgauss);
	pp_bfield.query("disk_zscale_kpc", disk_zscale_kpc);
	sim.userData_.B_0 = magnetic_field_microgauss * 1.0e-6 / std::sqrt(4.0 * M_PI);
	sim.userData_.z_d = disk_zscale_kpc * (1.0e3 * C::parsec);

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
