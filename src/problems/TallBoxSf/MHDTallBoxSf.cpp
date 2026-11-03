/// \file MHDTallBoxSf.cpp
/// \brief MHD version of TallBoxSf: a galactic patch (tall box) with self-consistent star formation and SN feedback, a horizontal magnetic field,
/// and the MHD diode (outflow, no-inflow) boundary condition on the vertical (z) boundaries.
///
/// The gas set-up (vertical disk profile, external potential, cooling, turbulence driving, star particles) follows testTallBoxSf.cpp.
/// The initial magnetic field follows the SILCC set-up (Walch et al. 2015, MNRAS 454, 238): a horizontal field along x whose strength scales with the
/// gas density, B_x(z) = B_0 sqrt(rho(z) / rho_0), with rho_0 the midplane density and a midplane strength B_0 = 3 microgauss (the default of
/// problem.B0_uG). This field depends on z only, so it is divergence-free.

#include <array>
#include <cmath>
#include <fstream>
#include <string>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_Random.H"
#include "AMReX_Reduce.H"
#include "AMReX_SPACE.H"
#include "util/BC.hpp"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "util/DataTable.hpp"
#include "util/time_units.hpp"

constexpr double mu = 1.0 * C::m_p;

struct MHDTallBoxSf {};

template <> struct SimulationData<MHDTallBoxSf> {
	Real initial_scalar_density = NAN; // scalar density in cgs units

	std::string stars_file; // default: no stars
	std::string IC_file;	// Initial disk vertical structure

	// Initial conditions table: z -> (g_1, g_ext, phi_tot)
	quokka::DataTable<1, 3, quokka::OutOfBounds::clamp> ic_table;

	// Galaxy parameters (default is solar neighborhood)
	Real rho01 = 4.320441e-24; // 2.58 m_p/cm^3
	Real sigma1 = 700000.0;

	// midplane magnetic field strength in microgauss (Gaussian units)
	Real B0_uG = 3.0;

	// total outflow rates are measured through the planes |z| = outflow_height
	Real outflow_height = 1.0e3 * C::parsec; // 1 kpc
	std::string outflow_file = "MHDTallBoxSf_outflow.txt";
};

template <> struct Particle_Traits<MHDTallBoxSf> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<MHDTallBoxSf> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct quokka::EOS_Traits<MHDTallBoxSf> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = mu;
	using EOSBackend = quokka::EOSTabulated<MHDTallBoxSf>;
};

template <> struct Physics_Traits<MHDTallBoxSf> : DefaultPhysicsTraits {
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numPassiveScalars = numMassScalars + 1; // number of passive scalars
};

namespace
{
// Quokka's MHD uses E_B = B^2 / 2 (Heaviside-Lorentz form with cgs base units), so a field B_G in gauss (Gaussian units, E_B = B_G^2 / (8 pi))
// is B_G / sqrt(4 pi) in code units.
constexpr double microgauss_to_code = 1.0e-6 / 2.0 / 1.7724538509055160273; // 1e-6 / sqrt(4 pi)

/// Two-component isothermal disk of testTallBoxSf.cpp: returns {rho, P} at height z.
template <typename Table>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto diskState(Table const &ic_table, Real z, Real rho01, Real sigma1) -> amrex::GpuArray<Real, 2>
{
	const Real sigma2 = 10.0 * sigma1;
	const Real rho02 = 1.0e-5 * rho01;
	std::array<amrex::Real, 1> const point = {std::abs(z)};
	auto const ic_values = ic_table.interpolate(point);
	// ic_values[0] = g_1, ic_values[1] = g_ext, ic_values[2] = phi_tot
	const Real phi_tot = ic_values[2];
	const Real rho1 = rho01 * std::exp(-phi_tot / (sigma1 * sigma1));
	const Real rho2 = rho02 * std::exp(-phi_tot / (sigma2 * sigma2));
	return {rho1 + rho2, rho1 * sigma1 * sigma1 + rho2 * sigma2 * sigma2};
}

/// B_x(z) = B_0 sqrt(rho(z) / rho_0) in code units; rho_0 = rho01 + rho02 is the midplane density (phi_tot(0) = 0).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto fieldBx(Real rho, Real rho01, Real B0_code) -> Real
{
	return B0_code * std::sqrt(rho / (rho01 * (1.0 + 1.0e-5)));
}
} // namespace

template <> void QuokkaSimulation<MHDTallBoxSf>::createInitialStochasticStellarPopParticles()
{
	if (userData_.stars_file.empty()) {
		amrex::Print() << "No stars file specified. Skipping particle creation.\n";
		return;
	}

	// Read particles from ASCII file (real components only); the integer components are set below.
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	StochasticStellarPopParticles->SetVerbose(1);
	StochasticStellarPopParticles->InitFromAsciiFile(userData_.stars_file, nreal_extra, nullptr);

	// loop over the actual particle container, see testTallBoxSf.cpp and https://github.com/AMReX-Codes/amrex/issues/4896
	for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
			if (np == 0) {
				continue;
			}
			auto *pdata = particle_array().data();
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
			});
		}
	}
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<MHDTallBoxSf>::preCalculateInitialConditions()
{
	static bool isSamplingDone = false;
	if (!isSamplingDone) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!userData_.IC_file.empty(), "No IC file specified. Please specify problem.IC_file in the input file.");
		amrex::Print() << "Reading initial conditions from: " << userData_.IC_file << "\n";
		userData_.ic_table = quokka::DataTable<1, 3, quokka::OutOfBounds::clamp>::CSVReader(userData_.IC_file, quokka::TransformType::linear);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.ic_table.is_initialized(), "Initial conditions table failed to load.");
		isSamplingDone = true;
	}
}

template <> void QuokkaSimulation<MHDTallBoxSf>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const Real sigma1 = userData_.sigma1;
	const Real rho01 = userData_.rho01;
	const Real B0_code = userData_.B0_uG * microgauss_to_code;
	const auto &ic_table = userData_.ic_table.const_tables();
	const amrex::Real initial_scalar_density = userData_.initial_scalar_density;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const z = prob_lo[2] + ((k + static_cast<amrex::Real>(0.5)) * dx[2]);
		auto const disk = diskState(ic_table, z, rho01, sigma1);
		const double rho = disk[0];
		const double P = disk[1];
		AMREX_ASSERT(!std::isnan(rho));

		// both x-faces of this cell hold B_x(z_k) (see setInitialConditionsOnGridFaceVars), so the face-averaged field is B_x(z_k)
		const double Bx = fieldBx(rho, rho01, B0_code);
		const double Emag = 0.5 * Bx * Bx;

		const auto gamma = quokka::EOS_Traits<MHDTallBoxSf>::gamma;
		state_cc(i, j, k, HydroSystem<MHDTallBoxSf>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<MHDTallBoxSf>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDTallBoxSf>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDTallBoxSf>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDTallBoxSf>::internalEnergy_index) = P / (gamma - 1.);
		state_cc(i, j, k, HydroSystem<MHDTallBoxSf>::energy_index) = P / (gamma - 1.) + Emag;
		state_cc(i, j, k, HydroSystem<MHDTallBoxSf>::scalar0_index) = initial_scalar_density;
	});
}

template <> void QuokkaSimulation<MHDTallBoxSf>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	const Real sigma1 = userData_.sigma1;
	const Real rho01 = userData_.rho01;
	const Real B0_code = userData_.B0_uG * microgauss_to_code;
	const auto &ic_table = userData_.ic_table.const_tables();

	// x-faces get B_x at the cell-centre height of their cell layer; y- and z-faces get zero, so the discrete div B vanishes exactly
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double b = 0.;
		if (dir == quokka::direction::x) {
			amrex::Real const z = prob_lo[2] + ((k + static_cast<amrex::Real>(0.5)) * dx[2]);
			b = fieldBx(diskState(ic_table, z, rho01, sigma1)[0], rho01, B0_code);
		}
		state_fc(i, j, k, Physics_Indices<MHDTallBoxSf>::mhdFirstIndex) = b;
	});
}

template <>
void QuokkaSimulation<MHDTallBoxSf>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in,
						       amrex::MultiFab const &state_cc, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
	const int ncomp = ncomp_in;
	auto const &output = mf.arrays();
	auto const &state = state_cc.const_arrays();

	if (dname == "gpot") {
		auto const &phi_arr = phi[lev].const_arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
	} else if (dname == "temperature") {
		auto const &b1 = state_fc[0].const_arrays();
		auto const &b2 = state_fc[1].const_arrays();
		auto const &b3 = state_fc[2].const_arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const fc = {b1[bx], b2[bx], b3[bx]};
			Real const rho = state[bx](i, j, k, HydroSystem<MHDTallBoxSf>::density_index);
			Real const Eint = HydroSystem<MHDTallBoxSf>::ComputeInternalEnergy(state[bx], i, j, k, &fc);
			output[bx](i, j, k, ncomp) = quokka::EOS<MHDTallBoxSf>::ComputeTgasFromEint(rho, Eint);
		});
	} else if (dname == "gas_z_outflow_rate" || dname == "metal_z_outflow_rate") {
		// outward (away from the midplane) vertical mass flux density of gas or metals [g cm^-2 s^-1]; scalar_0 is the metal density
		const int comp = (dname == "gas_z_outflow_rate") ? HydroSystem<MHDTallBoxSf>::density_index : HydroSystem<MHDTallBoxSf>::scalar0_index;
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const z = prob_lo[2] + (k + 0.5) * dx[2];
			Real const vz =
			    state[bx](i, j, k, HydroSystem<MHDTallBoxSf>::x3Momentum_index) / state[bx](i, j, k, HydroSystem<MHDTallBoxSf>::density_index);
			output[bx](i, j, k, ncomp) = state[bx](i, j, k, comp) * vz * std::copysign(1.0, z);
		});
	}
	amrex::Gpu::streamSynchronizeAll();
}

// Total gas and metal outflow rates through the planes |z| = outflow_height, appended to outflow_file every step.
// The flux is taken at the centres of the cells that contain the planes, on level 0.
template <> void QuokkaSimulation<MHDTallBoxSf>::computeAfterTimestep()
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[0].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[0].CellSizeArray();
	const Real z_out = userData_.outflow_height;
	auto const &state = state_new_cc_[0].const_arrays();

	auto const [mdot_gas, mdot_metal] =
	    amrex::ParReduce(amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{}, amrex::TypeList<amrex::Real, amrex::Real>{}, state_new_cc_[0],
			     amrex::IntVect(0), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
				     Real const z_lo = prob_lo[2] + k * dx[2];
				     Real const z_c = z_lo + 0.5 * dx[2];
				     bool const in_plane = ((z_lo <= z_out) && (z_out < z_lo + dx[2])) || ((z_lo <= -z_out) && (-z_out < z_lo + dx[2]));
				     if (!in_plane) {
					     return {0.0, 0.0};
				     }
				     Real const rho = state[bx](i, j, k, HydroSystem<MHDTallBoxSf>::density_index);
				     Real const vz_out = state[bx](i, j, k, HydroSystem<MHDTallBoxSf>::x3Momentum_index) / rho * std::copysign(1.0, z_c);
				     Real const area = dx[0] * dx[1];
				     return {rho * vz_out * area, state[bx](i, j, k, HydroSystem<MHDTallBoxSf>::scalar0_index) * vz_out * area};
			     });
	std::array<amrex::Real, 2> rates = {mdot_gas, mdot_metal};
	amrex::ParallelDescriptor::ReduceRealSum(rates.data(), static_cast<int>(rates.size()));

	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr Real msun_per_yr = C::M_solar / quokka::yr_in_s; // g/s
		std::ofstream file(userData_.outflow_file, std::ios::app);
		file << tNew_[0] / quokka::Myr_in_s << " " << rates[0] / msun_per_yr << " " << rates[1] / msun_per_yr << "\n";
	}
}

// Strang-split source term for the external fixed potential
template <> void QuokkaSimulation<MHDTallBoxSf>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real /*time*/, amrex::Real dt_lev) // NOLINT
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const Real dt = dt_lev;
	const auto &ic_table = userData_.ic_table.const_tables();

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<MHDTallBoxSf>::density_index);
			const Real x3mom = state(i, j, k, HydroSystem<MHDTallBoxSf>::x3Momentum_index);

			// g_ext is g_z at z > 0 (negative); the vertical acceleration is g_z = g_ext * sign(z)
			double const z = prob_lo[2] + (k + 0.5) * dx[2];
			std::array<amrex::Real, 1> const point = {std::abs(z)};
			const double g_ext = ic_table.interpolate(point)[1];
			const Real x3mom_new = x3mom + dt * rho * g_ext * std::copysign(1.0, z);
			AMREX_ASSERT(!std::isnan(x3mom_new));

			// the internal and magnetic energies are unchanged, so the total energy changes by the change in kinetic energy only
			state(i, j, k, HydroSystem<MHDTallBoxSf>::x3Momentum_index) = x3mom_new;
			state(i, j, k, HydroSystem<MHDTallBoxSf>::energy_index) += 0.5 * (x3mom_new * x3mom_new - x3mom * x3mom) / rho;
		});
	}
}

// MHD diode (outflow, no-inflow) in z: cell-centred ext_dir with setDiodeBC, face-centred ghost field filled by applyMHDDiodeBC
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDTallBoxSf>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
							 amrex::GeometryData const &geom, const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
							 int /*orig_comp*/)
{
	setDiodeBCLo<2>(iv, consVar, geom);
	setDiodeBCHi<2>(iv, consVar, geom);
}

template <> auto AMRSimulation<MHDTallBoxSf>::isMHDDiodeBoundary(int dir, int /*side*/) -> bool { return dir == 2; }

auto problem_main() -> int
{
	// set random state
	const int rank = amrex::ParallelDescriptor::MyProc();
	const int seed = 42 + rank;
	amrex::InitRandom(seed, 1);

	// x, y: periodic; z: MHD diode (the face-centred foextrap is a placeholder, overwritten by applyMHDDiodeBC)
	using BT = quokka::BCType::mathematicalBndryTypes;
	auto BCs_cc = quokka::BC_cc<MHDTallBoxSf>(BT::periodic, BT::periodic, BT::ext_dir);
	auto BCs_fc = quokka::BC_fc<MHDTallBoxSf>(BT::periodic, BT::periodic, BT::foextrap);
	QuokkaSimulation<MHDTallBoxSf> sim(BCs_cc, BCs_fc);

	amrex::ParmParse const pp("problem");
	pp.query("stars_file", sim.userData_.stars_file);
	pp.query("IC_file", sim.userData_.IC_file);
	pp.query("rho01", sim.userData_.rho01);
	pp.query("sigma1", sim.userData_.sigma1);
	pp.query("B0_uG", sim.userData_.B0_uG);
	pp.query("outflow_height", sim.userData_.outflow_height);
	pp.query("outflow_file", sim.userData_.outflow_file);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.userData_.B0_uG > 0.0, "problem.B0_uG must be positive");
	pp.query("initial_scalar_density", sim.userData_.initial_scalar_density);
	AMREX_ALWAYS_ASSERT(!std::isnan(sim.userData_.initial_scalar_density));

	// particles.scalar_yield_per_SN must exceed (initial_scalar_density * (128 pc)^3), see testTallBoxSf.cpp
	amrex::ParmParse const pp_particles("particles");
	double scalar_yield_per_SN = NAN;
	pp_particles.query("scalar_yield_per_SN", scalar_yield_per_SN);
	AMREX_ALWAYS_ASSERT(!std::isnan(scalar_yield_per_SN));
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(scalar_yield_per_SN > sim.userData_.initial_scalar_density * std::pow(128.0 * C::parsec, 3),
					 "particles.scalar_yield_per_SN must be greater than (initial_scalar_density * (128 pc)^3)");

	// ic_table must be initialised even when restarting from a checkpoint
	sim.preCalculateInitialConditions();
	sim.setInitialConditions();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream file(sim.userData_.outflow_file);
		file << "# total outflow rates through |z| = " << sim.userData_.outflow_height / C::parsec << " pc\n";
		file << "# time [Myr]  Mdot_gas [Msun/yr]  Mdot_metal [Msun/yr]\n";
	}
	sim.evolve();

	// div B check over the valid cells, normalised by the midplane field: dx |div B| / B_0
	auto const dx = sim.geom[0].CellSizeArray();
	constexpr int bIdx = Physics_Indices<MHDTallBoxSf>::mhdFirstIndex;
	auto const &bx = sim.state_new_fc_[0][0].const_arrays();
	auto const &by = sim.state_new_fc_[0][1].const_arrays();
	auto const &bz = sim.state_new_fc_[0][2].const_arrays();
	amrex::Real divB = amrex::ParReduce(amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real>{}, sim.state_new_cc_[0], amrex::IntVect(0),
					    [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
						    amrex::Real const div = (bx[box](i + 1, j, k, bIdx) - bx[box](i, j, k, bIdx)) / dx[0] +
									    (by[box](i, j + 1, k, bIdx) - by[box](i, j, k, bIdx)) / dx[1] +
									    (bz[box](i, j, k + 1, bIdx) - bz[box](i, j, k, bIdx)) / dx[2];
						    return {std::abs(div)};
					    });
	amrex::ParallelDescriptor::ReduceRealMax(divB);
	const amrex::Real divB_rel = divB * dx[0] / (sim.userData_.B0_uG * microgauss_to_code);

	amrex::Print() << "\nMHDTallBoxSf: max dx |div B| / B_0 = " << divB_rel << "\n";
	constexpr double divB_tol = 1.0e-10;
	const bool pass = divB_rel < divB_tol;
	amrex::Print() << (pass ? "Test passed.\n" : "Test FAILED.\n");
	return pass ? 0 : 1;
}
