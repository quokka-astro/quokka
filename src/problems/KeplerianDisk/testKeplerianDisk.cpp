//==============================================================================
// Copyright 2026 Quokka developers.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// Hydro-only circular disk in a static, regularized point-mass potential.

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <string>
#include <vector>

#include "QuokkaSimulation.hpp"
#include "util/BC.hpp"

struct KeplerianDisk {};

template <> struct quokka::EOS_Traits<KeplerianDisk> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<KeplerianDisk> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
};

template <> struct SimulationData<KeplerianDisk> {
	std::ofstream history;
	amrex::Real densityError = 0.;
	amrex::Real radialVelocityRms = 0.;
	bool transportWritten = false;
};

namespace disk
{
constexpr double gm = 1.;
constexpr double coreRadius = 0.25;
constexpr double pressure = 1.e-3;
constexpr double backgroundDensity = 1.e-2;
using Hydro = HydroSystem<KeplerianDisk>;

// A uniform-density gravitating sphere inside coreRadius; exactly -GM/r outside.
// The core is part of the prescribed potential, not a gas/particle gravity solve.
AMREX_GPU_HOST_DEVICE auto omegaSquared(double r2) -> double
{
	const double radius = std::sqrt(std::max(r2, coreRadius * coreRadius));
	return gm / (radius * radius * radius);
}

AMREX_GPU_HOST_DEVICE auto potential(double r2) -> double
{
	if (r2 < coreRadius * coreRadius) {
		return gm * (r2 / (coreRadius * coreRadius) - 3.) / (2. * coreRadius);
	}
	return -gm / std::sqrt(r2);
}

AMREX_GPU_HOST_DEVICE auto density(double r2) -> double
{
	const double s = (r2 - 1.) / 0.3;
	return backgroundDensity + std::exp(-s * s);
}

AMREX_GPU_DEVICE void setState(amrex::Array4<amrex::Real> const &state, int i, int j, int k, double x, double y)
{
	const double r2 = x * x + y * y;
	const double rho = density(r2);
	const double omega = std::sqrt(omegaSquared(r2));
	const double eint = pressure / (quokka::EOS_Traits<KeplerianDisk>::gamma - 1.);
	state(i, j, k, Hydro::density_index) = rho;
	state(i, j, k, Hydro::x1Momentum_index) = -rho * omega * y;
	state(i, j, k, Hydro::x2Momentum_index) = rho * omega * x;
	state(i, j, k, Hydro::x3Momentum_index) = 0.;
	state(i, j, k, Hydro::energy_index) = eint + 0.5 * rho * omega * omega * r2;
	state(i, j, k, Hydro::internalEnergy_index) = eint;
}

struct TransportBin {
	double lower = 0.;
	double upper = 0.;
	double massFlux = 0.;
	double angularMomentumFlux = 0.;
	double coverage = 0.;
};

// Volume-weighted annular averages, converted to outward surface-integrated
// ADVECTIVE fluxes. These are not the solver's dissipative numerical face fluxes.
auto transportProfile(amrex::MultiFab const &mf, amrex::Geometry const &geometry, int nbins, double rmax) -> std::vector<TransportBin>
{
	const auto dx = geometry.CellSizeArray();
	const auto lo = geometry.ProbLoArray();
	const auto hi = geometry.ProbHiArray();
	const double inscribedRadius = std::min({-lo[0], -lo[1], hi[0], hi[1]});
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(rmax) && rmax > 0. && rmax <= inscribedRadius,
					 "disk.transport_rmax must lie inside the largest complete circle centered at the origin.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nbins > 0 && nbins <= rmax / std::max(dx[0], dx[1]),
					 "disk.transport_nbins must be positive and give bins at least one cell wide.");
	const double dr = rmax / nbins;
	const double volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	double height = 1.; // 2D rates are per unit vertical length.
#if AMREX_SPACEDIM == 3
	height = hi[2] - lo[2];
#endif
	amrex::MultiFab integrands(mf.boxArray(), mf.DistributionMap(), 3, 0);
	const auto out = integrands.arrays();
	const auto state = mf.const_arrays();
	std::vector<TransportBin> profile;
	profile.reserve(nbins);
	for (int bin = 0; bin < nbins; ++bin) {
		const double lower = bin * dr;
		const double upper = (bin + 1) * dr;
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
			const double x = lo[0] + (i + 0.5) * dx[0];
			const double y = lo[1] + (j + 0.5) * dx[1];
			const double r = std::sqrt(x * x + y * y);
			out[bx](i, j, k, 0) = 0.;
			out[bx](i, j, k, 1) = 0.;
			out[bx](i, j, k, 2) = 0.;
			if (r > 0. && r >= lower && r < upper) {
				const double rho = state[bx](i, j, k, Hydro::density_index);
				const double px = state[bx](i, j, k, Hydro::x1Momentum_index);
				const double py = state[bx](i, j, k, Hydro::x2Momentum_index);
				const double radialMassFlux = (x * px + y * py) / r;
				const double specificAngularMomentum = (x * py - y * px) / rho;
				out[bx](i, j, k, 0) = radialMassFlux;
				out[bx](i, j, k, 1) = radialMassFlux * specificAngularMomentum;
				out[bx](i, j, k, 2) = 1.;
			}
		});
		// MultiFab::sum includes all MPI ranks and only valid cells.
		const double massSum = integrands.sum(0);
		const double angularMomentumSum = integrands.sum(1);
		const double count = integrands.sum(2);
		AMREX_ALWAYS_ASSERT(count > 0. && std::isfinite(massSum) && std::isfinite(angularMomentumSum));
		const double area = std::numbers::pi * (lower + upper) * height;
		profile.push_back({lower, upper, area * massSum / count, area * angularMomentumSum / count, count * volume / (area * dr)});
	}
	return profile;
}

// Manufactured inflow/outflow with rho=2, |vr|=0.25, and specific Lz=0.5.
// Constant flux densities give exact bin averages despite Cartesian sampling.
void checkTransportProfile(amrex::MultiFab const &mf, amrex::Geometry const &geometry, int nbins, double rmax)
{
	amrex::MultiFab manufactured(mf.boxArray(), mf.DistributionMap(), mf.nComp(), 0);
	manufactured.setVal(0.);
	const auto state = manufactured.arrays();
	const auto lo = geometry.ProbLoArray();
	const auto dx = geometry.CellSizeArray();
	for (const double sign : {-1., 1.}) {
		amrex::ParallelFor(manufactured, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
			const double x = lo[0] + (i + 0.5) * dx[0];
			const double y = lo[1] + (j + 0.5) * dx[1];
			const double r2 = x * x + y * y;
			state[bx](i, j, k, Hydro::density_index) = 2.;
			if (r2 > 0.) {
				state[bx](i, j, k, Hydro::x1Momentum_index) = 2. * (sign * 0.25 * x / std::sqrt(r2) - 0.5 * y / r2);
				state[bx](i, j, k, Hydro::x2Momentum_index) = 2. * (sign * 0.25 * y / std::sqrt(r2) + 0.5 * x / r2);
			}
		});
		double height = 1.;
#if AMREX_SPACEDIM == 3
		height = geometry.ProbHi(2) - geometry.ProbLo(2);
#endif
		for (auto const &bin : transportProfile(manufactured, geometry, nbins, rmax)) {
			const double expectedMassFlux = sign * 0.5 * std::numbers::pi * (bin.lower + bin.upper) * height;
			AMREX_ALWAYS_ASSERT(std::abs(bin.massFlux / expectedMassFlux - 1.) < 1.e-12);
			AMREX_ALWAYS_ASSERT(std::abs(bin.angularMomentumFlux / (0.5 * expectedMassFlux) - 1.) < 1.e-12);
		}
	}
}
} // namespace disk

template <> void QuokkaSimulation<KeplerianDisk>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const auto dx = grid_elem.dx_;
	const auto lo = grid_elem.prob_lo_;
	const auto state = grid_elem.array_;
	amrex::ParallelFor(grid_elem.indexRange_,
			   [=] AMREX_GPU_DEVICE(int i, int j, int k) { disk::setState(state, i, j, k, lo[0] + (i + 0.5) * dx[0], lo[1] + (j + 0.5) * dx[1]); });
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<KeplerianDisk>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &state, int /*dcomp*/, int /*numcomp*/,
							  amrex::GeometryData const &geometry, amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
							  int /*bcomp*/, int /*orig_comp*/)
{
	const auto [i, j, k] = iv.dim3();
	// GeometryData is required by this callback. Read scalar coordinates directly;
	// no host pointer is captured or passed into a GPU kernel.
	const double x = geometry.ProbLo(0) + (i + 0.5) * geometry.CellSize(0);
	const double y = geometry.ProbLo(1) + (j + 0.5) * geometry.CellSize(1);
	disk::setState(state, i, j, k, x, y);
}

template <> void QuokkaSimulation<KeplerianDisk>::addStrangSplitSources(amrex::MultiFab &state_mf, int lev, amrex::Real /*time*/, amrex::Real dt)
{
	const auto state = state_mf.arrays();
	const auto dx = geom[lev].CellSizeArray();
	const auto lo = geom[lev].ProbLoArray();
	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const double x = lo[0] + (i + 0.5) * dx[0];
		const double y = lo[1] + (j + 0.5) * dx[1];
		const double omega2 = disk::omegaSquared(x * x + y * y);
		const double rho = state[bx](i, j, k, disk::Hydro::density_index);
		const double px = state[bx](i, j, k, disk::Hydro::x1Momentum_index);
		const double py = state[bx](i, j, k, disk::Hydro::x2Momentum_index);
		const double dpx = -dt * rho * omega2 * x;
		const double dpy = -dt * rho * omega2 * y;
		state[bx](i, j, k, disk::Hydro::x1Momentum_index) += dpx;
		state[bx](i, j, k, disk::Hydro::x2Momentum_index) += dpy;
		// Exact kinetic-energy change for the fixed-density kick. No thermal heating.
		state[bx](i, j, k, disk::Hydro::energy_index) += ((px + 0.5 * dpx) * dpx + (py + 0.5 * dpy) * dpy) / rho;
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<KeplerianDisk>::computeAfterTimestep()
{
	// This diagnostic intentionally supports a single level: no double-counted AMR cells.
	const auto &mf = state_new_cc_[0];
	amrex::MultiFab diagnostics(mf.boxArray(), mf.DistributionMap(), 10, 0);
	const auto result = diagnostics.arrays();
	const auto state = mf.const_arrays();
	const auto dx = geom[0].CellSizeArray();
	const auto lo = geom[0].ProbLoArray();
	const double volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const double x = lo[0] + (i + 0.5) * dx[0];
		const double y = lo[1] + (j + 0.5) * dx[1];
		const double r2 = x * x + y * y;
		const double rho = state[bx](i, j, k, disk::Hydro::density_index);
		const double px = state[bx](i, j, k, disk::Hydro::x1Momentum_index);
		const double py = state[bx](i, j, k, disk::Hydro::x2Momentum_index);
		const double pz = state[bx](i, j, k, disk::Hydro::x3Momentum_index);
		const double ke = (px * px + py * py + pz * pz) / (2. * rho);
		const double etot = state[bx](i, j, k, disk::Hydro::energy_index);
		const double pr = x * px + y * py;
		result[bx](i, j, k, 0) = rho;
		result[bx](i, j, k, 1) = x * py - y * px;
		result[bx](i, j, k, 2) = ke;
		result[bx](i, j, k, 3) = etot + rho * disk::potential(r2);
		result[bx](i, j, k, 4) = std::abs(rho - disk::density(r2));
		result[bx](i, j, k, 5) = (r2 > 0.) ? pr * pr / (rho * r2) : 0.;
		result[bx](i, j, k, 6) = disk::density(r2);
		result[bx](i, j, k, 7) = (r2 < 1.) ? rho : 0.;
		result[bx](i, j, k, 8) = (r2 < 1.) ? x * py - y * px : 0.;
		result[bx](i, j, k, 9) = (etot - ke) * (quokka::EOS_Traits<KeplerianDisk>::gamma - 1.);
	});
	std::array<double, 9> sums{};
	for (int n = 0; n < 9; ++n) {
		sums[n] = diagnostics.sum(n) * volume;
		AMREX_ALWAYS_ASSERT(std::isfinite(sums[n]));
	}
	AMREX_ALWAYS_ASSERT(diagnostics.min(0) > 0. && diagnostics.min(9) > 0.);
	userData_.densityError = sums[4] / sums[6];
	userData_.radialVelocityRms = std::sqrt(sums[5] / sums[0]);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (!userData_.history.is_open()) {
			amrex::ParmParse pp("disk");
			std::string filename = "keplerian_disk_history.txt";
			pp.query("history_file", filename);
			userData_.history.open(filename);
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.history.good(), "Cannot open disk history file.");
			userData_.history
			    << "# time mass Lz kinetic_energy gas_plus_potential_energy density_relative_L1 radial_velocity_rms mass_r_lt_1 Lz_r_lt_1\n";
			userData_.history << std::setprecision(17);
		}
		userData_.history << tNew_[0];
		for (int n = 0; n < 4; ++n) {
			userData_.history << ' ' << sums[n];
		}
		userData_.history << ' ' << userData_.densityError << ' ' << userData_.radialVelocityRms << ' ' << sums[7] << ' ' << sums[8] << '\n';
		userData_.history.flush();
	}
}

template <> void QuokkaSimulation<KeplerianDisk>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*initSumCons*/)
{
	amrex::ParmParse pp("disk");
	const auto lo = geom[0].ProbLoArray();
	const auto hi = geom[0].ProbHiArray();
	const auto dx = geom[0].CellSizeArray();
	double rmax = std::min({-lo[0], -lo[1], hi[0], hi[1]});
	pp.query("transport_rmax", rmax);
	AMREX_ALWAYS_ASSERT(std::isfinite(rmax) && rmax > 0. && rmax <= std::min({-lo[0], -lo[1], hi[0], hi[1]}));
	int nbins = std::max(1, static_cast<int>(rmax / (2. * std::max(dx[0], dx[1]))));
	pp.query("transport_nbins", nbins);
	const auto profile = disk::transportProfile(state_new_cc_[0], geom[0], nbins, rmax);
	bool check = false;
	pp.query("check_transport", check);
	if (check) {
		disk::checkTransportProfile(state_new_cc_[0], geom[0], nbins, rmax);
	}
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::string filename = "keplerian_disk_transport.txt";
		pp.query("transport_file", filename);
		std::ofstream file(filename);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file.good(), "Cannot open disk transport profile.");
		file << std::setprecision(17);
		file << "# time = " << tNew_[0] << "\n# Instantaneous advective fluxes; positive = outward. Numerical viscosity fluxes are excluded.\n";
		file << "# Volume-weighted annular mean times circumference (times slab height in 3D); 2D rates are per unit vertical length.\n";
		file << "# radius r_lower r_upper mass_flux_outward Lz_flux_outward sampled_volume_over_annulus_volume\n";
		for (auto const &bin : profile) {
			file << 0.5 * (bin.lower + bin.upper) << ' ' << bin.lower << ' ' << bin.upper << ' ' << bin.massFlux << ' ' << bin.angularMomentumFlux
			     << ' ' << bin.coverage << '\n';
		}
		file.close();
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file.good(), "Cannot write disk transport profile.");
		amrex::Print() << "Final radial transport profile: " << filename << '\n';
	}
	userData_.transportWritten = true;
}

auto problem_main() -> int
{
	static_assert(AMREX_SPACEDIM >= 2);
	auto bcs = quokka::BC<KeplerianDisk>(quokka::BCType::ext_dir, quokka::BCType::ext_dir, quokka::BCType::int_dir);
	QuokkaSimulation<KeplerianDisk> sim(bcs);
	amrex::ParmParse amr_pp("amr");
	int max_level = 0;
	amr_pp.query("max_level", max_level);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_level == 0, "KeplerianDisk diagnostics require amr.max_level = 0.");
	sim.setInitialConditions();
	sim.computeAfterTimestep(); // Record the same quadrature used by subsequent diagnostics at t=0.
	sim.evolve();
	// evolve() skips computeAfterEvolve when no cell updates are requested.
	if (!sim.userData_.transportWritten) {
		amrex::Vector<amrex::Real> unused;
		sim.computeAfterEvolve(unused);
	}
	amrex::ParmParse pp("disk");
	double tolerance = -1.;
	pp.query("test_tolerance", tolerance);
	amrex::Print() << "Disk density relative L1 = " << sim.userData_.densityError << ", radial velocity RMS = " << sim.userData_.radialVelocityRms << '\n';
	if (tolerance >= 0. && (sim.userData_.densityError > tolerance || sim.userData_.radialVelocityRms > tolerance)) {
		return 1;
	}
	return 0;
}
