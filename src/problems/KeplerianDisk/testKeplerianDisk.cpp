//==============================================================================
// Copyright 2026 Quokka developers.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// Hydro-only version of the ring experiment in Krumholz et al. (2004), section 3.4.2.

#include <AMReX_BoxIterator.H>
#include <gcem.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <string>
#include <vector>

#include "QuokkaSimulation.hpp"
#include "ring_alpha.hpp"
#include "util/BC.hpp"

namespace disk
{
// Units: length = one paper cell, mass = M_sun, time = sqrt(length^3 / GM_sun).
// The paper's 2.1e13 cm = 1.85 AU is inconsistent; use its AU value, which
// agrees with the independently quoted Bondi radius of about 1.36e4 cells.
constexpr double cellLengthCgs = 1.85 * 1.495978707e13;
constexpr double soundSpeedSquared = (C::k_B * 10. / (2.33 * C::m_p)) * cellLengthCgs / (C::Gconst * C::M_solar);
AMREX_GPU_MANAGED double ringRadius = 20.;   // NOLINT
AMREX_GPU_MANAGED bool powerLawDisk = false; // NOLINT
constexpr double outerRadius = 2.e15 / cellLengthCgs;
// Power-law surface-density unit is Sigma_0 = 0.1 g/cm^2; no gas self-gravity.
constexpr double surfaceDensityUnitCgs = 0.1;
} // namespace disk

struct KeplerianDisk {};

template <> struct quokka::EOS_Traits<KeplerianDisk> {
	static constexpr double gamma = 1.;
	static constexpr double cs_isothermal = gcem::sqrt(disk::soundSpeedSquared);
	static constexpr double mean_molecular_weight = 2.33 * C::m_p;
};

template <> struct HydroSystem_Traits<KeplerianDisk> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<KeplerianDisk> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
};

template <> struct SimulationData<KeplerianDisk> {
	std::ofstream history;
	std::ofstream alphaHistory;
	bool measureAlpha = true;
	double alphaIntervalOrbits = 0.01;
	double lastAlphaTime = -1.;
	std::string alphaFilename = "keplerian_disk_alpha.txt";
	ring_alpha::Average alphaAverage;
	ring_alpha::Fit alphaFit;
	amrex::Real densityError = 0.;
	amrex::Real radialVelocityRms = 0.;
	bool transportWritten = false;
};

namespace disk
{
constexpr double gm = 1.;
constexpr double backgroundDensity = 1.e-6;
using Hydro = HydroSystem<KeplerianDisk>;

// Unsoftened fixed point mass. The origin is excluded from valid cell centers;
// its value here is a harmless convention for face/quadrature diagnostics.
AMREX_GPU_HOST_DEVICE auto omegaSquared(double r2) -> double { return (r2 > 0.) ? gm / (r2 * std::sqrt(r2)) : 0.; }

AMREX_GPU_HOST_DEVICE auto potential(double r2) -> double { return (r2 > 0.) ? -gm / std::sqrt(r2) : 0.; }

AMREX_GPU_HOST_DEVICE auto density(double r2) -> double
{
	if (powerLawDisk) {
		const double r = std::sqrt(r2);
		return (r > 0. && r <= outerRadius) ? outerRadius / r : backgroundDensity;
	}
	// Two-cell full radial width, sampled at Cartesian cell centers.
	return (std::abs(std::sqrt(r2) - ringRadius) < 1.) ? 1. : backgroundDensity;
}

AMREX_GPU_DEVICE void setState(amrex::Array4<amrex::Real> const &state, int i, int j, int k, double x, double y)
{
	const double r2 = x * x + y * y;
	const double rho = density(r2);
	const double omega = std::sqrt(omegaSquared(r2));
	const double eint = 0.; // Isothermal EOS: total energy stores kinetic energy only.
	state(i, j, k, Hydro::density_index) = rho;
	state(i, j, k, Hydro::x1Momentum_index) = -rho * omega * y;
	state(i, j, k, Hydro::x2Momentum_index) = rho * omega * x;
	state(i, j, k, Hydro::x3Momentum_index) = 0.;
	state(i, j, k, Hydro::energy_index) = eint + 0.5 * rho * omega * omega * r2;
	state(i, j, k, Hydro::internalEnergy_index) = eint;
}

// Compare initialized grid integrals with independent continuum annulus moments.
// The tolerance accommodates cell-center sampling of the two sharp edges.
void checkInitialRing(amrex::MultiFab const &mf, amrex::Geometry const &geometry)
{
	amrex::MultiFab moments(mf.boxArray(), mf.DistributionMap(), 3, 0);
	const auto out = moments.arrays();
	const auto state = mf.const_arrays();
	const auto lo = geometry.ProbLoArray();
	const auto dx = geometry.CellSizeArray();
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const double x = lo[0] + (i + 0.5) * dx[0];
		const double y = lo[1] + (j + 0.5) * dx[1];
		const double r = std::sqrt(x * x + y * y);
		const double rho = state[bx](i, j, k, Hydro::density_index);
		const double excess = rho - backgroundDensity;
		out[bx](i, j, k, 0) = excess;
		out[bx](i, j, k, 1) = excess * (r - ringRadius) * (r - ringRadius);
		out[bx](i, j, k, 2) = (excess / rho) * (x * state[bx](i, j, k, Hydro::x2Momentum_index) - y * state[bx](i, j, k, Hydro::x1Momentum_index));
	});
	const double mass = moments.sum(0);
	// Independent lattice counts for half-integer cell centers in each annulus.
	const double expectedCells = (ringRadius == 20.) ? 272. : ((ringRadius == 40.) ? 492. : 724.);
	double height = 1.;
#if AMREX_SPACEDIM == 3
	height = geometry.ProbHi(2) - geometry.ProbLo(2);
#endif
	const double volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	AMREX_ALWAYS_ASSERT(std::abs(mass * volume / ((1. - backgroundDensity) * expectedCells * height) - 1.) < 1.e-12);
	const double expectedMass = (1. - backgroundDensity) * 4. * std::numbers::pi * ringRadius * height;
	const double expectedLz =
	    (1. - backgroundDensity) * 4. * std::numbers::pi / 5. * std::sqrt(gm) * (std::pow(ringRadius + 1., 2.5) - std::pow(ringRadius - 1., 2.5)) * height;
	AMREX_ALWAYS_ASSERT(std::abs(mass * volume / expectedMass - 1.) < 0.1);
	AMREX_ALWAYS_ASSERT(std::abs(moments.sum(1) / mass - 1. / 3.) < 0.08);
	AMREX_ALWAYS_ASSERT(std::abs(moments.sum(2) * volume / expectedLz - 1.) < 0.1);
	amrex::Print() << "Initial ring mass, radial width, and angular momentum checks passed.\n";
}

// Independent continuum mass check for Sigma/Sigma_0 = r0/r inside r0.
void checkInitialPowerLaw(amrex::MultiFab const &mf, amrex::Geometry const &geometry)
{
	const auto dx = geometry.CellSizeArray();
	double height = 1.;
#if AMREX_SPACEDIM == 3
	height = geometry.ProbHi(2) - geometry.ProbLo(2);
#endif
	const double area = (geometry.ProbHi(0) - geometry.ProbLo(0)) * (geometry.ProbHi(1) - geometry.ProbLo(1));
	const double expected =
	    height * (2. * std::numbers::pi * outerRadius * outerRadius + backgroundDensity * (area - std::numbers::pi * outerRadius * outerRadius));
	const double measured = mf.sum(Hydro::density_index) * AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	AMREX_ALWAYS_ASSERT(std::abs(measured / expected - 1.) < 0.01);
	amrex::Print() << "Initial power-law disk continuum mass check passed.\n";
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
// Only the density component is copied to the host, at diagnostic cadence.
// All valid cells and MPI ranks contribute; the host fit runs on the IO rank.
void measureAlpha(amrex::MultiFab const &mf, amrex::Geometry const &geometry, double time, SimulationData<KeplerianDisk> &data, bool force = false)
{
	const double period = 2. * std::numbers::pi * std::sqrt(ringRadius * ringRadius * ringRadius / gm);
	if (!data.measureAlpha || time <= 0. || time == data.lastAlphaTime ||
	    (!force && data.lastAlphaTime >= 0. && time - data.lastAlphaTime < data.alphaIntervalOrbits * period)) {
		return;
	}
	data.lastAlphaTime = time;
	const auto lo = geometry.ProbLoArray();
	const auto hi = geometry.ProbHiArray();
	const auto dx = geometry.CellSizeArray();
	const double rmax = std::min({-lo[0], -lo[1], hi[0], hi[1]});
	const int nbins = static_cast<int>(std::floor(rmax / std::max(dx[0], dx[1])));
	const double dr = rmax / nbins;
	const double volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	double height = 1.;
#if AMREX_SPACEDIM == 3
	height = hi[2] - lo[2];
#endif
	amrex::MultiFab host(mf.boxArray(), mf.DistributionMap(), 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
	amrex::MultiFab::Copy(host, mf, Hydro::density_index, 0, 1, 0);
	amrex::Gpu::streamSynchronize();
	std::vector<double> sums(2 * nbins + 1, 0.);
	for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
		const auto a = host.const_array(mfi);
		for (amrex::BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
			const auto [i, j, k] = bit().dim3();
			const double x = lo[0] + (i + 0.5) * dx[0];
			const double y = lo[1] + (j + 0.5) * dx[1];
			const double r2 = x * x + y * y;
			// Fixed INITIAL excess mass, not the remaining mass at this snapshot.
			sums[2 * nbins] += (density(r2) - backgroundDensity) * volume;
			const int bin = static_cast<int>(std::sqrt(r2) / dr);
			if (bin < nbins) {
				sums[bin] += a(i, j, k) - backgroundDensity;
				sums[nbins + bin] += 1.;
			}
		}
	}
	amrex::ParallelDescriptor::ReduceRealSum(sums.data(), static_cast<int>(sums.size()));
	if (!amrex::ParallelDescriptor::IOProcessor()) {
		return;
	}
	std::vector<double> sigma(nbins);
	for (int bin = 0; bin < nbins; ++bin) {
		AMREX_ALWAYS_ASSERT(sums[nbins + bin] > 0.);
		sigma[bin] = height * sums[bin] / sums[nbins + bin];
	}
	const auto fit = ring_alpha::fit(sigma, dr, ringRadius, sums[2 * nbins]);
	data.alphaFit = fit;
	const double nu = (fit.status == 0) ? ring_alpha::viscosity(fit.tau, ringRadius, time) : NAN;
	const double alpha = ring_alpha::alpha(nu, 2. * std::numbers::pi / period, soundSpeedSquared);
	data.alphaAverage.add(time / period, alpha);
	if (!data.alphaHistory.is_open()) {
		data.alphaHistory.open(data.alphaFilename);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(data.alphaHistory.good(), "Cannot open ring alpha history.");
		data.alphaHistory << std::setprecision(17)
				  << "# Krumholz et al. (2004), eqs. (23) and (20). Fixed initial excess mass; unweighted surface-density least squares.\n"
				  << "# R0 = " << ringRadius << ", initial_mass = " << sums[2 * nbins] << ", cs_squared = " << soundSpeedSquared
				  << ", orbital_period = " << period << ", bin_width = " << dr << ", rmax = " << rmax << '\n'
				  << "# status: 0=interior_fit, 1=tau_bound, 2=nonfinite_data, 3=no_positive_signal. Invalid nu/alpha are nan.\n"
				  << "# Average covers valid segments in [0.09,0.9] orbits during this invocation only. Full coverage is 0.81.\n"
				  << "# time orbits tau nu alpha fit_relative_L2 status alpha_mean covered_orbits\n";
	}
	data.alphaHistory << time << ' ' << time / period << ' ' << fit.tau << ' ' << nu << ' ' << alpha << ' ' << fit.relativeL2 << ' ' << fit.status << ' '
			  << data.alphaAverage.mean() << ' ' << data.alphaAverage.covered << '\n';
	data.alphaHistory.flush();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(data.alphaHistory.good(), "Cannot write ring alpha history.");
}
// Exercise Cartesian binning, background subtraction, MPI reduction, and the fit
// together using a manufactured spreading profile with known tau.
void checkAlphaProfile(amrex::MultiFab const &mf, amrex::Geometry const &geometry)
{
	amrex::MultiFab manufactured(mf.boxArray(), mf.DistributionMap(), 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
	const auto lo = geometry.ProbLoArray();
	const auto dx = geometry.CellSizeArray();
	constexpr double tau = 0.05;
	const double massPerHeight = (1. - backgroundDensity) * ((ringRadius == 20.) ? 272. : ((ringRadius == 40.) ? 492. : 724.));
	for (amrex::MFIter mfi(manufactured); mfi.isValid(); ++mfi) {
		const auto a = manufactured.array(mfi);
		for (amrex::BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
			const auto [i, j, k] = bit().dim3();
			const double x = lo[0] + (i + 0.5) * dx[0];
			const double y = lo[1] + (j + 0.5) * dx[1];
			a(i, j, k) = backgroundDensity + massPerHeight / (std::numbers::pi * ringRadius * ringRadius) *
							     ring_alpha::kernel(std::sqrt(x * x + y * y) / ringRadius, tau);
		}
	}
	SimulationData<KeplerianDisk> diagnostic;
	diagnostic.alphaFilename = "keplerian_disk_alpha_manufactured.txt";
	const double period = 2. * std::numbers::pi * std::sqrt(ringRadius * ringRadius * ringRadius / gm);
	measureAlpha(manufactured, geometry, 0.3 * period, diagnostic);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		AMREX_ALWAYS_ASSERT(diagnostic.alphaFit.status == 0 && std::abs(diagnostic.alphaFit.tau / tau - 1.) < 0.02);
		AMREX_ALWAYS_ASSERT(diagnostic.alphaFit.relativeL2 < 0.02);
		amrex::Print() << "Manufactured Cartesian alpha profile recovered tau = " << diagnostic.alphaFit.tau << " (expected 0.05).\n";
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
		result[bx](i, j, k, 7) = (r2 < disk::ringRadius * disk::ringRadius) ? rho : 0.;
		result[bx](i, j, k, 8) = (r2 < disk::ringRadius * disk::ringRadius) ? x * py - y * px : 0.;
		result[bx](i, j, k, 9) = rho * disk::soundSpeedSquared;
	});
	std::array<double, 9> sums{};
	for (int n = 0; n < 9; ++n) {
		sums[n] = diagnostics.sum(n) * volume;
		AMREX_ALWAYS_ASSERT(std::isfinite(sums[n]));
	}
	AMREX_ALWAYS_ASSERT(diagnostics.min(0) > 0. && diagnostics.min(9) > 0.);
	userData_.densityError = sums[4] / sums[6];
	userData_.radialVelocityRms = std::sqrt(sums[5] / sums[0]);
	disk::measureAlpha(mf, geom[0], tNew_[0], userData_);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (!userData_.history.is_open()) {
			amrex::ParmParse pp("disk");
			std::string filename = "keplerian_disk_history.txt";
			pp.query("history_file", filename);
			userData_.history.open(filename);
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.history.good(), "Cannot open disk history file.");
			userData_.history
			    << "# time mass Lz kinetic_energy gas_plus_potential_energy density_relative_L1 radial_velocity_rms mass_r_lt_R0 Lz_r_lt_R0\n";
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
	disk::measureAlpha(state_new_cc_[0], geom[0], tNew_[0], userData_, true);
	if (userData_.measureAlpha && amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Ring fitted alpha mean = " << userData_.alphaAverage.mean() << "; covered " << userData_.alphaAverage.covered
			       << " of 0.81 orbits in [0.09,0.9] (partial coverage is not the full paper average).\n";
	}
	userData_.transportWritten = true;
}

namespace disk
{
// Optional one-step experiment. Export cell data rather than binning on-device so
// angular fits and annular budgets can be changed without rerunning the solver.
void writeResidual(std::string const &filename, amrex::MultiFab const &initial, amrex::MultiFab const &rate, amrex::Geometry const &geometry, double dt,
		   amrex::MultiFab const *torqueDefect = nullptr)
{
	amrex::MultiFab data(initial.boxArray(), initial.DistributionMap(), 5, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
	amrex::MultiFab::Copy(data, initial, Hydro::density_index, 0, 1, 0);
	amrex::MultiFab::Copy(data, rate, Hydro::density_index, 1, 3, 0);
	data.setVal(0., 4, 1, 0);
	if (torqueDefect != nullptr) {
		amrex::MultiFab::Copy(data, *torqueDefect, 0, 4, 1, 0);
	}
	amrex::Gpu::streamSynchronize();
	std::ofstream file(filename);
	AMREX_ALWAYS_ASSERT(file.good());
	file << std::setprecision(17) << "# dt = " << dt
	     << "\n# torque_defect is only measured in the spatial file; zero placeholder in step and analytic files.\n";
	file << "# x y rho0 density_rate mx_rate my_rate torque_defect\n";
	const auto dx = geometry.CellSizeArray();
	const auto lo = geometry.ProbLoArray();
	for (amrex::MFIter mfi(data); mfi.isValid(); ++mfi) {
		const auto a = data.const_array(mfi);
		for (amrex::BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
			const auto [i, j, k] = bit().dim3();
			file << lo[0] + (i + 0.5) * dx[0] << ' ' << lo[1] + (j + 0.5) * dx[1];
			for (int n = 0; n < 5; ++n) {
				file << ' ' << a(i, j, k, n);
			}
			file << '\n';
		}
	}
	file.close();
	AMREX_ALWAYS_ASSERT(file.good());
}

// Identical analytic states on both sides of a face remove reconstruction and
// jump dissipation. Check the actual HLLC solver against the physical Euler flux.
AMREX_GPU_DEVICE auto analyticFlux(double x, double y, int direction) -> amrex::GpuArray<double, 3>
{
	const double rho = density(x * x + y * y);
	const double pressure = rho * soundSpeedSquared;
	const double omega = std::sqrt(omegaSquared(x * x + y * y));
	const double vx = -omega * y;
	const double vy = omega * x;
	quokka::HydroState<0, 0> state{};
	state.rho = rho;
	state.u = (direction == 0) ? vx : vy;
	state.v = (direction == 0) ? vy : vx;
	state.P = pressure;
	state.cs = std::sqrt(quokka::EOS_Traits<KeplerianDisk>::gamma * pressure / rho);
	state.Eint = 0.;
	state.E = state.Eint + 0.5 * rho * (vx * vx + vy * vy);
	const auto flux = quokka::Riemann::HLLC<KeplerianDisk, 0, 0, 6>(state, state, quokka::EOS_Traits<KeplerianDisk>::gamma, 0., 0.);
	const amrex::GpuArray<double, 3> exact{rho * state.u, rho * state.u * state.u + pressure, rho * state.u * state.v};
	for (int n = 0; n < 3; ++n) {
		AMREX_ALWAYS_ASSERT(std::abs(flux[n] - exact[n]) < 1.e-12 * (1. + std::abs(exact[n])));
	}
	return {flux[0], flux[(direction == 0) ? 1 : 2], flux[(direction == 0) ? 2 : 1]};
}

void writeAnalyticResiduals(QuokkaSimulation<KeplerianDisk> &sim, std::string const &prefix)
{
	const auto &state = sim.state_new_cc_[0];
	const auto dx = sim.geom[0].CellSizeArray();
	const auto lo = sim.geom[0].ProbLoArray();
	for (const int nq : {1, 2, 4}) {
		// Gauss-Legendre nodes/weights for AVERAGES on [-1/2,1/2].
		amrex::GpuArray<double, 4> nodes{0., 0., 0., 0.};
		amrex::GpuArray<double, 4> weights{1., 0., 0., 0.};
		if (nq == 2) {
			nodes = {-0.28867513459481288225, 0.28867513459481288225, 0., 0.};
			weights = {0.5, 0.5, 0., 0.};
		} else if (nq == 4) {
			nodes = {-0.43056815579702628761, -0.16999052179242813240, 0.16999052179242813240, 0.43056815579702628761};
			weights = {0.17392742256872692869, 0.32607257743127307131, 0.32607257743127307131, 0.17392742256872692869};
		}
		std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			auto ba = state.boxArray();
			ba.surroundingNodes(dir);
			flux[dir].define(ba, state.DistributionMap(), state.nComp(), 0);
			flux[dir].setVal(0.);
			const auto f = flux[dir].arrays();
			amrex::ParallelFor(flux[dir], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
				const double x = lo[0] + (i + ((dir == 0) ? 0. : 0.5)) * dx[0];
				const double y = lo[1] + (j + ((dir == 1) ? 0. : 0.5)) * dx[1];
				for (int q = 0; q < nq; ++q) {
					const double xq = x + ((dir == 1) ? nodes[q] * dx[0] : 0.);
					const double yq = y + ((dir == 0) ? nodes[q] * dx[1] : 0.);
					const auto value = analyticFlux(xq, yq, dir);
					for (int n = 0; n < 3; ++n) {
						f[bx](i, j, k, n) += weights[q] * value[n];
					}
				}
			});
		}
		amrex::MultiFab rhs(state.boxArray(), state.DistributionMap(), state.nComp(), 0);
		Hydro::ComputeRhsFromFluxes(rhs, flux, dx, state.nComp());
		const auto out = rhs.arrays();
		amrex::ParallelFor(rhs, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
			const double x = lo[0] + (i + 0.5) * dx[0];
			const double y = lo[1] + (j + 0.5) * dx[1];
			for (int qx = 0; qx < nq; ++qx) {
				for (int qy = 0; qy < nq; ++qy) {
					const double xq = x + nodes[qx] * dx[0];
					const double yq = y + nodes[qy] * dx[1];
					const double r2 = xq * xq + yq * yq;
					const double factor = weights[qx] * weights[qy] * density(r2) * omegaSquared(r2);
					out[bx](i, j, k, Hydro::x1Momentum_index) -= factor * xq;
					out[bx](i, j, k, Hydro::x2Momentum_index) -= factor * yq;
				}
			}
		});
		// Keep rho0 normalization identical to the existing diagnostic. Only the
		// operator is changed: nq>1 approximates cell-average flux/source balance.
		writeResidual(prefix + "-analytic-q" + std::to_string(nq) + ".txt", state, rhs, sim.geom[0], 0.);
	}
}

void writeSpatialResidual(QuokkaSimulation<KeplerianDisk> &sim, std::string const &prefix)
{
	auto &state = sim.state_new_cc_[0];
	sim.fillBoundaryConditions(state, state, 0, 0., quokka::centering::cc, quokka::direction::na, sim.PreInterpState, sim.PostInterpState);
	auto [flux, faceVelocity, waveSpeed] = sim.computeHydroFluxes(state, sim.state_new_fc_[0], state.nComp(), 0, 0);
	amrex::ignore_unused(faceVelocity, waveSpeed);
	amrex::MultiFab rhs(state.boxArray(), state.DistributionMap(), state.nComp(), 0);
	Hydro::ComputeRhsFromFluxes(rhs, flux, sim.geom[0].CellSizeArray(), state.nComp());
	amrex::MultiFab defect(state.boxArray(), state.DistributionMap(), 1, 0);
	const auto out = rhs.arrays();
	const auto u = state.const_arrays();
	const auto d = defect.arrays();
	const auto fx = flux[0].const_arrays();
	const auto fy = flux[1].const_arrays();
	const auto dx = sim.geom[0].CellSizeArray();
	const auto lo = sim.geom[0].ProbLoArray();
	amrex::ParallelFor(rhs, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const double x = lo[0] + (i + 0.5) * dx[0];
		const double y = lo[1] + (j + 0.5) * dx[1];
		const double rhoOmega2 = u[bx](i, j, k, Hydro::density_index) * omegaSquared(x * x + y * y);
		out[bx](i, j, k, Hydro::x1Momentum_index) -= rhoOmega2 * x;
		out[bx](i, j, k, Hydro::x2Momentum_index) -= rhoOmega2 * y;
		// R_Lz = -div(x_face F_my - y_face F_mx) + defect.
		// This identity distinguishes boundary transport from a Cartesian torque defect.
		d[bx](i, j, k, 0) = 0.5 * (fx[bx](i, j, k, Hydro::x2Momentum_index) + fx[bx](i + 1, j, k, Hydro::x2Momentum_index) -
					   fy[bx](i, j, k, Hydro::x1Momentum_index) - fy[bx](i, j + 1, k, Hydro::x1Momentum_index));
	});
	writeResidual(prefix + "-spatial.txt", state, rhs, sim.geom[0], 0., &defect);
}
} // namespace disk

auto problem_main() -> int
{
	static_assert(AMREX_SPACEDIM >= 2);
	auto bcs = quokka::BC<KeplerianDisk>(quokka::BCType::ext_dir, quokka::BCType::ext_dir, quokka::BCType::int_dir);
	QuokkaSimulation<KeplerianDisk> sim(bcs);
	amrex::ParmParse amr_pp("amr");
	int max_level = 0;
	amr_pp.query("max_level", max_level);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_level == 0, "KeplerianDisk diagnostics require amr.max_level = 0.");
	amrex::ParmParse residualPP("disk");
	std::string residualPrefix;
	residualPP.query("residual_prefix", residualPrefix);
	if (!residualPrefix.empty()) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 2 && amrex::ParallelDescriptor::NProcs() == 1 && sim.maxTimesteps_ == 1,
						 "disk.residual_prefix requires 2D, one MPI rank, and max_timesteps=1.");
	}
	amrex::ParmParse diskPP("disk");
	std::string profile = "ring";
	diskPP.query("profile", profile);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(profile == "ring" || profile == "power_law", "disk.profile must be ring or power_law.");
	disk::powerLawDisk = (profile == "power_law");
	sim.userData_.measureAlpha = !disk::powerLawDisk;
	diskPP.query("measure_alpha", sim.userData_.measureAlpha);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!disk::powerLawDisk || !sim.userData_.measureAlpha, "Ring alpha fits are invalid for a power-law disk.");
	diskPP.query("alpha_interval_orbits", sim.userData_.alphaIntervalOrbits);
	diskPP.query("alpha_file", sim.userData_.alphaFilename);
	AMREX_ALWAYS_ASSERT(std::isfinite(sim.userData_.alphaIntervalOrbits) && sim.userData_.alphaIntervalOrbits > 0.);
	diskPP.query("ring_radius_cells", disk::ringRadius);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(disk::ringRadius == 20. || disk::ringRadius == 40. || disk::ringRadius == 60.,
					 "The paper's ring radii are 20, 40, and 60 cells.");
	const double extent = disk::powerLawDisk ? disk::outerRadius : disk::ringRadius + 1.;
	const auto &geometry = sim.geom[0];
	for (int dir = 0; dir < 2; ++dir) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(geometry.CellSize(dir) - 1.) < 1.e-12,
						 "Ring units require dx = dy = 1; change domain bounds together with amr.n_cell.");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geometry.ProbLo(dir) < -extent && geometry.ProbHi(dir) > extent,
						 "The entire initial disk or ring must fit inside the domain.");
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(geometry.ProbLo(dir) - std::round(geometry.ProbLo(dir))) < 1.e-12,
						 "Place the point mass on cell faces, not at a singular cell center.");
	}
	const double referenceRadius = disk::powerLawDisk ? 4. : disk::ringRadius;
	const double period = 2. * std::numbers::pi * std::sqrt(std::pow(referenceRadius, 3) / disk::gm);
	double orbits = disk::powerLawDisk ? 50. : 0.9;
	diskPP.query("orbits", orbits);
	AMREX_ALWAYS_ASSERT(std::isfinite(orbits) && orbits >= 0.);
	amrex::ParmParse rootPP;
	if (!rootPP.contains("stop_time")) {
		sim.stopTime_ = orbits * period;
	}
	if (disk::powerLawDisk) {
		amrex::Print() << "Power-law disk: Sigma = " << disk::surfaceDensityUnitCgs << " (r0/r) g/cm^2, r0/dx = " << disk::outerRadius
			       << ", orbital reference radius/dx = 4, period = " << period << '\n';
	} else {
		amrex::Print() << "Ring R0/dx = " << disk::ringRadius << ", full width/dx = 2, r_B/dx = " << disk::gm / disk::soundSpeedSquared
			       << ", orbital period = " << period << '\n';
	}
	sim.setInitialConditions();
	bool checkInitial = false;
	diskPP.query("check_initial", checkInitial);
	if (checkInitial) {
		if (disk::powerLawDisk) {
			disk::checkInitialPowerLaw(sim.state_new_cc_[0], geometry);
		} else {
			disk::checkInitialRing(sim.state_new_cc_[0], geometry);
		}
	}
	bool checkAlpha = false;
	diskPP.query("check_alpha", checkAlpha);
	if (checkAlpha) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!disk::powerLawDisk, "disk.check_alpha requires the ring profile.");
		disk::checkAlphaProfile(sim.state_new_cc_[0], geometry);
	}
	amrex::MultiFab initial;
	if (!residualPrefix.empty()) {
		const auto &state = sim.state_new_cc_[0];
		initial.define(state.boxArray(), state.DistributionMap(), state.nComp(), 0);
		amrex::MultiFab::Copy(initial, state, 0, 0, state.nComp(), 0);
		disk::writeSpatialResidual(sim, residualPrefix);
		disk::writeAnalyticResiduals(sim, residualPrefix);
	}
	sim.computeAfterTimestep(); // Record the same quadrature used by subsequent diagnostics at t=0.
	sim.evolve();
	if (!residualPrefix.empty()) {
		AMREX_ALWAYS_ASSERT(sim.istep[0] == 1 && sim.tNew_[0] > 0.);
		amrex::MultiFab rate(initial.boxArray(), initial.DistributionMap(), initial.nComp(), 0);
		amrex::MultiFab::Copy(rate, sim.state_new_cc_[0], 0, 0, initial.nComp(), 0);
		amrex::MultiFab::Subtract(rate, initial, 0, 0, initial.nComp(), 0);
		rate.mult(1. / sim.tNew_[0], 0, initial.nComp(), 0);
		disk::writeResidual(residualPrefix + "-step.txt", initial, rate, sim.geom[0], sim.tNew_[0]);
	}
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
