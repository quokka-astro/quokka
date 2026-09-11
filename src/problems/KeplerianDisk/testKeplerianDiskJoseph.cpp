// Copyright 2026 Quokka developers. Released under the MIT license.
// Joseph et al. (2023), inviscid Cartesian ring, with Appendix E cooling.
#include "QuokkaSimulation.hpp"
#include "ring_alpha.hpp"
#include "util/BC.hpp"
#include <AMReX_BoxIterator.H>
#include <fstream>
#include <iomanip>
#include <numbers>

struct JosephRing {};
namespace joseph
{
constexpr double h = 0.005;
constexpr double tau0 = 0.018;
constexpr double background = 1.e-7;
constexpr double dampingRadius = 0.2;
constexpr double gamma = 1.4;		      // Appendix E does not specify gamma.
AMREX_GPU_MANAGED double kernelAtOne = 0.;    // NOLINT
AMREX_GPU_MANAGED double dampingPeriods = 1.; // NOLINT
// Stable scaled Bessel evaluation; matches ring_alpha::kernel for tau=tau0.
AMREX_GPU_HOST_DEVICE auto density(double r) -> double
{
	if (r < 0.2) {
		return background;
	} // analytic contribution is < 1e-14 here
	const double z = 2 * r / tau0;
	double bessel = 0.;
	if (z <= 50.) {
		double term = std::pow(.5 * z, .25) / std::tgamma(1.25);
		double sum = term;
		for (int k = 1; k < 200; ++k) {
			term *= .25 * z * z / (k * (k + .25));
			sum += term;
			if (term < 1.e-16 * sum) {
				break;
			}
		}
		bessel = std::exp(-z) * sum;
	} else {
		double term = 1.;
		double sum = 1.;
		for (int k = 1; k <= 12; ++k) {
			term *= ((2. * k - 1.) * (2. * k - 1.) - .25) / (8. * z * k);
			sum += term;
		}
		bessel = sum / std::sqrt(2. * std::numbers::pi * z);
	}
	return background + std::exp(-(r - 1.) * (r - 1.) / tau0) * bessel / (tau0 * std::pow(r, .25) * kernelAtOne);
}
} // namespace joseph
template <> struct quokka::EOS_Traits<JosephRing> {
	static constexpr double gamma = joseph::gamma;
	static constexpr double mean_molecular_weight = 1.;
};
template <> struct Physics_Traits<JosephRing> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
};
template <> struct SimulationData<JosephRing> {
	double nextProfile = 0.;
	double profileInterval = 2. * std::numbers::pi;
	double lastProfile = -1.;
	std::string prefix = "joseph";
};
namespace joseph
{
using Hydro = HydroSystem<JosephRing>;
AMREX_GPU_DEVICE void initialize(amrex::Array4<amrex::Real> const &u, int i, int j, int k, double x, double y)
{
	const double r = std::hypot(x, y);
	const double rho = density(r);
	const double omega = std::sqrt((1. - h * h) / (r * r * r));
	const double px = -rho * omega * y;
	const double py = rho * omega * x;
	const double eint = rho * h * h / (r * (gamma - 1.));
	u(i, j, k, Hydro::density_index) = rho;
	u(i, j, k, Hydro::x1Momentum_index) = px;
	u(i, j, k, Hydro::x2Momentum_index) = py;
	u(i, j, k, Hydro::x3Momentum_index) = 0.;
	u(i, j, k, Hydro::internalEnergy_index) = eint;
	u(i, j, k, Hydro::energy_index) = eint + (px * px + py * py) / (2 * rho);
}
// Exact gravity kick and exponential cooling/damping substep.
AMREX_GPU_DEVICE void sources(amrex::Array4<amrex::Real> const &u, int i, int j, int k, double x, double y, double dt)
{
	const double r = std::hypot(x, y);
	double rho = u(i, j, k, Hydro::density_index);
	double px = u(i, j, k, Hydro::x1Momentum_index);
	double py = u(i, j, k, Hydro::x2Momentum_index);
	double pz = u(i, j, k, Hydro::x3Momentum_index);
	const double thermal = u(i, j, k, Hydro::internalEnergy_index);
	// Dual energy is enabled: use its synchronized positive internal energy.
	const double omega = 1. / std::sqrt(r * r * r);
	px -= dt * rho * x * omega * omega;
	py -= dt * rho * y * omega * omega;
	double eint = rho * h * h / (r * (gamma - 1.)) + (thermal - rho * h * h / (r * (gamma - 1.))) * std::exp(-dt * omega / .01);
	if (r < dampingRadius) {
		const double ramp = (1. - r / dampingRadius) * (1. - r / dampingRadius);
		const double decay = std::exp(-dt * ramp / (dampingPeriods * 2. * std::numbers::pi * std::pow(dampingRadius, 1.5)));
		const double vr = (px * x + py * y) / (rho * r) * decay;
		const double vp = (-px * y + py * x) / (rho * r);
		const double rhoNew = background + (rho - background) * decay;
		eint *= rhoNew / rho;
		pz *= rhoNew / rho;
		rho = rhoNew;
		px = rho * (vr * x - vp * y) / r;
		py = rho * (vr * y + vp * x) / r;
	}
	u(i, j, k, Hydro::density_index) = rho;
	u(i, j, k, Hydro::x1Momentum_index) = px;
	u(i, j, k, Hydro::x2Momentum_index) = py;
	u(i, j, k, Hydro::x3Momentum_index) = pz;
	u(i, j, k, Hydro::internalEnergy_index) = eint;
	u(i, j, k, Hydro::energy_index) = eint + (px * px + py * py + pz * pz) / (2 * rho);
}
} // namespace joseph
template <> void QuokkaSimulation<JosephRing>::setInitialConditionsOnGrid(quokka::grid const &g)
{
	const auto dx = g.dx_;
	const auto lo = g.prob_lo_;
	const auto u = g.array_;
	amrex::ParallelFor(g.indexRange_,
			   [=] AMREX_GPU_DEVICE(int i, int j, int k) { joseph::initialize(u, i, j, k, lo[0] + (i + .5) * dx[0], lo[1] + (j + .5) * dx[1]); });
}
template <> void QuokkaSimulation<JosephRing>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real /*time*/, amrex::Real dt)
{
	const auto u = mf.arrays();
	const auto dx = geom[lev].CellSizeArray();
	const auto lo = geom[lev].ProbLoArray();
	amrex::ParallelFor(
	    mf, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) { joseph::sources(u[b], i, j, k, lo[0] + (i + .5) * dx[0], lo[1] + (j + .5) * dx[1], dt); });
}
template <> void QuokkaSimulation<JosephRing>::computeAfterTimestep()
{
	const double t = tNew_[0];
	if (t < userData_.nextProfile && t < stopTime_) {
		return;
	}
	if (t == userData_.lastProfile) {
		return;
	}
	auto const &mf = state_new_cc_[0];
	const auto dx = geom[0].CellSizeArray();
	const auto lo = geom[0].ProbLoArray();
	const int nx = geom[0].Domain().length(0);
	const int ny = geom[0].Domain().length(1);
	const int nbins = nx / 2;
	// slice sums/counts then annular sums/counts. Copy density only, at output cadence.
	std::vector<double> sums(2 * nx + 2 * nbins, 0.);
	amrex::MultiFab host(mf.boxArray(), mf.DistributionMap(), 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
	amrex::MultiFab::Copy(host, mf, joseph::Hydro::density_index, 0, 1, 0);
	amrex::Gpu::streamSynchronize();
	for (amrex::MFIter it(host); it.isValid(); ++it) {
		const auto a = host.const_array(it);
		for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell) {
			auto const iv = cell();
			const int i = iv[0];
			const int j = iv[1];
			const double rho = a(iv, 0);
			if (j == ny / 2 - 1 || j == ny / 2) {
				sums[i] += rho;
				sums[nx + i] += 1.;
			}
			const double r = std::hypot(lo[0] + (i + .5) * dx[0], lo[1] + (j + .5) * dx[1]);
			const int bin = static_cast<int>(r / dx[0]);
			if (bin < nbins) {
				sums[2 * nx + bin] += rho;
				sums[2 * nx + nbins + bin] += 1.;
			}
		}
	}
	amrex::ParallelDescriptor::ReduceRealSum(sums.data(), static_cast<int>(sums.size()));
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::ofstream out(userData_.prefix + "_profile_" + amrex::Concatenate("", istep[0], 8) + ".txt");
		out << std::setprecision(17) << "# time = " << t << ", orbits = " << t / (2 * std::numbers::pi) << "\n";
		out << "# R slice_y0 Sigma_annular initial_Sigma\n";
		for (int b = 0; b < nbins; ++b) {
			const int i = nx / 2 + b;
			const double r = (b + .5) * dx[0];
			AMREX_ALWAYS_ASSERT(sums[nx + i] == 2. && sums[2 * nx + nbins + b] > 0.);
			out << r << ' ' << sums[i] / sums[nx + i] << ' ' << sums[2 * nx + b] / sums[2 * nx + nbins + b] << ' ' << joseph::density(r) << '\n';
		}
		out.close();
		AMREX_ALWAYS_ASSERT(out.good());
	}
	userData_.lastProfile = t;
	userData_.nextProfile = (std::floor(t / userData_.profileInterval) + 1.) * userData_.profileInterval;
}
template <> void QuokkaSimulation<JosephRing>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*unused*/)
{
	userData_.nextProfile = 0.;
	computeAfterTimestep();
}
void checkJosephSources()
{
	const amrex::Box box(amrex::IntVect(0), amrex::IntVect(0));
	const amrex::BoxArray ba(box);
	const amrex::DistributionMapping dm(ba);
	amrex::MultiFab mf(ba, dm, 6, 0);
	const auto arrays = mf.arrays();
	for (const double radius : {1., .1}) {
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) {
			auto const u = arrays[b];
			u(i, j, k, 0) = 2.;
			u(i, j, k, 1) = 0.;
			u(i, j, k, 2) = 2.;
			u(i, j, k, 3) = 0.;
			const double target = 2. * joseph::h * joseph::h / (radius * (joseph::gamma - 1.));
			u(i, j, k, 4) = 1. + 2. * target;
			u(i, j, k, 5) = 2. * target;
			joseph::sources(u, i, j, k, radius, 0., .001);
		});
		const double omega = std::pow(radius, -1.5);
		const double damping = radius < .2 ? std::exp(-.001 * .25 / (joseph::dampingPeriods * 2. * std::numbers::pi * std::pow(.2, 1.5))) : 1.;
		const double rho = joseph::background + (2. - joseph::background) * damping;
		const double eint = rho * joseph::h * joseph::h / (radius * (joseph::gamma - 1.)) * (1. + std::exp(-.001 * omega / .01));
		AMREX_ALWAYS_ASSERT(std::abs(mf.sum(0) / rho - 1.) < 1.e-13);
		AMREX_ALWAYS_ASSERT(std::abs(mf.sum(1) / (-rho * .001 / (radius * radius) * damping) - 1.) < 1.e-13);
		AMREX_ALWAYS_ASSERT(std::abs(mf.sum(2) / rho - 1.) < 1.e-13);
		AMREX_ALWAYS_ASSERT(std::abs(mf.sum(5) / eint - 1.) < 1.e-13);
		const double expected = eint + .5 * rho * (1. + std::pow(.001 / (radius * radius) * damping, 2));
		AMREX_ALWAYS_ASSERT(std::abs(mf.sum(4) / expected - 1.) < 1.e-13);
	}
	amrex::Print() << "Joseph gravity, thermal relaxation and damping checks passed.\n";
}
auto problem_main() -> int
{
	joseph::kernelAtOne = ring_alpha::kernel(1., joseph::tau0);
	amrex::ParmParse pp("disk");
	pp.query("damping_periods", joseph::dampingPeriods);
	AMREX_ALWAYS_ASSERT(joseph::dampingPeriods > 0.);
	auto bcs = quokka::BC<JosephRing>(quokka::BCType::foextrap);
	QuokkaSimulation<JosephRing> sim(bcs);
	AMREX_ALWAYS_ASSERT(sim.geom[0].Domain().length(0) == sim.geom[0].Domain().length(1));
	AMREX_ALWAYS_ASSERT(sim.geom[0].Domain().length(0) % 2 == 0);
	amrex::ParmParse amr("amr");
	int level = 0;
	amr.query("max_level", level);
	AMREX_ALWAYS_ASSERT(level == 0);
	for (int d = 0; d < 2; ++d) {
		AMREX_ALWAYS_ASSERT(sim.geom[0].ProbLo(d) == -2. && sim.geom[0].ProbHi(d) == 2.);
	}
	double orbits = 748.;
	pp.query("orbits", orbits);
	AMREX_ALWAYS_ASSERT(orbits >= 0.);
	sim.stopTime_ = orbits * 2. * std::numbers::pi;
	double cadence = 1.;
	pp.query("profile_interval_orbits", cadence);
	AMREX_ALWAYS_ASSERT(cadence > 0.);
	sim.userData_.profileInterval = cadence * 2. * std::numbers::pi;
	pp.query("profile_prefix", sim.userData_.prefix);
	// Independent analytic kernel checks across the fitted radial domain.
	for (double r : {.2, .4, .7, 1., 1.3, 1.8}) {
		const double expected = ring_alpha::kernel(r, joseph::tau0) / joseph::kernelAtOne + joseph::background;
		AMREX_ALWAYS_ASSERT(std::abs(joseph::density(r) - expected) < 1.e-12);
	}
	using Hyper = HyperbolicSystem<JosephRing>;
	AMREX_ALWAYS_ASSERT(std::abs(Hyper::SlopeFunc<SlopeLimiter::vanleer>(1., 3.) - 1.5) < 1.e-14);
	AMREX_ALWAYS_ASSERT(Hyper::SlopeFunc<SlopeLimiter::vanleer>(-1., 3.) == 0.);
	AMREX_ALWAYS_ASSERT(Hyper::SlopeFunc<SlopeLimiter::vanleer>(0., 0.) == 0.);
	checkJosephSources();
	sim.setInitialConditions();
	sim.computeAfterTimestep();
	sim.evolve();
	sim.userData_.nextProfile = 0.;
	sim.computeAfterTimestep();
	return 0;
}
