//==============================================================================
// Copyright 2025 Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testOrszagTang.cpp
/// \brief
///   This problem is based on the implementation here:
///   https://github.com/PrincetonUniversity/athena/blob/master/src/pgen/orszag_tang.cpp.
///	  (Phil Hopkins made several typos on this page, do not use:
///	  https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html)
///

#include <array>
#include <cmath>
#include <iomanip>
#include <string>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Gpu.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct OrszagTang {
};

template <> struct quokka::EOS_Traits<OrszagTang> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<OrszagTang> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

constexpr double B0 = 1.0 / gcem::sqrt(4.0 * PI);

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto A_z(double x, double y) -> double
{
	return B0 / (4.0 * M_PI) * (std::cos(4.0 * M_PI * x) - 2.0 * std::cos(2.0 * M_PI * y));
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_x(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return (A_z(xL, yL + dx[1]) - A_z(xL, yL)) / dx[1];
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_y(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return -(A_z(xL + dx[0], yL) - A_z(xL, yL)) / dx[0];
};

template <> void QuokkaSimulation<OrszagTang>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	constexpr double gamma_gas = quokka::EOS_Traits<OrszagTang>::gamma;
	constexpr double rho0 = 25. / (36. * M_PI);
	constexpr double P0 = 5. / (12. * M_PI);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);

		const double vx = std::sin(2 * M_PI * y);
		const double vy = -std::sin(2 * M_PI * x);

		const double Bx = 0.5 * (B_x(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + B_x(x + 0.5 * dx[0], y - 0.5 * dx[1], dx));
		const double By = 0.5 * (B_y(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + B_y(x - 0.5 * dx[0], y + 0.5 * dx[1], dx));

		const double Ekin = 0.5 * rho0 * (vx * vx + vy * vy);
		const double Eint = P0 / (gamma_gas - 1.0);
		const double Emag = 0.5 * (Bx * Bx + By * By);

		state_cc(i, j, k, HydroSystem<OrszagTang>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<OrszagTang>::x1Momentum_index) = rho0 * vx;
		state_cc(i, j, k, HydroSystem<OrszagTang>::x2Momentum_index) = rho0 * vy;
		state_cc(i, j, k, HydroSystem<OrszagTang>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<OrszagTang>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<OrszagTang>::energy_index) = Eint + Ekin + Emag;
	});
}

template <> void QuokkaSimulation<OrszagTang>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double xL = prob_lo[0] + (i * dx[0]);
		const double yL = prob_lo[1] + (j * dx[1]);

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<OrszagTang>::mhdFirstIndex) = B_x(xL, yL, dx);
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<OrszagTang>::mhdFirstIndex) = B_y(xL, yL, dx);
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<OrszagTang>::mhdFirstIndex) = 0;
		}
	});
}

namespace
{
enum class Rot180Parity { Even, Odd };

// map a single-axis index under 180-degree rotation about the domain centre; generalised for both
// cell-centred (range [0, n-1] -> n-1-i) and face/nodal (range [0, n] -> n-i) axes.
AMREX_FORCE_INLINE auto rot180Index(int i, int n, bool is_nodal) -> int { return is_nodal ? (n - i) : (n - 1 - i); }

// gather `comp` of `mf` onto a single box on the IO processor (mf may be domain-decomposed across
// any number of boxes/ranks) and check that every cell's value exactly matches (Even) or exactly
// negates (Odd) its 180-degree-rotation partner.
auto checkRot180Symmetry(const amrex::MultiFab &mf, int comp, const amrex::Box &domain, Rot180Parity parity, const std::string &label) -> int
{
	const amrex::BoxArray ba_single(amrex::convert(domain, mf.ixType()));
	const amrex::DistributionMapping dm_single(amrex::Vector<int>{amrex::ParallelDescriptor::IOProcessorNumber()});
	amrex::MultiFab mf_single(ba_single, dm_single, 1, 0);
	mf_single.ParallelCopy(mf, comp, 0, 1);

	int num_diffs = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const amrex::MFIter mfi(mf_single);
		const amrex::Box &box = mfi.validbox();
#ifdef AMREX_USE_GPU
		amrex::FArrayBox host_fab(box, 1, amrex::The_Pinned_Arena());
		static_cast<void>(mf_single[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, 1, host_fab.dataPtr()));
		amrex::Gpu::synchronize();
#else
		amrex::FArrayBox host_fab(box, 1);
		static_cast<void>(mf_single[mfi].template copyToMem<amrex::RunOn::Host>(box, 0, 1, host_fab.dataPtr()));
#endif
		const auto &field = host_fab.const_array();
		const amrex::IntVect is_nodal = mf.ixType().toIntVect();
		const std::array<int, 2> n{domain.length(0), domain.length(1)};

		for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
			for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
				for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
					const int i_rot = rot180Index(i, n[0], is_nodal[0] != 0);
					const int j_rot = rot180Index(j, n[1], is_nodal[1] != 0);
					if ((i_rot < i) || (i_rot == i && j_rot < j)) {
						continue; // each pair only needs checking once
					}
					const amrex::Real val = field(i, j, k);
					const amrex::Real val_rot = field(i_rot, j_rot, k);
					const amrex::Real expected = (parity == Rot180Parity::Even) ? val : -val;
					if (val_rot != expected) {
						++num_diffs;
						amrex::Print()
						    << "[rot180] " << label << " mismatch: (" << i << "," << j << "," << k << ")=" << std::setprecision(17)
						    << val << " vs (" << i_rot << "," << j_rot << "," << k << ")=" << std::setprecision(17) << val_rot
						    << " (expected " << std::setprecision(17) << expected << ")\n";
					}
				}
			}
		}

		if (num_diffs == 0) {
			amrex::Print() << "[rot180] " << label << ": exact match under 180-degree rotation.\n";
		} else {
			amrex::Print() << "[rot180] " << label << ": found " << num_diffs << " mismatched cell pairs under 180-degree rotation.\n";
		}
	}
	return num_diffs;
}
} // namespace

template <> void QuokkaSimulation<OrszagTang>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*initSumCons*/)
{
	// the Orszag-Tang vortex's IC is point-symmetric about the domain centre, so the exact solution stays
	// bit-exactly rot180-symmetric for all time; a floating-point non-associativity bug in
	// reconstruction, EMF averaging, or the Riemann solver breaks this (see PR #1997).
	const amrex::Box &domain = Geom(0).Domain();
	int num_diffs = 0;
	num_diffs += checkRot180Symmetry(state_new_cc_[0], HydroSystem<OrszagTang>::density_index, domain, Rot180Parity::Even, "density");
	num_diffs += checkRot180Symmetry(state_new_cc_[0], HydroSystem<OrszagTang>::x1Momentum_index, domain, Rot180Parity::Odd, "x1Momentum");
	num_diffs += checkRot180Symmetry(state_new_cc_[0], HydroSystem<OrszagTang>::x2Momentum_index, domain, Rot180Parity::Odd, "x2Momentum");
	num_diffs += checkRot180Symmetry(state_new_fc_[0][0], Physics_Indices<OrszagTang>::mhdFirstIndex, domain, Rot180Parity::Odd, "Bx");
	num_diffs += checkRot180Symmetry(state_new_fc_[0][1], Physics_Indices<OrszagTang>::mhdFirstIndex, domain, Rot180Parity::Odd, "By");

	if (amrex::ParallelDescriptor::IOProcessor()) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(num_diffs == 0, "OrszagTang rot180 symmetry check failed: the MHD solver broke bit-exact point-reflection "
								 "symmetry (see PR #1997); this indicates a regression in floating-point term-pairing for "
								 "WENO reconstruction, EMF averaging, or HLLD.");
	}
}

auto problem_main() -> int
{
	QuokkaSimulation<OrszagTang> sim;
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
