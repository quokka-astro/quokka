//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDBlast.cpp
/// \brief Defines a test problem for a 3D explosion with magnetic fields.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"

struct MHDBlast {
};

template <> struct quokka::EOS_Traits<MHDBlast> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<MHDBlast> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<MHDBlast> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

template <> void QuokkaSimulation<MHDBlast>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// problem parameters
	constexpr Real rho = 1.0;      // g cm^-3
	constexpr Real R_init = 0.125; // cm

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);
		const double z = prob_lo[2] + ((k + 0.5) * dx[2]);
		const double R = std::sqrt(x * x + y * y + z * z);

		double P = NAN;
		double const Emag = 50.;
		if (R < R_init) {
			P = 100.;
		} else {
			P = 1.0;
		}

		state_cc(i, j, k, HydroSystem<MHDBlast>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<MHDBlast>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<MHDBlast>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<MHDBlast>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<MHDBlast>::energy_index) = P / (quokka::EOS_Traits<MHDBlast>::gamma - 1.) + Emag;
		state_cc(i, j, k, HydroSystem<MHDBlast>::internalEnergy_index) = P / (quokka::EOS_Traits<MHDBlast>::gamma - 1.);
	});
}

template <> void QuokkaSimulation<MHDBlast>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		constexpr double bx = 10. / gcem::sqrt(2.);
		constexpr double by = 10. / gcem::sqrt(2.);
		constexpr double bz = 0;

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<MHDBlast>::mhdFirstIndex) = bx;
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<MHDBlast>::mhdFirstIndex) = by;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<MHDBlast>::mhdFirstIndex) = bz;
		}
	});
}

template <> void QuokkaSimulation<MHDBlast>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	const amrex::Real eta_threshold = 0.1; // gradient refinement threshold
	const auto &state_fc = state_new_fc_[lev];

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
		    AMREX_D_DECL(state_fc[0].const_array(mfi), state_fc[1].const_array(mfi), state_fc[2].const_array(mfi))};

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const P = HydroSystem<MHDBlast>::ComputePressure(state, i, j, k, &cons_fc);

			amrex::Real const P_xplus = HydroSystem<MHDBlast>::ComputePressure(state, i + 1, j, k, &cons_fc);
			amrex::Real const P_xminus = HydroSystem<MHDBlast>::ComputePressure(state, i - 1, j, k, &cons_fc);
			amrex::Real const P_yplus = HydroSystem<MHDBlast>::ComputePressure(state, i, j + 1, k, &cons_fc);
			amrex::Real const P_yminus = HydroSystem<MHDBlast>::ComputePressure(state, i, j - 1, k, &cons_fc);
			amrex::Real const P_zplus = HydroSystem<MHDBlast>::ComputePressure(state, i, j, k + 1, &cons_fc);
			amrex::Real const P_zminus = HydroSystem<MHDBlast>::ComputePressure(state, i, j, k - 1, &cons_fc);

			amrex::Real const Dx = 0.5 * (P_xplus - P_xminus);
			amrex::Real const Dy = 0.5 * (P_yplus - P_yminus);
			amrex::Real const Dz = 0.5 * (P_zplus - P_zminus);
			amrex::Real const rel_gradient = std::sqrt(Dx * Dx + Dy * Dy + Dz * Dz) / P;

			if (rel_gradient > eta_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <>
void QuokkaSimulation<MHDBlast>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp,
						   amrex::MultiFab const & /*state_cc*/, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
	// compute derived variables and save in 'mf'
	if (dname == "magnetic_divergence") {
		const amrex::Geometry &geom_lev = geom[lev];
		const auto dx = geom_lev.CellSizeArray();
		auto output = mf.arrays();

		// Get the face-centered magnetic field arrays
		auto const &Bx_arr = state_fc[0].const_arrays();
		auto const &By_arr = state_fc[1].const_arrays();
		auto const &Bz_arr = state_fc[2].const_arrays();

		amrex::ParallelFor(mf, {0, 0, 0}, [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept {
			// Compute divergence using finite differences
			constexpr int idx = Physics_Indices<MHDBlast>::mhdFirstIndex;
			amrex::Real const Bx_p = Bx_arr[box](i + 1, j, k, idx);
			amrex::Real const Bx_m = Bx_arr[box](i, j, k, idx);
			amrex::Real const By_p = By_arr[box](i, j + 1, k, idx);
			amrex::Real const By_m = By_arr[box](i, j, k, idx);
			amrex::Real const Bz_p = Bz_arr[box](i, j, k + 1, idx);
			amrex::Real const Bz_m = Bz_arr[box](i, j, k, idx);
			amrex::Real const divB_x = (Bx_p - Bx_m) / dx[0];
			amrex::Real const divB_y = (By_p - By_m) / dx[1];
			amrex::Real const divB_z = (Bz_p - Bz_m) / dx[2];
			output[box](i, j, k, ncomp) = divB_x + divB_y + divB_z;
		});
	}
	amrex::Gpu::streamSynchronizeAll();
}

auto problem_main() -> int
{
	QuokkaSimulation<MHDBlast> sim;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	return 0;
}
