#ifndef TURBULENCE_GEN_AmreX_H
#define TURBULENCE_GEN_AmreX_H

#include "../../TurbGen.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "AMReX.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArray.H"
#include "AMReX_FabFactory.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuControl.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"

class TurbGenEx : public TurbGen
{
      private:
	amrex::Gpu::DeviceVector<double> modes_gpu[3], aka_gpu[3],
	    akb_gpu[3]; // Mode arrays,
	amrex::GpuArray<double *, 3> modes_pointers, aka_pointers, akb_pointers;
	amrex::Gpu::DeviceVector<double> ampl_gpu;
	amrex::Gpu::DeviceVector<double> ampl_factor_gpu;

	void initial_sync_to_gpu()
	{
		ampl_gpu.resize(ampl.size());
		ampl_factor_gpu.resize(3);

		for (int dim = 0; dim < 3; dim++) {
			modes_gpu[dim].resize(mode[dim].size());
			aka_gpu[dim].resize(aka[dim].size());
			akb_gpu[dim].resize(akb[dim].size());

			modes_pointers[dim] = modes_gpu[dim].data();
			aka_pointers[dim] = aka_gpu[dim].data();
			akb_pointers[dim] = akb_gpu[dim].data();

			amrex::Gpu::copy(amrex::Gpu::hostToDevice, mode[dim].begin(), mode[dim].end(), modes_gpu[dim].begin());
		}
	}

	void sync_to_gpu()
	{
		{
			amrex::Gpu::PinnedVector<double> pinnedAmpl(ampl.size());

			// pre-compute amplitude including normalisation factors ()
			for (int m = 0; m < nmodes; m++)
				pinnedAmpl[m] = 2.0 * sol_weight_norm * ampl[m];
			amrex::Gpu::copy(amrex::Gpu::hostToDevice, pinnedAmpl.begin(), pinnedAmpl.end(), ampl_gpu.begin());
		}

		amrex::Gpu::copy(amrex::Gpu::hostToDevice, ampl_factor, ampl_factor + 3, ampl_factor_gpu.begin());

		for (int dim = 0; dim < 3; dim++) {
			{
				amrex::Gpu::PinnedVector<double> pinnedAka(aka[dim].size());
				std::copy(aka[dim].begin(), aka[dim].end(), pinnedAka.begin());
				amrex::Gpu::copy(amrex::Gpu::hostToDevice, pinnedAka.begin(), pinnedAka.end(), aka_gpu[dim].begin());
			}
			{
				amrex::Gpu::PinnedVector<double> pinnedakb(akb[dim].size());
				std::copy(akb[dim].begin(), akb[dim].end(), pinnedakb.begin());
				amrex::Gpu::copy(amrex::Gpu::hostToDevice, pinnedakb.begin(), pinnedakb.end(), akb_gpu[dim].begin());
			}
		}
	} // sync_to_gpu

      public:
	// Prevents overloaded methods from hiding base class methods of different
	// signature.
	using TurbGen::check_for_update;
	using TurbGen::get_turb_vector_unigrid;
	using TurbGen::init_driving;

	TurbGenEx()
        : TurbGen(amrex::ParallelDescriptor::MyProc()) {}

   int init_driving(const std::map<std::string, std::string> &params) override
	{
		TurbGen::init_driving(params);
		initial_sync_to_gpu();
		return 0;
	}

	bool check_for_update(const double time, const double v_turb[]) override
	{
		// ******************************************************
		// Update driving pattern based on input 'time'.
		// If it is 'time' to update the pattern, call OU noise update
		// and update the decomposition coefficients; otherwise, simply return.
		// Identical to base class expect for call to sync_to_gpu() if an update has
		// occurred.
		// ******************************************************
		bool pattern_changed = TurbGen::check_for_update(time, v_turb);
		if (pattern_changed)
			sync_to_gpu();
		return pattern_changed;
	};

	void get_turb_vector_unigrid(amrex::FArrayBox &fab, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes)
	{
		// ******************************************************
		// Compute physical turbulent vector field on a uniform grid. provided
		// Takes a FArrayBox which contains the index space of the uniform grid,
		// which should have a number of components equal to the dimension.
		// (Integer, not 1.5, 2.5) Also takes the physical sizes of cells on the
		// level. Returns into
		// ******************************************************

		const amrex::Box &box = fab.box();
		amrex::Array4<amrex::Real> fieldArray = fab.array();

		if (verbose > 1)
			TurbGen_printf(FuncSig(__func__) + "entering.\n");

		// pre-compute grid position geometry, and trigonometry, to speed-up loops
		// over modes below
		// Get axis to loop over.
		amrex::Box xSpace(amrex::IntVect(AMREX_D_DECL(box.smallEnd()[0], 0, 0)), amrex::IntVect(AMREX_D_DECL(box.bigEnd()[0], 0, 0)));
		amrex::Box ySpace(amrex::IntVect(AMREX_D_DECL(0, box.smallEnd()[1], 0)), amrex::IntVect(AMREX_D_DECL(0, box.bigEnd()[0], 0)));
		amrex::Box zSpace(amrex::IntVect(AMREX_D_DECL(0, 0, box.smallEnd()[2])), amrex::IntVect(AMREX_D_DECL(0, 0, box.bigEnd()[2])));

		// Along each axis there is a sin and cos component for each mode.
		int comps = nmodes * 2;

		amrex::FArrayBox xPrecompFab(xSpace, comps);
		amrex::FArrayBox yPrecompFab(ySpace, comps);
		amrex::FArrayBox zPrecompFab(zSpace, comps);

		amrex::Array4 xPrecomp = xPrecompFab.array();
		amrex::Array4 yPrecomp = yPrecompFab.array();
		amrex::Array4 zPrecomp = zPrecompFab.array();

		// Loops over the 2D space of x and modes. Fills Array4 which has no y,z
		// size hence effectively a 2d array of x - modes The
		// "modes_pointers=this->modes_pointers,nmodes=this->nmodes" prevents
		// capture of class
		amrex::ParallelFor(xSpace, nmodes,
				   [=, modesPointers = this->modes_pointers, nmodes = this->nmodes] AMREX_GPU_DEVICE(int i, int j, int k, int m) {
					   const int SIN_INDEX = m;
					   const int COS_INDEX = m + nmodes;

					   xPrecomp(i, j, k, SIN_INDEX) = sin(modesPointers[X][m] * (i * cellSizes[X]));
					   xPrecomp(i, j, k, COS_INDEX) = cos(modesPointers[X][m] * (i * cellSizes[X]));
				   });

		amrex::ParallelFor(ySpace, nmodes,
				   [=, modesPointers = this->modes_pointers, nmodes = this->nmodes] AMREX_GPU_DEVICE(int i, int j, int k, int m) {
					   const int SIN_INDEX = m;
					   const int COS_INDEX = m + nmodes;
					   if (AMREX_SPACEDIM > 1) {
						   yPrecomp(i, j, k, SIN_INDEX) = sin(modesPointers[Y][m] * (j * cellSizes[Y]));
						   yPrecomp(i, j, k, COS_INDEX) = cos(modesPointers[Y][m] * (j * cellSizes[Y]));
					   } else {
						   yPrecomp(i, j, k, SIN_INDEX) = 0.0;
						   yPrecomp(i, j, k, COS_INDEX) = 1.0;
					   }
				   });

		amrex::ParallelFor(zSpace, nmodes,
				   [=, modesPointers = this->modes_pointers, nmodes = this->nmodes] AMREX_GPU_DEVICE(int i, int j, int k, int m) {
					   const int SIN_INDEX = m;
					   const int COS_INDEX = m + nmodes;

					   zPrecomp(i, j, k, SIN_INDEX) = AMREX_SPACEDIM > 2 ? sin(modesPointers[Z][m] * (k * cellSizes[Z])) : 0.0;
					   zPrecomp(i, j, k, COS_INDEX) = AMREX_SPACEDIM > 2 ? cos(modesPointers[Z][m] * (k * cellSizes[Z])) : 1.0;
				   });

		// Get pointers to pass to parallelFor (Necessary for GPU)
		double *ampl_factorGPU_prt = ampl_factor_gpu.data();
		double *ampGPU_prt = ampl_gpu.data();

		// Calculate forcing field for every point on the uniform grid using the
		// precomputed information.
		amrex::ParallelFor(box, [=, akaPointers = this->aka_pointers, akbPointers = this->akb_pointers, modesPointers = this->modes_pointers,
					 nmodes = this->nmodes] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real real, imag;

			amrex::Real v[3];

			v[X] = 0.0;
			v[Y] = 0.0;
			v[Z] = 0.0;
			// loop over modes
			for (int m = 0; m < nmodes; m++) {

				const int SIN_INDEX = m;
				const int COS_INDEX = m + nmodes;

				// these are the real and imaginary parts, respectively, of
				//  e^{ i \vec{k} \cdot \vec{x} } = cos(kx*x + ky*y + kz*z) + i
				//  sin(kx*x + ky*y + kz*z)

				real = (xPrecomp(i, 0, 0, COS_INDEX) * yPrecomp(0, j, 0, COS_INDEX) -
					xPrecomp(i, 0, 0, SIN_INDEX) * yPrecomp(0, j, 0, SIN_INDEX)) *
					   zPrecomp(0, 0, k, COS_INDEX) -
				       (xPrecomp(i, 0, 0, SIN_INDEX) * yPrecomp(0, j, 0, COS_INDEX) +
					xPrecomp(i, 0, 0, COS_INDEX) * yPrecomp(0, j, 0, SIN_INDEX)) *
					   zPrecomp(0, 0, k, SIN_INDEX);

				imag = (yPrecomp(0, j, 0, COS_INDEX) * zPrecomp(0, 0, k, SIN_INDEX) +
					yPrecomp(0, j, 0, SIN_INDEX) * zPrecomp(0, 0, k, COS_INDEX)) *
					   xPrecomp(i, 0, 0, COS_INDEX) +
				       (yPrecomp(0, j, 0, COS_INDEX) * zPrecomp(0, 0, k, COS_INDEX) -
					yPrecomp(0, j, 0, SIN_INDEX) * zPrecomp(0, 0, k, SIN_INDEX)) *
					   xPrecomp(i, 0, 0, SIN_INDEX);

				// accumulate total v as sum over modes
				v[X] += ampGPU_prt[m] * (akaPointers[X][m] * real - akbPointers[X][m] * imag);
				if constexpr (AMREX_SPACEDIM > 1)
					v[Y] += ampGPU_prt[m] * (akaPointers[Y][m] * real - akbPointers[Y][m] * imag);
				if constexpr (AMREX_SPACEDIM > 2)
					v[Z] += ampGPU_prt[m] * (akaPointers[Z][m] * real - akbPointers[Z][m] * imag);
			}
			// copy into return grid
			fieldArray(i, j, k, 0) = v[X] * ampl_factorGPU_prt[X];
			if constexpr (AMREX_SPACEDIM > 1)
				fieldArray(i, j, k, 1) = v[Y] * ampl_factorGPU_prt[Y];
			if constexpr (AMREX_SPACEDIM > 2)
				fieldArray(i, j, k, 2) = v[Z] * ampl_factorGPU_prt[Z];
		});
	} // get_turb_vector_unigrid (AMReX overload)
};

#endif // TURBULENCE_GEN_AmreX_H *NOT* defined
