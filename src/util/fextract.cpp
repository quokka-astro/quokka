#include <limits>
#include <tuple>

#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_SPACE.H"
#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

using namespace amrex; // NOLINT

auto fextract(MultiFab &mf, Geometry &geom, const int idir, const Real slice_coord, const bool center = false)
    -> std::tuple<Vector<Real>, Vector<Gpu::HostVector<Real>>>
{
	AMREX_D_TERM(Real xcoord = slice_coord;, Real ycoord = slice_coord;, Real zcoord = slice_coord;)

	GpuArray<Real, AMREX_SPACEDIM> problo = geom.ProbLoArray();
	GpuArray<Real, AMREX_SPACEDIM> dx0 = geom.CellSizeArray();
	Box probdom0 = geom.Domain();
	const auto lo0 = amrex::lbound(probdom0);
	const auto hi0 = amrex::ubound(probdom0);

	// compute the index of the center or lower left of the domain on the
	// coarse grid.  These are used to set the position of the slice in
	// the transverse direction.

	AMREX_D_TERM(int iloc = 0;, int jloc = 0;, int kloc = 0;)
	if (center) {
		AMREX_D_TERM(iloc = (hi0.x - lo0.x + 1) / 2 + lo0.x;, jloc = (hi0.y - lo0.y + 1) / 2 + lo0.y;, kloc = (hi0.z - lo0.z + 1) / 2 + lo0.z;)
	}

	if (idir == 0) {
		// we specified the x value to pass through
		iloc = hi0.x;
		for (int i = lo0.x; i <= hi0.x; ++i) {
			amrex::Real xc = problo[0] + (i + 0.5) * dx0[0];
			if (xc > xcoord) {
				iloc = i;
				break;
			}
		}
	}

#if AMREX_SPACEDIM >= 2
	if (idir == 1) {
		// we specified the y value to pass through
		jloc = hi0.y;
		for (int j = lo0.y; j <= hi0.y; ++j) {
			amrex::Real yc = problo[1] + (j + 0.5) * dx0[1];
			if (yc > ycoord) {
				jloc = j;
				break;
			}
		}
	}
#endif

#if AMREX_SPACEDIM == 3
	if (idir == 2) {
		// we specified the z value to pass through
		kloc = hi0.z;
		for (int k = lo0.z; k <= hi0.z; ++k) {
			amrex::Real zc = problo[2] + (k + 0.5) * dx0[2];
			if (zc > zcoord) {
				kloc = k;
				break;
			}
		}
	}
#endif

	if (idir < 0 || idir >= AMREX_SPACEDIM) {
		amrex::Abort("invalid direction!");
	}

	const IntVect ivloc{AMREX_D_DECL(iloc, jloc, kloc)};

	Vector<Real> pos;
	Vector<Gpu::HostVector<Real>> data(mf.nComp());

	IntVect rr{1};
	Box slice_box(ivloc * rr, ivloc * rr);
	slice_box.setSmall(idir, std::numeric_limits<int>::lowest());
	slice_box.setBig(idir, std::numeric_limits<int>::max());

	GpuArray<Real, AMREX_SPACEDIM> dx = dx0;

	// First pass: determine per-box offsets along the slice and total points
	Vector<int> offsets;
	int total_pts = 0;
	for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const Box bx = mfi.validbox() & slice_box;
		if (bx.ok()) {
			offsets.push_back(total_pts);
			total_pts += bx.length(idir);
		}
	}

	pos.resize(total_pts);
	for (auto &vec : data) {
		vec.resize(total_pts);
	}

	// compute position coordinates using contiguous local indices
	int box_idx = 0;
	for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const Box bx = mfi.validbox() & slice_box;
		if (bx.ok()) {
			const int offset = offsets[box_idx];
			const int start_dir = bx.smallEnd(idir);
			const int local_len = bx.length(idir);
			amrex::LoopOnCpu(bx, [problo, dx, idir, offset, start_dir, local_len, &pos](int i, int j, int k) {
				Array<Real, AMREX_SPACEDIM> p = {AMREX_D_DECL(problo[0] + static_cast<Real>(i + 0.5) * dx[0],
									      problo[1] + static_cast<Real>(j + 0.5) * dx[1],
									      problo[2] + static_cast<Real>(k + 0.5) * dx[2])};
				int idx = offset;
				if (idir == 0) {
					idx += i - start_dir;
				}
#if AMREX_SPACEDIM >= 2
				if (idir == 1) {
					idx += j - start_dir;
				}
#endif
#if AMREX_SPACEDIM == 3
				if (idir == 2) {
					idx += k - start_dir;
				}
#endif
				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(idx >= offset && idx < offset + local_len, "fextract: position index out of bounds");
				pos[idx] = p[idir];
			});
			++box_idx;
		}
	}

	// fill data arrays with the same contiguous indexing
	box_idx = 0;
	for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const Box bx = mfi.validbox() & slice_box;
		if (bx.ok()) {
			const int offset = offsets[box_idx];
			const int start_dir = bx.smallEnd(idir);
			const int local_len = bx.length(idir);
			const auto &fab = mf.array(mfi);
			for (int ivar = 0; ivar < mf.nComp(); ++ivar) {
				auto *dataptr = data[ivar].data();
				ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					int idx = offset;
					if (idir == 0) {
						idx += i - start_dir;
					}
#if AMREX_SPACEDIM >= 2
					if (idir == 1) {
						idx += j - start_dir;
					}
#endif
#if AMREX_SPACEDIM == 3
					if (idir == 2) {
						idx += k - start_dir;
					}
#endif
					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(idx >= offset && idx < offset + local_len, "fextract: data index out of bounds");
					dataptr[idx] = fab(i, j, k, ivar); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				});
			}
			++box_idx;
		}
	}

#ifdef AMREX_USE_MPI
	{
		const int numpts = static_cast<int>(pos.size());
		auto numpts_vec = ParallelDescriptor::Gather(numpts, ParallelDescriptor::IOProcessorNumber());
		Vector<int> recvcnt;
		Vector<int> disp;
		Vector<Real> allpos;
		Vector<Gpu::HostVector<Real>> alldata(data.size());
		if (ParallelDescriptor::IOProcessor()) {
			recvcnt.resize(numpts_vec.size());
			disp.resize(numpts_vec.size());
			int ntot = 0;
			disp[0] = 0;
			for (int i = 0, N = static_cast<int>(numpts_vec.size()); i < N; ++i) {
				ntot += numpts_vec[i];
				recvcnt[i] = numpts_vec[i];
				if (i + 1 < N) {
					disp[i + 1] = disp[i] + numpts_vec[i];
				}
			}
			allpos.resize(ntot);
			alldata.resize(data.size());
			for (auto &v : alldata) {
				v.resize(ntot);
			}
		} else {
			recvcnt.resize(1);
			disp.resize(1);
			allpos.resize(1);
			for (auto &v : alldata) {
				v.resize(1);
			}
		}
		// Handle empty vectors to avoid null pointer issues in MPI gather operations
		// Use static Real addresses when vectors are empty to provide non-null pointers
		// MPI won't read from these addresses when numpts is 0
		static Real static_real = 0.0;
		const Real *pos_ptr = pos.empty() ? &static_real : pos.data();
		Real *allpos_ptr = allpos.empty() ? &static_real : allpos.data();

		ParallelDescriptor::Gatherv(pos_ptr, numpts, allpos_ptr, recvcnt, disp, ParallelDescriptor::IOProcessorNumber());
		for (int i = 0; i < data.size(); ++i) {
			const Real *data_ptr = data[i].empty() ? &static_real : data[i].data();
			Real *alldata_ptr = alldata[i].empty() ? &static_real : alldata[i].data();
			ParallelDescriptor::Gatherv(data_ptr, numpts, alldata_ptr, recvcnt, disp, ParallelDescriptor::IOProcessorNumber());
		}
		if (ParallelDescriptor::IOProcessor()) {
			pos = std::move(allpos);
			data = std::move(alldata);
		}
	}
#endif // AMREX_USE_MPI

	return std::make_tuple(pos, data);
}
