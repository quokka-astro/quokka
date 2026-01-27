/// \file testDiskGalaxyInterpLimiter.cpp
/// \brief Compare AMReX cell-centered interpolators for a rotating exponential disk.

#include "AMReX.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FillPatchUtil.H"
#include "AMReX_Geometry.H"
#include "AMReX_MFInterpolater.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PhysBCFunct.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "AMReX_RealBox.H"

#include <array>
#include <cmath>

namespace
{
static_assert(AMREX_SPACEDIM == 3, "DiskGalaxyInterpLimiter requires AMREX_SPACEDIM == 3.");
constexpr int ncomp = 4; // rho, momx, momy, momz

struct NullFill {
	AMREX_GPU_DEVICE
	void operator()(const amrex::IntVect & /*iv*/, amrex::Array4<amrex::Real> const & /*dest*/, int /*dcomp*/, int /*numcomp*/,
			amrex::GeometryData const & /*geom*/, amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/) const
	{
		// no physical boundaries to fill (periodic)
	}
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto disk_density(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real rho0, amrex::Real rscale,
							   amrex::Real zscale) -> amrex::Real
{
	const amrex::Real R = std::sqrt(x * x + y * y);
	return rho0 * std::exp(-R / rscale) * std::exp(-std::abs(z) / zscale);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto disk_vphi(amrex::Real x, amrex::Real y, amrex::Real v0, amrex::Real rturn) -> amrex::Real
{
	const amrex::Real R = std::sqrt(x * x + y * y);
	if (R <= amrex::Real(0.0)) {
		return amrex::Real(0.0);
	}
	return v0 * (amrex::Real(1.0) - std::exp(-R / rturn));
}

void fill_disk(amrex::MultiFab &mf, const amrex::Geometry &geom, amrex::Real rho0, amrex::Real rscale, amrex::Real zscale, amrex::Real v0, amrex::Real rturn)
{
	const auto prob_lo = geom.ProbLoArray();
	const auto dx = geom.CellSizeArray();

	for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto const &state = mf.array(mfi);
		amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
			const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
			const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];

			const amrex::Real rho = disk_density(x, y, z, rho0, rscale, zscale);
			const amrex::Real vphi = disk_vphi(x, y, v0, rturn);
			const amrex::Real R = std::sqrt(x * x + y * y);

			amrex::Real vx = 0.0;
			amrex::Real vy = 0.0;
			if (R > amrex::Real(0.0)) {
				vx = -vphi * y / R;
				vy = vphi * x / R;
			}

			state(i, j, k, 0) = rho;
			state(i, j, k, 1) = rho * vx;
			state(i, j, k, 2) = rho * vy;
			state(i, j, k, 3) = amrex::Real(0.0);
		});
	}
}

auto interp_and_write(const std::string &label, amrex::MultiFab const &coarse, amrex::Geometry const &cgeom, amrex::Geometry const &fgeom,
		      amrex::BoxArray const &fba, amrex::MFInterpolater *mapper, const amrex::Vector<amrex::BCRec> &bcs, amrex::IntVect const &ratio,
		      amrex::MultiFab const &exact, const std::array<std::string, ncomp> &names, amrex::MultiFab *delta_out) -> void
{
	amrex::DistributionMapping const fdm(fba);
	amrex::MultiFab fine(fba, fdm, ncomp, 0);

	amrex::GpuBndryFuncFab<NullFill> bndry_func(NullFill{});
	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<NullFill>> cphys(cgeom, bcs, bndry_func);
	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<NullFill>> fphys(fgeom, bcs, bndry_func);

	amrex::InterpFromCoarseLevel(fine, 0.0, coarse, 0, 0, ncomp, cgeom, fgeom, cphys, 0, fphys, 0, ratio, mapper, bcs, 0);

	amrex::MultiFab err(fba, fdm, ncomp, 0);
	amrex::MultiFab::Copy(err, fine, 0, 0, ncomp, 0);
	err.minus(exact, 0, ncomp, 0);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Interpolation error norms for " << label << ":\n";
		for (int n = 0; n < ncomp; ++n) {
			const amrex::Real err_l1 = err.norm1(n, 0, true);
			const amrex::Real err_linf = err.norminf(n, 0, true);
			amrex::Print() << "  comp " << n << " L1=" << err_l1 << " Linf=" << err_linf << "\n";
		}
	}

	amrex::Vector<std::string> plot_names(names.begin(), names.end());
	amrex::WriteSingleLevelPlotfile(label, fine, plot_names, fgeom, 0.0, 0);

	if (delta_out != nullptr) {
		amrex::MultiFab::Copy(*delta_out, fine, 0, 0, ncomp, 0);
	}
}

} // namespace

auto problem_main() -> int
{
	amrex::ParmParse pp_amr("amr");
	amrex::Vector<int> n_cell;
	pp_amr.getarr("n_cell", n_cell);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_cell.size() == 3, "amr.n_cell must have 3 components.");

	amrex::ParmParse pp_geom("geometry");
	amrex::Vector<amrex::Real> prob_lo;
	amrex::Vector<amrex::Real> prob_hi;
	pp_geom.getarr("prob_lo", prob_lo);
	pp_geom.getarr("prob_hi", prob_hi);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(prob_lo.size() == 3 && prob_hi.size() == 3, "geometry.prob_lo/hi must have 3 components.");

	amrex::Vector<int> is_periodic(3, 1);
	pp_geom.queryarr("is_periodic", is_periodic);

	int max_grid_size = 128;
	pp_amr.query("max_grid_size", max_grid_size);

	amrex::Vector<int> ref_ratio_vec;
	pp_amr.queryarr("ref_ratio", ref_ratio_vec);
	const int ref_ratio = ref_ratio_vec.empty() ? 2 : ref_ratio_vec[0];
	const amrex::IntVect ratio(AMREX_D_DECL(ref_ratio, ref_ratio, ref_ratio));

	const amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
	const amrex::IntVect dom_hi(AMREX_D_DECL(n_cell[0] - 1, n_cell[1] - 1, n_cell[2] - 1));
	const amrex::Box domain(dom_lo, dom_hi);

	const amrex::RealBox real_box({prob_lo[0], prob_lo[1], prob_lo[2]}, {prob_hi[0], prob_hi[1], prob_hi[2]});
	const amrex::Geometry cgeom(domain, real_box, 0, {is_periodic[0], is_periodic[1], is_periodic[2]});
	const amrex::Box fine_domain = amrex::refine(domain, ratio);
	const amrex::Geometry fgeom(fine_domain, real_box, 0, {is_periodic[0], is_periodic[1], is_periodic[2]});

	amrex::ParmParse pp("disk_interp");
	amrex::Real rho0 = 1.0;
	amrex::Real rscale = 0.2 * (prob_hi[0] - prob_lo[0]);
	amrex::Real zscale = 0.02 * (prob_hi[2] - prob_lo[2]);
	amrex::Real v0 = 10.0;
	amrex::Real rturn = rscale;
	pp.query("rho0", rho0);
	pp.query("rscale", rscale);
	pp.query("zscale", zscale);
	pp.query("v0", v0);
	pp.query("rturn", rturn);

	amrex::Vector<int> ref_lo;
	amrex::Vector<int> ref_hi;
	pp.getarr("refine_box_lo", ref_lo);
	pp.getarr("refine_box_hi", ref_hi);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ref_lo.size() == 3 && ref_hi.size() == 3, "disk_interp.refine_box_lo/hi must have 3 components.");
	amrex::Box coarse_ref_box(amrex::IntVect{AMREX_D_DECL(ref_lo[0], ref_lo[1], ref_lo[2])}, amrex::IntVect{AMREX_D_DECL(ref_hi[0], ref_hi[1], ref_hi[2])});
	coarse_ref_box &= domain;
	amrex::Box fine_ref_box = amrex::refine(coarse_ref_box, ratio);

	amrex::BoxArray cba(domain);
	cba.maxSize(max_grid_size);
	amrex::DistributionMapping cdm(cba);
	amrex::MultiFab coarse(cba, cdm, ncomp, 0);
	fill_disk(coarse, cgeom, rho0, rscale, zscale, v0, rturn);

	amrex::BoxArray fba(fine_ref_box);
	fba.maxSize(max_grid_size * ref_ratio);
	amrex::DistributionMapping fdm(fba);
	amrex::MultiFab exact(fba, fdm, ncomp, 0);
	fill_disk(exact, fgeom, rho0, rscale, zscale, v0, rturn);

	amrex::Vector<amrex::BCRec> bcs(ncomp);
	for (int n = 0; n < ncomp; ++n) {
		amrex::BCRec bc;
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			bc.setLo(d, amrex::BCType::int_dir);
			bc.setHi(d, amrex::BCType::int_dir);
		}
		bcs[n] = bc;
	}

	const std::array<std::string, ncomp> names = {"rho", "momx", "momy", "momz"};
	amrex::WriteSingleLevelPlotfile("plt_disk_interp_exact", exact, amrex::Vector<std::string>(names.begin(), names.end()), fgeom, 0.0, 0);
	amrex::WriteSingleLevelPlotfile("plt_disk_interp_coarse", coarse, amrex::Vector<std::string>(names.begin(), names.end()), cgeom, 0.0, 0);

	amrex::MultiFab fine_ll(fba, fdm, ncomp, 0);
	amrex::MultiFab fine_minmax(fba, fdm, ncomp, 0);

	interp_and_write("plt_disk_interp_pc", coarse, cgeom, fgeom, fba, &amrex::mf_pc_interp, bcs, ratio, exact, names, nullptr);
	interp_and_write("plt_disk_interp_ll", coarse, cgeom, fgeom, fba, &amrex::mf_lincc_interp, bcs, ratio, exact, names, &fine_ll);
	interp_and_write("plt_disk_interp_mc", coarse, cgeom, fgeom, fba, &amrex::mf_cell_cons_interp, bcs, ratio, exact, names, nullptr);
	interp_and_write("plt_disk_interp_ll_minmax", coarse, cgeom, fgeom, fba, &amrex::mf_linear_slope_minmax_interp, bcs, ratio, exact, names, &fine_minmax);

	amrex::MultiFab delta(fba, fdm, ncomp, 0);
	for (amrex::MFIter mfi(delta); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto const &darr = delta.array(mfi);
		auto const &ll = fine_ll.const_array(mfi);
		auto const &mm = fine_minmax.const_array(mfi);
		amrex::ParallelFor(bx, ncomp,
				   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept { darr(i, j, k, n) = std::abs(mm(i, j, k, n) - ll(i, j, k, n)); });
	}
	amrex::WriteSingleLevelPlotfile("plt_disk_interp_ll_minmax_delta", delta, amrex::Vector<std::string>(names.begin(), names.end()), fgeom, 0.0, 0);

	return 0;
}
