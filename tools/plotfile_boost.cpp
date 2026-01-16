#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_RealBox.H>

#include <array>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

auto FindComponentIndex(amrex::Vector<std::string> const &names,
                        std::string const &target) -> int {
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    if (names[i] == target) {
      return i;
    }
  }
  return -1;
}

void PrintUsage() {
  amrex::Print()
      << "Usage: plotfile_boost <plotfile_in> <plotfile_out> <vx> <vy> <vz>\n";
}

} // namespace

auto main(int argc, char **argv) -> int {
  amrex::Initialize(argc, argv);
  {
    if (argc != 6) {
      PrintUsage();
      amrex::Abort("Invalid arguments");
    }

    const std::string plotfile_in = argv[1];
    const std::string plotfile_out = argv[2];
    const amrex::Real boost_vx = std::strtod(argv[3], nullptr);
    const amrex::Real boost_vy = std::strtod(argv[4], nullptr);
    const amrex::Real boost_vz = std::strtod(argv[5], nullptr);
    const amrex::Real boost_v2 =
        (boost_vx * boost_vx) + (boost_vy * boost_vy) + (boost_vz * boost_vz);

    amrex::PlotFileData plotfile(plotfile_in);
    const int finest_level = plotfile.finestLevel();
    auto const &var_names = plotfile.varNames();

    const int rho_index = FindComponentIndex(var_names, "gasDensity");
    const int px_index = FindComponentIndex(var_names, "x-GasMomentum");
    const int py_index = FindComponentIndex(var_names, "y-GasMomentum");
    const int pz_index = FindComponentIndex(var_names, "z-GasMomentum");
    const int energy_index = FindComponentIndex(var_names, "gasEnergy");

    if (rho_index < 0 || px_index < 0 || py_index < 0 || pz_index < 0 ||
        energy_index < 0) {
      amrex::Abort(
          "Required plotfile variables not found "
          "(gasDensity/x-GasMomentum/y-GasMomentum/z-GasMomentum/gasEnergy)");
    }

    amrex::Vector<amrex::MultiFab> state(finest_level + 1);
    amrex::Vector<amrex::Geometry> geom(finest_level + 1);
    amrex::Vector<int> level_steps(finest_level + 1, 0);
    amrex::Vector<amrex::IntVect> ref_ratio;
    if (finest_level > 0) {
      ref_ratio.reserve(finest_level);
    }

    amrex::RealBox real_box(plotfile.probLo().data(), plotfile.probHi().data());
    std::array<int, AMREX_SPACEDIM> is_per = {AMREX_D_DECL(0, 0, 0)};

    for (int lev = 0; lev <= finest_level; ++lev) {
      state[lev] = plotfile.get(lev);
      level_steps[lev] = plotfile.levelStep(lev);
      geom[lev] = amrex::Geometry(plotfile.probDomain(lev), &real_box,
                                  plotfile.coordSys(), is_per.data());
      if (lev < finest_level) {
        const int rr = plotfile.refRatio(lev);
        ref_ratio.emplace_back(amrex::IntVect{AMREX_D_DECL(rr, rr, rr)});
      }

      for (amrex::MFIter mfi(state[lev]); mfi.isValid(); ++mfi) {
        const amrex::Box &box = mfi.validbox();
        auto const &arr = state[lev].array(mfi);

        amrex::ParallelFor(
            box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              const amrex::Real rho = arr(i, j, k, rho_index);
              const amrex::Real px = arr(i, j, k, px_index);
              const amrex::Real py = arr(i, j, k, py_index);
              const amrex::Real pz = arr(i, j, k, pz_index);
              const amrex::Real energy = arr(i, j, k, energy_index);

              const amrex::Real px_new = px - rho * boost_vx;
              const amrex::Real py_new = py - rho * boost_vy;
              const amrex::Real pz_new = pz - rho * boost_vz;
              const amrex::Real energy_new =
                  energy - (boost_vx * px + boost_vy * py + boost_vz * pz) +
                  static_cast<amrex::Real>(0.5) * rho * boost_v2;

              arr(i, j, k, px_index) = px_new;
              arr(i, j, k, py_index) = py_new;
              arr(i, j, k, pz_index) = pz_new;
              arr(i, j, k, energy_index) = energy_new;
            });
      }
    }

    amrex::Vector<const amrex::MultiFab *> mf_ptrs(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      mf_ptrs[lev] = &state[lev];
    }

    amrex::WriteMultiLevelPlotfile(plotfile_out, finest_level + 1, mf_ptrs,
                                   var_names, geom, plotfile.time(),
                                   level_steps, ref_ratio);
  }
  amrex::Finalize();
  return 0;
}
