// Compare every stored cell/face component in a DiskGalaxy checkpoint pair.
// Build against a CPU AMReX installation; no simulation or GPU is required.
#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>

auto main(int argc, char **argv) -> int {
  amrex::Initialize(argc, argv, false);
  {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        argc == 4, "Usage: compare_checkpoint_mesh LEFT_CHECKPOINT "
                   "RIGHT_CHECKPOINT FINEST_LEVEL");
    const int finest = std::stoi(argv[3]);
    AMREX_ALWAYS_ASSERT(finest >= 0);
    for (int lev = 0; lev <= finest; ++lev) {
      for (const auto *name :
           std::array{"Cell", "Face_x", "Face_y", "Face_z"}) {
        const auto prefix =
            std::string("Level_") + std::to_string(lev) + "/" + name;
        amrex::MultiFab left;
        amrex::MultiFab right;
        amrex::VisMF::Read(left, std::string(argv[1]) + "/" + prefix);
        amrex::VisMF::Read(right, std::string(argv[2]) + "/" + prefix);
        AMREX_ALWAYS_ASSERT(left.boxArray() == right.boxArray());
        AMREX_ALWAYS_ASSERT(left.DistributionMap() == right.DistributionMap());
        AMREX_ALWAYS_ASSERT(left.nComp() == right.nComp() && left.nComp() > 0);
        AMREX_ALWAYS_ASSERT(left.nGrow() == 0 && right.nGrow() == 0);
        AMREX_ALWAYS_ASSERT(!left.contains_nan() && !right.contains_nan());
        AMREX_ALWAYS_ASSERT(!left.contains_inf() && !right.contains_inf());
        amrex::MultiFab difference(left.boxArray(), left.DistributionMap(),
                                   left.nComp(), 0);
        // Preserve each FAB's copy of shared faces when comparing raw output.
        amrex::MultiFab::Copy(difference, right, 0, 0, left.nComp(), 0);
        amrex::MultiFab::Subtract(difference, left, 0, 0, left.nComp(), 0);
        for (int comp = 0; comp < left.nComp(); ++comp) {
          const auto scale = std::max(left.norm0(comp), right.norm0(comp));
          const auto error = difference.norm0(comp);
          amrex::Print().SetPrecision(17)
              << "ERROR " << prefix << "/" << comp << " " << error << " "
              << (scale > 0 ? error / scale : 0) << "\n";
        }
      }
    }
  }
  amrex::Finalize();
  return EXIT_SUCCESS;
}
