#include "AMReX.H"
#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "AMReX_Utility.H"

#include "io/DiagBase.H"
#include "io/DiagFramePlane.H"
#include "io/DiagPDF.H"
#if AMREX_SPACEDIM == 3
#include "io/DiagVolumeRender.H"
#endif

#include "yaml-cpp/yaml.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_set>
#include <vector>

struct OfflineProblem {};

namespace {
auto stripTrailingSlash(std::string path) -> std::string {
  while (!path.empty() && path.back() == '/') {
    path.pop_back();
  }
  return path;
}

void PrintUsage() {
  std::cout << "Usage:\n"
            << "  quokka_diag_from_plotfile --inputs <inputs.in> --plotfile "
               "<plt> [options]\n\n"
            << "Options:\n"
            << "  --inputs <file>     Quokka inputs file (required)\n"
            << "  --plotfile <dir>    AMReX plotfile directory (required)\n"
            << "  --force             Run diagnostics regardless of interval "
               "settings\n"
            << "  --step <n>          Override plotfile step number\n"
            << "  --time <t>          Override plotfile time\n"
            << "  param=value         Any AMReX/Quokka ParmParse overrides\n";
}

auto ParseArgs(int argc, char **argv, std::string &inputs,
               std::string &plotfile, bool &force,
               std::optional<int> &stepOverride,
               std::optional<amrex::Real> &timeOverride,
               std::vector<std::string> &amrexOverrides) -> bool {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--inputs" && i + 1 < argc) {
      inputs = argv[++i];
    } else if (arg == "--plotfile" && i + 1 < argc) {
      plotfile = argv[++i];
    } else if (arg == "--force") {
      force = true;
    } else if (arg == "--step" && i + 1 < argc) {
      stepOverride = std::stoi(argv[++i]);
    } else if (arg == "--time" && i + 1 < argc) {
      timeOverride = amrex::Real(std::stod(argv[++i]));
    } else if (arg.rfind("plotfile=", 0) == 0) {
      plotfile = arg.substr(std::string("plotfile=").size());
    } else if (arg.rfind("-", 0) == 0) {
      std::cerr << "Unknown option: " << arg << "\n";
      return false;
    } else if (arg.find('=') != std::string::npos) {
      amrexOverrides.push_back(arg);
    } else if (inputs.empty()) {
      inputs = arg;
    } else if (plotfile.empty()) {
      plotfile = arg;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  return !(inputs.empty() || plotfile.empty());
}

auto BuildAmrexArgv(char **argv, const std::string &inputs,
                    const std::vector<std::string> &overrides,
                    std::vector<std::string> &amrexArgsStr,
                    std::vector<char *> &amrexArgs) -> void {
  amrexArgsStr.clear();
  amrexArgs.clear();
  amrexArgsStr.push_back(argv[0]);
  amrexArgsStr.push_back(inputs);
  for (auto const &arg : overrides) {
    amrexArgsStr.push_back(arg);
  }
  amrexArgs.reserve(amrexArgsStr.size() + 1);
  for (auto &arg : amrexArgsStr) {
    amrexArgs.push_back(arg.data());
  }
  amrexArgs.push_back(nullptr);
}

auto LoadMetadata(const std::string &plotfile) -> YAML::Node {
  YAML::Node metadata(YAML::NodeType::Map);
  std::string metadataPath = plotfile + "/metadata.yaml";
  if (std::filesystem::exists(metadataPath)) {
    metadata = YAML::LoadFile(metadataPath);
  } else {
    metadata["diagnostics_source_plotfile"] = plotfile;
    metadata["diagnostics_generated_by"] = "quokka_diag_from_plotfile";
  }
  return metadata;
}

} // namespace

// Offline specialization avoids the AMRSimulation dependency in DiagFramePlane.
template <>
void DiagFramePlane::processDiag<OfflineProblem>(int a_nstep,
                                                 const amrex::Real &a_time) {
  BL_PROFILE("DiagFramePlane::processDiag()");

  // Access diagnostic data through protected members
  auto const &a_state = *m_diagMF;
  auto const &simulationMetadata = *m_metadata;

  // Interpolate data to slice
  amrex::Vector<amrex::MultiFab> planeData(a_state.size());
  for (int lev = 0; lev < a_state.size(); ++lev) {
    planeData[lev].define(m_sliceBA[lev], m_sliceDM[lev],
                          static_cast<int>(m_fieldNames.size()), 0);
    auto const &geom = (*m_geoms)[lev];
    auto const problo = geom.ProbLoArray();
    auto const dx = geom.CellSizeArray();
    amrex::Real dist =
        (m_center[m_normal] - (problo[m_normal] + 0.5 * dx[m_normal])) /
        dx[m_normal];
    int p0 = static_cast<int>(std::round(dist));
    dist -= static_cast<amrex::Real>(p0);
    int const lo = geom.Domain().smallEnd(m_normal);
    int const hi = geom.Domain().bigEnd(m_normal);
    amrex::GpuArray<amrex::Real, 3> intwgt{0.0, 1.0, 0.0};
    if (p0 <= lo) {
      p0 = lo;
    } else if (p0 >= hi) {
      p0 = hi;
    } else {
      if (dist > 0.0) {
        intwgt[1] = 1.0 - dist;
        intwgt[2] = dist;
      } else if (dist < 0.0) {
        intwgt[0] = -dist;
        intwgt[1] = 1.0 + dist;
      }
      if (p0 - 1 < lo || p0 + 1 > hi) {
        intwgt = {0.0, 1.0, 0.0};
      }
    }
    int const p0m1 = (p0 - 1 < lo) ? lo : (p0 - 1);
    int const p0p1 = (p0 + 1 > hi) ? hi : (p0 + 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(planeData[lev], amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      const auto &bx = mfi.tilebox();
      const int state_idx = m_dmConvert[lev][mfi.index()];
      auto const &state = a_state[lev]->const_array(state_idx, 0);
      auto const &plane = planeData[lev].array(mfi);
      auto *idx_d_p = m_fieldIndices_d.dataPtr();
      if (m_normal == 0) {
        amrex::ParallelFor(
            bx, m_fieldNames.size(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
              int const stIdx = idx_d_p
                  [n]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
              plane(i, j, k, n) = intwgt[0] * state(p0m1, i, j, stIdx) +
                                  intwgt[1] * state(p0, i, j, stIdx) +
                                  intwgt[2] * state(p0p1, i, j, stIdx);
            });
      } else if (m_normal == 1) {
        amrex::ParallelFor(
            bx, m_fieldNames.size(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
              int const stIdx = idx_d_p
                  [n]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
              plane(i, j, k, n) = intwgt[0] * state(i, p0m1, j, stIdx) +
                                  intwgt[1] * state(i, p0, j, stIdx) +
                                  intwgt[2] * state(i, p0p1, j, stIdx);
            });
      } else if (m_normal == 2) {
        amrex::ParallelFor(
            bx, m_fieldNames.size(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
              int const stIdx = idx_d_p
                  [n]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
              plane(i, j, k, n) = intwgt[0] * state(i, j, p0m1, stIdx) +
                                  intwgt[1] * state(i, j, p0, stIdx) +
                                  intwgt[2] * state(i, j, p0p1, stIdx);
            });
      }
    }
  }

  // Count the number of level where the cut exists
  int nlevs = 0;
  for (int lev = 0; lev < a_state.size(); lev++) {
    if (!m_sliceBA[lev].empty()) {
      nlevs += 1;
    }
  }

  if (nlevs > 0) {
    // Build up a z-normal 2D Geom
    amrex::Vector<amrex::Geometry> pltGeoms(nlevs);
    pltGeoms[0] = m_geomLev0;
    amrex::Vector<amrex::IntVect> ref_ratio;
    amrex::IntVect const rref(AMREX_D_DECL(2, 2, 1));
    for (int lev = 1; lev < nlevs; ++lev) {
      pltGeoms[lev] = amrex::refine(pltGeoms[lev - 1], rref);
      ref_ratio.push_back(rref);
    }

    // File name based on step or time
    std::string diagfile;
    if (m_per > 0.0) {
      diagfile = m_diagfile + std::to_string(a_time);
    } else {
      diagfile = amrex::Concatenate(m_diagfile, a_nstep, 6);
    }
    amrex::Vector<int> const step_array(nlevs, a_nstep);
    Write2DMultiLevelPlotfile(diagfile, nlevs, GetVecOfConstPtrs(planeData),
                              m_fieldNames, pltGeoms, a_time, step_array,
                              ref_ratio, simulationMetadata);
  }
}

auto main(int argc, char **argv) -> int {
  std::string inputs;
  std::string plotfile;
  bool force = false;
  std::optional<int> stepOverride;
  std::optional<amrex::Real> timeOverride;
  std::vector<std::string> amrexOverrides;

  if (!ParseArgs(argc, argv, inputs, plotfile, force, stepOverride,
                 timeOverride, amrexOverrides)) {
    PrintUsage();
    return 1;
  }

  plotfile = stripTrailingSlash(plotfile);

  std::vector<std::string> amrexArgsStr;
  std::vector<char *> amrexArgs;
  BuildAmrexArgv(argv, inputs, amrexOverrides, amrexArgsStr, amrexArgs);
  int amrexArgc = static_cast<int>(amrexArgs.size()) - 1;

  int amrexArgcRef = amrexArgc;
  char **amrexArgvRef = amrexArgs.data();
  amrex::Initialize(amrexArgcRef, amrexArgvRef);
  int exitCode = 0;
  {
    amrex::Print() << "Loading plotfile: " << plotfile << "\n";
    amrex::PlotFileData pf(plotfile);

    const auto pfVars = pf.varNames();
    int finestLevel = pf.finestLevel();
    int nlevels = finestLevel + 1;

    amrex::Vector<amrex::Geometry> geoms(nlevels);
    amrex::Vector<amrex::BoxArray> grids(nlevels);
    amrex::Vector<amrex::DistributionMapping> dmaps(nlevels);

    amrex::Array<amrex::Real, AMREX_SPACEDIM> probLo = pf.probLo();
    amrex::Array<amrex::Real, AMREX_SPACEDIM> probHi = pf.probHi();
    amrex::RealBox realBox(probLo, probHi);
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 0)};

    for (int lev = 0; lev < nlevels; ++lev) {
      grids[lev] = pf.boxArray(lev);
      dmaps[lev] = pf.DistributionMap(lev);
      geoms[lev].define(pf.probDomain(lev), realBox, pf.coordSys(), isPeriodic);
    }

    amrex::Vector<amrex::IntVect> refRatio;
    refRatio.reserve(std::max(0, finestLevel));
    for (int lev = 0; lev < finestLevel; ++lev) {
      int rr = pf.refRatio(lev);
      refRatio.push_back(amrex::IntVect(AMREX_D_DECL(rr, rr, rr)));
    }

    amrex::ParmParse pp("quokka");
    int nDiags = pp.countval("diagnostics");
    if (nDiags <= 0) {
      amrex::Print() << "No diagnostics configured in inputs file.\n";
      amrex::Finalize();
      return 0;
    }

    amrex::Vector<std::string> diagNames(nDiags);
    for (int n = 0; n < nDiags; ++n) {
      pp.get("diagnostics", diagNames[n], n);
    }

    std::unordered_set<std::string> supportedTypes = {"DiagFramePlane",
                                                      "DiagPDF"};
#if AMREX_SPACEDIM == 3
    supportedTypes.insert("VolumeRender");
#endif
    std::vector<std::unique_ptr<DiagBase>> diagnostics;
    diagnostics.reserve(nDiags);

    for (auto const &diagName : diagNames) {
      std::string const diagPrefix = "quokka." + diagName;
      amrex::ParmParse ppd(diagPrefix);
      std::string diagType;
      ppd.get("type", diagType);

      if (!supportedTypes.contains(diagType)) {
        amrex::Print() << "Skipping diagnostic '" << diagName << "' (type "
                       << diagType << ") - not supported by offline tool.\n";
        continue;
      }

      auto diag = DiagBase::create(diagType);
      diag->init(diagPrefix, diagName);
      diagnostics.push_back(std::move(diag));
    }

    if (diagnostics.empty()) {
      amrex::Print() << "No supported diagnostics requested.\n";
      amrex::Finalize();
      return 0;
    }

    amrex::Vector<std::string> diagVars;
    for (auto &diag : diagnostics) {
      diag->addVars(diagVars);
    }

    std::ranges::sort(diagVars);
    auto last = std::ranges::unique(diagVars);
    diagVars.erase(last.begin(), last.end());

    for (auto const &var : diagVars) {
      if (std::find(pfVars.begin(), pfVars.end(), var) == pfVars.end()) {
        amrex::Abort("Required diagnostic variable missing from plotfile: " +
                     var);
      }
    }

    int const nGrow = 1;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> diagMFVec(nlevels);
    for (int lev = 0; lev < nlevels; ++lev) {
      diagMFVec[lev] = std::make_unique<amrex::MultiFab>(
          grids[lev], dmaps[lev], static_cast<int>(diagVars.size()), nGrow);
      for (int v = 0; v < diagVars.size(); ++v) {
        amrex::MultiFab src = pf.get(lev, diagVars[v]);
        amrex::MultiFab::Copy(*diagMFVec[lev], src, 0, v, 1, nGrow);
      }
      diagMFVec[lev]->FillBoundary(geoms[lev].periodicity());
    }

    amrex::Vector<const amrex::MultiFab *> diagMFVecPtr;
    diagMFVecPtr.reserve(nlevels);
    for (auto &mf : diagMFVec) {
      diagMFVecPtr.push_back(mf.get());
    }

    YAML::Node metadata = LoadMetadata(plotfile);

    int nstep = pf.levelStep(0);
    amrex::Real time = pf.time();
    if (stepOverride.has_value()) {
      nstep = stepOverride.value();
    }
    if (timeOverride.has_value()) {
      time = timeOverride.value();
    }

    amrex::Print() << "Using step=" << nstep << " time=" << time << "\n";

    for (auto &diag : diagnostics) {
      if (diag->needUpdate()) {
        diag->prepare(nlevels, geoms, grids, dmaps, diagVars);
      }

      diag->setDiagData<OfflineProblem>(nullptr, &diagMFVecPtr, &diagVars,
                                        &geoms, &refRatio, &metadata);

      bool willDo = force || diag->doDiag(time, nstep);
      if (!willDo) {
        amrex::Print() << "Skipping diagnostic (interval not matched). Use "
                          "--force to override.\n";
        continue;
      }

      if (auto *framePlane = dynamic_cast<DiagFramePlane *>(diag.get())) {
        if (!framePlane->getParticleTypes().empty()) {
          amrex::Print() << "DiagFramePlane requests particles; offline tool "
                            "ignores particle output.\n";
        }
        framePlane->processDiag<OfflineProblem>(nstep, time);
      } else if (auto *pdf = dynamic_cast<DiagPDF *>(diag.get())) {
        pdf->processDiag<OfflineProblem>(nstep, time);
      }
#if AMREX_SPACEDIM == 3
      else if (auto *volume = dynamic_cast<DiagVolumeRender *>(diag.get())) {
        volume->processDiag<OfflineProblem>(nstep, time);
      }
#endif
      else {
        amrex::Abort("Unsupported diagnostic type reached in offline tool.");
      }
    }
  }
  amrex::Finalize();
  return exitCode;
}
