/// Minimal test problem to validate WR and AGB continuous metal feedback.
/// Creates two star particles (one high-mass WR, one low-mass AGB), forces their
/// evolutionary stage, and runs a short simulation so `updateChemicalFeedback`
/// deposits WR/AGB yields to the gas.

#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"

struct ChuhanProblem_wind {};

constexpr Real gamma_ = 5. / 3.;
constexpr Real year = 3.15576e+07;
static Real n0 = 1.0e4; // NOLINT
static Real Tamb = 10.0; // NOLINT
static std::string initial_particles_file = "../inputs/ChuhanProblem_wind_particles.txt"; // NOLINT

template <> struct quokka::EOS_Traits<ChuhanProblem_wind> {
    static constexpr double gamma = gamma_;
    static constexpr double mean_molecular_weight = 1.0;
};

template <> struct Particle_Traits<ChuhanProblem_wind> {
    static constexpr ParticleSwitch particle_switch = ParticleSwitch::StochasticStellarPop;
};

template <> struct HydroSystem_Traits<ChuhanProblem_wind> {
    static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<ChuhanProblem_wind> {
    static constexpr bool is_self_gravity_enabled = false;
    static constexpr bool is_hydro_enabled = true;
    static constexpr bool is_radiation_enabled = false;
    static constexpr bool is_dust_enabled = false;
    static constexpr int nDustGroups = 1;
    static constexpr bool is_mhd_enabled = false;
    static constexpr int numMassScalars = 0;
    static constexpr int numPassiveScalars = 3; // track 3 isotopes by default
    static constexpr int nGroups = 1;
    static constexpr UnitSystem unit_system = UnitSystem::CGS;
    static constexpr double boltzmann_constant = C::k_B;
    static constexpr double gravitational_constant = C::Gconst;
    static constexpr double c_light = C::c_light;
    static constexpr double radiation_constant = C::a_rad;
};

template <> void QuokkaSimulation<ChuhanProblem_wind>::createInitialStochasticStellarPopParticles()
{
    const int nreal_extra = quokka::StochasticStellarPopParticleRealComps<ChuhanProblem_wind>;
    StochasticStellarPopParticles->SetVerbose(1);
    StochasticStellarPopParticles->InitFromAsciiFile(initial_particles_file, nreal_extra, nullptr);

    // Force particle metadata: particle 0 = high-mass (WR), particle 1 = low-mass (AGB)
    for (auto &kv : StochasticStellarPopParticles->GetParticles()) {
        for (auto &ikv : kv) {
            auto &particle_array = ikv.second.GetArrayOfStructs();
            const int np = particle_array.numParticles();
            if (np == 0) continue;
            auto *pdata = particle_array().data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
                // default: mark all as on the WR/AGB paths; we'll pick first as WR, second as AGB
                if (i == 0) {
                    pdata[i].idata(quokka::StochasticStellarPopParticleStageIdx) = static_cast<int>(quokka::StellarEvolutionStage::HighMassNonExploding);
                    // mass at birth ~ 20 Msun
                    pdata[i].rdata(quokka::StochasticStellarPopParticleMassAtBirthIdx) = 20.0 * C::M_solar;
                    pdata[i].rdata(quokka::StochasticStellarPopParticleBirthTimeIdx) = 0.0;
                    pdata[i].rdata(quokka::StochasticStellarPopParticleDeathTimeIdx) = 1.0e18; // very large so continuous WR active window covers it
                } else {
                    pdata[i].idata(quokka::StochasticStellarPopParticleStageIdx) = static_cast<int>(quokka::StellarEvolutionStage::LowMassComposite);
                    // mass at birth ~ 2 Msun
                    pdata[i].rdata(quokka::StochasticStellarPopParticleMassAtBirthIdx) = 2.0 * C::M_solar;
                    pdata[i].rdata(quokka::StochasticStellarPopParticleBirthTimeIdx) = 0.0;
                    pdata[i].rdata(quokka::StochasticStellarPopParticleDeathTimeIdx) = 1.0e18;
                }
            });
        }
    }

    amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<ChuhanProblem_wind>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
    const amrex::Box &indexRange = grid_elem.indexRange_;
    const amrex::Array4<double> &state_cc = grid_elem.array_;

    const double rho = n0 * 1.0;
    const double e_int = 1.0 / (gamma_ - 1.0) * rho * C::k_B * Tamb;

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        state_cc(i, j, k, HydroSystem<ChuhanProblem_wind>::density_index) = rho;
        state_cc(i, j, k, HydroSystem<ChuhanProblem_wind>::x1Momentum_index) = 0.0;
        state_cc(i, j, k, HydroSystem<ChuhanProblem_wind>::x2Momentum_index) = 0.0;
        state_cc(i, j, k, HydroSystem<ChuhanProblem_wind>::x3Momentum_index) = 0.0;
        state_cc(i, j, k, HydroSystem<ChuhanProblem_wind>::energy_index) = e_int;
        state_cc(i, j, k, HydroSystem<ChuhanProblem_wind>::internalEnergy_index) = e_int;
        for (int n = 0; n < Physics_Traits<ChuhanProblem_wind>::numPassiveScalars; ++n) {
            state_cc(i, j, k, HydroSystem<ChuhanProblem_wind>::scalar0_index + n) = 0.0;
        }
    });
}

auto problem_main() -> int
{
    QuokkaSimulation<ChuhanProblem_wind> sim;

    sim.reconstructionOrder_ = 3;
    sim.cflNumber_ = 0.5;
    sim.stopTime_ = 1.0e14; // short run sufficient for continuous deposition

    const int seed = 42;
    amrex::InitRandom(seed, 1);

    amrex::ParmParse const ppp("problem");
    ppp.query("Tamb", Tamb);
    ppp.query("n0", n0);
    ppp.query("initial_particles_file", initial_particles_file);

    sim.setInitialConditions();

    sim.evolve();
    amrex::Print() << "ChuhanProblem_wind completed\n";
    return 0;
}
