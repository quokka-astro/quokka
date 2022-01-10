/*
 *      ____                         __ __
 *     / __ \ ____ __ ___   __  __  / // /
 *    / /_/ // __// // _ \ /  \/ / / // /
 *    \____//_/  /_/ \___//_/\__/ /_//_/
 *
 *    Based on Pluto 3 and Chombo.
 *
 *    Please refer to COPYING in Pluto's root directory and
 *    Copyright.txt, in Chombo's root directory.
 *
 *    Modification: Orional code based on PLUTO3.0b2
 *    1) PS(09/25/08): Modified original code for CT scheme and include
 *                     turbulence driving.
 *    2) PS(12/03/08): Remove variable m_pump for turbulence driving by a
 *                     preproceesing flag PUMP.
 *    3) PS(12/04/08): Upgrade to PLUTO3 final release.
 *    4) PS(02/20/09): Use FluxBox class for face-centered magnetic field.
 *    5) PS(03/17/09): Add AMR coarsening and PiecewiseLinearFillFacePluto
 *                     for face-centered magnetic field.
 *    6) PS(03/16/09): Simplify checkpointing to read/write all concervative
 *                     cell- and face-centered state variables.
 *    7) PS(05/18/09): Add Dan Martin's code projectNewB and related codes
 *                     to ensure div(B)=0 in refined grids. A new MHDTools
 *                     is added in Chombo library.
 *                     Add a new fineInterpDrive for driving refinement.
 *    8) PS(07/16/09): Revised the driving subroutine to drive only at base
 *                     grid.
 *    9) PS(11/05/09): Revised Dan's B field correction in regridding.
 *                     Add U and Bs boundary update after fillpatching.
 *                     Add U boundary update after pumping.
 *   10) PS(01/05/10): Add divB calculation and divB dumpout for diagnostic.
 *   11) PS(08/06/10): Rename Pluto to Orion.
 *   12) PS(08/18/10): Output turbulence driving energy injection.
 *   13) PS(09/31/10): Modifiy driving function to all levels.
 *   14) PS(11/18/10): Add shear flow tagging scheme.
 *   15) PS(05/12/11): Implement gravity solver.
 */


#include "BoxIterator.H"
#include "LayoutIterator.H"
#include "ParmParse.H"

#include "AMRLevel.H"
#include "AMRLevelOrion.H"

#include "computeNorm.H"
#include "computeSum.H"

#include "Ancillae.H"
#include "ParallelHelper.H"

// PS: gravity
#ifdef GRAVITY

#define SOLVER_RESET_PHI false
#define SOLVER_NORM_TYPE (0) // AJC
#define SOLVER_MAX_ITER (50)
#define SOLVER_MIN_ITER (2)
#ifdef CH_USE_FLOAT
#define SOLVER_TOLERANCE (1.0e-4)
#else
#define SOLVER_TOLERANCE (1.0e-8)
#endif
//#undef  USE_RELAX_SOLVER //ALR-try new gravity solver
#define USE_RELAX_SOLVER // new gravity solver ALR
#define SOLVER_NUM_SMOOTH 8
#define SOLVER_NUM_MG 1
#define SOLVER_HANG 1.e-14
// SOLVER_NORM_THRES is L_\infty norm of the absolute error phi in units of 4pi
// G
#define SOLVER_NORM_THRES (1.e-5 * 4. * CONST_PI * CONST_G * SMALL_DN)
//#include "SelfGravityPhysics.H"
#include "AMRMultiGrid.H"
#include "AMRPoissonOp.H"
#include "BCFunc.H"
#include "BiCGStabSolver.H"
#include "RelaxSolver.H" //newgravity solver header ALR

int s_verbosity = 3;

// this is a dummy function for periodic BCs
void NoOpBc(FArrayBox &a_state, const Box &a_box, const ProblemDomain &a_domain,
            Real a_dx, bool a_homogeneous) {}

class GlobalBCRS {
public:
  static std::vector<bool> s_printedThatLo, s_printedThatHi;
  static std::vector<int> s_bcLo, s_bcHi;
  static RealVect s_trigvec;
  static bool s_areBCsParsed, s_valueParsed, s_trigParsed;
};

std::vector<bool> GlobalBCRS::s_printedThatLo =
    std::vector<bool>(SpaceDim, false);
std::vector<bool> GlobalBCRS::s_printedThatHi =
    std::vector<bool>(SpaceDim, false);
std::vector<int> GlobalBCRS::s_bcLo = std::vector<int>();
std::vector<int> GlobalBCRS::s_bcHi = std::vector<int>();
RealVect GlobalBCRS::s_trigvec = RealVect::Zero;
bool GlobalBCRS::s_areBCsParsed = false;
bool GlobalBCRS::s_valueParsed = false;
bool GlobalBCRS::s_trigParsed = false;

void ParseValue(Real *pos, int *dir, Side::LoHiSide *side, Real *a_values) {
  ParmParse pp;
  Real bcVal;
  pp.get("bc_value", bcVal);
  a_values[0] = bcVal; // always set to zero in orion2.ini for gravity!
}

void ParseBC(FArrayBox &a_state, const Box &a_valid,
             const ProblemDomain &a_domain, Real a_dx, bool a_homogeneous) {

  if (!a_domain.domainBox().contains(a_state.box())) {

    if (!GlobalBCRS::s_areBCsParsed) {
      ParmParse pp;
      pp.getarr("bc_lo", GlobalBCRS::s_bcLo, 0, SpaceDim);
      pp.getarr("bc_hi", GlobalBCRS::s_bcHi, 0, SpaceDim);
      GlobalBCRS::s_areBCsParsed = true;
    }

    Box valid = a_valid;
    for (int i = 0; i < CH_SPACEDIM; ++i) {
      // don't do anything if periodic
      if (!a_domain.isPeriodic(i)) {
        Box ghostBoxLo = adjCellBox(valid, i, Side::Lo, 1);
        Box ghostBoxHi = adjCellBox(valid, i, Side::Hi, 1);
        if (!a_domain.domainBox().contains(ghostBoxLo)) {
          if (GlobalBCRS::s_bcLo[i] == 1) {
            if (!GlobalBCRS::s_printedThatLo[i]) {
              if (a_state.nComp() != 1) {
                MayDay::Error("using scalar bc function for vector");
              }
              GlobalBCRS::s_printedThatLo[i] = true;
              if (s_verbosity > 5)
                pout() << "const neum bcs lo for direction " << i << endl;
            }
            NeumBC(a_state, valid, a_dx, a_homogeneous, ParseValue, i,
                   Side::Lo);
          } else if (GlobalBCRS::s_bcLo[i] == 0) {
            if (!GlobalBCRS::s_printedThatLo[i]) {
              if (a_state.nComp() != 1) {
                MayDay::Error("using scalar bc function for vector");
              }
              GlobalBCRS::s_printedThatLo[i] = true;
              if (s_verbosity > 5)
                pout() << "const diri bcs lo for direction " << i << endl;
            }
            DiriBC(a_state, valid, a_dx, a_homogeneous, ParseValue, i, Side::Lo,
                   1);
          } else {
            MayDay::Error("bogus bc flag lo");
          }
        }

        if (!a_domain.domainBox().contains(ghostBoxHi)) {
          if (GlobalBCRS::s_bcHi[i] == 1) {
            if (!GlobalBCRS::s_printedThatHi[i]) {
              if (a_state.nComp() != 1) {
                MayDay::Error("using scalar bc function for vector");
              }
              GlobalBCRS::s_printedThatHi[i] = true;
              if (s_verbosity > 5)
                pout() << "const neum bcs hi for direction " << i << endl;
            }
            NeumBC(a_state, valid, a_dx, a_homogeneous, ParseValue, i,
                   Side::Hi);
          } else if (GlobalBCRS::s_bcHi[i] == 0) {
            if (!GlobalBCRS::s_printedThatHi[i]) {
              if (a_state.nComp() != 1) {
                MayDay::Error("using scalar bc function for vector");
              }
              GlobalBCRS::s_printedThatHi[i] = true;
              if (s_verbosity > 5)
                pout() << "const diri bcs hi for direction " << i << endl;
            }
            DiriBC(a_state, valid, a_dx, a_homogeneous, ParseValue, i, Side::Hi,
                   1);
          } else {
            MayDay::Error("bogus bc flag hi");
          }
        }
      } // end if is not periodic in ith direction
    }
  }
}
#endif /* GRAVITY */

// Constructor
AMRLevelOrion::AMRLevelOrion() {
  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion default constructor" << endl;
  }

  m_cfl = 0.8;
  m_domainLength = 1.0;
  m_initial_dt_multiplier = 1.e-4;
  m_patchOrionFactory = NULL;
  m_patchOrion = NULL;
  // PS: gravity
#ifdef GRAVITY
  m_gradient = NULL;
  m_paramsDefined = false;
#endif /* GRAVITY */
}

// Destructor
AMRLevelOrion::~AMRLevelOrion() {
  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion destructor" << endl;
  }

  // Get rid of the patch integrators (factory and object)
  if (m_patchOrionFactory != NULL) {
    delete m_patchOrionFactory;
  }

  if (m_patchOrion != NULL) {
    delete m_patchOrion;
  }

  // PS: gravity
#ifdef GRAVITY
  if (m_gradient != NULL) {
    delete m_gradient;
  }
#endif /* GRAVITY */
}

// Define new AMR level
void AMRLevelOrion::define(AMRLevel *a_coarserLevelPtr,
                           const ProblemDomain &a_problemDomain, int a_level,
                           int a_refRatio) {
  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::define " << a_level << endl;
  }

  // Call inherited define
  AMRLevel::define(a_coarserLevelPtr, a_problemDomain, a_level, a_refRatio);

  // Get setup information from the next coarser level
  if (a_coarserLevelPtr != NULL) {
    AMRLevelOrion *amrGodPtr = dynamic_cast<AMRLevelOrion *>(a_coarserLevelPtr);

    if (amrGodPtr != NULL) {
      m_cfl = amrGodPtr->m_cfl;
      m_domainLength = amrGodPtr->m_domainLength;
      m_tagBufferSize = amrGodPtr->m_tagBufferSize;
    } else {
      MayDay::Error("AMRLevelOrion::define: a_coarserLevelPtr is not castable "
                    "to AMRLevelOrion*");
    }
  }

  // Compute the grid spacing
  m_dx = m_domainLength / a_problemDomain.domainBox().longside();

  // Nominally, one layer of ghost cells is maintained permanently and
  // individual computations may create local data with more
  m_numGhost = GET_NGHOST(NULL);

  // PS: gravity
#ifdef GRAVITY
  m_numRhsGhost = 1;
#endif /* GRAVITY */
#if defined(GRAVITY) || defined(SINKPARTICLE)
  m_numForceGhost = 2;
#endif // GRAVITY || SINKPARTICLE

  // Remove old patch integrator (if any), create a new one, and initialize
  if (m_patchOrion != NULL) {
    delete m_patchOrion;
  }

  m_patchOrion = m_patchOrionFactory->new_patchOrion();
  m_patchOrion->define(m_problem_domain, m_dx, m_level, m_numGhost);

  // Get additional information from the patch integrator
  m_numStates = m_patchOrion->numConserved();
  m_stateNames = m_patchOrion->stateNames();

  // PS: self gravity
#ifdef GRAVITY
  // Set the stencil for differentiating the grav. potential:
  if (m_gradient != NULL) {
    delete m_gradient;
    m_gradient = NULL;
  }

  m_gradient = new TwoPtsGradient();
  m_numPhiGhost = 1;
  const bool a_useDeltaPhiCorr = true;

  // whether or not the deltaPhi correction should be applied

  m_useDeltaPhiCorr = a_useDeltaPhiCorr;
  m_paramsDefined = true;

#endif /* GRAVITY */
}

// Advance by one timestep
Real AMRLevelOrion::advance() {
  CH_assert(isDefined());

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::advance level " << m_level << " to time "
           << m_time + m_dt << endl;
  }

  // Copy the new to the old

  DataIterator dit = m_UNew.dataIterator();
  for (; dit.ok(); ++dit) {
    m_UOld[dit()].copy(m_UNew[dit()]);
  }

  Real newDt = 0.0;

  // Set up arguments to LevelOrion::step based on whether there are
  // coarser and finer levels
  // Undefined flux register in case we need it
  LevelFluxRegister dummyFR;

  // Undefined leveldata in case we need it
  const LevelData<FArrayBox> dummyData;

  // Set arguments to dummy values and then fix if real values are available
  LevelFluxRegister *coarserFR = &dummyFR;
  LevelFluxRegister *finerFR = &dummyFR;

  const LevelData<FArrayBox> *coarserDataOld = &dummyData;
  const LevelData<FArrayBox> *coarserDataNew = &dummyData;

  Real tCoarserOld = 0.0;
  Real tCoarserNew = 0.0;

  // we don't need the flux in the simple hyperbolic case...
  LevelData<FArrayBox> flux[SpaceDim];

  // PS: self gravity
#ifdef GRAVITY
  CH_TIMERS("AMRLevelOrion::advance_grav");
  CH_TIMER("AMRLevelOrion::advance_grav::elliptic_solve", Adv_Ell_Solve);

  // save new Phi -> old Phi
  // save new Phi/Force -> old Phi/Force
  CH_START(Adv_Ell_Solve);
  for (DataIterator di = m_phiNew.dataIterator(); di.ok(); ++di) {
    m_phiOld[di].copy(m_phiNew[di]);
    m_forceOld[di].copy(m_forceNew[di]);
  }

  if (m_hasFiner && !m_isThisFinestLev) {
    ellipticSolver(m_level, true);
    computeForce(m_forceNew, m_phiNew, m_time);
  }

  m_dtOld = m_dt;
  CH_STOP(Adv_Ell_Solve);

  flag = checkState(0, 3);
  if (flag != 0) {
    // pout() << "Warning! Bad state detected after Gravity update" << endl;
    if (flag == 2) {
      pout() << "Warning! NAN detected after Gravity update" << endl;
      MayDay::Abort("Encountered NAN after Gravity update");
    }
    // if (flag >= 3)
    //  pout() << "Warning! Bad state detected after Gravity update. flag = " <<
    //  flag << endl;
  }
#endif /* GRAVITY */

  // Update the time and store the new timestep
  m_time += m_dt;
  Real returnDt = m_cfl * newDt;

  m_dtNew = returnDt;

  return returnDt;
}

// Things to do after a timestep
void AMRLevelOrion::postTimeStep() {
  CH_TIMERS("AMRLevelOrion::postTimeStep");
  CH_TIMER("AMRLevelOrion::postTimeStep::reflux", timeReflux);
  CH_TIMER("AMRLevelOrion::postTimeStep::drive", timeDrive);
  CH_TIMER("AMRLevelOrion::postTimeStep::gravity", timeGravity);

  CH_assert(isDefined());

  int flag;

  // Used for conservation tests
  static Real orig_integral = 0.0;
  static Real last_integral = 0.0;
  static bool first = true;

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::postTimeStep " << m_level << endl;
  }

  // PS: self gravity
#ifdef GRAVITY
  CH_START(timeGravity);
  // if there is a force term, first we solve for the potential
  // (Poisson eq.) on all levels of the grid hierarchy that are
  // aligned in time. This is done by gravity() which also calls, in
  // the proper order, functions that apply 2nd order corrections and
  // compute the force field taking into account interlevel and domain
  // boundary conditions. After all this has been taken care of, if
  // necessary, particles are reassigned to the a new grid patch and
  // hierarchy level by manageParticles(). If no force term is present
  // only manageParticles() needs to be called.
  const Real eps = 5.e-2;
  Real crseTime = -one;
  if (m_level > 0)
    crseTime = m_coarser_level_ptr->time();
  const bool stepsLeft = abs(crseTime - m_time) > (eps * m_dt);

  if (m_level == 0 || stepsLeft)
    gravity(m_level);

  flag = checkState(0, 10);
  if (flag != 0) {
    // pout() << "Warning! Bad state detected after gravity fine coarsening" <<
    // endl;
    if (flag == 2) {
      pout() << "Warning! NAN detected after gravity fine coarsening" << endl;
      MayDay::Abort("Encountered NAN after gravity fine coarsening");
    }
    // if (flag >= 3)
    //  pout() << "Warning! Bad state detected after gravity fine coarsening.
    //  flag = " << flag << endl;
  }

  CH_STOP(timeGravity);
#endif

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::postTimeStep " << m_level << " finished" << endl;
  }
}

// Set up data on this level after regridding
void AMRLevelOrion::regrid(const Vector<Box> &a_newGrids) {
  CH_TIMERS("AMRLevelOrion::regrid");

  CH_TIMER("AMRLevelOrion::regrid::loadbalance", timeloadBalance);
  CH_TIMER("AMRLevelOrion::regrid::saveData", timesaveData);
  CH_TIMER("AMRLevelOrion::regrid::newgrids", timenewgrids);
  CH_TIMER("AMRLevelOrion::regrid::Interpolate", timeInterpolate);
  CH_TIMER("AMRLevelOrion::regrid::copystate", timecopystate);
  CH_TIMER("AMRLevelOrion::regrid::projectNewB", timeProjectNewB);
  CH_TIMER("AMRLevelOrion::regrid::conclude", timeConclude);

  CH_assert(isDefined());

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::regrid " << m_level << endl;
  }

  CH_START(timeloadBalance);

  // Save original grids and load balance
  m_level_grids = a_newGrids;
  m_grids = loadBalance(a_newGrids);

  CH_STOP(timeloadBalance);

  if (s_verbosity >= 4) {
    // Indicate/guarantee that the indexing below is only for reading
    // otherwise an error/assertion failure occurs

    const DisjointBoxLayout &constGrids = m_grids;

    pout() << "new grids: " << endl;

    for (LayoutIterator lit = constGrids.layoutIterator(); lit.ok(); ++lit) {
      pout() << constGrids[lit()] << endl;
    }
  }

  CH_START(timesaveData);

  // Save data for later

  DataIterator dit = m_UNew.dataIterator();
  for (dit.reset(); dit.ok(); ++dit) {
    m_UOld[dit()].copy(m_UNew[dit()]);
  }

  CH_STOP(timesaveData);

  CH_START(timenewgrids);

  // Reshape state with new grids

  IntVect ivGhost = m_numGhost * IntVect::Unit;
  m_UNew.define(m_grids, m_numStates, ivGhost);

  // PS: self gravity
#ifdef GRAVITY
  for (DataIterator di = m_phiNew.dataIterator(); di.ok(); ++di) {
    m_phiOld[di].copy(m_phiNew[di]);
  }
  m_phiNew.define(m_grids, 1, m_numPhiGhost * IntVect::Unit);
  resetToZero(m_phiNew);

  if (m_useDeltaPhiCorr) {
    m_deltaPhi.define(m_grids, 1, m_numPhiGhost * IntVect::Unit);
    resetToZero(m_deltaPhi);
  }

  m_forceNew.define(m_grids, SpaceDim, m_numForceGhost * IntVect::Unit);
  m_forceOld.define(m_grids, SpaceDim, m_numForceGhost * IntVect::Unit);

  m_rhs.define(m_grids, 1, m_numRhsGhost * IntVect::Unit);
  resetToZero(m_rhs);
#endif

  // Set up data structures
  levelSetup();

  CH_STOP(timenewgrids);

  CH_START(timeInterpolate);

  // Interpolate from coarser level

  if (m_hasCoarser) {
    AMRLevelOrion *amrGodCoarserPtr = getCoarserLevel();

#if DOEINTAVE != YES
    m_fineInterp.interpToFine(m_UNew, amrGodCoarserPtr->m_UNew);
#else
    m_fineInterp.interpToFineEInt(m_UNew, amrGodCoarserPtr->m_UNew, m_dx);
#endif

    // PS: self gravity
#ifdef GRAVITY
    m_fineInterpPhi.interpToFine(m_phiNew, amrGodCoarserPtr->m_phiNew);
#endif
  }

  CH_STOP(timeInterpolate);

  CH_START(timecopystate);

  // Copy from old state

  m_UOld.copyTo(m_UOld.interval(), m_UNew, m_UNew.interval());

  CH_STOP(timecopystate);

  int flag = checkState(0, 13);
  if (flag != 0) {
    if (flag == 2) {
      pout() << "Warning! NAN detected after grid init" << endl;
      MayDay::Abort("Encountered NAN after grid init");
    }

    // if (flag >= 3)
    //  pout() << "Warning! Bad state detected after grid init. flag = " << flag
    //  << " m_level = " << m_level <<  endl;
  }

  CH_START(timeConclude);

  // PS: self gravity
#ifdef GRAVITY
  m_phiOld.copyTo(m_phiOld.interval(), m_phiNew, m_phiNew.interval());
  m_phiOld.define(m_grids, 1, m_numPhiGhost * IntVect::Unit);
  dit = m_phiOld.dataIterator();
  for (; dit.ok(); ++dit)
    m_phiOld[dit()].copy(m_phiNew[dit()]);
#endif

  flag = checkState(0, 14);
  if (flag != 0) {
    if (flag == 2) {
      pout() << "Warning! NAN detected after divb regridding" << endl;
      MayDay::Abort("Encountered NAN after regridding");
    }

    // if (flag >= 3)
    //  pout() << "Warning! Bad state detected after divB regridding. flag = "
    //  << flag << " m_level = " << m_level <<  endl;
  }

  CH_STOP(timeConclude);
}

#ifdef GRAVITY
/*PS: Add postRegrid to ORION2 */
void AMRLevelOrion::postRegrid(int a_base_level) {
  CH_TIMERS("AMRLevelOrion::postRegrid");

  CH_TIMER("AMRLevelOrion::postRegrid::all", timeAll);

  CH_START(timeAll);

  if (s_verbosity >= 3) {
    pout() << " AMRLevelOrion::postRegrid " << m_level << " base level "
           << a_base_level << endl;
  }

  if (m_hasFiner) {
    // define finest level
    AMRLevelOrion *amrGodFinerPtr = getFinerLevel();
    m_isThisFinestLev = (amrGodFinerPtr->m_grids.size() <= 0);
  }

  if (m_level == a_base_level + 1) {
    // define the new finest level
    AMRLevelOrion *amrCoarserPtr = getCoarserLevel();
    amrCoarserPtr->isThisFinestLev(!m_grids.size() > 0);

    // in AMRC::timestep() we make sure that regrid() is called only
    // after postTimeStep() has been executed for all synchronized
    // levels. Thus this regrid() was called from the coarsest among
    // the synchronized levels, which, however, is not necessarily the
    // same as a_base_level ! In any case, this means that at this
    // point all regridding operations have been carried out.  Since
    // the grid hierarchy has changed, we now need to call gravity()
    // to compute the new multilevel gravitational potential. That
    // call must be made from the coarsest among the synchronized
    // levels: so first task is to climb down the ranks and find out
    // that level. But note that if m_level==1, the level we look for
    // can only be a_base_level=0.

    // we now we are in a_base_level+1;
    int crsestSynchLev = m_level - 1;
    AMRLevelOrion *amrSynchLevPtr = amrCoarserPtr;

    if (m_level > 1) {
      // look for the right level
      const Real eps = 5.e-2;
      const Real smallT = eps * m_dt;
      AMRLevelOrion *amrCoarserPtr = amrSynchLevPtr->getCoarserLevel();

      while (crsestSynchLev > 0) {
        const Real dTimeLevs = amrSynchLevPtr->time() - amrCoarserPtr->time();
        if (abs(dTimeLevs) > smallT)
          break;

        // need to update allFinerPcls for all synchronized levels
        --crsestSynchLev;

        amrSynchLevPtr = amrCoarserPtr;
        amrCoarserPtr = amrSynchLevPtr->getCoarserLevel();
      }
    }

    // kind of need to call gravity() from the synch lev because of
    // reference to data of the base level there.
    const bool srceCorr = false;
    amrSynchLevPtr->gravity(crsestSynchLev, srceCorr);
  }

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::postRegrid done " << endl;
  }
  CH_STOP(timeAll);
}
#endif /*GRAVITY */

// Initialize grids
void AMRLevelOrion::initialGrid(const Vector<Box> &a_newGrids) {
  CH_assert(isDefined());

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::initialGrid " << m_level << endl;
  }

  // Save original grids and load balance
  m_level_grids = a_newGrids;
  m_grids = loadBalance(a_newGrids);

  if (s_verbosity >= 4) {
    // Indicate/guarantee that the indexing below is only for reading
    // otherwise an error/assertion failure occurs
    const DisjointBoxLayout &constGrids = m_grids;

    pout() << "new grids: " << endl;
    for (LayoutIterator lit = constGrids.layoutIterator(); lit.ok(); ++lit) {
      pout() << constGrids[lit()] << endl;
    }
  }

  // Define old and new state data structures
  IntVect ivGhost = m_numGhost * IntVect::Unit;
  m_UNew.define(m_grids, m_numStates, ivGhost);
  m_UOld.define(m_grids, m_numStates, ivGhost);

#ifdef STAGGERED_MHD
  m_BsNew.define(m_grids, 1, ivGhost);
  m_BsOld.define(m_grids, 1, ivGhost);
#endif

#ifdef PUMP
  // Turbulence driving: define driving arrays
  m_DriveNew.define(m_grids, 3, ivGhost);
  m_DriveOld.define(m_grids, 3, ivGhost);
#endif /* PUMP */

#ifdef SINKPARTICLE
  m_sinkAccel.define(m_grids, SpaceDim, m_numForceGhost * IntVect::Unit);
#endif

  // PS: self gravity
#ifdef GRAVITY
  m_phiNew.define(m_grids, 1, m_numPhiGhost * IntVect::Unit);
  m_phiOld.define(m_grids, 1, m_numPhiGhost * IntVect::Unit);
  resetToZero(m_phiNew);

  if (m_useDeltaPhiCorr) {
    m_deltaPhi.define(m_grids, 1, ivGhost);
    resetToZero(m_deltaPhi);
  }

  m_forceNew.define(m_grids, SpaceDim, m_numForceGhost * IntVect::Unit);
  m_forceOld.define(m_grids, SpaceDim, m_numForceGhost * IntVect::Unit);

  const IntVect ivRhsGhost = m_numRhsGhost * IntVect::Unit;
  m_rhs.define(m_grids, 1, ivRhsGhost);
  resetToZero(m_rhs);
#endif /* GRAVITY */

  // Set up data structures
  levelSetup();
}

// Initialize data
void AMRLevelOrion::initialData() {
  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::initialData " << m_level << endl;
  }

  // Initialize data.  For driving turbulence, initialize turbulence driving.

  // ATM we need this if using gravity with periodic boundary conditions
#ifdef GRAVITY
  if (getCoarserLevel() == NULL) {
    Real sumRHS = computeSum(m_UNew, NULL, 1, m_dx, Interval(0, 0));
    Real domainVol = pow(m_domainLength, SpaceDim);
    m_globalMeanDensity = sumRHS / domainVol;
  }
#endif // GRAVITY
}

// Things to do after initialization
void AMRLevelOrion::postInitialize() {
  CH_assert(isDefined());

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::postInitialize " << m_level << endl;
  }

#ifdef GRAVITY
  if (m_hasFiner) {
    AMRLevelOrion *amrGodFinerPtr = getFinerLevel();
    m_isThisFinestLev = (amrGodFinerPtr->m_grids.size() <= 0);
  }

  if (!m_isThisFinestLev) {
    AMRLevelOrion *amrGodFinerPtr = getFinerLevel();
    amrGodFinerPtr->m_coarseAverage.averageToCoarse(m_UNew,
                                                    amrGodFinerPtr->m_UNew);
  }
  const bool srceCorr = false;
  if (m_level == 0)
    gravity(m_level, srceCorr);
#endif /* GRAVITY */
}

// Setup menagerie of data structures
void AMRLevelOrion::levelSetup() {
  CH_TIMERS("AMRLevelOrion::levelSetup");

  CH_TIMER("AMRLevelOrion::levelSetup::define", timedefine);
  CH_TIMER("AMRLevelOrion::levelSetup::fluxRegister", timefluxRegister);
  CH_TIMER("AMRLevelOrion::levelSetup::FC_fluxRegister", timeFC_fluxRegister);

  CH_assert(isDefined());

  DisjointBoxLayout finerDomain;

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::levelSetup " << m_level << endl;
  }

  AMRLevelOrion *amrGodCoarserPtr = getCoarserLevel();
  AMRLevelOrion *amrGodFinerPtr = getFinerLevel();

  m_hasCoarser = (amrGodCoarserPtr != NULL);
  m_hasFiner = (amrGodFinerPtr != NULL);

#ifdef GRAVITY
  // this will be finally set in postInitialize or postRegrid because
  // I can't really tell till all levels have been defined
  m_isThisFinestLev = true;
#endif

  // Mark split/unsplit cells
  int locGhost = GET_NGHOST(NULL);
  m_split_tags.define(m_grids, 1, IntVect::Zero);
  m_cover_tags.define(m_grids, 1, locGhost * IntVect::Unit);

#ifdef SKIP_SPLIT_CELLS
  for (DataIterator dit = m_grids.dataIterator(); dit.ok(); ++dit) {
    m_split_tags[dit()].setVal(1.);
    m_cover_tags[dit()].setVal(1.);
  }
#endif

  if (m_hasCoarser) {
    int nRefCrse = m_coarser_level_ptr->refRatio();

    m_coarseAverage.define(m_grids, m_numStates, nRefCrse);

    m_fineInterp.define(m_grids, m_numStates, nRefCrse, m_problem_domain);
    // ATM

#ifdef PUMP
    m_fineInterpDrive.define(m_grids, 3, nRefCrse, m_problem_domain);
#endif

    // PS: AMR for face-centered box
    m_coarseAverageFace.define(m_grids, 1, nRefCrse);

    m_fineInterpFace.define(m_grids, 1, nRefCrse, m_problem_domain);

    CH_START(timedefine);

    const DisjointBoxLayout &coarserLevelDomain = amrGodCoarserPtr->m_grids;

#ifdef SKIP_SPLIT_CELLS
    // Mark split/unsplit cells of the coarser level
    amrGodCoarserPtr->mark_split(m_grids);
#endif

    // Maintain levelOrion
    m_levelOrion.define(m_grids, coarserLevelDomain, m_problem_domain, nRefCrse,
                        m_level, m_dx, m_patchOrionFactory, m_hasCoarser,
                        m_hasFiner);

    // PS: self gravity
#ifdef GRAVITY
    const int numComp = 1;
    m_coarseAveragePhi.define(m_grids, numComp, nRefCrse);

    m_fineInterpPhi.define(m_grids, numComp, nRefCrse, m_problem_domain);

    // Maintain QuadCFInterp
    m_quadCFInterp.define(m_grids, &coarserLevelDomain, m_dx, nRefCrse, numComp,
                          m_problem_domain);

    m_forcePatcher.define(m_grids, coarserLevelDomain, SpaceDim,
                          amrGodCoarserPtr->m_problem_domain,
                          amrGodCoarserPtr->m_ref_ratio, m_numForceGhost);
#endif /* GRAVITY */

    CH_STOP(timedefine);

    // This may look twisted but you have to do this this way because the
    // coarser levels get setup before the finer levels so, since a flux
    // register lives between this level and the next FINER level, the finer
    // level has to do the setup because it is the only one with the
    // information at the time of construction.

    CH_START(timefluxRegister);

    // Maintain flux registers
    amrGodCoarserPtr->m_fluxRegister.define(
        m_grids, amrGodCoarserPtr->m_grids, m_problem_domain,
        amrGodCoarserPtr->m_ref_ratio, m_numStates);
    amrGodCoarserPtr->m_fluxRegister.setToZero();

    CH_STOP(timefluxRegister);

  } else {
    m_levelOrion.define(m_grids, DisjointBoxLayout(), m_problem_domain,
                        m_ref_ratio, m_level, m_dx, m_patchOrionFactory,
                        m_hasCoarser, m_hasFiner);
  }
}

// Get the next coarser level
AMRLevelOrion *AMRLevelOrion::getCoarserLevel() const {
  CH_assert(isDefined());

  AMRLevelOrion *amrGodCoarserPtr = NULL;

  if (m_coarser_level_ptr != NULL) {
    amrGodCoarserPtr = dynamic_cast<AMRLevelOrion *>(m_coarser_level_ptr);

    if (amrGodCoarserPtr == NULL) {
      MayDay::Error("AMRLevelOrion::getCoarserLevel: dynamic cast failed");
    }
  }

  return amrGodCoarserPtr;
}

// Get the next finer level
AMRLevelOrion *AMRLevelOrion::getFinerLevel() const {
  CH_assert(isDefined());

  AMRLevelOrion *amrGodFinerPtr = NULL;

  if (m_finer_level_ptr != NULL) {
    amrGodFinerPtr = dynamic_cast<AMRLevelOrion *>(m_finer_level_ptr);

    if (amrGodFinerPtr == NULL) {
      MayDay::Error("AMRLevelOrion::getFinerLevel: dynamic cast failed");
    }
  }

  return amrGodFinerPtr;
}

// PS: self gravity
#ifdef GRAVITY
// return pointer to
LevelData<FArrayBox> *AMRLevelOrion::getPhi(const Real &a_time) {
  if (s_verbosity >= 3) {
    pout() << " AMRLevelOrion::getPhi " << m_level << " a_time= " << a_time
           << " m_time " << m_time << endl;
  }
  LevelData<FArrayBox> *phiPtr = NULL;

  const Real eps = 0.01 * m_dt;
  if (Abs(a_time - m_time) <= eps) // case alpha=1; new synchronization point
  {
    phiPtr = &m_phiNew;
  } else if (Abs(a_time - (m_time - m_dtOld)) <
             eps) // case alpha=0; old synch point
  {
    phiPtr = &m_phiOld;
  } else {
    // define phiInt
    m_phiInt.define(m_grids, 1, m_numPhiGhost * IntVect::Unit);
    // m_phiInt.define(m_grids,1);

    // need time interpolation
    Real alpha = (a_time - (m_time - m_dtOld)) / m_dtOld;

    interpolateInTime(m_phiInt, m_phiOld, m_phiNew, a_time, m_time,
                      m_time - m_dtOld, m_dt, Interval(0, 0), Interval(0, 0));

    // add deltaPhi
    if (m_useDeltaPhiCorr && !m_isThisFinestLev) {
      for (DataIterator di = m_phiInt.dataIterator(); di.ok(); ++di) {
        m_phiInt[di].plus(m_deltaPhi[di], alpha);
      }
    }

    phiPtr = &m_phiInt;
  }

  if (phiPtr == NULL) {
    MayDay::Error("AMRLevelOrion::getPhi: something failed");
  }

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion:: getPhi done " << endl;
  }
  return phiPtr;
}

// set boolean stating whether or not this is the finest lev.
void AMRLevelOrion::isThisFinestLev(const bool a_isThisFinestLev) {
  m_isThisFinestLev = a_isThisFinestLev;
}

// return boolean stating whether or not this is the finest lev.
bool AMRLevelOrion::isThisFinestLev() const { return m_isThisFinestLev; }

bool AMRLevelOrion::allDefined() const {
  return isDefined() && m_paramsDefined;
}
#endif /* GRAVITY */

// PS: self gravity
#ifdef GRAVITY
LevelData<FArrayBox> *AMRLevelOrion::getPoissonRhs(const Real &a_time) {
  if (s_verbosity >= 3) {
    pout() << " AMRLevelOrion:: getPoissonRhs " << m_level << endl;
  }
  LevelData<FArrayBox> *poissonRhs = NULL;

  makePoissonRhs(m_rhs, m_UNew, a_time);

  poissonRhs = &m_rhs;

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::getPoissonRhs done " << endl;
  }
  return poissonRhs;
}
#endif /* GRAVITY */

// PS: self gravity
#ifdef GRAVITY
void AMRLevelOrion::gravity(const int a_baseLevel, const bool a_srceCorr) {
  CH_assert(allDefined());

  if (s_verbosity >= 3) {
    pout() << " AMRLevelOrion::gravity start: baseLevel = " << a_baseLevel
           << endl;
  }

  ellipticSolver(a_baseLevel, false);

  // need to do the two calls in this order, to ensure that boundary
  // condiions can be properly applied
  if (a_srceCorr) {
    // applies 2nd order corrections and compute the new force
    secondOrderCorrection();

    // same thing for finer levels
    AMRLevelOrion *thisSelfGravityPtr = this;
    while (!thisSelfGravityPtr->isThisFinestLev()) {
      thisSelfGravityPtr = thisSelfGravityPtr->getFinerLevel();

      thisSelfGravityPtr->secondOrderCorrection();
    }
  } else {
    // compute the new force
    computeForce(m_forceNew, m_phiNew, m_time);

    // same thing for finer levels
    AMRLevelOrion *thisSelfGravityPtr = this;
    while (!thisSelfGravityPtr->isThisFinestLev()) {
      thisSelfGravityPtr = thisSelfGravityPtr->getFinerLevel();

      thisSelfGravityPtr->computeForce(thisSelfGravityPtr->m_forceNew,
                                       thisSelfGravityPtr->m_phiNew, m_time);
    }
  }

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::gravity done " << endl << endl;
  }
}
#endif /* GRAVITY */

#ifdef GRAVITY

void AMRLevelOrion::ellipticSolver(const int a_baseLevel,
                                   const bool a_isLevelSolve) {
  CH_assert(allDefined());

  if (s_verbosity >= 3) {
    pout() << " AMRLevelOrion::ellipticSolver start: baseLevel = "
           << a_baseLevel << endl;
    pout() << " is level solve ? " << a_isLevelSolve << endl;
  }

  ParmParse pp("gravity");
  int LCOARSE = 50;
  pp.query("lcoarse", LCOARSE);

  AMRLevelOrion *thisSelfGravityPtr = this;
  AMRLevelOrion *gptr = this;
  AMRLevelOrion *gptr_coarse = gptr->getCoarserLevel();

  CH_assert(a_baseLevel == thisSelfGravityPtr->m_level);

  // if this is a levelSolve then consider only one (this) level
  if (!a_isLevelSolve) {
    while (thisSelfGravityPtr->m_hasFiner &&
           !thisSelfGravityPtr->isThisFinestLev()) {
      thisSelfGravityPtr = thisSelfGravityPtr->getFinerLevel();
    }
  }
  // maxLevel = maximum refinement level anywhere
  int maxLevel = thisSelfGravityPtr->m_level;

  thisSelfGravityPtr = this;
  if (!a_isLevelSolve) {
    while (thisSelfGravityPtr->m_hasFiner &&
           !thisSelfGravityPtr->isThisFinestLev() &&
           (thisSelfGravityPtr->m_level < LCOARSE)) {
      thisSelfGravityPtr = thisSelfGravityPtr->getFinerLevel();
    }
  }
  int finestLevel = thisSelfGravityPtr->m_level;

  // ctss gravity coarsening modification
  if (a_baseLevel > LCOARSE) {

    // Here, we don't want to compute gravity,
    // but simply interpolate it from coarser level

    if (a_isLevelSolve)
      maxLevel = a_baseLevel;
    for (int lev = a_baseLevel; lev <= maxLevel; ++lev) {
      gptr_coarse = gptr->getCoarserLevel();
      gptr->m_fineInterpPhi.interpToFine(gptr->m_phiNew,
                                         *gptr_coarse->getPhi(m_time));
      gptr = gptr->getFinerLevel();
    }
    return;
  }

  // ctss gravity coarsening modification
  finestLevel = min(thisSelfGravityPtr->m_level, LCOARSE);

  // At this point, thisSelfGravityPtr should
  // refer to whatever finestLevel is:

  if (s_verbosity >= 3) {
    pout() << "... and finestLevel = " << finestLevel << endl;
    pout() << "... and maxLevel = " << maxLevel << endl;
    pout() << "... and thisSelfGravityPtr->m_level = "
           << thisSelfGravityPtr->m_level << endl;
  }

  // up to finest level
  Vector<AMRLevelOrion *> amrLevel(finestLevel + 1, NULL);
  Vector<LevelData<FArrayBox> *> amrPhi(finestLevel + 1, NULL);

  const int bndryLevel = Max(0, a_baseLevel - 1);

  // setup level pointers, up to finest level
  for (int lev = finestLevel; lev >= 0; --lev) {
    amrLevel[lev] = thisSelfGravityPtr;
    thisSelfGravityPtr = thisSelfGravityPtr->getCoarserLevel();
  }

  // set up potential, up to finest level
  for (int lev = bndryLevel; lev <= finestLevel; ++lev) {
    amrPhi[lev] = amrLevel[lev]->getPhi(m_time);
  }

  // up to max gravity level
  Vector<LevelData<FArrayBox> *> amrRhs(finestLevel + 1, NULL);
  Vector<DisjointBoxLayout> amrBoxes(finestLevel + 1);
  Vector<ProblemDomain> amrProbDom(finestLevel + 1);
  Vector<Real> amrDx(finestLevel + 1);
  Vector<int> amrRefRat(finestLevel + 1, 1);

  // setup data structure pointers, up to max gravity level
  for (int lev = finestLevel; lev >= 0; --lev) {
    amrDx[lev] = amrLevel[lev]->m_dx;
    amrBoxes[lev] = amrLevel[lev]->m_grids;
    amrRefRat[lev] = amrLevel[lev]->refRatio();
    amrProbDom[lev] = amrLevel[lev]->problemDomain();
  }

  // setup Poisson eq. right hand side, up to max gravity level
  for (int lev = a_baseLevel; lev <= finestLevel; ++lev) {
    amrRhs[lev] = amrLevel[lev]->getPoissonRhs(m_time);
  }

  bool isDomainCovered = (a_baseLevel == 0);
  if (m_problem_domain.isPeriodic()) {
    // should be: sum(rhs)=0; if not, then offset residual
    // and synchronize
    if (!isDomainCovered) {
      long numPtsDomain = amrProbDom[a_baseLevel].domainBox().numPts();

      // count number of cells on this level
      long numPtsLevel = 0;
      const DisjointBoxLayout &baseGrids = amrBoxes[a_baseLevel];
      for (LayoutIterator lit = baseGrids.layoutIterator(); lit.ok(); ++lit) {
        numPtsLevel += amrBoxes[a_baseLevel][lit].numPts();
      }

      isDomainCovered = (numPtsDomain == numPtsLevel);
    }

    // compute sum(rhs)
    if (!a_isLevelSolve) {
      if (isDomainCovered) {
        m_rhsOffset = computeSum(amrRhs, amrRefRat, amrDx[a_baseLevel],
                                 Interval(0, 0), a_baseLevel);

        // divide offset by domain volume
        Real domainVol = pow(m_domainLength, SpaceDim);
        m_rhsOffset /= domainVol;

        if (s_verbosity >= 3) {
          pout() << " gravity::rhs_resid: " << m_rhsOffset << " level "
                 << a_baseLevel << endl;
        }

        for (int lev = finestLevel; lev > a_baseLevel; --lev) {
          amrLevel[lev]->m_rhsOffset = m_rhsOffset;
        }
      } else {
        // we needed this if the levels were created after
        // rhsOffset was computed
        for (int lev = finestLevel; lev > bndryLevel; --lev) {
          amrLevel[lev]->m_rhsOffset = amrLevel[bndryLevel]->m_rhsOffset;
        }
      }

      // enforce sum(rhs)=0
      for (int lev = a_baseLevel; lev <= finestLevel; ++lev) {
        offset(*amrRhs[lev], m_rhsOffset);
      }

      if (s_verbosity >= 3) {
        if (isDomainCovered) {
          // compute the new sum(rhs)
          pout() << " gravity: rhs_resid: lev= : " << a_baseLevel << " "
                 << computeSum(amrRhs, amrRefRat, amrDx[a_baseLevel],
                               Interval(0, 0), a_baseLevel)
                 << endl;
        }
      }
    } else if (isDomainCovered) {
      Real rhsOffset =
          computeSum(*amrRhs[a_baseLevel], NULL, 1, amrDx[a_baseLevel]);
      // divide offset by domain volume
      Real domainVol = pow(m_domainLength, SpaceDim);
      rhsOffset /= domainVol;
      offset(*amrRhs[a_baseLevel], rhsOffset);
    }
  }

  bool reset = (a_baseLevel == 0 ? true : SOLVER_RESET_PHI);

  AMRPoissonOpFactory opFactory;
  if (m_problem_domain.isPeriodic(0))
  // Use dummy function if BCs are periodic
  {
    if (s_verbosity > 3) {
      pout() << "Use dummy function if BCs are periodic ..." << endl;
    }
    opFactory.define(amrProbDom[0], amrBoxes, amrRefRat, amrDx[0], &NoOpBc);

  } else {
    // Otherwise parse values bc_hi, bc_lo

    if (s_verbosity > 3) {
      pout() << "Parse boundary conditions based on bc_lo bc_hi ..." << endl;
    }
    opFactory.define(amrProbDom[0], amrBoxes, amrRefRat, amrDx[0], &ParseBC);
  }

#ifdef USE_RELAX_SOLVER
  RelaxSolver<LevelData<FArrayBox>> bottomSolver;
  bottomSolver.m_imax = 4 * SOLVER_NUM_SMOOTH;
  bottomSolver.m_eps = 1.0e-3;
  bottomSolver.m_verbosity = s_verbosity;
#else
  BiCGStabSolver<LevelData<FArrayBox>> bottomSolver;
  bottomSolver.m_verbosity = s_verbosity;
  bottomSolver.m_normType = SOLVER_NORM_TYPE;
  bottomSolver.m_small = SOLVER_NORM_THRES * SOLVER_TOLERANCE; // AJC
#endif

  //
  AMRMultiGrid<LevelData<FArrayBox>> amrSolver;

  amrSolver.define(amrProbDom[0], opFactory, &bottomSolver, finestLevel + 1);

  amrSolver.setSolverParameters(
      SOLVER_NUM_SMOOTH, SOLVER_NUM_SMOOTH, SOLVER_NUM_SMOOTH, SOLVER_NUM_MG,
      SOLVER_MAX_ITER, SOLVER_TOLERANCE, SOLVER_HANG, SOLVER_NORM_THRES);

  /*PS: set verbosity before calling solver */
  amrSolver.m_verbosity = s_verbosity;

  // AJC DEBUG CODE for C99 NAN traps
  // For some reason the innards of the eliptic solve cause
  // floating point overflows all the time and this seems to
  // not cause a problem in the acual solition.
  // fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  // END DEBUG

  amrSolver.solve(amrPhi, amrRhs, finestLevel, a_baseLevel, reset);

  // AJC DEBUG CODE for C99 NAN traps
  // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  // END DEBUG

  if (!a_isLevelSolve) {
    // do postelliptic kind of operations
    for (int lev = finestLevel; lev > a_baseLevel; --lev) {
      // Average Phi from finer level data
      amrLevel[lev]->m_coarseAveragePhi.averageToCoarse(*amrPhi[lev - 1],
                                                        *amrPhi[lev]);
    }

    if (m_useDeltaPhiCorr) {
      //
      for (int lev = finestLevel - 1; lev >= a_baseLevel; --lev) {
        //
        LevelData<FArrayBox> &phi = (*amrPhi[lev]);
        LevelData<FArrayBox> &dPhi = (amrLevel[lev]->m_deltaPhi);

        // save a copy of phi in dPhi and use phi with the
        // solver which needs ghost data
        for (DataIterator di = phi.dataIterator(); di.ok(); ++di) {
          dPhi[di].copy(phi[di]);
        }

        // point amrPhi[lev] to deltaPhi; phi should still point
        // to m_phiNew of level lev
        //                  amrPhi[lev] = &(dPhi);

        // undo the previous offsetting, simlar to levelSolve case
        if (m_problem_domain.isPeriodic()) {
          // if (m_level==0)
          if (lev == a_baseLevel && isDomainCovered) {
            Real volume;
            Real theOffset =
                computeSum(volume, (*amrRhs[lev]), NULL, 1, amrDx[lev]);
            theOffset /= volume;
            offset((*amrRhs[lev]), theOffset);
          } else {
            offset((*amrRhs[lev]), -m_rhsOffset);
          }
        }

        // AJC DEBUG CODE for C99 NAN traps
        // For some reason the innards of the eliptic solve cause
        // floating point overflows all the time and this seems to
        // not cause a problem in the acual solition.
        // fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
        // END DEBUG

        // solve again, but on a single level
        amrSolver.solve(amrPhi, amrRhs, lev, lev, reset);

        //
        Real normDeltaPhi = zero;
        const DisjointBoxLayout &grids = amrBoxes[lev];
        for (DataIterator di = grids.dataIterator(); di.ok(); ++di) {
          // temp to allow data swapping
          FArrayBox temp(grids.get(di), 1);

          // pass copy of phi to temp
          temp.copy(dPhi[di]);

          // define dPhi = phi^comp-phi^level
          dPhi[di] -= phi[di];

          // restore phi
          phi[di].copy(temp);

          normDeltaPhi += dPhi[di].norm(grids.get(di));
        }

        //
        if (s_verbosity >= 3) {
          pout() << " normDeltaPhi = " << normDeltaPhi << endl;
        }
        // AJC DEBUG CODE for C99 NAN traps
        // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
        // END DEBUG
      }
    }
  } // if !levelSolve

  // Here we need to do some final bookkeeping for gravity coarsening
  //
  // For level solves, we're good with the interpolation at the beginning,
  // but for full solves, all the above levels need to have filled data.

  // Only if this was a full solve do we need to
  // fill levels finer than LCOARSE with valid phi
  if (!a_isLevelSolve && maxLevel > LCOARSE) {

    gptr = this;

    while (gptr->m_level != LCOARSE + 1) {
      gptr = gptr->getFinerLevel();
    }
    // gptr now refers to LCOARSE+1 level

    for (int lev = LCOARSE + 1; lev <= maxLevel; ++lev) {
      gptr_coarse = gptr->getCoarserLevel();
      gptr->m_fineInterpPhi.interpToFine(gptr->m_phiNew,
                                         *gptr_coarse->getPhi(m_time));
      gptr = gptr->getFinerLevel();
    }
  }

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::ellipticSolver done " << endl << endl;
  }
}

#endif /* GRAVITY */

// PS: self gravity
#ifdef GRAVITY
// Allow for second order source term corrections
void AMRLevelOrion::secondOrderCorrection() {
  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::secondOrderCorrection: " << m_level << endl;
  }

  IntVect ivGhost = m_numForceGhost * IntVect::Unit;
  LevelData<FArrayBox> dForce(m_grids, SpaceDim, ivGhost);

  // first set to old force
  AMRLevelOrion *thisADPtr = this;
  DataIterator di = thisADPtr->m_grids.dataIterator();
  for (di.begin(); di.ok(); ++di) {
    dForce[di].copy(m_forceOld[di]);
  }

  // then compute new force
  computeForce(m_forceNew, m_phiNew, m_time);

  for (di.begin(); di.ok(); ++di) {
    dForce[di] -= m_forceNew[di];
    dForce[di] *= (-0.500e0);
  }

  for (di.begin(); di.ok(); ++di) {
    double ***UU[NVAR], ***gforce[SpaceDim];
    int nv, ibg, ieg, jbg, jeg, kbg, keg, nxtot, nytot, nztot;
    int i, j, k, im, jm, km, nxtotm2, nytotm2, nztotm2, nxtotmg, nytotmg,
        nztotmg, dghost;
    double vx, vy, vz, rho, irho, rhot, ke, pr;

    FArrayBox &m_U = m_UNew[di()];
    FArrayBox &m_F = dForce[di()];

    ibg = m_U.loVect()[0];
    ieg = m_U.hiVect()[0];
#if DIMENSIONS > 1
    jbg = m_U.loVect()[1];
    jeg = m_U.hiVect()[1];
#endif
#if DIMENSIONS > 2
    kbg = m_U.loVect()[2];
    keg = m_U.hiVect()[2];
#endif
    nxtot = ieg - ibg + 1;
    nytot = jeg - jbg + 1;
    nztot = keg - kbg + 1;
    nxtotm2 = nxtot - (m_numGhost + 1);
    nytotm2 = nytot - (m_numGhost + 1);
    nztotm2 = nztot - (m_numGhost + 1);
    nxtotmg = nxtot - 2 * m_numGhost;
    nytotmg = nytot - 2 * m_numGhost;
    nztotmg = nztot - 2 * m_numGhost;

    for (nv = 0; nv < NVAR; nv++)
      UU[nv] = chmatrix3(nztot, nytot, nxtot,
                         &m_U.dataPtr(0)[nv * nztot * nytot * nxtot]);

    nxtot = nxtotmg + 2 * m_numForceGhost;
    nytot = nytotmg + 2 * m_numForceGhost;
    nztot = nztotmg + 2 * m_numForceGhost;

    for (nv = 0; nv < 3; nv++)
      gforce[nv] = chmatrix3(nztot, nytot, nxtot,
                             &m_F.dataPtr(0)[nv * nztot * nytot * nxtot]);

    dghost = m_numGhost - m_numForceGhost;

    for (k = m_numGhost; k <= nztotm2; k++) {
      km = k - dghost;
      for (j = m_numGhost; j <= nytotm2; j++) {
        jm = j - dghost;
        for (i = m_numGhost; i <= nxtotm2; i++) {
          im = i - dghost;
          rho = UU[0][k][j][i];
          rhot = rho * m_dt;
          irho = 1.0 / rho;
          vx = UU[1][k][j][i];
          vy = UU[2][k][j][i];
          vz = UU[3][k][j][i];
          ke = 0.5 * (vx * vx + vy * vy + vz * vz) * irho;
#if EOS != ISOTHERMAL
          pr = (UU[EN][k][j][i] - ke) * (gmm - 1.0);
#endif
          UU[1][k][j][i] += gforce[0][km][jm][im] * rhot;
          UU[2][k][j][i] += gforce[1][km][jm][im] * rhot;
          UU[3][k][j][i] += gforce[2][km][jm][im] * rhot;
          vx = UU[1][k][j][i];
          vy = UU[2][k][j][i];
          vz = UU[3][k][j][i];
          ke = 0.5 * (vx * vx + vy * vy + vz * vz) * irho;
#if EOS != ISOTHERMAL
          UU[EN][k][j][i] = ke + pr / (gmm - 1.0);
#endif
        }
      }
    }
    // clearup storage
    for (nv = 0; nv < NVAR; nv++)
      free_chmatrix3(UU[nv]);
    for (nv = 0; nv < SpaceDim; nv++)
      free_chmatrix3(gforce[nv]);
  }

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::secondOrderCorrection: done " << endl;
  }
}
#endif /* GRAVITY */

// PS: self gravity
#ifdef GRAVITY
void AMRLevelOrion::makePoissonRhs(LevelData<FArrayBox> &a_rhs,
                                   const LevelData<FArrayBox> &a_U,
                                   const Real &a_time) {
  // I am assuming they are all defined on the same grids; assert() this along
  // the way

  if (s_verbosity >= 3) {
    pout() << "AMRLevelOrion::makePoissonRhs: " << m_level << endl;
  }

  const DisjointBoxLayout &grids = a_rhs.getBoxes();
  DataIterator di = grids.dataIterator();

  // 0. set rhs = 0
  resetToZero(a_rhs);

  // 1. add gas
  if (a_U.isDefined()) {
    CH_assert(a_rhs.getBoxes() == a_U.getBoxes());
    for (di.begin(); di.ok(); ++di) {
      a_rhs[di].plus(a_U[di], grids.get(di), 0, 0);
    }
  }

  int numCells = 0;
  for (di.begin(); di.ok(); ++di) {
    const Box &curBox = grids.get(di);
    numCells += curBox.numPts();

    // ATM ensure that sum(RHS) = 0
    Real globalMeanDensity = getGlobalMeanDensity();
    if (m_problem_domain.isPeriodic())
      a_rhs[di].plus(-globalMeanDensity, curBox);

    /* AJC:  set the right hand side to 4 pi G rho
             and scale into code units  */
    a_rhs[di].mult(4.0 * CONST_PI * CONST_G *
                       pow(UNIT_VELOCITY / UNIT_LENGTH, 2) * UNIT_DENSITY,
                   curBox);
  }

  if (s_verbosity > 1) {
    const bool notPerVol = false;
    const Real totalRhs = globalAverage(a_rhs, 0, notPerVol);

    pout() << "  " << endl;
    pout() << " level  " << m_level << endl;
    pout() << " Total   RHS   = " << totalRhs << endl;
    pout() << " # cells       = " << numCells << endl;
    pout() << "  " << endl;
  }

  if (s_verbosity >= 3) {
    pout() << " AMRLevelOrion::makePoissonRhs: done  " << endl;
  }
}

Real AMRLevelOrion::getGlobalMeanDensity() {
  if (m_level == 0) {
    return m_globalMeanDensity;
  } else {
    AMRLevelOrion *amrGodCoarserPtr = getCoarserLevel();
    return amrGodCoarserPtr->getGlobalMeanDensity();
  }
}

#endif /* GRAVITY */

// PS: self gravity
#ifdef GRAVITY
// overloaded version that can be called from gravity()
void AMRLevelOrion::computeForce() {
  computeForce(m_forceNew, m_phiNew, m_time);
}
#endif /* GRAVITY */

// PS: self gravity
#if defined(GRAVITY) || defined(SINKPARTICLE)
// FM 6.7.05: a_phi=>phi: temporary hack around solver misbehavior when phi
// ghosts>1
// function that computes F=-grad(phi)
void AMRLevelOrion::computeForce(LevelData<FArrayBox> &a_force,
                                 LevelData<FArrayBox> &phi,
                                 const Real &a_time) {

  if (s_verbosity >= 4) {
    pout() << "AMRLevelOrion::computeForce " << m_level << endl;
  }

#ifdef GRAVITY
  CH_assert(allDefined());

  // FM 6.7.05: temporary hack around solver misbehavior when phi ghosts>1
  int numPhiGhost = 2;
  LevelData<FArrayBox> a_phi(m_grids, 1, numPhiGhost * IntVect::Unit);
  phi.copyTo(a_phi.interval(), a_phi, a_phi.interval());

  a_phi.exchange(a_phi.interval());

  Real alpha;
  AMRLevelOrion *amrCoarserPtr = NULL;
  if (m_hasCoarser) {
    amrCoarserPtr = getCoarserLevel();
    const Real tNew = amrCoarserPtr->m_time;
    const Real tOld = tNew - amrCoarserPtr->m_dt;

    const Real eps = 0.01 * m_dt;
    if (Abs(a_time - tNew) <= eps) // case alpha=1; synchronization point;
    {
      alpha = one;
      m_quadCFInterp.coarseFineInterp(a_phi, amrCoarserPtr->m_phiNew);
      // ,numPhiGhost);
    } else if (Abs(a_time - tOld) < eps) // case alpha=0
    {
      alpha = zero;
      //      pout() << " Warning:: alpha=0 should never happen ";
      //      pout() << " time " << a_time << " tOld " << tOld << " tNew " <<
      //      tNew; pout() << " dt " << m_dt << endl;
      m_quadCFInterp.coarseFineInterp(a_phi, amrCoarserPtr->m_phiOld);
      // ,numPhiGhost);
    } else {
      CH_assert((tNew - tOld) > eps);
      alpha = (a_time - tOld) / (tNew - tOld);
      LevelData<FArrayBox> phiCrse(amrCoarserPtr->m_grids, 1,
                                   a_phi.ghostVect());
      interpolateInTime(phiCrse, amrCoarserPtr->m_phiOld,
                        amrCoarserPtr->m_phiNew, alpha, one, 0.0e0, m_dt);
      m_quadCFInterp.coarseFineInterp(a_phi, phiCrse); //,numPhiGhost);
    }
  }

  const DisjointBoxLayout &grids = a_phi.getBoxes();
  for (DataIterator di = grids.dataIterator(); di.ok(); ++di) {
    // The current box
    const Box &box = grids.get(di);

    // f = -grad(phi), hence pass -dx
    m_gradient->gradient(a_force[di], a_phi[di], m_problem_domain, (-m_dx),
                         box);
  }

  if (m_hasCoarser) {
    m_forcePatcher.fillInterp(a_force, amrCoarserPtr->m_forceOld,
                              amrCoarserPtr->m_forceNew, alpha, 0, 0, SpaceDim);

    if (s_verbosity >= 4) {
      pout() << "AMRLevelOrion:: computeForce CFInterp OK!" << endl;
    }
  }
#endif // GRAVITY

  // exchange here to get force on one zone outside the valid region:
  // needed for the predictor step and the source term
  a_force.exchange(a_force.interval());

  /*
  if (s_verbosity >= 3 && s_verbosity < 4)
  {
    for (DataIterator di = grids.dataIterator(); di.ok(); ++di)
    {
      const FArrayBox& fx = a_force[di].getFlux(0);
      const Box& box = fx.box();

      for (BoxIterator bi(box); bi.ok(); ++bi)
      {
        const Real& f = fx(bi(),0);
        pout() << " IV : ";
        for (int i=0;i<SpaceDim;i++) pout() << bi()[i] << " ";
        pout() << "    fx : " << f << endl;
      }
    }
  }
  */

  if (s_verbosity >= 4) {
    pout() << "AMRLevelOrion::computeForce done " << endl;
  }
}
#endif /* GRAVITY */
