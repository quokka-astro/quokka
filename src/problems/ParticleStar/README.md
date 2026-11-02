# ParticleStar reference tracks

## Orion2007

`orion2007_track_mdot1e-5.dat` and `orion2007_track_mdot1e-4.dat` are protostellar evolution tracks from Orion2007, the 2007 standalone version of the ORION star particle module (`StarParticleData.cpp`; an earlier version of the model published by Offner et al. 2009, Appendix B), run at a constant accretion rate of $10^{-5}$ and $10^{-4}\,M_\odot\,\mathrm{yr}^{-1}$. `testParticleStar.cpp` evolves `quokka::Orion2007StellarModel` with the same steps and compares the radius, luminosity, polytropic index, deuterium mass and evolutionary phase with these tracks.

The physical constants in `StarParticleData.cpp` were replaced by the values Quokka uses, so that the comparison tests the algorithm and not the constants:

| Orion2007 | Value used | Quokka symbol |
|---|---|---|
| `G` = 6.67e-8 | 6.67428e-8 | `C::Gconst` |
| `A` = 7.56e-15 | $4\sigma_{\rm SB}/c$ | `C::a_rad` |
| `KB` = 1.38e-16 | 1.380649e-16 | `C::k_B` |
| `MH` = 1.67e-24 | 1.67262192595e-24 | `C::m_p` |
| `SIGMA` = 5.67e-5 | 5.670374419184432e-5 | `C::sigma_SB` |
| `PI` = 3.1415927 | `M_PI` | |
| `MSUN` = 1.99e33 | 1.9884e33 | `C::M_solar` |
| `RSUN` = 6.96e10 | 6.957e10 | `C::R_solar` |
| `LSUN` = 3.90e33 | 3.828e33 | `StellarPhysics::L_solar` |
| `YRSEC` = 3.16e7 | 3.15576e7 | `quokka::yr_in_s` |
| `NAVOG` = 6.022e23 | 6.02214076e23 | `C::n_A` |
| `ERGEV` = 1.6e-12 | 1.602176634e-12 | `C::ev2erg` |

The tracks were written by the driver below, compiled with the modified `StarParticleData.cpp`, as `dump_track 1e-5 10 1e-4 100` and `dump_track 1e-4 20 1e-4 200`. Like the Orion2007 driver `main.cpp`, it starts from $10^{-3}\,M_\odot$ and adds $10^{-4}\,M_\odot$ per step.

```cpp
#include "StarParticleData.H"
#include <cstdio>
#include <cstdlib>
int main(int argc, char **argv)
{
	const double msun = 1.9884e33;
	const double yr = 3.15576e7;
	const double mdot_in = atof(argv[1]);
	const double mdot = mdot_in * msun / yr;
	const double mmax = atof(argv[2]) * msun;
	const double mstep = atof(argv[3]) * msun;
	const int nout = atoi(argv[4]);
	StarParticleData star(1.0e-3 * msun, mdot);
	printf("# Orion2007 (standalone ORION StarParticleData, 2007) protostellar model with Quokka physical constants (see README.md)\n");
	printf("# m0 = 1e-3 Msun, mdot = %g Msun/yr, dm = %g Msun per step, dt = dm/mdot; one row every %d steps\n", mdot_in, atof(argv[3]), nout);
	printf("# state: 0 PreCollapse, 1 NoBurning, 2 CoreDeuteriumFixedTc, 3 CoreDeuteriumVariableTc, 4 ShellDeuterium, 5 MainSequence\n");
	printf("# step  m[g]  r[cm]  n  mdeut[g]  state  L[erg/s]\n");
	for (long step = 1; star.m < mmax * (1.0 - 1.0e-12); ++step) {
		const double dt = mstep / mdot;
		star.m += mstep;
		star.updateState(dt);
		if (step % nout == 0) {
			const int state = star.initDone ? static_cast<int>(star.burnState) + 1 : 0;
			printf("%ld %.10e %.10e %.10e %.10e %d %.10e\n", step, star.m, star.r, star.n, star.mdeut, state, star.luminosity());
		}
	}
}
```

## Orion2

`orion2_track_mdot1e-5.dat` (to $10\,M_\odot$, one row every 100 steps) and `orion2_track_mdot1e-4.dat` (to $20\,M_\odot$, one row every 200 steps) are tracks from the protostellar model of the ORION2 code (`Src/Star_Particles/StarParticleData.cpp` of the ORION2 release), run with the same steps as the Orion2007 tracks. `testParticleStar.cpp` evolves `quokka::Orion2StellarModel` with these steps and compares the radius, luminosity, polytropic index, deuterium mass, evolutionary phase and carried-over negative luminosity `netlum` with them.

`StarParticleData.cpp` and `StarParticleData.H` were compiled outside ORION2/Chombo with these changes:

- ORION2's constants (`MSUN` = 1.989e33, `LSUN` = 3.84e33, `RSUN` = 6.96e10; `G`, `A`, `KB`, `MH`, `SIGMA`, `NAVOG`, `ERGEV` as in Orion2007) were replaced by the Quokka values of the table above, with `KB` = 1.3806490000000002e-16 (the double value of `C::k_B`) and `A` = 4 `SIGMA` / 29979245800. The literal `6.96e10` in the negative-radius reset was replaced by `RSUN`.
- Two bugs in `beta()` were fixed, as in `Orion2StellarModel`: the table row is `floor((logm-logm0) / dlogm)` (ORION2 has `floor((logm-logm0 / dlogm))`), clamped to `nm-2` with the mass weight clamped to 1; and for `n == 3` in the table interior the column is `nn-2` with weight 1 (ORION2 uses `nn-1`, which reads column `nn`, outside the table).
- `NHIST` is undefined (no averaging of the accretion rate over past steps, no `l_hist`), so `mdot = (m - mlast)/dt`; the per-step `pout()` print of that branch was removed.
- The stream operators were removed (they use `l_hist`, which exists only with `NHIST`).

The Chombo dependencies are replaced by this `SinkParticleData.H` (plus a `CH_assert.H` that maps `CH_assert` to `assert`):

```cpp
// Minimal stand-in for ORION2/Chombo dependencies of StarParticleData
#ifndef SHIM_SINKPARTICLEDATA_H_
#define SHIM_SINKPARTICLEDATA_H_
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;
typedef double Real;
#define CH_SPACEDIM 3

namespace MayDay { inline void Error(const char *s) { fprintf(stderr, "MayDay::Error: %s\n", s); exit(1); } }
inline std::ostream &pout() { return std::cerr; }
namespace SerialTask { enum task { compute }; }
inline int CHprocID() { return 0; }
inline int uniqueProc(SerialTask::task) { return -1; } // never "this proc": silences the init message
class SinkParticleData {
public:
  SinkParticleData(Real *newpos, Real *newmom, Real *newangmom, Real newmass) : m(newmass) {
    for (int i = 0; i < CH_SPACEDIM; i++) { pos[i] = newpos[i]; mom[i] = newmom[i]; angmom[i] = newangmom[i]; }
  }
  SinkParticleData() { m = -1.0; }
  virtual ~SinkParticleData() {}
  Real m;
  Real pos[CH_SPACEDIM], mom[CH_SPACEDIM], angmom[CH_SPACEDIM];
  int angmom_method = 0;
};
#endif
```

The tracks were written by the driver below as `dump_track 1e-5 10 1e-4 100` and `dump_track 1e-4 20 1e-4 200`. After each `updateState(dt)` it applies the rule of `StarParticleList::starRadiation`: the luminosity `luminosity(netlum)` includes `netlum` if it is negative, and a negative result is stored in `netlum` and replaced by 0. The ORION2 constructor does not set `netlum`; the driver sets it to 0. The `state` column uses ORION2's `burnState` (0 `Uninitialized`, 1 `None`, 2 `VariableCoreDeuterium`, 3 `SteadyCoreDeuterium`, 4 `ShellDeuterium`, 5 `ZAMS`), which has the numbering of `burn_state` in Quokka. In state 0 the columns `r` and `n` are the constructor's values for $10^{-4}\,M_\odot\,\mathrm{yr}^{-1}$, which Quokka does not store, so the test compares them only from state 1 on.

```cpp
// Driver: ORION2 StarParticleData at constant accretion rate (see Quokka src/problems/ParticleStar/README.md)
#define protected public
#define private public
#include "StarParticleData.H"
#undef protected
#undef private
#include <cstdio>
#include <cstdlib>
int main(int argc, char **argv)
{
	const double msun = 1.9884e33;
	const double yr = 3.15576e7;
	const double mdot_in = atof(argv[1]);
	const double mdot = mdot_in * msun / yr;
	const double mmax = atof(argv[2]) * msun;
	const double mstep = atof(argv[3]) * msun;
	const int nout = atoi(argv[4]);
	Real zero[3] = {0.0, 0.0, 0.0};
	StarParticleData star(zero, zero, zero, 1.0e-3 * msun);
	star.netlum = 0.0; // not set by the ORION2 constructor
	int last_state = star.burnState;
	printf("# ORION2 StarParticleData protostellar model with Quokka physical constants and two beta() fixes (see README.md)\n");
	printf("# m0 = 1e-3 Msun, mdot = %g Msun/yr, dm = %g Msun per step, dt = dm/mdot; one row every %d steps\n", mdot_in, atof(argv[3]), nout);
	printf("# state: 0 Uninitialized, 1 None, 2 VariableCoreDeuterium, 3 SteadyCoreDeuterium, 4 ShellDeuterium, 5 ZAMS\n");
	printf("# step  m[g]  r[cm]  n  mdeut[g]  state  L[erg/s]  netlum[erg/s]\n");
	for (long step = 1; star.m < mmax * (1.0 - 1.0e-12); ++step) {
		const double dt = mstep / mdot;
		star.m += mstep;
		star.updateState(dt);
		// StarParticleList::starRadiation: a negative luminosity is carried to the next step
		double lum = star.luminosity(star.netlum);
		if (lum < 0.0) {
			star.netlum = lum;
			lum = 0.0;
		} else {
			star.netlum = 0.0;
		}
		if (star.burnState != last_state) {
			fprintf(stderr, "step %ld: state %d at m = %.6f Msun\n", step, static_cast<int>(star.burnState), star.m / msun);
			last_state = star.burnState;
		}
		if (step % nout == 0) {
			printf("%ld %.10e %.10e %.10e %.10e %d %.10e %.10e\n", step, star.m, star.r, star.n, star.mdeut, static_cast<int>(star.burnState), lum, star.netlum);
		}
	}
}
```

The only algorithmic difference between `Orion2StellarModel` and this code is the root solver for the central temperature: ORION2 bisects to a relative tolerance of $10^{-7}$, Quokka uses TOMS 748 to $10^{-12}$. The radius, luminosity and deuterium mass agree to $3\times10^{-8}$, and this difference disappears when the bisection tolerance is reduced to $10^{-15}$.
