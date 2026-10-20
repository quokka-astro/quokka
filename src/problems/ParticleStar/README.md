# ParticleStar reference tracks

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
