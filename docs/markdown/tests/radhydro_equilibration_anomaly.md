# Moving matter-radiation equilibration anomaly

This test problem demonstrates the anomalous acceleration described by Lowrie, Wollaeger, and Morel for Newtonian hydrodynamics coupled to radiation transport. The setup is a uniform infinite-medium equilibration problem viewed in a frame moving with respect to the material rest frame. In the fully relativistic solution, the gas velocity is constant. In Quokka's mixed-frame Newtonian radiation hydrodynamics update, the gas momentum changes during the matter-radiation source update.

The test is diagnostic rather than a validation against a desired invariant solution. It is intended to make the anomalous behavior easy to reproduce and quantify.

## Setup

The gas is uniform, periodic, and initially moving in the \\(x\\)-direction. The material and radiation are out of thermal equilibrium:

<script type="math/tex; mode=display">
\begin{aligned}
\rho_0 &= 1, \\
c &= \hat c = 100, \\
v_0 &= 10^{-2} c = 1, \\
T_0 &= 1, \\
a_r &= 1, \\
\kappa_P &= \kappa_F = 1.
\end{aligned}
</script>

The default CTest input uses \\(E_{r,0}=4\\), while the long diagnostic input uses \\(E_{r,0}=3 \times 10^5\\). In both cases the gas internal energy law is

<script type="math/tex; mode=display">
e_\mathrm{gas} = T^4 ,
</script>

so that the radiation-matter energy exchange is a simple equilibration problem with \\(a_rT_0^4 = 1\\). The radiation flux is initialized as the \\(O(v/c)\\) boosted isotropic field,

<script type="math/tex; mode=display">
F_{r,x,0} = \frac{4}{3} v_0 E_{r,0}.
</script>

## Expected behavior

For this problem the fully relativistic solution has no acceleration. A Newtonian conservative momentum update coupled to the lab-frame radiation momentum deposition instead gives, to leading order at early time,

<script type="math/tex; mode=display">
\frac{dv}{dt} =
\frac{\kappa_F v}{c}
\left(E_r - a_r T^4\right).
</script>

Thus the short test asserts that the measured velocity increment is positive and close to the small-step Newtonian prediction. This verifies that Quokka is exercising the anomalous coupling path.

## Scaling note

The anomalous term is small in powers of the velocity only if the radiation energy density is comparable to the flux energy density. Comparing the anomalous acceleration to the physical flux acceleration,

<script type="math/tex; mode=display">
\frac{a_\mathrm{anom}}{a_\mathrm{rad}}
\sim
\beta\frac{c(E_r-a_rT^4)}{F_r},
</script>

where \\(\beta=v/c\\) and \\(a_\mathrm{rad}=\kappa_F F_r/c\\). For optically thin, free-streaming radiation, \\(cE_r/F_r = O(1)\\), so the anomalous acceleration is only \\(O(\beta)\\) of the physical radiative acceleration. In an optically thick diffusive region, \\(cE_r/F_r = O(\tau)\\). If the radiation and material are also out of thermal equilibrium, so that \\(|E_r-a_rT^4| = O(E_r)\\), then the instantaneous anomalous acceleration can be \\(O(\beta\tau)\\) times the physical flux acceleration.

This force-level estimate does not by itself imply that every equilibration event produces an \\(O(\beta\tau)\\) velocity error. For a one-time relaxation, increasing the opacity also shortens the equilibration time, and the anomalous acceleration shuts off when \\(E_r=a_rT^4\\). The \\(O(\beta\tau)\\) scaling is most relevant when non-equilibrium is sustained over a hydrodynamic residence time, such as in a driven radiative shock, a Zel'dovich spike, or a source/transport imbalance that continually replenishes \\(E_r-a_rT^4\\). In the notation of Lowrie, Wollaeger, and Morel, the corresponding fluid-time measure is

<script type="math/tex; mode=display">
A =
\left(\frac{E_r}{\rho v_\mathrm{ref}^2}\right)
(\kappa_F\rho L)
\left(\frac{v_\mathrm{ref}}{c}\right),
</script>

with the thermal non-equilibrium fraction multiplying this estimate when \\(|E_r-a_rT^4| < E_r\\).

## Running

The registered test uses the short one-step input:

```bash
./scripts/bash/quokka buildrun -d 3d RadhydroEquilibrationAnomaly
```

The long diagnostic input writes a CSV history and drives the final velocity to order 2:

```bash
./scripts/bash/quokka run -d 3d RadhydroEquilibrationAnomaly --input inputs/RadhydroEquilibrationAnomalyLong.toml
```

The long run writes `tests/radhydro_equilibration_anomaly_velocity.csv`. The run used for this page reached a final velocity of \\(1.9522\\), with a transient peak of \\(2.9020\\).

## Result

![](attach/radhydro_equilibration_anomaly_velocity.png)

*Gas velocity as a function of time for the long diagnostic input. The velocity starts at 1, rapidly accelerates above 2, and relaxes to a final value of about 1.95 after radiation-matter equilibration.*
