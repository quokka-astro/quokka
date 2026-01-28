# Particles, star formation and feedback

## StochasticStellarPop Particle Type

StochasticStellarPop particles carry the following real-valued attributes:

- **Mass**: Current particle mass ($M_{\star}$)
- **Velocity**: Three-dimensional velocity vector $(v_x, v_y, v_z)$
- **Birth time**: Simulation time when the particle formed
- **Death time**: When the particle explodes as a supernova
- **Mass at birth**: Initial mass before mass loss
- **Luminosity**: Radiation luminosity (if radiation is enabled)

Each particle also stores an integer **evolution stage** that tracks its lifecycle:

- `SNProgenitor`: High-mass star ($M > 8 M_{\odot}$) that will explode as a supernova
- `SNRemnant`: Compact remnant left after supernova explosion
- `LowMassStar`: Low-mass star that will not explode
- `LowMassComposite`: Composite particle representing a population of low-mass stars

## Star Formation

### Overview

The star formation module adds star particles through a lightweight stochastic prescription that plugs into the Stochastic Stellar Population specialisation.

- Runs once per hydro timestep and evaluates each cell independently.
- Always spawns a low-mass star particle when a trial succeeds.
- Adds high-mass particles probabilistically so the Chabrier (2005) initial mass function (IMF) is satisfied in expectation.

### Jeans instability filter

Eligible cells are first identified through a Jeans-length check before any stochastic sampling occurs.

- Compute the Jeans length \(\lambda_J = c_s / \sqrt{G \rho}\) in every cell.
- Mark the cell as eligible when \(\lambda_J < J \cdot \Delta x\) with \(J = 0.5\).
- Only eligible cells continue to the sampling steps below.

### Controlling the formation rate

Two efficiency parameters tune how aggressively eligible gas is converted into star particles during each hydro step.

- \(\epsilon_{ff}\): efficiency per free-fall time, defined through \(\epsilon_{ff} = (\dot M_{\star} \, t_{ff}) / (M_{cell} \, \Delta t)\).
- \(\epsilon_{\star}\): fraction of the cell mass used when a particle is spawned.
- The target stellar mass for the step is \(\epsilon_{ff} M_{cell} (\Delta t / t_{ff})\).
- Bernoulli probability for spawning: \( P = \frac{\epsilon_{ff}}{\epsilon_{\star}} \frac{\Delta t}{t_{ff}} \).
- The expectation value \(\langle M_{\star} \rangle = P \epsilon_{\star} M_{cell}\) matches the target mass provided \(\Delta t < t_{ff}\); the CFL condition typically enforces that inequality.

### Sampling the stellar population

Once a cell passes the filter and the Bernoulli draw succeeds, we construct the composite stellar population represented by the spawned particles.

- Every accepted draw creates one low-mass particle that represents all stars with \(M < 8 M_{\odot}\).
- High-mass stars follow the Chabrier (2005) IMF: log-normal below \(1 M_{\odot}\) and a slope of \(2.35\) above it.
- Pre-computed IMF integrals provide the mass fraction \(f_{\star,high}\) and mean mass \(\langle m \rangle_{\star,high}\) of the high-mass component.
- The expected number of massive stars is \(f_{\star,high} \epsilon_{\star} M_{cell} / \langle m \rangle_{\star,high}\); a Poisson variate with that mean sets the actual count.
- Each massive star draws its mass from the high-mass end of the IMF, while the low-mass particle retains the remaining fraction \(1 - f_{\star,high}\) of the spawned mass.
- If the Poisson draw returns zero, only the low-mass particle is inserted and it inherits the local gas velocity.

### Assigning particle velocities

Sampled star particles receive velocities that combine the local bulk flow with an isotropic runaway kick.

- Each massive star draws a speed from a power-law distribution \(p(v) \propto v^{-1.8}\) truncated between 3 and 385 km s\(^{-1}\), then converts it to cgs units.
- A random direction is chosen by sampling \(\cos \theta\) uniformly in \([-1, 1]\) and \(\phi\) in \([0, 2\pi)\); the resulting kick is added to the gas velocity of the parent cell.
- The total momentum of all massive stars is accumulated, and the low-mass composite particle receives the opposite momentum so that the cell-level particle system conserves momentum.

### Practical considerations

A few implementation notes help interpret corner cases and limitations of the current recipe.

- Star formation is operator-split from the hydrodynamics. When \(t_{ff}\) is unresolved (\(\Delta t \gtrsim t_{ff}\)), the true star formation rate is not captured, and this scheme provides one possible approximation; no explicit limiter is enforced beyond the CFL-controlled hydro step.
- All spawned particles are inserted at the cell centre. Other physics modules are responsible for any subsequent repositioning or feedback coupling.


## Supernova Feedback

### Overview

Quokka includes a particle framework for modeling stellar populations and their feedback effects on the interstellar medium (ISM). The primary feedback mechanism is supernova (SN) explosions, which inject energy and momentum into the surrounding gas. The SN module implements several physically-motivated schemes that adapt to the local resolution and density conditions.

### Supernova Feedback

#### Physical Parameters

When a progenitor star reaches its death time, it explodes as a Type II supernova with the following canonical parameters:

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Blast energy | $E_{\text{blast}}$ | $10^{51}$ erg | Thermal energy injected into the ISM |
| Ejecta mass | $m_{\text{ej}}$ | $10 M_{\odot}$ | Mass expelled during explosion |
| Terminal momentum | $p_{\text{snr},0}$ | $2.8 \times 10^5 M_{\odot} \, \text{km s}^{-1}$ | Asymptotic momentum of the SNR |
| Remnant mass | $m_{\text{dead}}$ | $\geq 1.4 M_{\odot}$ | Mass of the compact remnant |

The terminal momentum is density-dependent and scales as:

$$
p_{\text{snr}} = p_{\text{snr},0} \, n_{\text{H}}^{-0.17}
$$

where $n_{\text{H}}$ is the ambient hydrogen number density averaged over the deposition kernel.

#### Deposition Kernel

SN feedback is deposited into a $(2 \times 3 + 1)^3 = 343$ cell cubic stencil centered on the particle's location. The kernel weights are pre-computed to approximate a spherical distribution with radius $r = 3 \Delta x$, where $\Delta x$ is the cell width, around the host cell center. The kernel weights $W_{ijk}$ are normalized such that $\sum_{i,j,k} W_{ijk} = 1$.

#### Resolution-Adaptive Schemes

The feedback implementation uses the **resolution factor** $R_M$ to determine whether the Sedov-Taylor phase is resolved:

$$
R_M = \frac{M_{\text{SNR}}}{M_{\text{sf}}}
$$

where:
- $M_{\text{SNR}} = M_{\text{gas}} + m_{\text{ej}}$ is the total SNR mass (gas in stencil plus ejecta)
- $M_{\text{sf}} = 1679 M_{\odot} \, n_{\text{H}}^{-0.26}$ is the shell-formation mass

When $R_M < 1$, the Sedov-Taylor phase is resolved and the blast wave dynamics can be captured. When $R_M \geq 1$, the resolution is insufficient and the code transitions to momentum-dominated feedback following Kim & Ostriker (2017).

##### Available Schemes

Quokka provides four SN feedback schemes controlled by `particles.SN_scheme`:

1. **`SN_thermal_only`**: Pure thermal energy injection
   - Deposits $E_{\text{blast}} + E_{\text{kin}}$ as thermal energy
   - Deposits ejecta mass and momentum $m_{\text{ej}} \vec{v}_{\text{ej}}$
   - No resolution-dependent momentum injection
   - Simplest scheme, appropriate when Sedov-Taylor phase is well-resolved. Should only be used for testing. 

2. **`SN_thermal_or_thermal_momentum`** (default):
   - When $R_M < 0.027$: Pure thermal (like `SN_thermal_only`)
   - When $0.027 \leq R_M < 1$: Partial momentum injection with $f = 0.529 \sqrt{R_M}$
   - When $R_M \geq 1$: Full momentum injection with $f = 1$
   - Based on Kim & Ostriker (2017) calibration: $f^2 = 0.28$ captures 28% kinetic energy fraction in the well-resolved limit

3. **`SN_thermal_kinetic_or_thermal_momentum`**:
   - When $R_M < 0.027$: Pure kinetic injection with $f = \sqrt{2 R_M}$
   - When $0.027 \leq R_M < 1$: Partial momentum injection with $f = 0.529 \sqrt{R_M}$
   - When $R_M \geq 1$: Full momentum injection with $f = 1$
   - In the well-resolved limit, converts thermal energy to kinetic to better represent early expansion

4. **`SN_pure_kinetic_or_thermal_momentum`**:
   - When $R_M < 1$: Pure kinetic injection with $f = 1$
   - When $R_M \geq 1$: Full momentum injection with $f = 1$
   - Always injects full terminal momentum regardless of resolution

The momentum fraction $f$ determines how much of the terminal momentum is injected:

$$
p_{\text{inject}} = f \, p_{\text{snr}}
$$

#### Momentum Deposition

For schemes with momentum injection, the momentum is distributed radially from the particle position. For each cell $(i,j,k)$ in the stencil:

1. Compute the unit vector $\hat{\mathbf{r}}_{ijk}$ from the particle to the cell center
2. Distribute momentum proportional to the kernel weight:
   $$
   \Delta \mathbf{p}_{ijk} = f \, p_{\text{snr}} \, W_{ijk} \, \hat{\mathbf{r}}_{ijk}
   $$

This creates an isotropic outward momentum distribution centered on the SN position.

#### Galilean Invariance Option

The parameter `particles.SN_use_galilean_invariant` (default: `1`) controls whether the momentum deposition is Galilean invariant:

##### Galilean-Invariant Formulation (`SN_use_galilean_invariant = 1`, default)

Uses a center-of-mass (COM) frame formulation that ensures the feedback is invariant under Galilean transformations:

1. **Compute COM velocity** of the SNR (gas + ejecta):
   $$
   \vec{v}_{\text{COM}} = \frac{\vec{P}_{\text{gas}} + m_{\text{ej}} \vec{v}_{\text{ej}}}{M_{\text{gas}} + m_{\text{ej}}}
   $$
   where $\vec{P}_{\text{gas}}$ is the total momentum of gas in the stencil (kernel-weighted).

2. **Deposit momentum** such that each cell receives:
   $$
   \Delta \vec{p}_{ijk} = \left[\rho_{\text{new}} \vec{v}_{\text{COM}} - \vec{p}_{\text{old}}\right] + \vec{p}_{\text{radial}}
   $$
   where $\vec{p}_{\text{radial}} = f \, p_{\text{snr}} \, W_{ijk} \, \hat{\mathbf{r}}_{ijk}$ is the radial momentum component.

3. **Energy deposition** includes a cross term:
   $$
   \Delta E_{ijk} = \left(E_{\text{blast}} + E_{\text{kin}}\right) W_{ijk} + \vec{v}_{\text{COM}} \cdot \vec{p}_{\text{radial}}
   $$

   The cross term $\vec{v}_{\text{COM}} \cdot \vec{p}_{\text{radial}}$ accounts for the kinetic energy change from "resetting" the cell velocity to the COM frame before adding radial expansion. This term sums to zero over all cells (momentum is conserved), ensuring Galilean invariance.

**Physical interpretation**: The gas velocity is first transformed to the COM frame, then isotropic expansion is added. This ensures that a boosted observer sees the same SN physics.

##### Lab-frame Formulation (`SN_use_galilean_invariant = 0`)

Uses a lab-frame formulation that is simpler but not Galilean invariant:

1. **Deposit momentum** proportionally to preserve cell velocity:
   $$
   \Delta \vec{p}_{ijk} = \frac{\Delta \rho_{ijk}}{\rho_{ijk}} \vec{p}_{\text{old}} + \vec{p}_{\text{radial}}
   $$
   where $\Delta \rho_{ijk} = m_{\text{ej}} W_{ijk}$ is the deposited mass.

2. **Energy deposition** uses lab-frame ejecta kinetic energy:
   $$
   \Delta E_{ijk} = \left(E_{\text{blast}} + \frac{1}{2} m_{\text{ej}} |\vec{v}_{\text{ej}}|^2\right) W_{ijk}
   $$

**Physical interpretation**: The first term in the momentum deposition increases the momentum proportionally to maintain the cell's original velocity as mass is added. The radial momentum is then added on top. This is the legacy formulation from the original implementation.

**When to use each**:
- Galilean-invariant (default): Physically correct for moving systems, better for simulations with bulk flows
- Lab-frame: Simpler, may give lower numerical errors in some static cases, preserves original behavior

#### Thermal-Only Scheme

The `SN_thermal_only` scheme always uses the energy-conserving (lab-frame) formulation since there is no radial momentum injection. It deposits:

- **Mass**: $\Delta \rho_{ijk} = m_{\text{ej}} W_{ijk}$
- **Momentum**: $\Delta \vec{p}_{ijk} = m_{\text{ej}} \vec{v}_{\text{ej}} W_{ijk}$ (ejecta momentum only)
- **Energy**: $\Delta E_{ijk} = \left(E_{\text{blast}} + \frac{1}{2} m_{\text{ej}} |\vec{v}_{\text{ej}}|^2\right) W_{ijk}$

### Runtime Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particles.SN_scheme` | String | `SN_thermal_or_thermal_momentum` | Feedback scheme (see above) |
| `particles.SN_use_galilean_invariant` | Boolean | `1` | Use Galilean-invariant formulation for momentum schemes |
| `particles.disable_SN_feedback` | Boolean | `0` | Disable SN feedback entirely |
| `particles.verbose` | Integer | `0` | Verbosity level for particle diagnostics |
| `particles.stellar_velocity_limit` | Float | $10^8$ cm/s | Maximum allowed stellar velocity (aborts if exceeded) |

### Implementation Notes

#### Operator Splitting

SN feedback is operator-split from the hydrodynamics and applied after each timestep:

1. Hydrodynamics advances the gas state by $\Delta t$
2. Particle positions are updated by drift
3. SN candidates (particles reaching death time) are identified
4. Feedback is deposited into a temporary buffer
5. Buffer is added to the state using a reproducibility-aware algorithm
6. Particle evolution stage is updated to `SNRemnant`

#### Numerical Reproducibility

The deposition uses a roundoff-resistant algorithm to ensure bit-for-bit reproducibility across different processor counts. The algorithm:

1. Deposits feedback into a temporary buffer with a count field
2. Applies the Kahan compensated summation algorithm
3. Removes low-order bits controlled by `particles.reproducibility_roundoff_redundancy`

#### AMR Considerations

- Feedback is always deposited at the finest level covering the particle
- If a particle sits at a coarse-fine boundary, feedback is deposited into fine-level cells (known limitation that needs to be fixed in the future)
- The reflux operation ensures conservation across AMR boundaries
- The kernel stencil size is fixed in cell widths, so the physical size adapts to local resolution
- The use of `setForceFinestLevel(true)` is recommended to ensure that all SN progenitors, and thus SN events, exist at finest level. 

#### Limitations

- Accurate radiative cooling must be present to resolve the pressure-confined snowplow phase
- The spherical kernel is approximated on a Cartesian grid, introducing mild anisotropy
- The terminal momentum formula assumes solar metallicity and Type II SNe physics
- No explicit treatment of metal enrichment (can be added via passive scalars)

### Physical Motivation

The resolution-dependent momentum injection is based on the realization that under-resolved SN explosions lose energy to artificial radiative losses within a few timesteps. The momentum-based schemes compensate by directly injecting momentum when the Sedov-Taylor phase cannot be captured.

The terminal momentum $p_{\text{snr}}$ represents the asymptotic momentum that a supernova remnant reaches in the pressure-confined snowplow phase, after most of the kinetic energy has been radiated away. This value (Kim & Ostriker 2017) is calibrated from high-resolution simulations and depends weakly on the ambient density.

The $R_M$ factor effectively measures whether the simulation has sufficient resolution and density to capture the Sedov-Taylor expansion. When $R_M \ll 1$, the blast wave will expand through many cells before reaching the shell-formation radius, allowing the hydrodynamics to naturally capture the momentum buildup. When $R_M \sim 1$ or larger, the explosion is under-resolved and the code preemptively injects the expected terminal momentum.

### Examples

#### Basic SN Test

The `SN` problem provides a canonical test with one or more SN progenitors in a uniform medium:

```
particles.SN_scheme = SN_thermal_or_thermal_momentum   # adaptive scheme
particles.SN_use_galilean_invariant = 1                # use Galilean-invariant formulation
particles.verbose = 1                                  # print SN diagnostics
```

#### Random Blast Test

The `RandomBlast` problem tests multiple SN explosions with various ambient conditions and bulk flows:

```
particles.SN_scheme = SN_thermal_or_thermal_momentum
particles.SN_use_galilean_invariant = 1                # critical for moving medium
```

The boost velocity tests demonstrate the importance of Galilean invariance. With `SN_use_galilean_invariant = 1`, the temperature and velocity profiles are identical in the boosted and rest frames (to numerical precision). With `SN_use_galilean_invariant = 0`, the profiles differ significantly.

