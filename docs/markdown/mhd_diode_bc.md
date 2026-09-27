# MHD Diode Boundary Condition

A diode boundary lets gas leave the domain freely and stops gas from entering it. This page describes how Quokka implements the diode for MHD with constrained transport (CT): the requirements, an analysis of the obvious candidate schemes and why they fail, the scheme that is used, and the reasons for each choice.

## Usage

For each boundary that should be a diode:

1. set the cell-centred boundary type to `ext_dir` and call `setDiodeBCLo<dir>` / `setDiodeBCHi<dir>` in `setCustomBoundaryConditions` (the same helpers as for a hydro-only diode);
2. set the face-centred boundary type to `foextrap`. This is only a placeholder fill; the diode pass overwrites it;
3. specialise `AMRSimulation<problem_t>::isMHDDiodeBoundary(int dir, int side)` to return `true` for this boundary (`side = 0` is the lower boundary, `side = 1` the upper one).

The test problem `src/problems/MHDDiode/testMHDDiode.cpp` is a complete example. After the cell-centred and face-centred ghost fills that precede every hydro stage, `QuokkaSimulation` calls `AMRSimulation::applyMHDDiodeBC`, which fills the magnetic field and corrects the total energy in the ghost cells of every diode boundary.

## Notation

We describe the lower \\(x\\) boundary; the upper boundary and the \\(y\\) and \\(z\\) boundaries follow by symmetry. Cell \\(i = 0\\) is the first valid cell, and the ghost cells are \\(i = -1, \dots, -n\_g\\). Face \\(i\\) is the left face of cell \\(i\\), so face \\(0\\) lies on the domain boundary and face \\(-m\\) is the left face of ghost cell \\(-m\\). A *column* is the set of cells with the same transverse indices \\((j, k)\\). The normal field is \\(B\_n = B\_x\\), stored on \\(x\\)-faces; the transverse field is \\(B\_t = (B\_y, B\_z)\\), stored on \\(y\\)- and \\(z\\)-faces. In the ghost region, the transverse faces sit at the same \\(x\\)-positions as the ghost cell centres.

The discrete divergence of cell \\((i, j, k)\\) is

<script type="math/tex; mode=display">
(\nabla \cdot \mathbf{B})_{i,j,k} = \frac{B_x(i+1,j,k) - B_x(i,j,k)}{\Delta x} + \frac{B_y(i,j+1,k) - B_y(i,j,k)}{\Delta y} + \frac{B_z(i,j,k+1) - B_z(i,j,k)}{\Delta z}.
</script>

## Requirements

- **R1 (no inflow).** Where the flow at the boundary points into the domain, the mass flux through the boundary face vanishes to round-off, for every reconstruction order (PLM, PPM, xPPM).
- **R2 (free outflow).** Where the flow points out of the domain, all quantities, including \\(\mathbf{B}\\), leave with zero-gradient extrapolation.
- **R3 (divergence-free field).** \\(\nabla \cdot \mathbf{B} = 0\\) to round-off in every valid cell **and every ghost cell**. This must hold where neighbouring columns differ (one inflow, one outflow), and when a column switches between inflow and outflow from one step to the next.
- **R4 (consistent gas state).** The gas pressure that `ComputePrimVars` computes in a ghost cell equals the pressure of the cell the ghost state was copied or mirrored from.

## Why MHD is harder than hydro

The hydro-only diode (`setDiodeBCLo/Hi`) works column by column on cell-centred data: outflow copies the first valid cell, inflow mirrors the interior with the normal momentum reversed. The magnetic field adds two difficulties.

1. **The boundary face is shared.** The normal field on face \\(0\\) belongs to both the first valid cell and the first ghost cell. It is valid data, evolved by CT, and the boundary fill must not change it. A ghost cell can therefore never be an independent copy of a valid cell.
2. **The switch between inflow and outflow.** Any fill rule that treats \\(B\\) differently in inflow and outflow columns must stay divergence-free where the two kinds of column meet and when a column changes type. This is the hard part of the problem: reflecting boundaries alone are easy.

Two facts about the current code also matter:

- The total energy stored in a cell includes the magnetic energy, and `ComputePrimVars` computes the pressure as \\(P = (\gamma - 1)(E - E\_{\rm kin} - E\_B)\\), with \\(E\_B\\) averaged from the **face** fields. So a ghost cell whose total energy was copied from one cell, but whose face field differs from that cell's, has the wrong pressure.
- The cell-centred state is filled before the face-centred state, and the face-centred boundary hook sees only the face array. It cannot see the momentum that decides between inflow and outflow. The diode therefore runs as a separate pass after both fills.

## Analysis of candidate schemes

### Copy everything at outflow

Copying the face fields into the ghost cells, \\(B\_x(-m) = B\_x(0)\\) (AMReX `foextrap`), gives a ghost cell \\(-1\\) with no normal-field difference but a copied transverse divergence:

<script type="math/tex; mode=display">
(\nabla \cdot \mathbf{B})_{-1} = 0 + \left( \frac{\partial B_y}{\partial y} + \frac{\partial B_z}{\partial z} \right)_{0} = -\frac{B_x(1) - B_x(0)}{\Delta x} \neq 0 .
</script>

In multi-dimensional flows this is non-zero wherever \\(B\_x\\) varies across the first valid cell. Copying the transverse field is fine; the normal field must be computed from the divergence constraint instead. In the `MHDDiode` tests, the `foextrap` fill alone gives \\(\Delta x |\nabla \cdot \mathbf{B}| / |\mathbf{B}| \approx 4 \times 10^{-2}\\) in the ghost cells.

### Copy and flip at inflow

Copying the first valid cell into all ghost cells and reversing the normal momentum gives zero mass flux only at first order. With PLM, PPM or xPPM, the ghost-side state at face \\(0\\) is the copied cell value, while the valid-side state is reconstructed from a stencil that reaches into the interior (and into the ghost cells). The two states are not mirror images, and mass crosses the boundary. In the `MHDDiode` wall test this scheme loses a fraction \\(2.7 \times 10^{-3}\\) of the mass. The gas must be mirrored geometrically (ghost \\(-m\\) takes the state of valid cell \\(m-1\\)), which is what the hydro diode already does.

### Conducting-wall reflection of the field

The usual reflecting boundary (for example the `reflect` boundary of Athena++) treats \\(\mathbf{B}\\) like a perfectly conducting wall: \\(B\_n\\) odd and \\(B\_t\\) even. On the staggered grid this means \\(B\_x(-m) = -B\_x(m)\\), and for \\(m = 0\\) it requires \\(B\_x(0) = 0\\). After an outflow phase, \\(B\_x(0)\\) is in general non-zero, and then

<script type="math/tex; mode=display">
(\nabla \cdot \mathbf{B})_{-1} = \frac{2 B_x(0)}{\Delta x} \neq 0 .
</script>

So a mirror image of a divergence-free field is itself divergence-free only if the normal field on the mirror plane vanishes. This rules out the conducting-wall reflection for a diode.

### Pseudo-vector mirror

Ideal MHD has a second mirror symmetry: the conducting-wall mirror combined with the global symmetry \\(\mathbf{B} \to -\mathbf{B}\\), which makes \\(B\_n\\) even and \\(B\_t\\) odd. It is consistent with any \\(B\_x(0)\\) and keeps \\(\nabla \cdot \mathbf{B} = 0\\). It has two drawbacks. It puts a current sheet at the wall (the tangential field changes sign across face \\(0\\)), and it flips the sign of the ghost tangential field every time a column switches between inflow and outflow. Each flip is a jump in the boundary-edge EMFs, and so in \\(B\_x(0)\\).

### Setting EMFs instead of fields

Pjanka & Stone (Sect. 3.5 of [@Pjanka2020]) use a diode in Athena++. At outflow they copy the cell-centred quantities and the edge EMFs into the ghost cells; at inflow they reflect the cell-centred quantities and set the ghost EMFs to zero. Acting on the EMFs guarantees \\(\nabla \cdot \mathbf{B} = 0\\) by construction, but it needed changes to the EMF update, and the paper does not say how the ghost face fields or the EMFs on the boundary plane are treated. Quokka does not need this: `SolveInductionEqn` updates every valid face, face \\(0\\) included, with the curl of the edge EMFs, which cannot change the divergence of any valid cell whatever the EMF values are. It is enough to fill the ghost face fields in a divergence-free way and leave the EMFs alone. We keep one idea from that paper: the gas is reflected, not copied, at inflow.

## The scheme

`applyMHDDiodeBC` runs three steps on the ghost region of each diode boundary.

### Step 1: transverse field

For each transverse ghost face at \\(x\\)-index \\(-m\\), the value is copied from the first valid cell or mirrored from valid cell \\(m-1\\), **with the same sign**:

<script type="math/tex; mode=display">
B_t(-m, \cdot) =
\begin{cases}
B_t(0, \cdot) & \text{outflow (copy)}, \\
B_t(m-1, \cdot) & \text{inflow (mirror)}.
\end{cases}
</script>

A transverse face lies between two columns (for example, a \\(y\\)-face between columns \\(j-1\\) and \\(j\\)). It is treated as inflow if **either** adjacent column is inflow.

### Step 2: normal field

The normal field on the ghost faces is integrated outward from the boundary face, one ghost cell at a time, so that each ghost cell is divergence-free. At the lower boundary, for \\(m = 1, \dots, n\_g\\):

<script type="math/tex; mode=display">
B_x(-m) = B_x(-m+1) + \Delta x \left[ \frac{B_y(-m, j+1) - B_y(-m, j)}{\Delta y} + \frac{B_z(-m, k+1) - B_z(-m, k)}{\Delta z} \right].
</script>

At the upper boundary, with last valid cell \\(N\\), boundary face \\(N+1\\) and ghost cells \\(N+m\\):

<script type="math/tex; mode=display">
B_x(N+1+m) = B_x(N+m) - \Delta x \left[ \frac{B_y(N+m, j+1) - B_y(N+m, j)}{\Delta y} + \frac{B_z(N+m, k+1) - B_z(N+m, k)}{\Delta z} \right].
</script>

The recursion starts from the boundary face, which is read but never written.

### Step 3: energy correction

The cell-centred fill copied or mirrored the total energy of a source cell (cell \\(0\\) at outflow, cell \\(m-1\\) at inflow), which contains the source-cell magnetic energy. After steps 1 and 2 the ghost face field is known, and the ghost total energy is corrected to

<script type="math/tex; mode=display">
E_{\rm ghost} \leftarrow E_{\rm ghost} - E_B^{\rm source} + E_B^{\rm ghost}, \qquad E_B = \frac{1}{2} \sum_{d} \left[ \frac{B_d(\text{left face}) + B_d(\text{right face})}{2} \right]^2 ,
</script>

with \\(E\_B\\) computed from the face averages exactly as in `ComputeMagneticEnergy`. The internal-energy variable, the density, the momentum and the passive scalars are left as the cell-centred fill set them.

### Per-column rule

The flag of a column is the sign of the normal momentum in its first valid cell, the same rule as `setDiodeBCLo/Hi`. At the lower boundary, \\((\rho v\_x)\_0 < 0\\) is outflow and anything else is inflow; at the upper boundary, \\((\rho v\_x)\_N > 0\\) is outflow. Because the flag reads only valid data, the cell-centred fill and the face-centred pass always agree.

| Quantity | Outflow column | Inflow column |
|---|---|---|
| \\(\rho\\), \\(\rho \mathbf{v}\_t\\), internal energy, passive scalars | copy of cell \\(0\\) | mirror of cell \\(m-1\\) |
| \\(\rho v\_n\\) | copy of cell \\(0\\) | mirror of cell \\(m-1\\), sign reversed |
| \\(B\_t\\) on ghost transverse faces | copy of the faces of cell \\(0\\) | mirror of the faces of cell \\(m-1\\), same sign |
| \\(B\_n\\) on ghost faces \\(-1, \dots, -n\_g\\) | divergence-free recursion (step 2) | divergence-free recursion (step 2) |
| \\(B\_n\\) on the boundary face | never written | never written |
| total energy | copy of cell \\(0\\), then step 3 | mirror of cell \\(m-1\\), then step 3 |

In a column whose transverse faces are all copied, step 2 gives a linear extrapolation of \\(B\_x\\) (a copy in 1D). In a column whose transverse faces are all mirrored, it gives

<script type="math/tex; mode=display">
B_x(-m) = 2 B_x(0) - B_x(m),
</script>

that is, the deviation \\(B\_x - B\_x(0)\\) is odd about the wall: the conducting-wall mirror applied to the deviation from the wall value.

## Justification of the choices

### The divergence constraint holds by construction

Step 2 makes every ghost cell divergence-free for **any** values of the transverse ghost faces, because each new normal face is chosen to close the divergence of its cell, starting from the fixed boundary face. The divergence-free property therefore does not depend on the inflow/outflow rule, on how inflow and outflow columns are mixed, or on whether a column changed type since the last step. The valid faces, including the boundary face, are never touched, so the divergence of the valid cells is controlled by CT alone. This separation is the central design choice: the inflow/outflow logic only decides the transverse field and the gas state, and it cannot break R3.

### Zero mass flux

The HLLD solver uses the face value \\(B\_x\\) for both sides and symmetric outer speeds \\(S\_L = -S\_R\\) when the fast speeds of the two states are equal. Its middle speed is (eq. 38 of [@Miyoshi2005])

<script type="math/tex; mode=display">
S_M = \frac{(S_R - u_R)\rho_R u_R - (S_L - u_L)\rho_L u_L - p_{T,R} + p_{T,L}}{(S_R - u_R)\rho_R - (S_L - u_L)\rho_L}, \qquad p_T = P + \frac{1}{2}\left(B_x^2 + |\mathbf{B}_t|^2\right).
</script>

If the two face states have equal \\(\rho\\), \\(P\\) and \\(|\mathbf{B}\_t|\\) and opposite \\(u\\), then the numerator is exactly zero, also in floating point, because IEEE subtraction is sign-symmetric. So \\(S\_M = 0\\), and the mass flux of the star states, \\(\rho^{\ast} S\_M\\), vanishes. Mirror-symmetric face states follow from mirror-symmetric ghost data, since the reconstruction schemes are symmetric under reflection. The data are mirror-symmetric only if the ghost pressure is right (step 3) and the ghost transverse field is mirrored (step 1). Only \\(|\mathbf{B}\_t|^2\\) enters \\(S\_M\\), so both signs of the mirrored \\(B\_t\\) would give zero mass flux; the sign is chosen for other reasons (below).

The energy correction is the part that matters most in practice. Without step 3, the copied total energy contains the source-cell magnetic energy while the ghost face field differs, so the ghost pressure is wrong, the face states are asymmetric, and mass is lost: in the `MHDDiode` wall test, a fraction \\(3.4 \times 10^{-6}\\) with PLM and \\(1.3 \times 10^{-5}\\) with xPPM. With PPM, the mirrored velocity makes the wall cell a local extremum, the limiter reduces the reconstruction to first order there, and the error is hidden. The wall test therefore uses xPPM.

### Same-sign mirror of the transverse field

At inflow the tangential field is mirrored with the same sign (\\(B\_t\\) even), not with the opposite sign (\\(B\_t\\) odd, the pseudo-vector mirror). The reasons:

- **Switching.** The ghost tangential field keeps its sign when a column switches between inflow and outflow. With the odd mirror it flips at every switch, which makes a jump in the boundary-edge EMFs and in \\(B\_x(0)\\).
- **No artificial current sheet.** The odd mirror forces \\(B\_t = 0\\) at the wall and a current sheet along the whole inflow boundary. In low-\\(\beta\\) gas this drives strong Lorentz forces and numerical reconnection at the boundary.
- **Consistency with outflow.** At outflow the tangential field is copied (zero gradient). The even mirror keeps a zero normal gradient of \\(B\_t\\) at the wall, so the two cases differ as little as possible.

The price is that the inflow wall is not an exact mirror symmetry of ideal MHD when \\(B\_x(0) \neq 0\\). The mass flux still vanishes (above), but the wall can carry a tangential magnetic stress and a Poynting flux (see Limitations).

### Faces shared by inflow and outflow columns

A transverse ghost face between an inflow and an outflow column is treated as inflow. Then every inflow column has all of its transverse faces mirrored, and its ghost data, and so its boundary Riemann problem, are exactly mirror-symmetric, which R1 needs. An outflow column next to an inflow column gets one mirrored face, which changes its ghost state only slightly and does not affect R2 in any important way. R3 is not affected by this choice at all.

### Flag from the first valid cell

The flag is the sign of the normal momentum in the first valid cell, as in the hydro diode. It reads only valid data, so the cell-centred and face-centred parts of the fill always make the same decision, and the MHD diode reduces to the hydro diode for the gas variables.

### No EMF boundary condition

The boundary face is evolved by CT with EMFs computed from the ghost states, like any other valid face. Its update is a discrete curl, so it keeps every valid cell divergence-free for any EMF values. There is no need to set, copy or zero EMFs on the boundary, as done by [@Pjanka2020]. The coarse-fine EMF correction (`EdgeFluxRegister`) treats the boundary plane in the same way as for any other boundary type.

## Limitations

- **Magnetic flux can enter through inflow faces.** The diode is a condition on the mass flux only. Where \\(B\_x(0) \neq 0\\), the tangential motion at the wall gives a non-zero EMF (for example \\(\mathcal{E}\_z \simeq v\_y B\_x\\)) on the boundary edges, so the boundary-face normal field still evolves: field-line footpoints slide with the flow along the wall, and magnetic flux can enter or leave through inflow faces. The wall can also exert a tangential magnetic stress (\\(-B\_x B\_t^{\ast}\\)) and carry a Poynting flux (\\(-B\_x \mathbf{v}^{\ast} \cdot \mathbf{B}^{\ast}\\)), so total energy is not conserved at inflow faces even though mass is.
- **Corners.** Where two diode boundaries meet, the corner ghost cells are filled by the pass that runs last, so their total energy correction may use a source state from the other direction. Corner ghost cells are divergence-free but not otherwise consistent.
- **Non-zero-gradient outflow.** In multi-dimensional outflow columns the ghost normal field is linearly extrapolated, not copied, because that is what the divergence constraint requires.

## Verification

The test problem `MHDDiode` runs in a thin periodic-\\(z\\) box (MHD in Quokka is 3D only), with an oblique divergence-free field whose normal component and tangential gradient are non-zero at the walls, and a density gradient at the walls. Both inputs check \\(\max \Delta x |\nabla \cdot \mathbf{B}| / |\mathbf{B}| < 10^{-12}\\) over all valid **and ghost** cells after the ghost fill, at every step.

- `MHDDiodeWall.toml`: converging flow \\(v\_x = -v\_0 x\\), so every boundary column is inflow; xPPM reconstruction. It checks that the total mass is conserved to \\(10^{-12}\\) and asserts that every column stayed inflow.
- `MHDDiodeMixed.toml`: \\(v\_x = v\_0 \sin(2 \pi y)\\), so half of each wall is outflow and half is inflow. It checks that the total mass never increases.

Two variants of the fill each fail these tests. Disabling the pass fails the divergence check in both inputs and the mass check in the wall input. Copying and flipping the gas at inflow fails the mass check.

## References

See the [bibliography](bibliography.html) for [@Pjanka2020] and [@Miyoshi2005].
