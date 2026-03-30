# Gaussian Particle Interpolator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gaussian kernel particle interpolator and use it for radiation deposition.

**Architecture:** A new `Gaussian<N>` struct in `amrex::ParticleInterpolator` namespace following the CRTP `Base<Derived, WeightType>` pattern used by `Linear` and `NearestEight`. Template parameter `N` controls stencil extent (default 3, max 6). `RadDeposition` switches from `Linear` to `Gaussian<>`.

**Tech Stack:** C++20, AMReX particle framework, GPU-compatible (`AMREX_GPU_DEVICE`)

---

### Task 1: Add the `Gaussian<N>` interpolator struct

**Files:**
- Modify: `src/particles/particle_deposition.hpp:19-50` (inside `amrex::ParticleInterpolator` namespace, after `NearestEight`)

- [ ] **Step 1: Add the `Gaussian` template struct after `NearestEight` (before the closing `}` of the namespace on line 50)**

Insert the following code after line 49 (`};` closing `NearestEight`) and before line 50 (`} // namespace`):

```cpp
/** \brief A class that implements Gaussian kernel interpolation.
 *
 *  Template parameter N controls stencil extent: the kernel covers
 *  2*N+1 cells per dimension (N cells in each direction from center).
 *  N is limited by the number of ghost cells (max 6).
 *  Weights are separable 1D Gaussians, normalized so the full 3D
 *  product sums to 1, ensuring exact conservation of deposited quantities.
 */
template <int N = 3>
struct Gaussian : public Base<Gaussian<N>, amrex::Real> {
	static constexpr int stencil_width = 2 * N + 1;
	static constexpr double sigma = 1.5; // Gaussian width in units of cell size (dx)

	static constexpr int nx = (AMREX_SPACEDIM >= 1) ? stencil_width - 1 : 0; // NOLINT
	static constexpr int ny = (AMREX_SPACEDIM >= 2) ? stencil_width - 1 : 0; // NOLINT
	static constexpr int nz = (AMREX_SPACEDIM >= 3) ? stencil_width - 1 : 0; // NOLINT

	amrex::Real weights[3 * stencil_width]; // NOLINT

	template <typename P>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE Gaussian(const P &p, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, // NOLINT
						     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi)
	{
		this->w = &weights[0]; // NOLINT
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			const amrex::Real l = (p.pos(i) - plo[i]) * dxi[i] + 0.5;
			this->index[i] = static_cast<int>(amrex::Math::floor(l)) - N;
			const amrex::Real frac = l - amrex::Math::floor(l);
			// Compute unnormalized 1D Gaussian weights
			amrex::Real sum = 0.0;
			for (int j = 0; j <= 2 * N; ++j) {
				const amrex::Real d = static_cast<amrex::Real>(N - j) + frac - 1.0;
				const amrex::Real wt = std::exp(-0.5 * d * d / (sigma * sigma));
				this->w[stencil_width * i + j] = wt;
				sum += wt;
			}
			// Normalize so weights sum to 1 in this dimension
			const amrex::Real inv_sum = 1.0 / sum;
			for (int j = 0; j <= 2 * N; ++j) {
				this->w[stencil_width * i + j] *= inv_sum;
			}
		}
		for (int i = AMREX_SPACEDIM; i < 3; ++i) {
			this->index[i] = 0;
			this->w[stencil_width * i + 0] = 1.0;
			for (int j = 1; j < stencil_width; ++j) {
				this->w[stencil_width * i + j] = 0.0;
			}
		}
	}
};
```

Key design notes for the implementer:
- `l = (p.pos(i) - plo[i]) * dxi[i] + 0.5` — same position calculation as `Linear`.
- `index[i] = floor(l) - N` — the stencil spans cells `floor(l)-N` to `floor(l)+N`.
- Distance from particle to cell at offset `j`: `d = (N - j) + frac - 1` where `frac = l - floor(l)`. When `j = N` and `frac = 0.5`, `d = -0.5` (half a cell away — correct).
- Each dimension's weights are independently normalized to sum to 1. Since the 3D weight is the product of separable 1D weights, `sum_3D = (sum_x)(sum_y)(sum_z) = 1*1*1 = 1`.
- Must use `this->w` and `this->index` because `Base` is a dependent base class of a template.
- For unused dimensions (`>= AMREX_SPACEDIM`): first weight = 1.0, rest = 0.0 (same pattern as `Linear`).

- [ ] **Step 2: Build to verify compilation**

Run:
```bash
quokka build 3d ParticleRadiation
```
Expected: Compiles successfully with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/particles/particle_deposition.hpp
git commit -m "feat: add Gaussian<N> particle interpolator

Add a new ParticleInterpolator::Gaussian<N> struct following the AMReX
CRTP Base pattern. Uses separable normalized 1D Gaussian weights with
configurable stencil extent N (default 3) and sigma (default 1.5 dx)."
```

---

### Task 2: Update `RadDeposition` to use `Gaussian`

**Files:**
- Modify: `src/particles/particle_deposition.hpp:77`

- [ ] **Step 1: Replace `Linear` with `Gaussian<>` in `RadDeposition::operator()`**

Change line 77 from:
```cpp
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
```
to:
```cpp
		amrex::ParticleInterpolator::Gaussian<> interp(p, plo, dxi);
```

Also update the comment on line 71 from:
```cpp
	// Operator to perform radiation deposition using linear interpolation
```
to:
```cpp
	// Operator to perform radiation deposition using Gaussian kernel interpolation
```

- [ ] **Step 2: Build to verify compilation**

Run:
```bash
quokka build 3d ParticleRadiation
```
Expected: Compiles successfully.

- [ ] **Step 3: Run the ParticleRadiation test**

Run:
```bash
quokka run 3d ParticleRadiation
```
Expected: Test passes — "Test passed: change of total energy within tolerance." The Gaussian weights are normalized to sum to 1, so total deposited energy is conserved identically to Linear.

- [ ] **Step 4: Run the ParticleRadiationLog variant**

Run:
```bash
quokka run 3d ParticleRadiationLog
```
Expected: Test passes.

- [ ] **Step 5: Run the ParticleRadiationFastlog variant**

Run:
```bash
quokka run 3d ParticleRadiationFastlog
```
Expected: Test passes.

- [ ] **Step 6: Commit**

```bash
git add src/particles/particle_deposition.hpp
git commit -m "feat: switch RadDeposition from Linear to Gaussian interpolator

Use the new Gaussian<> (N=3, sigma=1.5 dx) interpolator for radiation
energy deposition, spreading particle luminosity over a wider, smoother
kernel instead of 8-cell linear (CIC) interpolation."
```
