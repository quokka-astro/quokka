#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>

using Real = amrex::Real;
using MultiFab = amrex::MultiFab;
using Box = amrex::Box;

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------
constexpr Real W0 = -1.0 / 12.0;
constexpr Real W1 = 7.0 / 12.0;
constexpr Real EPSILON = 1.0e-10;
constexpr int MAX_ITER = 6;

// ------------------------------------------------------------------
// Optimized Helper
// ------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cubic_interp_fast(Real v_m1, Real v_0, Real v_p1, Real v_p2) -> Real
{
	// Fused multiply-adds are generated better if written this way
	return W0 * (v_m1 + v_p2) + W1 * (v_0 + v_p1);
}

#if AMREX_SPACEDIM == 1
void ComputeBdsReconstruction1D(const MultiFab &input_mf, MultiFab &x_L, MultiFab &x_R, MultiFab &y_L, MultiFab &y_R, MultiFab &z_L, MultiFab &z_R,
				int num_ghost)
{
	amrex::ignore_unused(y_L, y_R, z_L, z_R);

	AMREX_ASSERT(num_ghost >= 0);
	AMREX_ASSERT(input_mf.nGrow() >= num_ghost + 2);
	AMREX_ASSERT(x_L.nGrow() >= num_ghost);
	AMREX_ASSERT(x_R.nGrow() >= num_ghost);
	AMREX_ASSERT(x_L.nComp() == input_mf.nComp());
	AMREX_ASSERT(x_R.nComp() == input_mf.nComp());

	for (amrex::MFIter mfi(input_mf); mfi.isValid(); ++mfi) {
		const Box &bx = mfi.growntilebox(num_ghost);
		int const ncomp = input_mf.nComp();

		auto const &src = input_mf.array(mfi);
		auto const &xl = x_L.array(mfi);
		auto const &xr = x_R.array(mfi);

		amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
			Real s_avg = src(i, j, k, n);

			Real left_min = amrex::min(src(i - 1, j, k, n), s_avg);
			Real left_max = amrex::max(src(i - 1, j, k, n), s_avg);
			Real right_min = amrex::min(s_avg, src(i + 1, j, k, n));
			Real right_max = amrex::max(s_avg, src(i + 1, j, k, n));

			Real c_left = cubic_interp_fast(src(i - 2, j, k, n), src(i - 1, j, k, n), s_avg, src(i + 1, j, k, n));
			Real c_right = cubic_interp_fast(src(i - 1, j, k, n), s_avg, src(i + 1, j, k, n), src(i + 2, j, k, n));

			Real shift = s_avg - 0.5 * (c_left + c_right);
			c_left += shift;
			c_right += shift;

			c_left = amrex::max(left_min, amrex::min(left_max, c_left));
			c_right = amrex::max(right_min, amrex::min(right_max, c_right));

			for (int iter = 0; iter < MAX_ITER; ++iter) {
				Real diff = (c_left + c_right) - 2.0 * s_avg;
				Real sign_mask = (diff > 0.0) ? 1.0 : -1.0;
				Real target_limit = s_avg + sign_mask * EPSILON;

				Real mask_left = (diff > 0.0) ? (c_left > target_limit ? 1.0 : 0.0) : (c_left < target_limit ? 1.0 : 0.0);
				Real mask_right = (diff > 0.0) ? (c_right > target_limit ? 1.0 : 0.0) : (c_right < target_limit ? 1.0 : 0.0);
				Real count = mask_left + mask_right;

				if (count < 0.5) {
					break;
				}

				Real correction = diff / count;
				Real abs_correction = amrex::Math::abs(correction);

				Real dist_left = amrex::Math::abs(c_left - s_avg);
				Real delta_left = -sign_mask * amrex::min(abs_correction, dist_left);
				c_left += mask_left * delta_left;

				Real dist_right = amrex::Math::abs(c_right - s_avg);
				Real delta_right = -sign_mask * amrex::min(abs_correction, dist_right);
				c_right += mask_right * delta_right;
			}

			xl(i, j, k, n) = c_left;
			xr(i, j, k, n) = c_right;
		});
	}
}
#endif

#if AMREX_SPACEDIM == 2
void ComputeBdsReconstruction2D(const MultiFab &input_mf, MultiFab &x_L, MultiFab &x_R, MultiFab &y_L, MultiFab &y_R, MultiFab &z_L, MultiFab &z_R, int num_ghost)
{
	amrex::ignore_unused(z_L, z_R);

	AMREX_ASSERT(num_ghost >= 0);
	AMREX_ASSERT(input_mf.nGrow() >= num_ghost + 2);
	AMREX_ASSERT(x_L.nGrow() >= num_ghost);
	AMREX_ASSERT(x_R.nGrow() >= num_ghost);
	AMREX_ASSERT(y_L.nGrow() >= num_ghost);
	AMREX_ASSERT(y_R.nGrow() >= num_ghost);
	AMREX_ASSERT(x_L.nComp() == input_mf.nComp());
	AMREX_ASSERT(x_R.nComp() == input_mf.nComp());
	AMREX_ASSERT(y_L.nComp() == input_mf.nComp());
	AMREX_ASSERT(y_R.nComp() == input_mf.nComp());

	for (amrex::MFIter mfi(input_mf); mfi.isValid(); ++mfi) {
		const Box &bx = mfi.growntilebox(num_ghost);
		int const ncomp = input_mf.nComp();

		auto const &src = input_mf.array(mfi);
		auto const &xl = x_L.array(mfi);
		auto const &xr = x_R.array(mfi);
		auto const &yl = y_L.array(mfi);
		auto const &yr = y_R.array(mfi);
		auto const &zl = z_L.array(mfi);
		auto const &zr = z_R.array(mfi);

		amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
			Real nbr[3][3]; // NOLINT

			for (int dy = 0; dy < 3; ++dy) {
				for (int dx = 0; dx < 3; ++dx) {
					nbr[dy][dx] = src(i + dx - 1, j + dy - 1, k, n);
				}
			}

			Real s_avg = nbr[1][1];

			Real c[4];     // corner values // NOLINT
			Real b_min[4]; // min bound // NOLINT
			Real b_max[4]; // max bound // NOLINT

#pragma unroll
			for (int corn_idx = 0; corn_idx < 4; ++corn_idx) {
				int ky = corn_idx / 2;
				int kx = corn_idx % 2;

				Real local_min = 1.0e30;
				Real local_max = -1.0e30;

				for (int y = 0; y < 2; ++y) {
					for (int x = 0; x < 2; ++x) {
						Real val = nbr[ky + y][kx + x];
						local_min = amrex::min(local_min, val);
						local_max = amrex::max(local_max, val);
					}
				}
				b_min[corn_idx] = local_min;
				b_max[corn_idx] = local_max;

				int bi = i + kx - 2;
				int bj = j + ky - 2;

				Real y_lines[4]; // NOLINT
				for (int y_iter = 0; y_iter < 4; ++y_iter) {
					Real v0 = src(bi + 0, bj + y_iter, k, n);
					Real v1 = src(bi + 1, bj + y_iter, k, n);
					Real v2 = src(bi + 2, bj + y_iter, k, n);
					Real v3 = src(bi + 3, bj + y_iter, k, n);
					y_lines[y_iter] = cubic_interp_fast(v0, v1, v2, v3);
				}
				c[corn_idx] = cubic_interp_fast(y_lines[0], y_lines[1], y_lines[2], y_lines[3]);
			}

			Real sum_corners = 0.0;
			for (int idx = 0; idx < 4; ++idx) {
				sum_corners += c[idx];
			}
			Real shift = s_avg - (sum_corners * 0.25);

			for (int idx = 0; idx < 4; ++idx) {
				c[idx] += shift;
				c[idx] = amrex::max(b_min[idx], amrex::min(b_max[idx], c[idx]));
			}

			for (int iter = 0; iter < MAX_ITER; ++iter) {
				Real sum_curr = 0.0;
				for (int idx = 0; idx < 4; ++idx) {
					sum_curr += c[idx];
				}

				Real diff = sum_curr - 4.0 * s_avg;
				Real sign_mask = (diff > 0.0) ? 1.0 : -1.0;
				Real target_limit = s_avg + sign_mask * EPSILON;

				Real count = 0.0;
				Real mask[4]; // NOLINT
				for (int idx = 0; idx < 4; ++idx) {
					bool is_cand = (diff > 0.0) ? (c[idx] > target_limit) : (c[idx] < target_limit);
					mask[idx] = is_cand ? 1.0 : 0.0;
					count += mask[idx];
				}

				if (count < 0.5) {
					break;
				}

				Real correction = diff / count;
				Real abs_correction = amrex::Math::abs(correction);

				for (int idx = 0; idx < 4; ++idx) {
					Real dist = amrex::Math::abs(c[idx] - s_avg);
					Real act_chg = amrex::min(abs_correction, dist);
					Real delta = -sign_mask * act_chg;
					c[idx] += mask[idx] * delta;
				}
			}

			xl(i, j, k, n) = 0.5 * (c[0] + c[2]);
			xr(i, j, k, n) = 0.5 * (c[1] + c[3]);

			yl(i, j, k, n) = 0.5 * (c[0] + c[1]);
			yr(i, j, k, n) = 0.5 * (c[2] + c[3]);

			// z-face states are unused in 2D but set to avoid uninitialized data.
			zl(i, j, k, n) = 0.0;
			zr(i, j, k, n) = 0.0;
		});
	}
}
#endif

#if AMREX_SPACEDIM == 3
void ComputeBdsReconstruction3D(const MultiFab &input_mf, MultiFab &x_L, MultiFab &x_R, MultiFab &y_L, MultiFab &y_R, MultiFab &z_L, MultiFab &z_R, int num_ghost)
{
	AMREX_ASSERT(num_ghost >= 0);
	AMREX_ASSERT(input_mf.nGrow() >= num_ghost + 2);
	AMREX_ASSERT(x_L.nGrow() >= num_ghost);
	AMREX_ASSERT(x_R.nGrow() >= num_ghost);
	AMREX_ASSERT(y_L.nGrow() >= num_ghost);
	AMREX_ASSERT(y_R.nGrow() >= num_ghost);
	AMREX_ASSERT(z_L.nGrow() >= num_ghost);
	AMREX_ASSERT(z_R.nGrow() >= num_ghost);
	AMREX_ASSERT(x_L.nComp() == input_mf.nComp());
	AMREX_ASSERT(x_R.nComp() == input_mf.nComp());
	AMREX_ASSERT(y_L.nComp() == input_mf.nComp());
	AMREX_ASSERT(y_R.nComp() == input_mf.nComp());
	AMREX_ASSERT(z_L.nComp() == input_mf.nComp());
	AMREX_ASSERT(z_R.nComp() == input_mf.nComp());

	for (amrex::MFIter mfi(input_mf); mfi.isValid(); ++mfi) {
		const Box &bx = mfi.growntilebox(num_ghost);
		int const ncomp = input_mf.nComp();

		auto const &src = input_mf.array(mfi);
		auto const &xl = x_L.array(mfi);
		auto const &xr = x_R.array(mfi);
		auto const &yl = y_L.array(mfi);
		auto const &yr = y_R.array(mfi);
		auto const &zl = z_L.array(mfi);
		auto const &zr = z_R.array(mfi);

		amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
			// ----------------------------------------------------------
			// Optimization 1: Register Cache for 3x3x3 Neighborhood
			// ----------------------------------------------------------
			// We load the immediate neighbors (i-1 to i+1) into registers.
			// These are used heavily for the bounds check (Constraint 2).
			// Mapping: nbr[1][1][1] is center (i,j,k)
			Real nbr[3][3][3]; // NOLINT

			// Manual unroll helps compiler fuse loads
			for (int dz = 0; dz < 3; ++dz) {
				for (int dy = 0; dy < 3; ++dy) {
					for (int dx = 0; dx < 3; ++dx) {
						nbr[dz][dy][dx] = src(i + dx - 1, j + dy - 1, k + dz - 1, n);
					}
				}
			}

			Real s_avg = nbr[1][1][1];

			// Corner Storage: Flat array is often better for register indexing
			Real c[8];     // The corner values // NOLINT
			Real b_min[8]; // Min bound // NOLINT
			Real b_max[8]; // Max bound // NOLINT

			// ----------------------------------------------------------
			// Part A: Tricubic Interpolation & Bounds Calculation
			// ----------------------------------------------------------
#pragma unroll
			for (int corn_idx = 0; corn_idx < 8; ++corn_idx) {
				// Decode corner index to 0 (Low) / 1 (High)
				int kz = corn_idx / 4;
				int ky = (corn_idx / 2) % 2;
				int kx = corn_idx % 2;

				// -- 1. Bounds Check (Using Register Cache) --
				// We need min/max of the 8 cells surrounding this corner.
				// The corner is at (kx-0.5, ky-0.5, kz-0.5) relative to center.
				// The 8 cells correspond to indices [kx-1 .. kx] in the nbr array.
				// Note: nbr index 0 is i-1, index 1 is i.
				// If kx=0 (left), we look at nbr 0 and 1. If kx=1 (right), we look at 1 and 2.

				Real local_min = 1.0e30;
				Real local_max = -1.0e30;

				// Loop over the 2x2x2 block in the Register Cache
				for (int z = 0; z < 2; ++z) {
					for (int y = 0; y < 2; ++y) {
						for (int x = 0; x < 2; ++x) {
							Real val = nbr[kz + z][ky + y][kx + x];
							local_min = amrex::min(local_min, val);
							local_max = amrex::max(local_max, val);
						}
					}
				}
				b_min[corn_idx] = local_min;
				b_max[corn_idx] = local_max;

				// -- 2. Tricubic Interpolation --
				// We must fall back to 'src' for the full 4x4x4 stencil,
				// but the central 2x2x2 part is already in 'nbr'.
				// For simplicity and code size, we just call src() here,
				// relying on L2 cache for the wider stencil points.
				int bi = i + kx - 2;
				int bj = j + ky - 2;
				int bk = k + kz - 2;

				Real z_lines[4]; // NOLINT
				for (int z_iter = 0; z_iter < 4; ++z_iter) {
					Real y_lines[4]; // NOLINT
					for (int y_iter = 0; y_iter < 4; ++y_iter) {
						// Note: Could optimize to use 'nbr' when indices align,
						// but divergence cost might outweigh load savings.
						Real v0 = src(bi + 0, bj + y_iter, bk + z_iter, n);
						Real v1 = src(bi + 1, bj + y_iter, bk + z_iter, n);
						Real v2 = src(bi + 2, bj + y_iter, bk + z_iter, n);
						Real v3 = src(bi + 3, bj + y_iter, bk + z_iter, n);
						y_lines[y_iter] = cubic_interp_fast(v0, v1, v2, v3);
					}
					z_lines[z_iter] = cubic_interp_fast(y_lines[0], y_lines[1], y_lines[2], y_lines[3]);
				}
				c[corn_idx] = cubic_interp_fast(z_lines[0], z_lines[1], z_lines[2], z_lines[3]);
			}

			// ----------------------------------------------------------
			// Part B: Optimization 2 - Branchless Limiting
			// ----------------------------------------------------------

			// 1. Enforce Mean
			Real sum_corners = 0.0;
			for (int k = 0; k < 8; ++k) {
				sum_corners += c[k];
			}
			Real shift = s_avg - (sum_corners * 0.125);

			for (int k = 0; k < 8; ++k) {
				c[k] += shift;
				// Clamp
				c[k] = amrex::max(b_min[k], amrex::min(b_max[k], c[k]));
			}

			// 2. Iterative Heuristic (Branchless)
			for (int iter = 0; iter < MAX_ITER; ++iter) {
				Real sum_curr = 0.0;
				for (int k = 0; k < 8; ++k) {
					sum_curr += c[k];
				}

				Real diff = sum_curr - 8.0 * s_avg;

				// Early exit check - requires reduction across warp or all threads continuing
				// Since we are in ParallelFor (SIMT), strict early exit causes divergence.
				// Better to just run the iterations (only 6) or use a predicate mask.
				// However, if diff is 0, the math below results in 0 correction, so it's safe.

				Real sign_mask = (diff > 0.0) ? 1.0 : -1.0;
				// If diff > 0, target > s_avg. If diff < 0, target < s_avg.
				Real target_limit = s_avg + sign_mask * EPSILON;

				Real count = 0.0;
				Real mask[8]; // 1.0 if candidate, 0.0 if not   // NOLINT

				for (int k = 0; k < 8; ++k) {
					// Logic: if (sign > 0) return val > target; else return val < target;
					bool is_cand = (diff > 0.0) ? (c[k] > target_limit) : (c[k] < target_limit);
					mask[k] = is_cand ? 1.0 : 0.0;
					count += mask[k];
				}

				// Avoid divide by zero if count == 0
				if (count < 0.5) {
					break;
				}

				Real correction = diff / count;
				Real abs_correction = amrex::Math::abs(correction);

				for (int k = 0; k < 8; ++k) {
					// Calculate max allowed change: |val - s_avg|
					Real dist = amrex::Math::abs(c[k] - s_avg);

					// Actual change = min(|correction|, dist)
					Real act_chg = amrex::min(abs_correction, dist);

					// Apply direction: if diff > 0, we subtract. if diff < 0, we add.
					// Direction = -sign_mask
					Real delta = -sign_mask * act_chg;

					// Apply mask
					c[k] += mask[k] * delta;
				}
			}

			// ----------------------------------------------------------
			// Part C: Face Averaging
			// ----------------------------------------------------------
			// Macros to map c[0..7] to z,y,x
			// 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111

			// X Faces (Stride 1)
			xl(i, j, k, n) = 0.25 * (c[0] + c[2] + c[4] + c[6]); // x=0
			xr(i, j, k, n) = 0.25 * (c[1] + c[3] + c[5] + c[7]); // x=1

			// Y Faces (Stride 2)
			yl(i, j, k, n) = 0.25 * (c[0] + c[1] + c[4] + c[5]); // y=0
			yr(i, j, k, n) = 0.25 * (c[2] + c[3] + c[6] + c[7]); // y=1

			// Z Faces (Stride 4)
			zl(i, j, k, n) = 0.25 * (c[0] + c[1] + c[2] + c[3]); // z=0
			zr(i, j, k, n) = 0.25 * (c[4] + c[5] + c[6] + c[7]); // z=1
		});
	}
}
#endif

void ComputeBDSReconstructionOptimized(const MultiFab &input_mf, MultiFab &x_L, MultiFab &x_R, MultiFab &y_L, MultiFab &y_R, MultiFab &z_L, MultiFab &z_R, int num_ghost)
{
	AMREX_ASSERT(num_ghost >= 0);
	AMREX_ASSERT(input_mf.nGrow() >= num_ghost + 2);
	AMREX_ASSERT(x_L.nGrow() >= num_ghost);
	AMREX_ASSERT(x_R.nGrow() >= num_ghost);
	AMREX_ASSERT(x_L.nComp() == input_mf.nComp());
	AMREX_ASSERT(x_R.nComp() == input_mf.nComp());
#if AMREX_SPACEDIM >= 2
	AMREX_ASSERT(y_L.nGrow() >= num_ghost);
	AMREX_ASSERT(y_R.nGrow() >= num_ghost);
	AMREX_ASSERT(y_L.nComp() == input_mf.nComp());
	AMREX_ASSERT(y_R.nComp() == input_mf.nComp());
#endif
#if AMREX_SPACEDIM == 3
	AMREX_ASSERT(z_L.nGrow() >= num_ghost);
	AMREX_ASSERT(z_R.nGrow() >= num_ghost);
	AMREX_ASSERT(z_L.nComp() == input_mf.nComp());
	AMREX_ASSERT(z_R.nComp() == input_mf.nComp());
#endif

#if AMREX_SPACEDIM == 3
	ComputeBdsReconstruction3D(input_mf, x_L, x_R, y_L, y_R, z_L, z_R, num_ghost);
#elif AMREX_SPACEDIM == 2
	// z_L/z_R are ignored in 2D but we keep the parameters for a uniform interface.
	ComputeBdsReconstruction2D(input_mf, x_L, x_R, y_L, y_R, z_L, z_R, num_ghost);
#else
	ComputeBdsReconstruction1D(input_mf, x_L, x_R, y_L, y_R, z_L, z_R, num_ghost);
#endif
}
