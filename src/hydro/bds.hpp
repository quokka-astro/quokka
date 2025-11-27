#ifndef BDS_HPP_
#define BDS_HPP_

#include "AMReX_MultiFab.H"
#include <array>

namespace bds
{
	constexpr amrex::Real weight0 = -1.0 / 12.0;
	constexpr amrex::Real weight1 = 7.0 / 12.0;
	constexpr amrex::Real epsilon = 1.0e-12;
	constexpr int max_iter = 20;

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cubic_interp_fast(amrex::Real v_m1, amrex::Real v_0, amrex::Real v_p1, amrex::Real v_p2) -> amrex::Real
	{
		return weight0 * (v_m1 + v_p2) + weight1 * (v_0 + v_p1);
	}

	template <int dir0, int dir1>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeCornerValues2D(amrex::Array4<const amrex::Real> const &src, int i, int j, int k, int n)
	    -> std::array<amrex::Real, 4>
	{
		static_assert(dir0 != dir1, "dir0 and dir1 must be different");
		std::array<amrex::Real, 4> c{};
		amrex::Real b_min[4];
		amrex::Real b_max[4];

		auto sample = [&](int offset0, int offset1) -> amrex::Real {
			std::array<int, 3> idx{AMREX_D_DECL(i, j, k)};
			idx[dir0] += offset0;
			idx[dir1] += offset1;
			return src(idx[0], idx[1], idx[2], n);
		};

		amrex::Real nbr[3][3];
		for (int dy = 0; dy < 3; ++dy) {
			for (int dx = 0; dx < 3; ++dx) {
				nbr[dy][dx] = sample(dx - 1, dy - 1);
			}
		}

		amrex::Real s_avg = nbr[1][1];

		for (int corn_idx = 0; corn_idx < 4; ++corn_idx) {
			int ky = corn_idx / 2;
			int kx = corn_idx % 2;

			amrex::Real local_min = 1.0e30;
			amrex::Real local_max = -1.0e30;
			for (int y = 0; y < 2; ++y) {
				for (int x = 0; x < 2; ++x) {
					amrex::Real val = sample(kx + x - 1, ky + y - 1);
					local_min = amrex::min(local_min, val);
					local_max = amrex::max(local_max, val);
				}
			}
			b_min[corn_idx] = local_min;
			b_max[corn_idx] = local_max;

			int bi = i + kx - 2;
			int bj = j + ky - 2;

			amrex::Real y_lines[4];
			for (int y_iter = 0; y_iter < 4; ++y_iter) {
				amrex::Real v0 = sample(bi - i + 0, bj - j + y_iter);
				amrex::Real v1 = sample(bi - i + 1, bj - j + y_iter);
				amrex::Real v2 = sample(bi - i + 2, bj - j + y_iter);
				amrex::Real v3 = sample(bi - i + 3, bj - j + y_iter);
				y_lines[y_iter] = cubic_interp_fast(v0, v1, v2, v3);
			}
			c[corn_idx] = cubic_interp_fast(y_lines[0], y_lines[1], y_lines[2], y_lines[3]);
		}

		amrex::Real sum_corners = 0.0;
		for (auto const val : c) {
			sum_corners += val;
		}
		amrex::Real shift = s_avg - (sum_corners * 0.25);

		for (int idx = 0; idx < 4; ++idx) {
			c[idx] += shift;
			c[idx] = amrex::max(b_min[idx], amrex::min(b_max[idx], c[idx]));
		}

		for (int iter = 0; iter < max_iter; ++iter) {
			amrex::Real max_abs = amrex::Math::abs(s_avg);
			for (int idx = 0; idx < 4; ++idx) {
				max_abs = amrex::max(max_abs, amrex::Math::abs(c[idx]));
			}
			amrex::Real tol = epsilon * max_abs * 4.0;
			if (max_abs == 0.0) {
				tol = epsilon * static_cast<amrex::Real>(1.0e-40);
			}

			amrex::Real sum_curr = 0.0;
			for (int idx = 0; idx < 4; ++idx) {
				sum_curr += c[idx];
			}

			amrex::Real delta = sum_curr - 4.0 * s_avg;
			if (amrex::Math::abs(delta) <= tol) {
				break;
			}

			if (delta > 0.0) {
				int count = 0;
				bool is_cand[4];
				for (int idx = 0; idx < 4; ++idx) {
					is_cand[idx] = (c[idx] > (s_avg + tol));
					count += is_cand[idx] ? 1 : 0;
				}
				if (count == 0) {
					break;
				}
				for (int idx = 0; idx < 4 && delta > tol; ++idx) {
					if (!is_cand[idx]) {
						continue;
					}
					amrex::Real headroom = c[idx] - b_min[idx];
					amrex::Real share = delta / static_cast<amrex::Real>(count);
					amrex::Real gamma = amrex::min(share, headroom);
					c[idx] -= gamma;
					delta -= gamma;
					--count;
				}
			} else {
				delta = -delta;
				int count = 0;
				bool is_cand[4];
				for (int idx = 0; idx < 4; ++idx) {
					is_cand[idx] = (c[idx] < (s_avg - tol));
					count += is_cand[idx] ? 1 : 0;
				}
				if (count == 0) {
					break;
				}
				for (int idx = 0; idx < 4 && delta > tol; ++idx) {
					if (!is_cand[idx]) {
						continue;
					}
					amrex::Real headroom = b_max[idx] - c[idx];
					amrex::Real share = delta / static_cast<amrex::Real>(count);
					amrex::Real gamma = amrex::min(share, headroom);
					c[idx] += gamma;
					delta -= gamma;
					--count;
				}
			}
		}

		return c;
	}

	template <int dir>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Reconstruct1DToEdges(amrex::Array4<const amrex::Real> const &src,
						   amrex::Array4<amrex::Real> const &low_side,
						   amrex::Array4<amrex::Real> const &high_side, amrex::Box const &box, int ncomp = 1)
	{
		static_assert(dir >= 0 && dir < AMREX_SPACEDIM, "Invalid direction for BDS reconstruction");
		amrex::ParallelFor(box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
			auto sample = [&](int off) -> amrex::Real {
				if constexpr (dir == 0) {
					return src(i + off, j, k, n);
				} else if constexpr (dir == 1) {
					return src(i, j + off, k, n);
				} else {
					return src(i, j, k + off, n);
				}
			};

			amrex::Real s_avg = sample(0);

			amrex::Real left_min = amrex::min(sample(-1), s_avg);
			amrex::Real left_max = amrex::max(sample(-1), s_avg);
			amrex::Real right_min = amrex::min(s_avg, sample(1));
			amrex::Real right_max = amrex::max(s_avg, sample(1));

			amrex::Real c_left = cubic_interp_fast(sample(-2), sample(-1), s_avg, sample(1));
			amrex::Real c_right = cubic_interp_fast(sample(-1), s_avg, sample(1), sample(2));

			amrex::Real shift = s_avg - 0.5 * (c_left + c_right);
			c_left += shift;
			c_right += shift;

			c_left = amrex::max(left_min, amrex::min(left_max, c_left));
			c_right = amrex::max(right_min, amrex::min(right_max, c_right));

			for (int iter = 0; iter < max_iter; ++iter) {
				amrex::Real max_abs = amrex::Math::abs(s_avg);
				max_abs = amrex::max(max_abs, amrex::Math::abs(c_left));
				max_abs = amrex::max(max_abs, amrex::Math::abs(c_right));
				amrex::Real tol = epsilon * max_abs * 2.0;
				if (max_abs == 0.0) {
					tol = epsilon * static_cast<amrex::Real>(1.0e-40);
				}

				amrex::Real sum_curr = c_left + c_right;
				amrex::Real delta = sum_curr - 2.0 * s_avg;
				if (amrex::Math::abs(delta) <= tol) {
					break;
				}

				if (delta > 0.0) {
					int count = 0;
					bool left_cand = (c_left > (s_avg + tol));
					bool right_cand = (c_right > (s_avg + tol));
					count += left_cand ? 1 : 0;
					count += right_cand ? 1 : 0;
					if (count == 0) {
						break;
					}
					if (left_cand && delta > tol) {
						amrex::Real headroom = c_left - left_min;
						amrex::Real share = delta / static_cast<amrex::Real>(count);
						amrex::Real gamma = amrex::min(share, headroom);
						c_left -= gamma;
						delta -= gamma;
						--count;
					}
					if (right_cand && delta > tol) {
						amrex::Real headroom = c_right - right_min;
						amrex::Real share = (count > 0) ? (delta / static_cast<amrex::Real>(count)) : delta;
						amrex::Real gamma = amrex::min(share, headroom);
						c_right -= gamma;
						delta -= gamma;
					}
				} else {
					delta = -delta;
					int count = 0;
					bool left_cand = (c_left < (s_avg - tol));
					bool right_cand = (c_right < (s_avg - tol));
					count += left_cand ? 1 : 0;
					count += right_cand ? 1 : 0;
					if (count == 0) {
						break;
					}
					if (left_cand && delta > tol) {
						amrex::Real headroom = left_max - c_left;
						amrex::Real share = delta / static_cast<amrex::Real>(count);
						amrex::Real gamma = amrex::min(share, headroom);
						c_left += gamma;
						delta -= gamma;
						--count;
					}
					if (right_cand && delta > tol) {
						amrex::Real headroom = right_max - c_right;
						amrex::Real share = (count > 0) ? (delta / static_cast<amrex::Real>(count)) : delta;
						amrex::Real gamma = amrex::min(share, headroom);
						c_right += gamma;
						delta -= gamma;
					}
				}
			}

			if constexpr (dir == 0) {
				high_side(i, j, k, n) = c_left;
				low_side(i + 1, j, k, n) = c_right;
			} else if constexpr (dir == 1) {
				high_side(i, j, k, n) = c_left;
				low_side(i, j + 1, k, n) = c_right;
			} else {
				high_side(i, j, k, n) = c_left;
				low_side(i, j, k + 1, n) = c_right;
			}
		});
	}
} // namespace bds

void ComputeBDSReconstructionOptimized(amrex::MultiFab const &input_mf, amrex::MultiFab &x_L, amrex::MultiFab &x_R, amrex::MultiFab &y_L, amrex::MultiFab &y_R,
			       amrex::MultiFab &z_L, amrex::MultiFab &z_R, int num_ghost);

#endif // BDS_HPP_
