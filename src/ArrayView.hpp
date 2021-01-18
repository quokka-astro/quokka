// These macros are defined such that, e.g., Array4View<X2>::operator(LOOP_ORDER_X2(i,j,k)) == arr_(i,j,k).
// Therefore, they do NOT have the same index ordering as that inside the corresponding Array4View<>::operator()!
//#define REORDER_X1(i,j,k) i,j,k
//#define REORDER_X2(i,j,k) j,k,i
//#define REORDER_X3(i,j,k) k,i,j

enum array4ViewIndexOrderList { X1 = 0, X2 = 1, X3 = 2 };

template <int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex(int, int, int);

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<X1>(int i, int j, int k)
{
	return std::make_tuple(i,j,k);
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<X2>(int i, int j, int k)
{
	return std::make_tuple(j,k,i);
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<X3>(int i, int j, int k)
{
	return std::make_tuple(k,i,j);
}

template <class T, int N, class Enable = void> struct Array4View {
	amrex::Array4<T> arr_;
	constexpr static int indexOrder = N;

	Array4View(amrex::Array4<T> arr) : arr_(arr) {}
};

// X1

// if T is non-const
template <class T> struct Array4View<T, X1, std::enable_if_t<!std::is_const<T>::value>> {
	amrex::Array4<T> arr_;
	constexpr static int indexOrder = X1;

	Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double &operator()(int i, int j, int k) noexcept
	{
		return arr_(i, j, k);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double operator()(int i, int j,
								   int k) const noexcept
	{
		return arr_(i, j, k);
	}
};

// if T is const
template <class T> struct Array4View<T, X1, std::enable_if_t<std::is_const<T>::value>> {
	amrex::Array4<T> arr_;
	constexpr static int indexOrder = X1;

	Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double operator()(int i, int j,
								   int k) const noexcept
	{
		return arr_(i, j, k);
	}
};


// X2-flux

// if T is non-const
template <class T> struct Array4View<T, X2, std::enable_if_t<!std::is_const<T>::value>> {
	amrex::Array4<T> arr_;
	constexpr static int indexOrder = X2;

	Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double &operator()(int i, int j, int k) noexcept
	{
		return arr_(k, i, j);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double operator()(int i, int j,
								   int k) const noexcept
	{
		return arr_(k, i, j);
	}
};

// if T is const
template <class T> struct Array4View<T, X2, std::enable_if_t<std::is_const<T>::value>> {
	amrex::Array4<T> arr_;
	constexpr static int indexOrder = X2;

	Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double operator()(int i, int j,
								   int k) const noexcept
	{
		return arr_(k, i, j);
	}
};


// X3-flux

// if T is non-const
template <class T> struct Array4View<T, X3, std::enable_if_t<!std::is_const<T>::value>> {
	amrex::Array4<T> arr_;
	constexpr static int indexOrder = X3;

	Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double &operator()(int i, int j, int k) noexcept
	{
		return arr_(j, k, i);
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double operator()(int i, int j,
								   int k) const noexcept
	{
		return arr_(j, k, i);
	}
};

// if T is const
template <class T> struct Array4View<T, X3, std::enable_if_t<std::is_const<T>::value>> {
	amrex::Array4<T> arr_;
	constexpr static int indexOrder = X3;

	Array4View(amrex::Array4<T> arr) : arr_(arr) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double operator()(int i, int j,
								   int k) const noexcept
	{
		return arr_(j, k, i);
	}
};