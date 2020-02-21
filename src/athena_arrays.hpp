#ifndef ATHENA_ARRAYS_HPP_
#define ATHENA_ARRAYS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file athena_arrays.hpp
//  \brief provides array classes valid in 1D to 5D.
//
//  The operator() is overloaded, e.g. elements of a 4D array of size
//  [N4xN3xN2xN1] are accessed as:  A(n,k,j,i) = A[i + N1*(j + N2*(k + N3*n))]
//  	(NOTE: By default, index "i" is stride-1.)

// C++ headers
#include <cstddef> // size_t
#include <utility> // make_pair

// external headers
#include "Kokkos_Core.hpp"

template <typename T> class AthenaArray
{
      public:
	AthenaArray();
	~AthenaArray();
	// define copy constructor and overload assignment operator so both do
	// deep copies.
	AthenaArray(const AthenaArray<T> &t) noexcept;
	AthenaArray<T> &operator=(const AthenaArray<T> &t) noexcept;

	// public functions to allocate/deallocate memory for 1D-5D data
	void NewAthenaArray(int nx1) noexcept;
	void NewAthenaArray(int nx2, int nx1) noexcept;
	void NewAthenaArray(int nx3, int nx2, int nx1) noexcept;
	void NewAthenaArray(int nx4, int nx3, int nx2, int nx1) noexcept;
	void NewAthenaArray(int nx5, int nx4, int nx3, int nx2,
			    int nx1) noexcept;
	void DeleteAthenaArray();

	// public function to (shallow) swap data pointers of two equally-sized
	// arrays
	void SwapAthenaArray(AthenaArray<T> &array2);
	void ZeroClear();

	// functions to get array dimensions
	int GetDim1() const { return nx1_; }
	int GetDim2() const { return nx2_; }
	int GetDim3() const { return nx3_; }
	int GetDim4() const { return nx4_; }
	int GetDim5() const { return nx5_; }

	// a function to get the total size of the array
	int GetSize() const { return nx1_ * nx2_ * nx3_ * nx4_ * nx5_; }
	size_t GetSizeInBytes() const
	{
		return nx1_ * nx2_ * nx3_ * nx4_ * nx5_ * sizeof(T);
	}

	bool IsShallowCopy() { return (scopy_ == true); }
	T *data() { return pdata_; }
	const T *data() const { return pdata_; }

	// overload operator() to access 1d-5d data
	T &operator()(const int n) { return pdata_[n]; }
	T operator()(const int n) const { return pdata_[n]; }

	T &operator()(const int n, const int i) { return pdata_[i + nx1_ * n]; }
	T operator()(const int n, const int i) const
	{
		return pdata_[i + nx1_ * n];
	}

	T &operator()(const int n, const int j, const int i)
	{
		return pdata_[i + nx1_ * (j + nx2_ * n)];
	}
	T operator()(const int n, const int j, const int i) const
	{
		return pdata_[i + nx1_ * (j + nx2_ * n)];
	}

	T &operator()(const int n, const int k, const int j, const int i)
	{
		return pdata_[i + nx1_ * (j + nx2_ * (k + nx3_ * n))];
	}
	T operator()(const int n, const int k, const int j, const int i) const
	{
		return pdata_[i + nx1_ * (j + nx2_ * (k + nx3_ * n))];
	}

	T &operator()(const int m, const int n, const int k, const int j,
		      const int i)
	{
		return pdata_[i +
			      nx1_ * (j + nx2_ * (k + nx3_ * (n + nx4_ * m)))];
	}
	T operator()(const int m, const int n, const int k, const int j,
		     const int i) const
	{
		return pdata_[i +
			      nx1_ * (j + nx2_ * (k + nx3_ * (n + nx4_ * m)))];
	}

	// functions that initialize an array with shallow copy or slice from
	// another array
	void InitWithShallowCopy(AthenaArray<T> &src);
	void InitWithShallowSlice(AthenaArray<T> &src, const int dim,
				  const int indx, const int nvar);

      private:
	T *pdata_;
	int nx1_, nx2_, nx3_, nx4_, nx5_;
	bool
	    scopy_; // true if shallow copy (prevents source from being deleted)

	/*******************************************************************************
	 *    Kokkos components
	 *
	 *******************************************************************************/
      public:
	Kokkos::View<T *, Kokkos::LayoutRight> get_KView1D() const
	{
		return KView1D_;
	}
	Kokkos::View<T **, Kokkos::LayoutRight> get_KView2D() const
	{
		return KView2D_;
	}
	Kokkos::View<T ***, Kokkos::LayoutRight> get_KView3D() const
	{
		return KView3D_;
	}
	Kokkos::View<T ****, Kokkos::LayoutRight> get_KView4D() const
	{
		return KView4D_;
	}
	Kokkos::View<T *****, Kokkos::LayoutRight> get_KView5D() const
	{
		return KView5D_;
	}

      private:
	Kokkos::View<T *, Kokkos::LayoutRight, Kokkos::HostSpace> KView1D_;
	Kokkos::View<T **, Kokkos::LayoutRight, Kokkos::HostSpace> KView2D_;
	Kokkos::View<T ***, Kokkos::LayoutRight, Kokkos::HostSpace> KView3D_;
	Kokkos::View<T ****, Kokkos::LayoutRight, Kokkos::HostSpace> KView4D_;
	Kokkos::View<T *****, Kokkos::LayoutRight, Kokkos::HostSpace> KView5D_;
};

// constructor

template <typename T>
AthenaArray<T>::AthenaArray()
    : pdata_(NULL), nx1_(0), nx2_(0), nx3_(0), nx4_(0), nx5_(0), scopy_(true),
      KView1D_(), KView2D_(), KView3D_(), KView4D_(), KView5D_()
{
}

// destructor

template <typename T> AthenaArray<T>::~AthenaArray() { DeleteAthenaArray(); }

// copy constructor (does a deep copy)

template <typename T>
AthenaArray<T>::AthenaArray(const AthenaArray<T> &src) noexcept
{
	nx1_ = src.nx1_;
	nx2_ = src.nx2_;
	nx3_ = src.nx3_;
	nx4_ = src.nx4_;
	nx5_ = src.nx5_;
	if (src.pdata_) {

		if (KView1D_.data()) {
			KView1D_ =
			    Kokkos::View<T *, Kokkos::LayoutRight,
					 Kokkos::HostSpace>("1Darray", nx1_);
			pdata_ = KView1D_.data();
			Kokkos::deep_copy(KView1D_, src.KView1D_);

		} else if (KView2D_.data()) {
			KView2D_ = Kokkos::View<T **, Kokkos::LayoutRight,
						Kokkos::HostSpace>("2Darray",
								   nx2_, nx1_);
			pdata_ = KView2D_.data();
			Kokkos::deep_copy(KView2D_, src.KView2D_);

		} else if (KView3D_.data()) {
			KView3D_ = Kokkos::View<T ***, Kokkos::LayoutRight,
						Kokkos::HostSpace>(
			    "3Darray", nx3_, nx2_, nx1_);
			pdata_ = KView3D_.data();
			Kokkos::deep_copy(KView3D_, src.KView3D_);

		} else if (KView4D_.data()) {
			KView4D_ = Kokkos::View<T ****, Kokkos::LayoutRight,
						Kokkos::HostSpace>(
			    "4Darray", nx4_, nx3_, nx2_, nx1_);
			pdata_ = KView4D_.data();
			Kokkos::deep_copy(KView4D_, src.KView4D_);

		} else if (KView5D_.data()) {
			KView5D_ = Kokkos::View<T *****, Kokkos::LayoutRight,
						Kokkos::HostSpace>(
			    "5Darray", nx5_, nx4_, nx3_, nx2_, nx1_);
			pdata_ = KView5D_.data();
			Kokkos::deep_copy(KView5D_, src.KView5D_);

		} else {
			std::cout << "### This is bad. Array <-> Kokkos "
				     "assignments don't match...\n";
		}

		scopy_ = false;
	}
}

// assignment operator (does a deep copy).  Does not allocate memory for
// destination. THIS REQUIRES DESTINATION ARRAY BE ALREADY ALLOCATED AND SAME
// SIZE AS SOURCE

template <typename T>
AthenaArray<T> &AthenaArray<T>::operator=(const AthenaArray<T> &src) noexcept
{
	if (this != &src) {
		if (KView1D_.data()) {
			Kokkos::deep_copy(KView1D_, src.KView1D_);

		} else if (KView2D_.data()) {
			Kokkos::deep_copy(KView2D_, src.KView2D_);

		} else if (KView3D_.data()) {
			Kokkos::deep_copy(KView3D_, src.KView3D_);

		} else if (KView4D_.data()) {
			Kokkos::deep_copy(KView4D_, src.KView4D_);

		} else if (KView5D_.data()) {
			Kokkos::deep_copy(KView5D_, src.KView5D_);

		} else {
			std::cout << "### This is bad. Array <-> Kokkos "
				     "assignments don't match...\n";
		}

		scopy_ = false;
	}
	return *this;
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::InitWithShallowCopy()
//  \brief shallow copy of array (copies ptrs, but not data)

template <typename T>
void AthenaArray<T>::InitWithShallowCopy(AthenaArray<T> &src)
{
	nx1_ = src.nx1_;
	nx2_ = src.nx2_;
	nx3_ = src.nx3_;
	nx4_ = src.nx4_;
	nx5_ = src.nx5_;
	pdata_ = src.pdata_;
	scopy_ = true;
	// Kokkos view assignments are by default shallow copies
	KView1D_ = src.KView1D_;
	KView2D_ = src.KView2D_;
	KView3D_ = src.KView3D_;
	KView4D_ = src.KView4D_;
	KView5D_ = src.KView5D_;
	return;
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::InitWithShallowSlice()
//  \brief shallow copy of nvar elements in dimension dim of an array, starting
//  at index=indx.  Copies pointers to data, but not data itself.

template <typename T>
void AthenaArray<T>::InitWithShallowSlice(AthenaArray<T> &src, const int dim,
					  const int indx, const int nvar)
{

	if (dim == 5) {
		nx5_ = nvar;
		nx4_ = src.nx4_;
		nx3_ = src.nx3_;
		nx2_ = src.nx2_;
		nx1_ = src.nx1_;
		KView5D_ = Kokkos::subview(
		    src.KView5D_, std::make_pair(indx, indx + nvar),
		    Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
		pdata_ = KView5D_.data();
	} else if (dim == 4) {
		nx5_ = 1;
		nx4_ = nvar;
		nx3_ = src.nx3_;
		nx2_ = src.nx2_;
		nx1_ = src.nx1_;
		KView4D_ = Kokkos::subview(
		    src.KView4D_, std::make_pair(indx, indx + nvar),
		    Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
		pdata_ = KView4D_.data();
	} else if (dim == 3) {
		nx5_ = 1;
		nx4_ = 1;
		nx3_ = nvar;
		nx2_ = src.nx2_;
		nx1_ = src.nx1_;
		KView3D_ = Kokkos::subview(src.KView3D_,
					   std::make_pair(indx, indx + nvar),
					   Kokkos::ALL(), Kokkos::ALL());
		pdata_ = KView3D_.data();
	} else if (dim == 2) {
		nx5_ = 1;
		nx4_ = 1;
		nx3_ = 1;
		nx2_ = nvar;
		nx1_ = src.nx1_;
		KView2D_ = Kokkos::subview(src.KView2D_,
					   std::make_pair(indx, indx + nvar),
					   Kokkos::ALL());
		pdata_ = KView2D_.data();
	} else if (dim == 1) {
		nx5_ = 1;
		nx4_ = 1;
		nx3_ = 1;
		nx2_ = 1;
		nx1_ = nvar;
		KView1D_ = Kokkos::subview(src.KView1D_,
					   std::make_pair(indx, indx + nvar));
		pdata_ = KView1D_.data();
	}
	scopy_ = true;
	return;
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief allocate new 1D array with elements initialized to zero.

template <typename T> void AthenaArray<T>::NewAthenaArray(int nx1) noexcept
{
	scopy_ = false;
	nx1_ = nx1;
	nx2_ = 1;
	nx3_ = 1;
	nx4_ = 1;
	nx5_ = 1;
	KView1D_ = Kokkos::View<T *, Kokkos::LayoutRight, Kokkos::HostSpace>(
	    "1Darray", nx1_);
	pdata_ = KView1D_.data();
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief 2d data allocation

template <typename T>
void AthenaArray<T>::NewAthenaArray(int nx2, int nx1) noexcept
{
	scopy_ = false;
	nx1_ = nx1;
	nx2_ = nx2;
	nx3_ = 1;
	nx4_ = 1;
	nx5_ = 1;
	KView2D_ = Kokkos::View<T **, Kokkos::LayoutRight, Kokkos::HostSpace>(
	    "2Darray", nx2_, nx1_);
	pdata_ = KView2D_.data();
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief 3d data allocation

template <typename T>
void AthenaArray<T>::NewAthenaArray(int nx3, int nx2, int nx1) noexcept
{
	scopy_ = false;
	nx1_ = nx1;
	nx2_ = nx2;
	nx3_ = nx3;
	nx4_ = 1;
	nx5_ = 1;

	KView3D_ = Kokkos::View<T ***, Kokkos::LayoutRight, Kokkos::HostSpace>(
	    "3Darray", nx3_, nx2_, nx1_);
	pdata_ = KView3D_.data();
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief 4d data allocation

template <typename T>
void AthenaArray<T>::NewAthenaArray(int nx4, int nx3, int nx2, int nx1) noexcept
{
	scopy_ = false;
	nx1_ = nx1;
	nx2_ = nx2;
	nx3_ = nx3;
	nx4_ = nx4;
	nx5_ = 1;

	KView4D_ = Kokkos::View<T ****, Kokkos::LayoutRight, Kokkos::HostSpace>(
	    "4Darray", nx4_, nx3_, nx2_, nx1_);
	pdata_ = KView4D_.data();
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::NewAthenaArray()
//  \brief 5d data allocation

template <typename T>
void AthenaArray<T>::NewAthenaArray(int nx5, int nx4, int nx3, int nx2,
				    int nx1) noexcept
{
	scopy_ = false;
	nx1_ = nx1;
	nx2_ = nx2;
	nx3_ = nx3;
	nx4_ = nx4;
	nx5_ = nx5;

	KView5D_ =
	    Kokkos::View<T *****, Kokkos::LayoutRight, Kokkos::HostSpace>(
		"5Darray", nx5_, nx4_, nx3_, nx2_, nx1_);
	pdata_ = KView5D_.data();
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::DeleteAthenaArray()
//  \brief  free memory allocated for data array

template <typename T> void AthenaArray<T>::DeleteAthenaArray()
{
	if (scopy_) {
		pdata_ = NULL;
	} else {
		// All data is managed by Kokkos so we don't need to manually
		// delete
		//  delete[] pdata_;
		pdata_ = NULL;
		scopy_ = true;
	}
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::SwapAthenaArray()
//  \brief  swap pdata_ pointers of two equally sized AthenaArrays (shallow
//  swap)
// Does not allocate memory for either AthenArray
// THIS REQUIRES DESTINATION AND SOURCE ARRAYS BE ALREADY ALLOCATED AND HAVE THE
// SAME SIZES (does not explicitly check either condition)

template <typename T>
void AthenaArray<T>::SwapAthenaArray(AthenaArray<T> &array2)
{
	// scopy_ is essentially only tracked for correctness of delete[] in
	// DeleteAthenaArray() cache array1 data ptr
	T *tmp_pdata_ = pdata_;
	pdata_ = array2.pdata_;
	array2.pdata_ = tmp_pdata_;

	Kokkos::View<T *, Kokkos::LayoutRight, Kokkos::HostSpace> tmp_KView1D_ =
	    KView1D_;
	KView1D_ = array2.KView1D_;
	array2.KView1D_ = tmp_KView1D_;

	Kokkos::View<T **, Kokkos::LayoutRight, Kokkos::HostSpace>
	    tmp_KView2D_ = KView2D_;
	KView2D_ = array2.KView2D_;
	array2.KView2D_ = tmp_KView2D_;

	Kokkos::View<T ***, Kokkos::LayoutRight, Kokkos::HostSpace>
	    tmp_KView3D_ = KView3D_;
	KView3D_ = array2.KView3D_;
	array2.KView3D_ = tmp_KView3D_;

	Kokkos::View<T ****, Kokkos::LayoutRight, Kokkos::HostSpace>
	    tmp_KView4D_ = KView4D_;
	KView4D_ = array2.KView4D_;
	array2.KView4D_ = tmp_KView4D_;

	Kokkos::View<T *****, Kokkos::LayoutRight, Kokkos::HostSpace>
	    tmp_KView5D_ = KView5D_;
	KView5D_ = array2.KView5D_;
	array2.KView5D_ = tmp_KView5D_;
}

//----------------------------------------------------------------------------------------
//! \fn AthenaArray::ZeroClear()
//  \brief  fill the array with zero
//	With Kokkos, this will *ONLY* work if the data is on the host (or UVM)!

template <typename T> void AthenaArray<T>::ZeroClear()
{
	// allocate memory and initialize to zero
	std::memset(pdata_, 0, GetSizeInBytes());
}

#endif // ATHENA_ARRAYS_HPP_
