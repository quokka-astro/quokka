#ifndef HDFIO_H
#define HDFIO_H

#include <hdf5.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <cstring>
#include <cassert>
#include <typeinfo>

// The detault is to use this class with MPI support (to switch off MPI, the user needs to #define NO_MPI).
// Also note that this class makes use of H5_HAVE_PARALLEL defined in hdf5.h, if hdf5 has parallel support.
#ifdef NO_MPI
#ifndef MPI_Comm
#define MPI_Comm int
#endif
#ifndef MPI_COMM_NULL
#define MPI_COMM_NULL 0
#endif
#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD 0
#endif
#else
#include <mpi.h>
#endif

namespace NameSpaceHDFIO {
    static const unsigned int flash_str_len = 79; // string length for FLASH scalars and runtime parameters I/O
    template<typename> struct FlashScalarsParametersStruct; // structure for FLASH scalars and runtime parameters I/O
    template<> struct FlashScalarsParametersStruct<int> { char name[flash_str_len]; int value; };
    template<> struct FlashScalarsParametersStruct<double> { char name[flash_str_len]; double value; };
    template<> struct FlashScalarsParametersStruct<bool> { char name[flash_str_len]; bool value; };
    template<> struct FlashScalarsParametersStruct<std::string> { char name[flash_str_len]; char value[flash_str_len]; };
    // FLASHPropAssign template function; applies to string case (so we do a string-copy from string to c_str)
    template<typename T, typename Td> inline void FLASHPropAssign(T val, Td * data) { strcpy(*data, val.c_str()); };
    // FLASHPropAssign template functions that apply to int, double, bool cases
    template<> inline void FLASHPropAssign<int>    (int    val, int    * data) { *data = val; }; // simply assign
    template<> inline void FLASHPropAssign<double> (double val, double * data) { *data = val; }; // simply assign
    template<> inline void FLASHPropAssign<bool>   (bool   val, bool   * data) { *data = val; }; // simply assign
}

/**
 * HDFIO class
 *
 * This class represents an HDF5 object/file and provides basic HDF operations like opening a file,
 * reading data form a dataset of that file, creating new HDF files and datasets and writing datasets into a file.
 * There are also some functions for getting and setting different attributes, like the dimensionaltity of the HDF
 * dataset to be written or the datasetnames inside an HDF5 file.
 *
 * @author Christoph Federrath
 * @version 2007-2025
 */

class HDFIO
{
    private:
    std::string ClassSignature, Filename; // class signature and HDF5 filename
    int Rank; // dimensionality of the field
    hid_t   File_id, Dataset_id, Dataspace_id; // HDF5 stuff
    hsize_t HDFSize, HDFDims[4]; // HDF5 stuff; can maximally handle 4D datasets (but who would ever want more?)
    herr_t  HDF5_status, HDF5_error; // HDF5 stuff
    int Verbose; // verbose level for printing to stdout

    /// Constructors
    public: HDFIO(void)
    {
        // empty constructor, so we can define a global HDFIO object in application code
        Constructor("", 'r', MPI_COMM_NULL, 0);
    };
    public: HDFIO(const int verbose)
    {
        Constructor("", 'r', MPI_COMM_NULL, verbose);
    };
    public: HDFIO(const std::string filename)
    {
        Constructor(filename, 'r', MPI_COMM_NULL, 1);
    };
    public: HDFIO(const std::string filename, const int verbose)
    {
        Constructor(filename, 'r', MPI_COMM_NULL, verbose);
    };
    public: HDFIO(const std::string filename, const char read_write_char)
    {
        if (read_write_char == 'w')
            Constructor(filename, read_write_char, MPI_COMM_WORLD, 1);
        else
            Constructor(filename, read_write_char, MPI_COMM_NULL, 1);
    };
    public: HDFIO(const std::string filename, const char read_write_char, const int verbose)
    {
        if (read_write_char == 'w')
            Constructor(filename, read_write_char, MPI_COMM_WORLD, verbose);
        else
            Constructor(filename, read_write_char, MPI_COMM_NULL, verbose);
    };

    /// Destructor
    public: ~HDFIO()
    {
      if (Verbose > 1) std::cout<<"HDFIO: destructor called."<<std::endl;
    };

    private: void Constructor(const std::string filename, const char read_write_char, MPI_Comm comm, const int verbose)
    {
        Filename = filename; // HDF5 filename
        Verbose = verbose; // verbose (can be 0: no stdout, 1: default output, 2: more output)
        ClassSignature = "HDFIO: "; // class signature, when this class is printing to stdout
        Rank = 1; // dimensionality 1 (e.g., x only)
        File_id = 0; // HDF5 file id 0
        Dataset_id = 0; // HDF5 dataset id 0
        Dataspace_id = 0; // HDF5 data space id 0
        HDFSize = 0; // HDF5 buffer size 0
        HDF5_status = 0; // HDF5 status 0
        HDF5_error = -1; // HDF5 error -1
        for (unsigned int i = 0; i < 4; i++) HDFDims[i] = 0; // set Dimensions to (0,0,0,0)
        if (Filename != "") this->open(Filename, read_write_char, comm); // open file if provided in constructor
    };

    // get function signature for printing to stdout
    private: std::string FuncSig(const std::string func_name)
    { return ClassSignature+func_name+": "; };

    /**
     * open an HDF5 file in serial mode
     * @param Filename HDF5 filename
     * @param read_write_char 'r': read only flag, 'w': write flag
     */
    public: void open(const std::string Filename, const char read_write_char)
    {
        this->open(Filename, read_write_char, MPI_COMM_NULL); // open in serial mode
    };

    /**
     * open an HDF5 file (overloaded)
     * @param Filename HDF5 filename
     * @param read_write_char 'r': read only flag, 'w': write flag
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void open(const std::string Filename, const char read_write_char, MPI_Comm comm)
    {
        this->Filename = Filename;

        hid_t plist_id = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            plist_id = H5Pcreate(H5P_FILE_ACCESS);
            H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
            assert( HDF5_status != HDF5_error );
        }
#endif
        switch (read_write_char)
        {
            case 'r':
            {
                // open HDF5 file in read only mode
                File_id = H5Fopen(Filename.c_str(), H5F_ACC_RDONLY, plist_id);
                assert( File_id != HDF5_error );
                break;
            }
            case 'w':
            {
                // open HDF5 file in write mode
                File_id = H5Fopen(Filename.c_str(), H5F_ACC_RDWR, plist_id);
                assert( File_id != HDF5_error );
                break;
            }
            default:
            {
                // open HDF5 file in read only mode
                File_id = H5Fopen(Filename.c_str(), H5F_ACC_RDONLY, plist_id);
                assert( File_id != HDF5_error );
                break;
            }
        }
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) H5Pclose(plist_id);
#endif
    };

    /**
     * close HDF5 file
     */
    public: void close(void)
    {
        // close HDF5 file
        HDF5_status = H5Fclose(File_id);
        assert( HDF5_status != HDF5_error );
    };

    /**
     * read data from a dataset
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     */
    public: void read(void* const DataBuffer, const std::string Datasetname, const hid_t DataType)
    {
        this->read(DataBuffer, Datasetname, DataType, MPI_COMM_NULL); // serial mode
    };

    /**
     * read data from a dataset
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void read(void* const DataBuffer, const std::string Datasetname, const hid_t DataType, MPI_Comm comm)
    {
        // get dimensional information from dataspace and update HDFSize
        this->getDims(Datasetname);

        // open dataset
        Dataset_id = H5Dopen(File_id, Datasetname.c_str(), H5P_DEFAULT);
        assert( Dataset_id != HDF5_error );

        // open dataspace (to get dimensions)
        Dataspace_id = H5Dget_space(Dataset_id);
        assert( Dataspace_id != HDF5_error );

        /// create property list for collective dataset i/o
        hid_t plist_id = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        }
#endif
        // read buffer /memspaceid //filespaceid
        HDF5_status = H5Dread(Dataset_id, DataType, H5S_ALL, H5S_ALL, plist_id, DataBuffer);
        assert( HDF5_status != HDF5_error );
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            HDF5_status = H5Pclose(plist_id);
            assert( HDF5_status != HDF5_error );
        }
#endif
        HDF5_status = H5Sclose(Dataspace_id);
        assert( HDF5_status != HDF5_error );

        HDF5_status = H5Dclose(Dataset_id);
        assert( HDF5_status != HDF5_error );
    }; // read

    /**
     * read attribute from a dataset
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname name of dataset
     * @param Attributename name of attribute to be read
     * @param DataType (i.e. H5T_STD_I32LE)
     */
    public: void read_attribute(void* const DataBuffer, const std::string Datasetname,
                                const std::string Attributename, const hid_t DataType)
    {
        this->read_attribute(DataBuffer, Datasetname, Attributename, DataType, MPI_COMM_NULL); // serial mode
    };

    /**
     * read attribute from a dataset
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname name of dataset
     * @param Attributename name of attribute to be read
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void read_attribute(void* const DataBuffer, const std::string Datasetname,
                                const std::string Attributename, const hid_t DataType, MPI_Comm comm)
    {
        // open dataset
        Dataset_id = H5Dopen(File_id, Datasetname.c_str(), H5P_DEFAULT);
        assert( Dataset_id != HDF5_error );

        /// create property list for collective dataset i/o
        hid_t plist_id = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        }
#endif
        /// open attribute
        hid_t attr_id = H5Aopen(Dataset_id, Attributename.c_str(), plist_id);

        /// read attribute
        HDF5_status = H5Aread(attr_id, DataType, DataBuffer);
        assert( HDF5_status != HDF5_error );

#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            HDF5_status = H5Pclose(plist_id);
            assert( HDF5_status != HDF5_error );
        }
#endif
        HDF5_status = H5Aclose(attr_id);
        assert( HDF5_status != HDF5_error );

        HDF5_status = H5Dclose(Dataset_id);
        assert( HDF5_status != HDF5_error );

    }; // read_attribute

    /**
     * read a slab of data from a dataset (overloaded)
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param offset (offset array)
     * @param count (count array, i.e. the number of cells[dim] to be selected)
     * @param out_rank (rank of the output array selected)
     * @param out_offset (output offset array)
     * @param out_count (output count array, i.e. the number of cells[dim] in output)
     */
    public: void read_slab( void* const DataBuffer, const std::string Datasetname, const hid_t DataType,
                            const hsize_t offset[], const hsize_t count[],
                            const hsize_t out_rank, const hsize_t out_offset[], const hsize_t out_count[])
    {
        /// simply set total_out_count = out_count, in which case we selected a hyperslab that fits completely into the
        /// total size of the output array. Use the overloaded function below, if output goes into an offset hyperslab
        this->read_slab(DataBuffer, Datasetname, DataType, offset, count, out_rank, out_offset, out_count, out_count);
    };

    /**
     * read a slab of data from a dataset (overloaded)
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param offset (offset array)
     * @param count (count array, i.e. the number of cells[dim] to be selected)
     * @param out_rank (rank of the output array selected)
     * @param out_offset (output offset array)
     * @param out_count (output count array, i.e. the number of cells[dim] in output)
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void read_slab( void* const DataBuffer, const std::string Datasetname, const hid_t DataType,
                            const hsize_t offset[], const hsize_t count[],
                            const hsize_t out_rank, const hsize_t out_offset[], const hsize_t out_count[], MPI_Comm comm)
    {
        /// simply set total_out_count = out_count, in which case we selected a hyperslab that fits completely into the
        /// total size of the output array. Use the overloaded function below, if output goes into an offset hyperslab
        this->read_slab(DataBuffer, Datasetname, DataType, offset, count, out_rank, out_offset, out_count, out_count, comm);
    };

    /**
     * read a slab of data from a dataset
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param offset (offset array)
     * @param count (count array, i.e. the number of cells[dim] to be selected)
     * @param out_rank (rank of the output array selected)
     * @param out_offset (output offset array)
     * @param out_count (output count array, i.e. the number of cells[dim] in output)
     * @param total_out_count (total output count array, i.e. the total number of cells[dim] in output)
     */
    public: void read_slab( void* const DataBuffer, const std::string Datasetname, const hid_t DataType, 
                            const hsize_t offset[], const hsize_t count[], 
                            const hsize_t out_rank, const hsize_t out_offset[], const hsize_t out_count[],
                            const hsize_t total_out_count[])
    {
        this->read_slab(DataBuffer, Datasetname, DataType, offset, count, out_rank, out_offset, out_count,
                        total_out_count, MPI_COMM_NULL); // serial mode
    };

    /**
     * read a slab of data from a dataset
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param offset (offset array)
     * @param count (count array, i.e. the number of cells[dim] to be selected)
     * @param out_rank (rank of the output array selected)
     * @param out_offset (output offset array)
     * @param out_count (output count array, i.e. the number of cells[dim] in output)
     * @param total_out_count (total output count array, i.e. the total number of cells[dim] in output)
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void read_slab( void* const DataBuffer, const std::string Datasetname, const hid_t DataType,
                            const hsize_t offset[], const hsize_t count[],
                            const hsize_t out_rank, const hsize_t out_offset[], const hsize_t out_count[],
                            const hsize_t total_out_count[], MPI_Comm comm)
    {
        // get dimensional information from dataspace and update HDFSize
        this->getDims(Datasetname);

        // open dataset
        Dataset_id = H5Dopen(File_id, Datasetname.c_str(), H5P_DEFAULT);
        assert( Dataset_id != HDF5_error );

        // open dataspace (to get dimensions)
        Dataspace_id = H5Dget_space(Dataset_id);
        assert( Dataspace_id != HDF5_error );

        // select hyperslab
        HDF5_status = H5Sselect_hyperslab(Dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
        assert( HDF5_status != HDF5_error );

        // create memspace
        hid_t Memspace_id = H5Screate_simple(out_rank, total_out_count, NULL);
        HDF5_status = H5Sselect_hyperslab(Memspace_id, H5S_SELECT_SET, out_offset, NULL, out_count, NULL);
        assert( HDF5_status != HDF5_error );

        /// create property list for collective dataset i/o
        hid_t plist_id = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        }
#endif
        // read buffer
        HDF5_status = H5Dread(Dataset_id, DataType, Memspace_id, Dataspace_id, plist_id, DataBuffer );
        assert( HDF5_status != HDF5_error );
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            HDF5_status = H5Pclose(plist_id);
            assert( HDF5_status != HDF5_error );
        }
#endif
        HDF5_status = H5Sclose(Memspace_id);
        assert( HDF5_status != HDF5_error );

        HDF5_status = H5Sclose(Dataspace_id);
        assert( HDF5_status != HDF5_error );

        HDF5_status = H5Dclose(Dataset_id);
        assert( HDF5_status != HDF5_error );

    }; // read_slab

    /**
     * overwrite a slab of data from a dataset
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param offset (offset array)
     * @param count (count array, i.e. the number of cells[dim] to be selected)
     * @param out_rank (rank of the output array selected)
     * @param out_offset (output offset array)
     * @param out_count (output count array, i.e. the number of cells[dim] in output)
     */
    public: void overwrite_slab(void* const DataBuffer, const std::string Datasetname, const hid_t DataType, 
                                const hsize_t offset[], const hsize_t count[], 
                                const hsize_t out_rank, const hsize_t out_offset[], const hsize_t out_count[])
    {
        this->overwrite_slab(DataBuffer, Datasetname, DataType, 
                                offset, count, out_rank, out_offset, out_count, MPI_COMM_NULL); // serial mode
    };

    /**
     * overwrite a slab of data from a dataset
     * @param *DataBuffer pointer to double/float/int array to which data is to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param offset (offset array)
     * @param count (count array, i.e. the number of cells[dim] to be selected)
     * @param out_rank (rank of the output array selected)
     * @param out_offset (output offset array)
     * @param out_count (output count array, i.e. the number of cells[dim] in output)
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void overwrite_slab(void* const DataBuffer, const std::string Datasetname, const hid_t DataType,
                                const hsize_t offset[], const hsize_t count[],
                                const hsize_t out_rank, const hsize_t out_offset[], const hsize_t out_count[],
                                MPI_Comm comm)
    {
        // get dimensional information from dataspace and update HDFSize
        this->getDims(Datasetname);

        // open dataset
        Dataset_id = H5Dopen(File_id, Datasetname.c_str(), H5P_DEFAULT);
        assert( Dataset_id != HDF5_error );

        // open dataspace (to get dimensions)
        Dataspace_id = H5Dget_space(Dataset_id);
        assert( Dataspace_id != HDF5_error );

        // select hyperslab
        HDF5_status = H5Sselect_hyperslab(Dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
        assert( HDF5_status != HDF5_error );

        // create memspace
        hid_t Memspace_id = H5Screate_simple(out_rank, out_count, NULL);
        HDF5_status = H5Sselect_hyperslab(Memspace_id, H5S_SELECT_SET, out_offset, NULL, out_count, NULL);
        assert( HDF5_status != HDF5_error );

        /// create property list for collective dataset i/o
        hid_t plist_id = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        }
#endif
        // overwrite dataset
        HDF5_status = H5Dwrite(Dataset_id, DataType, Memspace_id, Dataspace_id, plist_id, DataBuffer);
        assert( HDF5_status != HDF5_error );
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            HDF5_status = H5Pclose(plist_id);
            assert( HDF5_status != HDF5_error );
        }
#endif
        HDF5_status = H5Sclose(Memspace_id);
        assert( HDF5_status != HDF5_error );

        HDF5_status = H5Sclose(Dataspace_id);
        assert( HDF5_status != HDF5_error );

        HDF5_status = H5Dclose(Dataset_id);
        assert( HDF5_status != HDF5_error );

    }; // overwrite_slab

    /**
     * overwrite data of an existing dataset (serial mode)
     * @param *DataBuffer pointer to double/float/int array containing data to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     */
    public: void overwrite(const void* const DataBuffer, const std::string Datasetname, const hid_t DataType)
    {
        this->overwrite(DataBuffer, Datasetname, DataType, MPI_COMM_NULL);
    };

    /**
     * overwrite data of an existing dataset (overloaded)
     * @param *DataBuffer pointer to double/float/int array containing data to be written
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void overwrite(const void* const DataBuffer, const std::string Datasetname, const hid_t DataType, MPI_Comm comm)
    {
        // get dimensional information from dataspace and update HDFSize
        this->getDims(Datasetname);

        // open dataset
        Dataset_id = H5Dopen(File_id, Datasetname.c_str(), H5P_DEFAULT);
        assert( Dataset_id != HDF5_error );

        // open dataspace (to get dimensions)
        Dataspace_id = H5Dget_space(Dataset_id);
        assert( Dataspace_id != HDF5_error );

        /// create property list for collective dataset i/o
        hid_t plist_id = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        }
#endif
        // overwrite dataset
        HDF5_status = H5Dwrite(Dataset_id, DataType, H5S_ALL, H5S_ALL, plist_id, DataBuffer);
        assert( HDF5_status != HDF5_error );
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            HDF5_status = H5Pclose(plist_id);
            assert( HDF5_status != HDF5_error );
        }
#endif
        HDF5_status = H5Sclose(Dataspace_id);
        assert( HDF5_status != HDF5_error );

        HDF5_status = H5Dclose(Dataset_id);
        assert( HDF5_status != HDF5_error );

    }; // overwrite

    /**
     * create new HDF5 file (serial mode)
     * @param Filename HDF5 filename
     */
    public: void create(const std::string Filename)
    {
        this->create(Filename, MPI_COMM_NULL);
    };

    /**
     * create new HDF5 file (overloaded)
     * @param Filename HDF5 filename
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void create(const std::string Filename, MPI_Comm comm)
    {
        this->Filename = Filename;

        hid_t plist_id = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            plist_id = H5Pcreate(H5P_FILE_ACCESS);
            H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
        }
#endif
        // create HDF5 file (overwrite)
        File_id = H5Fcreate(Filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        assert( File_id != HDF5_error );

#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) H5Pclose(plist_id);
#endif
    };

    /**
     * write HDF5 dataset (serial mode)
     * @param DataBuffer int/float/double array containing the data
     * @param Datasetname datasetname
     * @param Dimensions dataset dimensions
     * @param DataType (i.e. H5T_STD_I32LE)
     */
    public: void write( const void* const DataBuffer, const std::string Datasetname,
                        const std::vector<int> Dimensions, const hid_t DataType)
    {
        this->write(DataBuffer, Datasetname, Dimensions, DataType, MPI_COMM_NULL);
    };

    /**
     * write HDF5 dataset (overloaded)
     * @param DataBuffer int/float/double array containing the data
     * @param Datasetname datasetname
     * @param Dimensions dataset dimensions
     * @param DataType (i.e. H5T_STD_I32LE)
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void write( const void* const DataBuffer, const std::string Datasetname,
                        const std::vector<int> Dimensions, const hid_t DataType, MPI_Comm comm)
    {
        this->setDims(Dimensions); // set dimensions
        this->write(DataBuffer, Datasetname, DataType, comm); // call write
    };

    /**
     * write HDF5 dataset (serial mode)
     * @param *DataBuffer pointer to int/float/double array containing the data
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_IEEE_F32BE, H5T_STD_I32LE, ...)
     */
    public: void write(const void* const DataBuffer, const std::string Datasetname, const hid_t DataType)
    {
        this->write(DataBuffer, Datasetname, DataType, MPI_COMM_NULL);
    };

    /**
     * write HDF5 dataset (overloaded)
     * @param *DataBuffer pointer to int/float/double array containing the data
     * @param Datasetname datasetname
     * @param DataType (i.e. H5T_IEEE_F32BE, H5T_STD_I32LE, ...)
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void write(const void* const DataBuffer, const std::string Datasetname, const hid_t DataType, MPI_Comm comm)
    {
        // -------------- create dataspace
        Dataspace_id = H5Screate_simple(Rank, HDFDims, NULL);
        assert( Dataspace_id != HDF5_error );

        // -------------- create dataset
        Dataset_id = H5Dcreate(File_id, Datasetname.c_str(), DataType, Dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert( Dataset_id != HDF5_error );

        /// create property list for collective dataset i/o
        hid_t plist_id = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        }
#endif
        // -------------- write dataset
        HDF5_status = H5Dwrite(Dataset_id, DataType, H5S_ALL, H5S_ALL, plist_id, DataBuffer);
        assert( HDF5_status != HDF5_error );
#ifdef H5_HAVE_PARALLEL
        if (comm != MPI_COMM_NULL) {
            HDF5_status = H5Pclose(plist_id);
            assert( HDF5_status != HDF5_error );
        }
#endif
        // -------------- close dataset
        HDF5_status = H5Dclose(Dataset_id);
        assert( HDF5_status != HDF5_error );

        // -------------- close dataspace
        HDF5_status = H5Sclose(Dataspace_id);
        assert( HDF5_status != HDF5_error );
    }; // write

    /**
     * create empty HDF5 dataset
     * @param Datasetname datasetname
     * @param Dimensions dataset dimensions
     * @param DataType (i.e. H5T_IEEE_F32BE, H5T_STD_I32LE, ...)
     * @param comm: MPI communicator for parallel file I/O
     */
    public: void create_dataset(const std::string Datasetname, const std::vector<int> Dimensions,
                                const hid_t DataType, MPI_Comm comm)
    {
        this->setDims(Dimensions); // set dimensions

        // -------------- create dataspace
        Dataspace_id = H5Screate_simple(Rank, HDFDims, NULL);
        assert( Dataspace_id != HDF5_error );

        // -------------- create dataset
        Dataset_id = H5Dcreate(File_id, Datasetname.c_str(), DataType, Dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert( Dataset_id != HDF5_error );

        // -------------- close dataset
        HDF5_status = H5Dclose(Dataset_id);
        assert( HDF5_status != HDF5_error );

        // -------------- close dataspace
        HDF5_status = H5Sclose(Dataspace_id);
        assert( HDF5_status != HDF5_error );
    };

    /**
     * check if HDF5 dataset exists
     * @param Datasetname datasetname
     */
    public: bool dataset_exists(const std::string Datasetname)
    {
        std::vector<std::string> dsets_in_file = getDatasetnames();
        bool dset_in_file = false; // see if the Datasetname exists in the file
        for (unsigned int i = 0; i < dsets_in_file.size(); i++) {
            if (dsets_in_file[i] == Datasetname) { dset_in_file = true; break; }
        }
        return dset_in_file;
    };

    /**
     * delete HDF5 dataset
     * @param Datasetname datasetname
     */
    public: void delete_dataset(const std::string Datasetname)
    {
        if (this->dataset_exists(Datasetname)) { // if the Datasetname is in the file, delete it
            HDF5_status = H5Ldelete(File_id, Datasetname.c_str(), H5P_DEFAULT); // delete dataset
            assert( HDF5_status != HDF5_error );
        }
    };

    /**
     * delete all HDF5 datasets
     */
    public: void delete_datasets(void)
    {
        std::vector<std::string> dsets_in_file = getDatasetnames();
        for (unsigned int i = 0; i < dsets_in_file.size(); i++) {
            this->delete_dataset(dsets_in_file[i]);
        }
    };

    /**
     * set the rank of a dataset
     * @param Rank the rank, dimensionality (0,1,2)
     */
    public: void setRank(const int Rank)
    {
        this->Rank = Rank;
    };

    /**
     * set array dimension in different directions
     * @param dim dimension of the array
     */
    public: void setDimX(const int dim_x)
    {
        HDFDims[0] = static_cast<hsize_t>(dim_x);
    };

    public: void setDimY(const int dim_y)
    {
        HDFDims[1] = static_cast<hsize_t>(dim_y);
    };

    public: void setDimZ(const int dim_z)
    {
        HDFDims[2] = static_cast<hsize_t>(dim_z);
    };

    public: void setDims(const std::vector<int> Dimensions)
    {
        Rank = Dimensions.size();
        for(int i = 0; i < Rank; i++)
            HDFDims[i] = static_cast<hsize_t>(Dimensions[i]);
    };

    /**
     * get the rank of the current dataset
     * @return int dimensionality
     */
    public: int getRank(void) const
    {
        return Rank;
    };

    /**
     * get the rank of a dataset with name datasetname
     * @param Datasetname datasetname
     * @return int dimensionality
     */
    public: int getRank(const std::string Datasetname)
    {
        // get dimensional information from dataspace and update HDFSize
        this->getDims(Datasetname);
        return Rank;
    };

    /**
     * get array dimension in different directions
     * @return dimension of the array
     */
    public: int getDimX(void) const
    {
        return static_cast<int>(HDFDims[0]);
    };

    public: int getDimY(void) const
    {
        return static_cast<int>(HDFDims[1]);
    };

    public: int getDimZ(void) const
    {
        return static_cast<int>(HDFDims[2]);
    };

    /**
     * get dataset size of dataset with datasetname
     * @param Datasetname datasetname
     * @return size of the dataset
     */
    public: int getSize(const std::string Datasetname)
    {
        // open dataset
        Dataset_id = H5Dopen(File_id, Datasetname.c_str(), H5P_DEFAULT);
        if (Dataset_id == HDF5_error)
        {
            std::cout << "HDFIO:  getSize():  CAUTION: Datasetname '" << Datasetname << "' does not exists in file '"
                    <<this->getFilename()<<"'. Continuing..." << std::endl;
            return 0;
        }
        assert( Dataset_id != HDF5_error );

        // open dataspace
        Dataspace_id = H5Dget_space(Dataset_id);
        assert( Dataspace_id != HDF5_error );

        // get dimensional information from dataspace
        hsize_t HDFxdims[4], HDFmaxdims[4];
        Rank = H5Sget_simple_extent_dims(Dataspace_id, HDFxdims, HDFmaxdims);

        // from the dimensional info, calculate the size of the buffer.
        HDFSize = 1;
        for (int i = 0; i < Rank; i++) {
            HDFDims[i] = HDFxdims[i];
            HDFSize *= HDFDims[i];
        }

        HDF5_status = H5Sclose(Dataspace_id);
        assert( HDF5_status != HDF5_error );

        HDF5_status = H5Dclose(Dataset_id);
        assert( HDF5_status != HDF5_error );

        return static_cast<int>(HDFSize);
    }; // getSize

    /**
     * get array dimension in different directions and update HDFSize
     * @param Datasetname datasetname for which dimensional info is read
     * @return dimension of the array
     */
    public: std::vector<int> getDims(const std::string Datasetname)
    {
        // open dataset
        Dataset_id = H5Dopen(File_id, Datasetname.c_str(), H5P_DEFAULT);
        assert( Dataset_id != HDF5_error );

        // open dataspace
        Dataspace_id = H5Dget_space(Dataset_id);
        assert( Dataspace_id != HDF5_error );

        // get dimensional information from dataspace
        hsize_t HDFxdims[4], HDFmaxdims[4];

        Rank = H5Sget_simple_extent_dims(Dataspace_id, HDFxdims, HDFmaxdims);

        // from the dimensional info, calculate the size of the buffer.
        HDFSize = 1;
        for (int i = 0; i < Rank; i++) {
            HDFDims[i] = HDFxdims[i];
            HDFSize *= HDFDims[i];
        }

        HDF5_status = H5Sclose(Dataspace_id);
        assert( HDF5_status != HDF5_error );

        HDF5_status = H5Dclose(Dataset_id);
        assert( HDF5_status != HDF5_error );

        std::vector<int> ReturnDims(Rank);
        for(int i = 0; i < Rank; i++)
            ReturnDims[i] = static_cast<int>(HDFDims[i]);

        return ReturnDims;
    }; // getDims

    /**
     * get HDF5 file ID
     * @return File_id
     */
    public: hid_t getFileID(void)
    {
        return File_id;
    };

    /**
     * get HDF5 filename
     * @return filename
     */
    public: std::string getFilename(void)
    {
        return Filename;
    };

    /**
     * get number of datasets in HDF5 file
     * @return the number of datasets
     */
    public: int getNumberOfDatasets(void)
    {
        hsize_t *NumberofObjects = new hsize_t[1];
        H5Gget_num_objs(File_id, NumberofObjects);
        int returnNumber = static_cast<int>(NumberofObjects[0]);
        delete [] NumberofObjects;
        return returnNumber;
    };

    /**
     * get HDF5 datasetname
     * @param datasetnumber integer number identifying the dataset
     * @return datasetname
     */
    public: std::string getDatasetname(const int datasetnumber)
    {
        char *Name = new char[256];
        H5Gget_objname_by_idx(File_id, datasetnumber, Name, 256);
        std::string returnName = static_cast<std::string>(Name);
        delete [] Name;
        return returnName;
    };

    /**
     * getDatasetnames
     * @return vector<string> datasetnames
     */
    public: std::vector<std::string> getDatasetnames(void)
    {
        int nsets = this->getNumberOfDatasets();
        std::vector<std::string> ret(nsets);
        for (int i=0; i<nsets; i++)
        {
            ret[i] = this->getDatasetname(i);
            if (Verbose > 2) std::cout << "dataset in file: " << ret[i] << std::endl;
        }
        return ret;
    };

    // ************************************************************************* //
    // ************** reading FLASH scalars or runtime parameters ************** //
    // ************************************************************************* //
    /// returns FLASH integer, real, logical, string scalars or runtime parameters as maps
    public: std::map<std::string, int> ReadFlashIntegerScalars()
    { return GetFLASHProps<int>("integer scalars"); };
    public: std::map<std::string, int> ReadFlashIntegerParameters()
    { return GetFLASHProps<int>("integer runtime parameters"); };
    public: std::map<std::string, double> ReadFlashRealScalars()
    { return GetFLASHProps<double>("real scalars"); };
    public: std::map<std::string, double> ReadFlashRealParameters()
    { return GetFLASHProps<double>("real runtime parameters"); };
    public: std::map<std::string, bool> ReadFlashLogicalScalars()
    { return GetFLASHProps<bool>("logical scalars"); };
    public: std::map<std::string, bool> ReadFlashLogicalParameters()
    {   return GetFLASHProps<bool>("logical runtime parameters"); };
    public: std::map<std::string, std::string> ReadFlashStringScalars()
    { return GetFLASHProps<std::string>("string scalars"); };
    public: std::map<std::string, std::string> ReadFlashStringParameters()
    { return GetFLASHProps<std::string>("string runtime parameters"); };
    // returns map of FLASH integer, real, logical, string scalars or runtime parameters
    private: template<typename T> std::map<std::string, T> GetFLASHProps(std::string dsetname)
    {
        std::map<std::string, T> ret; // return object
        int nProps = this->getDims(dsetname)[0];
        hid_t h5string = H5Tcopy(H5T_C_S1); H5Tset_size(h5string, NameSpaceHDFIO::flash_str_len); hid_t dtype;
        if (typeid(T) == typeid(int))         dtype = H5T_NATIVE_INT;
        if (typeid(T) == typeid(double))      dtype = H5T_NATIVE_DOUBLE;
        if (typeid(T) == typeid(bool))        dtype = H5T_NATIVE_HBOOL;
        if (typeid(T) == typeid(std::string)) dtype = H5Tcopy(h5string);
        hid_t datatype = H5Tcreate(H5T_COMPOUND, sizeof(NameSpaceHDFIO::FlashScalarsParametersStruct<T>));
        H5Tinsert(datatype, "name", HOFFSET(NameSpaceHDFIO::FlashScalarsParametersStruct<T>, name), h5string);
        H5Tinsert(datatype, "value", HOFFSET(NameSpaceHDFIO::FlashScalarsParametersStruct<T>, value), dtype);
        NameSpaceHDFIO::FlashScalarsParametersStruct<T> * data = new NameSpaceHDFIO::FlashScalarsParametersStruct<T>[nProps];
        this->read(data, dsetname, datatype);
        for (int i = 0; i < nProps; i++) {
            std::string pname = Trim((std::string)data[i].name);
            ret[pname] = (T)data[i].value;
            if (Verbose > 1) std::cout<<FuncSig(__func__)<<"name, value = '"<<pname<<"', '"<<ret[pname]<<"'"<<std::endl;
        }
        delete [] data;
        return ret;
    };
    // clear whitespace and/or terminators at the end of a string
    private: std::string Trim(const std::string input)
    {   std::string ret = input;
        return ret.erase(input.find_last_not_of(" \n\r\t")+1);
    };

    // ************************************************************************* //
    // ************ overwriting FLASH scalars or runtime parameters ************ //
    // ************************************************************************* //
    public: void OverwriteFlashIntegerScalars(std::map<std::string, int> props)
    { for (std::map<std::string, int>::iterator it = props.begin(); it != props.end(); it++)
        OverwriteFLASHProp("integer scalars", it->first, it->second); };
    public: void OverwriteFlashIntegerParameters(std::map<std::string, int> props)
    { for (std::map<std::string, int>::iterator it = props.begin(); it != props.end(); it++)
        OverwriteFLASHProp("integer runtime parameters", it->first, it->second); };
    public: void OverwriteFlashRealScalars(std::map<std::string, double> props)
    { for (std::map<std::string, double>::iterator it = props.begin(); it != props.end(); it++)
        OverwriteFLASHProp("real scalars", it->first, it->second); };
    public: void OverwriteFlashRealParameters(std::map<std::string, double> props)
    { for (std::map<std::string, double>::iterator it = props.begin(); it != props.end(); it++)
        OverwriteFLASHProp("real runtime parameters", it->first, it->second); };
    public: void OverwriteFlashLogicalScalars(std::map<std::string, bool> props)
    { for (std::map<std::string, bool>::iterator it = props.begin(); it != props.end(); it++)
        OverwriteFLASHProp("logical scalars", it->first, it->second); };
    public: void OverwriteFlashLogicalParameters(std::map<std::string, bool> props)
    { for (std::map<std::string, bool>::iterator it = props.begin(); it != props.end(); it++)
        OverwriteFLASHProp("logical runtime parameters", it->first, it->second); };
    public: void OverwriteFlashStringScalars(std::map<std::string, std::string> props)
    { for (std::map<std::string, std::string>::iterator it = props.begin(); it != props.end(); it++)
        OverwriteFLASHProp("string scalars", it->first, it->second); };
    public: void OverwriteFlashStringParameters(std::map<std::string, std::string> props)
    { for (std::map<std::string, std::string>::iterator it = props.begin(); it != props.end(); it++)
        OverwriteFLASHProp("string runtime parameters", it->first, it->second); };
    // overwrite FLASH integer, real, logical, string scalars or runtime parameters
    private: template<typename T> void OverwriteFLASHProp(std::string datasetname, std::string fieldname, T fieldvalue)
    {
        hid_t dsetId = H5Dopen(File_id, datasetname.c_str(), H5P_DEFAULT); hid_t spaceId = H5Dget_space(dsetId);
        hsize_t locDims[1]; H5Sget_simple_extent_dims(spaceId, locDims, NULL); int num = locDims[0];
        hid_t h5string = H5Tcopy(H5T_C_S1); H5Tset_size(h5string, NameSpaceHDFIO::flash_str_len); hid_t dtype;
        if (typeid(T) == typeid(int))         dtype = H5T_NATIVE_INT;
        if (typeid(T) == typeid(double))      dtype = H5T_NATIVE_DOUBLE;
        if (typeid(T) == typeid(bool))        dtype = H5T_NATIVE_HBOOL;
        if (typeid(T) == typeid(std::string)) dtype = H5Tcopy(h5string);
        hid_t datatype = H5Tcreate(H5T_COMPOUND, sizeof(NameSpaceHDFIO::FlashScalarsParametersStruct<T>));
        H5Tinsert(datatype, "name", HOFFSET(NameSpaceHDFIO::FlashScalarsParametersStruct<T>, name), h5string);
        H5Tinsert(datatype, "value", HOFFSET(NameSpaceHDFIO::FlashScalarsParametersStruct<T>, value), dtype);
        NameSpaceHDFIO::FlashScalarsParametersStruct<T> *data = new NameSpaceHDFIO::FlashScalarsParametersStruct<T>[num];
        H5Dread(dsetId, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        for (int i = 0; i < num; i++)
            if (strncmp(data[i].name, fieldname.c_str(), fieldname.size()) == 0) {
                if (Verbose > 1) std::cout<<FuncSig(__func__)<<": Overwriting "<<fieldname<<" of "<<data[i].value
                                        <<" with "<<fieldvalue<<" in '"<<datasetname<<"'"<<std::endl;
                NameSpaceHDFIO::FLASHPropAssign<T>(fieldvalue, &data[i].value);
            }
        H5Dwrite(dsetId, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data); delete [] data;
        H5Tclose(datatype); H5Tclose(h5string); H5Sclose(spaceId); H5Dclose(dsetId);
    };

}; // end: HDFIO
#endif
