/*!
 * \file
 *
 * Copyright (c) 2024 Mark Mizzi
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __cuda_matrix3_h
#define __cuda_matrix3_h

#include "../vector.h"
#include "config.h"
#include "util.h"
#include "vector.h"

namespace cuda
{

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of auto ownership
// 3 - Display data contents when doing shallow copies
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

template <class T>
class matrix3_reference;

/*!
 * \brief   A three-dimensional array in device memory
 * \author  Mark Mizzi
 *
 * This class represents a '3D array in device memory'. It consists of two
 * parts:
 * 1) The host-side interface contains all the memory-allocation and data
 *    transfer routines. Copies of this object on the host create deep copies
 *    on the device.
 * 2) The device interface contains the data-access routines needed within
 *    device code. Copies of this object on the device create shallow copies
 *    (references to the same memory).
 *
 * Elements are stored in row-major order in a linear array.
 * x is the innermost index, followed by y, with the outermost index being z.
 * Elements with same y and z indices are stored consecutively.
 * A row in the array consists of elements with the same y and z indices but
 * varying x index; these are contiguous in memory. Each row is padded so that
 * the start of each row is aligned.
 *
 * \todo This class and its associated classes need some thoughtful
 *       reorganization, based on their intended use cases.
 */

template <class T>
class matrix3
{
private:
    // Class friends
    friend class matrix3_reference<T>;

protected:
    /*! \name Object representation */
    T* data __attribute__((
        aligned(8))); //!< Pointer to allocated memory in global device space
    //! Size in bytes of padded rows (a row consists of elements with same y, z
    //! but varying x)
    size_t pitch __attribute__((aligned(8)));
    //! Width along z-axis of the 3D matrix
    int zsize __attribute__((aligned(8)));
    //! Width along y-axis of the 3D matrix
    int ysize;
    //! Width along x-axis of the 3D matrix
    // @}
    int xsize;

protected:
    /*! \name Test and debug functions */
    /*! \brief Test the validity of the internal representation (host only)
     *
     * There are two possible internal states, determined by the 'data' element:
     * an empty matrix or an allocated one.
     */
    void test_invariant() const
    {
        if (data == NULL) {
            assert(xsize == 0 && ysize == 0 && zsize == 0);
            assert(pitch == 0);
        } else {
            assert(xsize > 0 && ysize > 0 && zsize > 0);
            assert(pitch >= xsize * sizeof(T));
        }
    }
    //! Outputs a standard debug header, identifying object type and address
    std::ostream& debug_header(std::ostream& sout) const
    {
        sout << "DEBUG (cuda::matrix3<" << typeid(T).name() << "> at " << this
             << "):";
        return sout;
    }
    //! Outputs a standard debug trailer, identifying object contents
    std::ostream& debug_trailer(std::ostream& sout) const
    {
        if (data == NULL) {
            sout << "empty matrix3" << std::endl;
        } else {
            sout << zsize << "×" << ysize << "x" << xsize << " elements (size "
                 << sizeof(T) << ") at " << data << " (pitch " << pitch << ")"
                 << std::endl;
        }

        return sout;
    }
    // @}

    /*! \name Data setting functions */
    //! shallow copy from an equivalent object
#ifdef __CUDACC__
    __device__
    __host__
#endif
    void copyfrom(const matrix3<T>& x)
    {
        data = x.data;
        xsize = x.xsize;
        ysize = x.ysize;
        zsize = x.zsize;
        pitch = x.pitch;
    }
    //! reset to a null matrix
#ifdef __CUDACC__
    __device__
    __host__
#endif
    void reset()
    {
        data = NULL;
        xsize = 0;
        ysize = 0;
        zsize = 0;
        pitch = 0;
    }
    // @}

    /*! \name Memory allocation functions */
    //! allocate requested number of elements
    void allocate(int n_zsize, int n_ysize, int n_xsize);
    //! free memory
    void free();
    // @}

    /*! \name Element access */
    /*! \brief Returns row start address (write-access)
     * \note Rows in this context consist of all elements with same y, z indices
     * but varying x
     * \note Performs boundary checking if used on host.
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    T* get_rowaddress(const int z, const int y)
    {
        cuda_assert(y >= 0 && y < ysize);
        cuda_assert(z >= 0 && z < zsize);
        return (T*)((char*)data + z * pitch * ysize + y * pitch);
    }
    /*! \brief Returns row start address (read-only access)
     * \note Rows in this context consist of all elements with same y, z indices
     * but varying x
     * \note Performs boundary checking if used on host.
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    const T* get_rowaddress(const int z, const int y) const
    {
        cuda_assert(y >= 0 && y < ysize);
        cuda_assert(z >= 0 && z < zsize);
        return (T*)((char*)data + z * pitch * ysize + y * pitch);
    }
    // @}

public:
    /*! \name Constructors */
    /*! \brief Default constructor
     * Does not allocate space.
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    matrix3() : data(NULL), pitch(0), xsize(0), ysize(0), zsize(0) {}
    // @}

    /*! \name Law of the Big Three */
    //! Destructor
#ifdef __CUDACC__
    __device__
    __host__
#endif
    ~matrix3()
    {
#ifndef __CUDA_ARCH__ // Host code path
        free();
#endif
    }
    /*! \brief Copy constructor
     * \note Copy construction on a host is a deep copy.
     * \note Copy construction on a device is a shallow copy.
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    matrix3(const matrix3<T>& x);
    /*! \brief Copy assignment operator
     * \note Copy assignment on a host is a deep copy.
     * \note Copy assignment on a device is a shallow copy.
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    matrix3<T>& operator=(const matrix3<T>& x);
    // @}

    /*! \name Memory operations */
    /*! \brief Set to given size, freeing if and as required
     *
     * This method leaves the object as it is if the size was already correct,
     * and frees/reallocates if necessary. This helps reduce redundant
     * free/alloc operations on objects which keep the same size.
     */
    void init(const int n_zsize, const int n_ysize, const int n_xsize)
    {
        if (xsize == n_xsize && ysize == n_ysize && zsize == n_zsize) {
            return;
        }

        free();
        allocate(n_zsize, n_ysize, n_xsize);
    }
    /*! \brief Set device memory to the given byte value
     *
     * This method assumes the device object has been allocated.
     */
    void fill(const unsigned char value)
    {
        cudaSafeMemset3D(data, value, pitch, xsize, ysize, zsize);
    }
    // @}

    /*! \name Information functions */
    //! Total number of elements
#ifdef __CUDACC__
    __device__
    __host__
#endif
    int size() const { return xsize * ysize * zsize; }
    //! Width along x-axis
#ifdef __CUDACC__
    __device__
    __host__
#endif
    int get_xsize() const { return xsize; }
    //! Width along y-axis
#ifdef __CUDACC__
    __device__
    __host__
#endif
    int get_ysize() const { return ysize; }
    //! Width along z-axis
#ifdef __CUDACC__
    __device__
    __host__
#endif
    int get_zsize() const { return zsize; }
    // @}

    /*! \name Conversion to/from equivalent host objects */
    //! copy from standard vector (matrix in row major order, with elements
    //! having same z being consecutive)
    matrix3<T>& operator=(const libbase::vector<T>& x);
    //! copy to standard vector (matrix in row major order, with elements having
    //! same z being consecutive)
    operator libbase::vector<T>() const;
    // @}

    /*! \name Element access */
    /*! \brief Row extraction (write-access)
     * This allows write access to row data without array copying.
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    vector_reference<T> extract_row(const int z, const int y)
    {
        return vector_reference<T>(get_rowaddress(z, y), xsize);
    }
    /*! \brief Row extraction (read-only access)
     * This allows read access to row data without array copying.
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    const vector_reference<T> extract_row(const int z, const int y) const
    {
        return vector_reference<T>(const_cast<T*>(get_rowaddress(z, y)), xsize);
    }
    // @}

    // Methods for device code only
#ifdef __CUDACC__
    /*! \name Element access */
    /*! \brief Index operator (write-access)
     * \note Does not perform boundary checking.
     */
    __device__
    T& operator()(const int z, const int y, const int x)
    {
        cuda_assert(x >= 0 && x < xsize);
        cuda_assert(y >= 0 && y < ysize);
        cuda_assert(z >= 0 && z < zsize);
        return get_rowaddress(z, y)[x];
    }
    /*! \brief Index operator (read-only access)
     * \note Does not perform boundary checking.
     */
    __device__
    const T& operator()(const int z, const int y, const int x) const
    {
        cuda_assert(x >= 0 && x < xsize);
        cuda_assert(y >= 0 && y < ysize);
        cuda_assert(z >= 0 && z < zsize);
        return get_rowaddress(z, y)[x];
    }
    // @}
#endif
};

#ifdef __CUDACC__
template <class T>
inline void
matrix3<T>::allocate(int n_zsize, int n_ysize, int n_xsize)
{
    test_invariant();
    // check input parameters
    assert((n_xsize > 0 && n_ysize > 0 && n_zsize > 0) ||
           (n_xsize == 0 && n_ysize == 0 && n_zsize == 0));
    // only allocate on an empty matrix
    assert(data == NULL);
    // if there is something to allocate, do it
    if (n_xsize > 0 && n_ysize > 0 && n_zsize > 0) {
        xsize = n_xsize;
        ysize = n_ysize;
        zsize = n_zsize;
        data = cudaSafeMalloc3D<T>(&pitch, xsize, ysize, zsize);
    }
    test_invariant();
}

template <class T>
inline void
matrix3<T>::free()
{
    test_invariant();
    // if there is something allocated, free it
    if (data != NULL) {
        // free device memory
        cudaSafeFree(data);
        // reset variables
        reset();
    }
    test_invariant();
}

template <class T>
inline matrix3<T>::matrix3(const matrix3<T>& x)
    : data(NULL), pitch(0), xsize(0), ysize(0), zsize(0)
{
#    ifdef __CUDA_ARCH__ // Device code path (for all compute capabilities)
    copyfrom(x);
#    else // Host code path
    if (x.data) {
        // allocate memory
        allocate(x.zsize, x.ysize, x.xsize);
        // copy data from device to device
        cudaSafeMemcpy3D(data,
                         pitch,
                         x.data,
                         x.pitch,
                         x.xsize,
                         x.ysize,
                         x.zsize,
                         cudaMemcpyDeviceToDevice);
    }
#    endif
}

template <class T>
inline matrix3<T>&
matrix3<T>::operator=(const matrix3<T>& x)
{
#    ifdef __CUDA_ARCH__ // Device code path (for all compute capabilities)
    copyfrom(x);
    return *this;
#    else // Host code path
    if (x.data == NULL) {
        // deallocate memory if needed
        free();
    } else {
        // (re-)allocate memory if needed
        init(x.zsize, x.ysize, x.xsize);
        // copy data from device to device
        cudaSafeMemcpy3D(data,
                         pitch,
                         x.data,
                         x.pitch,
                         xsize,
                         ysize,
                         zsize,
                         cudaMemcpyDeviceToDevice);
    }
    return *this;
#    endif
}

template <class T>
inline matrix3<T>&
matrix3<T>::operator=(const libbase::vector<T>& x)
{
    // can only copy from vector of the right size
    assertalways(x.size() == xsize * ysize * zsize);

    // copy data from host to device if necessary
    // Because of invariant, data == NULL if and only if
    //   xsize == ysize == zsize == 0
    //   and in this case x.size() = 0 as well due to invariant above.
    //   mitigating need to copy.
    if (data != NULL) {
        cudaSafeMemcpy3D(data,
                         pitch,
                         &x(0),
                         // vector is compact; no padding
                         xsize * sizeof(T),
                         xsize,
                         ysize,
                         zsize,
                         cudaMemcpyHostToDevice);
    }

    return *this;
}

template <class T>
inline matrix3<T>::operator libbase::vector<T>() const
{
    libbase::vector<T> x(xsize * ysize * zsize);

    // copy data from device to host if necessary
    if (data != NULL) {
        cudaSafeMemcpy3D(&x(0),
                         // vector is compact; no padding
                         xsize * sizeof(T),
                         data,
                         pitch,
                         xsize,
                         ysize,
                         zsize,
                         cudaMemcpyDeviceToHost);
    }

    return x;
}
#endif

// Prior definition of matrix class

template <class T>
class matrix3;

/*!
 * \brief   A reference to a three-dimensional array in device memory.
 * \author  Mark Mizzi
 *
 * A matrix reference is a matrix that does not own its allocated memory.
 * Consequently, all operations that require a resize are forbidden.
 * The data set is really just a reference to (part of) a regular matrix.
 * When an indirect matrix is destroyed, the actual allocated memory is not
 * released. This only happens when the referenced matrix is destroyed.
 * There is always a risk that the referenced matrix is destroyed before
 * the indirect references, in which case those references become stale.
 *
 * It is intended that for the user, the use of matrix references should be
 * essentially transparent (in that they can mostly be used in place of a
 * normal matrix). There is only one scenario where the user needs to create
 * one explicitly: when passing as an argument to a kernel, since these do
 * not take reference arguments in the usual way. Otherwise, creation should
 * happen only through a normal matrix's methods.
 */
template <class T>
class matrix3_reference : public matrix3<T>
{
private:
    // Class friends
    friend class matrix3<T>;
    // Shorthand for class hierarchy
    typedef matrix3<T> Base;

protected:
    /*! \name Test and debug functions */
    //! Outputs a standard debug header, identifying object type and address
    std::ostream& debug_header(std::ostream& sout) const
    {
        sout << "DEBUG (cuda::matrix3_reference<" << typeid(T).name() << "> at "
             << this << "):";
        return sout;
    }
    //! Outputs a standard debug trailer, identifying object contents
    std::ostream& debug_trailer(std::ostream& sout) const
    {
        return Base::debug_trailer(sout);
    }
    // @}

    /*! \name Resizing operations */
    /*! \brief Set to given size, freeing if and as required
     *
     * This method is disabled in matrix references.
     */
    void init(const int n_zsize, const int n_ysize, const int n_xsize)
    {
        failwith("Not supported.");
    }
    // @}
public:
    /*! \name Constructors */
    /*! \brief Principal constructor
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    matrix3_reference() {}
    /*! \brief Automatic conversion from normal matrix
     * \warning This allows modification of 'const' matrixs
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    matrix3_reference(const matrix3<T>& x)
    {
        // do not invoke the base constructor, to avoid a deep copy
        // note: this operation requires this class to be a friend of matrix
        Base::copyfrom(x);
    }
    // @}
    /*! \brief Assignment from normal matrix
     * \note Assignment is a shallow copy.
     * \warning This allows modification of 'const' matrixs
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    matrix3_reference<T>& operator=(const matrix3<T>& x)
    {
        Base::copyfrom(x);
        return *this;
    }

    /*! \name Law of the Big Three */
    //! Destructor
#ifdef __CUDACC__
    __device__
    __host__
#endif
    ~matrix3_reference()
    {
        // reset base class, in preparation for eventual destruction
        Base::reset();
    }
    /*! \brief Copy constructor
     * \note Copy construction is a shallow copy.
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    matrix3_reference(const matrix3_reference<T>& x)
    {
        // do not invoke the base constructor, to avoid a deep copy
        Base::copyfrom(x);
    }
    /*! \brief Copy assignment operator
     * \note Copy assignment is a shallow copy.
     */
#ifdef __CUDACC__
    __device__
    __host__
#endif
    matrix3_reference<T>& operator=(const matrix3_reference<T>& x)
    {
        Base::copyfrom(x);
        return *this;
    }
    // @}
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG
#endif

} // namespace cuda

#endif // __cuda_matrix3_h
