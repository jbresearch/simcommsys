/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
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

#ifndef __cuda_matrix_h
#define __cuda_matrix_h

#include "config.h"
#include "cuda-all.h"
#include "../matrix.h"

namespace cuda {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of auto ownership
// 3 - Display data contents when doing shallow copies
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class T>
class matrix_reference;

/*!
 * \brief   A two-dimensional array in device memory
 * \author  Johann Briffa
 *
 * This class represents a '2D array in device memory'. It consists of two
 * parts:
 * 1) The host-side interface contains all the memory-allocation and data
 *    transfer routines. Copies of this object on the host create deep copies
 *    on the device.
 * 2) The device interface contains the data-access routines needed within
 *    device code. Copies of this object on the device create shallow copies
 *    (references to the same memory).
 *
 * Elements are stored in row-major order in a linear array. Each row is
 * padded so that the start of each row is aligned.
 *
 * \todo This class and its associated classes need some thoughtful
 *       reorganization, based on their intended use cases.
 */

template <class T>
class matrix {
private:
   // Class friends
   friend class matrix_reference<T> ;

protected:
   /*! \name Object representation */
   T* data __attribute__((aligned(8))); //!< Pointer to allocated memory in global device space
   size_t pitch __attribute__((aligned(8))); //!< Padded length of each row in bytes
   int rows __attribute__((aligned(8))); //!< Number of matrix rows in elements
   int cols; //!< Number of matrix columns in elements
   // @}

protected:
   /*! \name Test and debug functions */
   /*! \brief Test the validity of the internal representation (host only)
    *
    * There are two possible internal states, determined by the 'data' element:
    * an empty matrix or an allocated one.
    */
   void test_invariant() const
      {
      if (data == NULL)
         {
         assert(rows == 0 && cols == 0);
         assert(pitch == 0);
         }
      else
         {
         assert(rows > 0 && cols > 0);
         assert(pitch >= cols * sizeof(T));
         }
      }
   //! Outputs a standard debug header, identifying object type and address
   std::ostream& debug_header(std::ostream& sout) const
      {
      sout << "DEBUG (cuda::matrix<" << typeid(T).name() << "> at " << this
            << "):";
      return sout;
      }
   //! Outputs a standard debug trailer, identifying object contents
   std::ostream& debug_trailer(std::ostream& sout) const
      {
      if (data == NULL)
         sout << "empty matrix" << std::endl;
      else
         sout << rows << "×" << cols << " elements (size " << sizeof(T)
               << ") at " << data << " (pitch " << pitch << ")" << std::endl;
      return sout;
      }
   // @}

   /*! \name Data setting functions */
   //! shallow copy from an equivalent object
#ifdef __CUDACC__
   __device__ __host__
#endif
   void copyfrom(const matrix<T>& x)
      {
      data = x.data;
      rows = x.rows;
      cols = x.cols;
      pitch = x.pitch;
      }
   //! reset to a null matrix
#ifdef __CUDACC__
   __device__ __host__
#endif
   void reset()
      {
      data = NULL;
      rows = 0;
      cols = 0;
      pitch = 0;
      }
   // @}

   /*! \name Memory allocation functions */
   //! allocate requested number of elements
   void allocate(int m, int n);
   //! free memory
   void free();
   // @}

   /*! \name Element access */
   /*! \brief Returns row start address (write-access)
    * \note Performs boundary checking if used on host.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   T* get_rowaddress(const int i)
      {
      cuda_assert(i >= 0 && i < rows);
      return (T*) ((char*) data + i * pitch);
      }
   /*! \brief Returns row start address (read-only access)
    * \note Performs boundary checking if used on host.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   const T* get_rowaddress(const int i) const
      {
      cuda_assert(i >= 0 && i < rows);
      return (T*) ((char*) data + i * pitch);
      }
   // @}

public:
   /*! \name Constructors */
   /*! \brief Default constructor
    * Does not allocate space.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   matrix() :
      data(NULL), pitch(0), rows(0), cols(0)
      {
      }
   // @}

   /*! \name Law of the Big Three */
   //! Destructor
#ifdef __CUDACC__
   __device__ __host__
#endif
   ~matrix()
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
   __device__ __host__
#endif
   matrix(const matrix<T>& x);
   /*! \brief Copy assignment operator
    * \note Copy assignment on a host is a deep copy.
    * \note Copy assignment on a device is a shallow copy.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   matrix<T>& operator=(const matrix<T>& x);
   // @}

   /*! \name Memory operations */
   /*! \brief Set to given size, freeing if and as required
    *
    * This method leaves the object as it is if the size was already correct,
    * and frees/reallocates if necessary. This helps reduce redundant free/alloc
    * operations on objects which keep the same size.
    */
   void init(const int m, const int n)
      {
      if (m == rows && n == cols)
         return;
      free();
      allocate(m, n);
      }
   /*! \brief Set device memory to the given byte value
    *
    * This method assumes the device object has been allocated.
    */
   void fill(const unsigned char value)
      {
      cudaSafeMemset2D(data, pitch, value, cols, rows);
      }
   // @}

   /*! \name Information functions */
   //! Total number of elements
#ifdef __CUDACC__
   __device__ __host__
#endif
   int size() const
      {
      return rows * cols;
      }
   //! Number of rows
#ifdef __CUDACC__
   __device__ __host__
#endif
   int get_rows() const
      {
      return rows;
      }
   //! Number of columns
#ifdef __CUDACC__
   __device__ __host__
#endif
   int get_cols() const
      {
      return cols;
      }
   // @}

   /*! \name Conversion to/from equivalent host objects */
   //! copy from standard matrix
   matrix<T>& operator=(const libbase::matrix<T>& x);
   //! copy to standard matrix
   operator libbase::matrix<T>() const;
   //! copy from standard vector (matrix in row major order)
   matrix<T>& operator=(const libbase::vector<T>& x);
   //! copy to standard vector (matrix in row major order)
   operator libbase::vector<T>() const;
   // @}

   /*! \name Element access */
   /*! \brief Row extraction (write-access)
    * This allows write access to row data without array copying.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   vector_reference<T> extract_row(const int i)
      {
      return vector_reference<T> (get_rowaddress(i), cols);
      }
   /*! \brief Row extraction (read-only access)
    * This allows read access to row data without array copying.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   const vector_reference<T> extract_row(const int i) const
      {
      return vector_reference<T> (const_cast<T*> (get_rowaddress(i)), cols);
      }
   // @}

   // Methods for device code only
#ifdef __CUDACC__
   /*! \name Element access */
   /*! \brief Index operator (write-access)
    * \note Does not perform boundary checking.
    */
   __device__
   T& operator()(const int i, const int j)
      {
      cuda_assert(i >= 0 && i < rows);
      cuda_assert(j >= 0 && j < cols);
      return get_rowaddress(i)[j];
      }
   /*! \brief Index operator (read-only access)
    * \note Does not perform boundary checking.
    */
   __device__
   const T& operator()(const int i, const int j) const
      {
      cuda_assert(i >= 0 && i < rows);
      cuda_assert(j >= 0 && j < cols);
      return get_rowaddress(i)[j];
      }
   // @}
#endif
};

#ifdef __CUDACC__
template <class T>
inline void matrix<T>::allocate(int m, int n)
   {
   test_invariant();
   // check input parameters
   assert((m > 0 && n > 0) || (m == 0 && n == 0));
   // only allocate on an empty matrix
   assert(data == NULL);
   // if there is something to allocate, do it
   if (m > 0 && n > 0)
      {
      rows = m;
      cols = n;
      data = cudaSafeMalloc2D<T> (&pitch, cols, rows);
      }
   test_invariant();
   }

template <class T>
inline void matrix<T>::free()
   {
   test_invariant();
   // if there is something allocated, free it
   if (data != NULL)
      {
      // free device memory
      cudaSafeFree(data);
      // reset variables
      reset();
      }
   test_invariant();
   }

template <class T>
inline matrix<T>::matrix(const matrix<T>& x) :
data(NULL), pitch(0), rows(0), cols(0)
   {
#ifdef __CUDA_ARCH__ // Device code path (for all compute capabilities)
   copyfrom(x);
#else // Host code path
   if (x.data)
      {
      // allocate memory
      allocate(x.rows, x.cols);
      // copy data from device to device
      cudaSafeMemcpy2D(data, pitch, x.data, x.pitch, cols, rows,
            cudaMemcpyDeviceToDevice);
      }
#endif
   }

template <class T>
inline matrix<T>& matrix<T>::operator=(const matrix<T>& x)
   {
#ifdef __CUDA_ARCH__ // Device code path (for all compute capabilities)
   copyfrom(x);
   return *this;
#else // Host code path
   if (x.data == NULL)
      {
      // deallocate memory if needed
      free();
      }
   else
      {
      // (re-)allocate memory if needed
      init(x.rows, x.cols);
      // copy data from device to device
      cudaSafeMemcpy2D(data, pitch, x.data, x.pitch, cols, rows,
            cudaMemcpyDeviceToDevice);
      }
   return *this;
#endif
   }

template <class T>
inline matrix<T>& matrix<T>::operator=(const libbase::matrix<T>& x)
   {
   // (re-)allocate memory if needed
   init(x.size().rows(), x.size().cols());

   // copy data from host to device if necessary
   if (data != NULL)
      {
      for (int i = 0; i < rows; i++)
         {
         cudaSafeMemcpy(get_rowaddress(i), &x(i, 0), cols,
               cudaMemcpyHostToDevice);
         }
      }

   return *this;
   }

template <class T>
inline matrix<T>::operator libbase::matrix<T>() const
   {
   libbase::matrix<T> x(rows, cols);

   // copy data from device to host if necessary
   if (data != NULL)
      {
      for (int i = 0; i < rows; i++)
         {
         cudaSafeMemcpy(&x(i, 0), get_rowaddress(i), cols, cudaMemcpyDeviceToHost);
         }
      }

   return x;
   }

template <class T>
inline matrix<T>& matrix<T>::operator=(const libbase::vector<T>& x)
   {
   // can only copy from vector of the right size
   assertalways(x.size() == rows * cols);

   // copy data from host to device if necessary
   if (data != NULL)
      {
      cudaSafeMemcpy2D(data, pitch, &x(0), cols * sizeof(T), cols, rows, cudaMemcpyHostToDevice);
      }

   return *this;
   }

template <class T>
inline matrix<T>::operator libbase::vector<T>() const
   {
   libbase::vector<T> x(rows * cols);

   // copy data from device to host if necessary
   if (data != NULL)
      {
      cudaSafeMemcpy2D(&x(0), cols * sizeof(T), data, pitch, cols, rows, cudaMemcpyDeviceToHost);
      }

   return x;
   }
#endif

// Prior definition of matrix class

template <class T>
class matrix;

/*!
 * \brief   A reference to a two-dimensional array in device memory.
 * \author  Johann Briffa
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
class matrix_reference : public matrix<T> {
private:
   // Class friends
   friend class matrix<T> ;
   // Shorthand for class hierarchy
   typedef matrix<T> Base;
protected:
   /*! \name Test and debug functions */
   //! Outputs a standard debug header, identifying object type and address
   std::ostream& debug_header(std::ostream& sout) const
      {
      sout << "DEBUG (cuda::matrix_reference<" << typeid(T).name() << "> at "
            << this << "):";
      return sout;
      }
   //! Outputs a standard debug trailer, identifying object contents
   std::ostream& debug_trailer(std::ostream& sout) const
      {
      return Base::debug_trailer(sout);
      }
   // @}

   /*! \name Internal functions */
   //TODO: add support for partial matrix extraction
   //! Unique constructor, can be called only by friends
   //#ifdef __CUDACC__
   //   __device__ __host__
   //#endif
   //   matrix_reference(T* start, const int n)
   //      {
   //      // update base class by shallow copy, as needed
   //      if (n > 0)
   //         {
   //         Base::length = n;
   //         Base::data = start;
   //         }
   //      }
   // @}
   /*! \name Resizing operations */
   /*! \brief Set to given size, freeing if and as required
    *
    * This method is disabled in matrix references.
    */
   void init(const int m, const int n)
      {
      failwith("Not supported.");
      }
   // @}
public:
   /*! \name Constructors */
   /*! \brief Principal constructor
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   matrix_reference()
      {
      }
   /*! \brief Automatic conversion from normal matrix
    * \warning This allows modification of 'const' matrixs
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   matrix_reference(const matrix<T>& x)
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
   __device__ __host__
#endif
   matrix_reference<T>& operator=(const matrix<T>& x)
      {
      Base::copyfrom(x);
      return *this;
      }

   /*! \name Law of the Big Three */
   //! Destructor
#ifdef __CUDACC__
   __device__ __host__
#endif
   ~matrix_reference()
      {
      // reset base class, in preparation for eventual destruction
      Base::reset();
      }
   /*! \brief Copy constructor
    * \note Copy construction is a shallow copy.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   matrix_reference(const matrix_reference<T>& x)
      {
      // do not invoke the base constructor, to avoid a deep copy
      Base::copyfrom(x);
      }
   /*! \brief Copy assignment operator
    * \note Copy assignment is a shallow copy.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   matrix_reference<T>& operator=(const matrix_reference<T>& x)
      {
      Base::copyfrom(x);
      return *this;
      }
   // @}

   /*! \name Conversion to/from equivalent host objects */
   //! copy from standard matrix
   matrix_reference<T>& operator=(const libbase::matrix<T>& x);
   //! copy to standard matrix
   operator libbase::matrix<T>() const
      {
      return Base::operator libbase::matrix<T>();
      }
   // @}
};

#ifdef __CUDACC__

template <class T>
inline matrix_reference<T>& matrix_reference<T>::operator=(const libbase::matrix<T>& x)
   {
   assert(x.size().rows() == Base::rows);
   assert(x.size().cols() == Base::cols);
   // copy data from host to device
   for (int i = 0; i < Base::rows; i++)
      {
      cudaSafeMemcpy(Base::get_rowaddress(i), &x(i, 0), Base::cols,
            cudaMemcpyHostToDevice);
      }
   return *this;
   }

#endif

/*!
 * \brief   A two-dimensional array in device memory - automatic
 * \author  Johann Briffa
 *
 * The first automatic object creates an actual matrix on the device.
 * Copies of this object (through copy construction) create shallow copies
 * (references to the same memory) on the device.
 */
template <class T>
class matrix_auto : public matrix<T> {
private:
   // Shorthand for class hierarchy
   typedef matrix<T> Base;
private:
   bool isowner __attribute__((aligned(8)));
protected:
   /*! \name Test and debug functions */
   //! Outputs a standard debug header, identifying object type and address
   std::ostream& debug_header(std::ostream& sout) const
      {
      sout << "DEBUG (cuda::matrix_auto<" << typeid(T).name() << "> at "
            << this << "):";
      return sout;
      }
   //! Outputs a standard debug trailer, identifying object contents
   std::ostream& debug_trailer(std::ostream& sout) const
      {
      if (!isowner)
         sout << "link to ";
      return Base::debug_trailer(sout);
      }
   // @}

private:
   /*! \name Memory allocation functions */
   //! allocate requested number of elements (host only)
   void allocate(int m, int n)
      {
      // this should only be called on an owned object
      assert(isowner);
      Base::allocate(m, n);
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " allocated ";
      debug_trailer(std::cerr);
#endif
      }
   //! free memory if we own it, and reset pointer (host only)
   void free()
      {
      if (isowner)
         {
#if DEBUG>=2
         debug_header(std::cerr);
         std::cerr << " deallocating ";
         debug_trailer(std::cerr);
#endif
         Base::free();
         }
      else
         {
#if DEBUG>=2
         debug_header(std::cerr);
         std::cerr << " unlinking ";
         debug_trailer(std::cerr);
#endif
         Base::reset();
         isowner = true;
         }
      }
   // @}
public:
   /*! \name Law of the Big Three */
   //! Destructor
#ifdef __CUDACC__
   __device__ __host__
#endif
   ~matrix_auto()
      {
#ifndef __CUDA_ARCH__ // Host code path
      // decide what to do before the base object is destroyed
      free();
#endif
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " destroyed" << std::endl;
#endif
      }
   /*! \brief Copy constructor
    * \note Copy construction is a shallow copy.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   matrix_auto(const matrix_auto<T>& x) :
      isowner(false)
      {
      // do not invoke the base constructor, to avoid a deep copy
      Base::copyfrom(x);
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " copy construction from " << &x << " - new ";
      debug_trailer(std::cerr);
#endif
      }
   /*! \brief Copy assignment operator
    * \note Copy assignment is a shallow copy.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   matrix_auto<T>& operator=(const matrix_auto<T>& x)
      {
#ifndef __CUDA_ARCH__
      // decide what to do before copying
      free();
#endif
      // do not invoke the base copy assignment, to avoid a deep copy
      Base::copyfrom(x);
      // determine if this should be an owner (only own an empty object)
      isowner = (Base::data == NULL);
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " copy assignment from " << &x << " - new ";
      debug_trailer(std::cerr);
#endif
      return *this;
      }
   // @}

   /*! \name Other Constructors */
   /*! \brief Default constructor
    * Does not allocate space.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   matrix_auto() :
      isowner(true)
      {
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " new ";
      debug_trailer(std::cerr);
#endif
      }
   // @}

   /*! \name Memory operations */
   /*! \brief Set to given size, freeing if and as required (host only)
    *
    * This method leaves the object as it is if the size was already correct,
    * and frees/reallocates if necessary. This helps reduce redundant free/alloc
    * operations on objects which keep the same size.
    *
    * If re-allocation is necessary:
    * - the old memory is only freed if this object is not a reference.
    * - the object becomes the owner of the newly allocated memory, even if it
    *   was only a reference to the old memory.
    */
   void init(const int m, const int n)
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " set size to " << m << "×" << n << " for ";
      debug_trailer(std::cerr);
#endif
      if (m == Base::rows && n == Base::cols)
         return;
      free();
      allocate(m, n);
      }
   // @}

   /*! \name Conversion to/from equivalent host objects */
   /*! \brief Copy from standard matrix
    *
    * This method re-allocates memory, taking ownership, if necessary
    */
   matrix_auto<T>& operator=(const libbase::matrix<T>& x)
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " copy from host object " << &x << std::endl;
#endif
      // (re-)allocate memory if needed
      init(x.size().rows(), x.size().cols());
      Base::operator=(x);
      return *this;
      }
   //! copy to standard matrix
   operator libbase::matrix<T>() const
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " copy to host object" << std::endl;
#endif
      return Base::operator libbase::matrix<T>();
      }
   // @}
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
