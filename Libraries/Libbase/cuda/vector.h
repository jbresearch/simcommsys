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

#ifndef __cuda_vector_h
#define __cuda_vector_h

#include "config.h"
#include "cuda-all.h"
#include "../vector.h"

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
class vector_reference;

/*!
 * \brief   A one-dimensional array in device memory
 * \author  Johann Briffa
 *
 * This class represents a '1D array in device memory'. It consists of two
 * parts:
 * 1) The host-side interface contains all the memory-allocation and data
 *    transfer routines. Copies of this object on the host create deep copies
 *    on the device.
 * 2) The device interface contains the data-access routines needed within
 *    device code. Copies of this object on the device create shallow copies
 *    (references to the same memory).
 *
 * \todo This class and its associated classes need some thoughtful
 *       reorganization, based on their intended use cases.
 */

template <class T>
class vector {
private:
   // Class friends
   friend class vector_reference<T> ;

protected:
   /*! \name Object representation */
   T* data __attribute__((aligned(8))); //!< Pointer to allocated memory in global device space
   int length __attribute__((aligned(8))); //!< Length of vector in elements
   // @}

protected:
   /*! \name Test functions */
   /*! \brief Test the validity of the internal representation (host only)
    *
    * There are two possible internal states, determined by the 'data' element:
    * an empty vector or an allocated one.
    */
   void test_invariant() const
      {
      if (data == NULL)
         {
         assert(length == 0);
         }
      else
         {
         assert(length > 0);
         }
      }
   //! Outputs a standard debug header, identifying object type and address
   std::ostream& debug_header(std::ostream& sout) const
      {
      sout << "DEBUG (cuda::vector<" << typeid(T).name() << "> at " << this
            << "):";
      return sout;
      }
   //! Outputs a standard debug trailer, identifying object contents
   std::ostream& debug_trailer(std::ostream& sout) const
      {
      if (data == NULL)
         sout << "empty vector" << std::endl;
      else
         sout << length << " elements (size " << sizeof(T) << ") at " << data
               << std::endl;
      return sout;
      }
   // @}

   /*! \name Data setting functions */
   //! shallow copy from an equivalent object
#ifdef __CUDACC__
   __device__ __host__
#endif
   void copyfrom(const vector<T>& x)
      {
      data = x.data;
      length = x.length;
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " copyfrom(" << &x << ") - ";
      debug_trailer(std::cerr);
#endif
      }
   //! reset to a null vector
#ifdef __CUDACC__
   __device__ __host__
#endif
   void reset()
      {
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " reset()" << std::endl;
#endif
      data = NULL;
      length = 0;
      }
   // @}

   /*! \name Memory allocation functions */
   //! allocate requested number of elements
   void allocate(int n);
   //! free memory
   void free();
   // @}

public:
   /*! \name Constructors */
   /*! \brief Default constructor
    * Does not allocate space.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   vector() :
      data(NULL), length(0)
      {
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " new ";
      debug_trailer(std::cerr);
#endif
      }
   // @}

   /*! \name Law of the Big Three */
   //! Destructor
#ifdef __CUDACC__
   __device__ __host__
#endif
   ~vector()
      {
#ifndef __CUDA_ARCH__ // Host code path
      free();
#endif
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " destroyed" << std::endl;
#endif
      }
   /*! \brief Copy constructor
    * \note Copy construction on a host is a deep copy.
    * \note Copy construction on a device is a shallow copy.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   vector(const vector<T>& x);
   /*! \brief Copy assignment operator
    * \note Copy assignment on a host is a deep copy.
    * \note Copy assignment on a device is a shallow copy.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   vector<T>& operator=(const vector<T>& x);
   // @}

   /*! \name Memory operations */
   /*! \brief Set to given size, freeing if and as required
    *
    * This method leaves the object as it is if the size was already correct,
    * and frees/reallocates if necessary. This helps reduce redundant free/alloc
    * operations on objects which keep the same size.
    */
   void init(const int n)
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " set size to " << n << " for ";
      debug_trailer(std::cerr);
#endif
      if (n == length)
         return;
      free();
      allocate(n);
      }
   /*! \brief Set device memory to the given byte value
    *
    * This method assumes the device object has been allocated.
    */
   void fill(const unsigned char value)
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " fill with " << int(value) << " for ";
      debug_trailer(std::cerr);
#endif
      cudaSafeMemset(data, value, length);
      }
   // @}

   /*! \name Information functions */
   //! Total number of elements
#ifdef __CUDACC__
   __device__ __host__
#endif
   int size() const
      {
      return length;
      }
   // @}

   /*! \name Conversion to/from equivalent host objects */
   //! copy from standard vector
   vector<T>& operator=(const libbase::vector<T>& x);
   //! copy to standard vector
   operator libbase::vector<T>() const;
   // @}

   /*! \name Element access */
   /*! \brief Extract a sub-vector as a reference into this vector
    * This allows read access to sub-vector data without array copying.
    * \note Performs boundary checking if used on host.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   const vector_reference<T> extract(const int start, const int n) const
      {
      cuda_assert(start >= 0);
      cuda_assert(n >= 0);
      cuda_assert(start + n <= length);
      return vector_reference<T> (const_cast<T*> (data + start), n);
      }
   /*! \brief Access part of this vector as a sub-vector
    * This allows write access to sub-vector data without array copying.
    * \note Performs boundary checking if used on host.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   vector_reference<T> segment(const int start, const int n)
      {
      cuda_assert(start >= 0);
      cuda_assert(n >= 0);
      cuda_assert(start + n <= length);
      return vector_reference<T> (data + start, n);
      }
   // @}

   // Methods for device code only
#ifdef __CUDACC__
   /*! \name Element access */
   /*! \brief Index operator (write-access)
    * \note Does not perform boundary checking.
    */
   __device__
   T& operator()(const int x)
      {
      cuda_assert(x >= 0 && x < length);
      return data[x];
      }
   /*! \brief Index operator (read-only access)
    * \note Does not performs boundary checking.
    */
   __device__
   const T& operator()(const int x) const
      {
      cuda_assert(x >= 0 && x < length);
      return data[x];
      }
   // @}
#endif
};

#ifdef __CUDACC__
template <class T>
inline void vector<T>::allocate(int n)
   {
   test_invariant();
   // check input parameters
   assert(n >= 0);
   // only allocate on an empty matrix
   assert(data == NULL);
   // if there is something to allocate, do it
   if (n > 0)
      {
      length = n;
      data = cudaSafeMalloc<T>(length);
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " allocated ";
      debug_trailer(std::cerr);
#endif
      }
   test_invariant();
   }

template <class T>
inline void vector<T>::free()
   {
   test_invariant();
   // if there is something allocated, free it
   if (data != NULL)
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " deallocating ";
      debug_trailer(std::cerr);
#endif
      // free device memory
      cudaSafeFree(data);
      // reset variables
      reset();
      }
   test_invariant();
   }

template <class T>
inline vector<T>::vector(const vector<T>& x) :
data(NULL), length(0)
   {
#ifdef __CUDA_ARCH__ // Device code path (for all compute capabilities)
   copyfrom(x);
#else // Host code path
   // allocate memory if needed
   allocate(x.length);
   // copy data from device to device if necessary
   if (data != NULL)
      {
      cudaSafeMemcpy(data, x.data, length, cudaMemcpyDeviceToDevice);
      }
#if DEBUG>=2
   debug_header(std::cerr);
   std::cerr << " copy construction from " << &x << " - new ";
   debug_trailer(std::cerr);
#endif
#endif
   }

template <class T>
inline vector<T>& vector<T>::operator=(const vector<T>& x)
   {
#ifdef __CUDA_ARCH__ // Device code path (for all compute capabilities)
   copyfrom(x);
   return *this;
#else // Host code path
   // (re-)allocate memory if needed
   init(x.length);
   // copy data from device to device if necessary
   if (data != NULL)
      {
      cudaSafeMemcpy(data, x.data, length, cudaMemcpyDeviceToDevice);
      }
#if DEBUG>=2
   debug_header(std::cerr);
   std::cerr << " copy assignment from " << &x << " - new ";
   debug_trailer(std::cerr);
#endif
   return *this;
#endif
   }

template <class T>
inline vector<T>& vector<T>::operator=(const libbase::vector<T>& x)
   {
#if DEBUG>=2
   debug_header(std::cerr);
   std::cerr << " copy from host object " << &x << std::endl;
#endif
   // (re-)allocate memory if needed
   init(x.size().length());
   // copy data from host to device if necessary
   if (data != NULL)
      {
      cudaSafeMemcpy(data, &x(0), length, cudaMemcpyHostToDevice);
      }
   return *this;
   }

template <class T>
inline vector<T>::operator libbase::vector<T>() const
   {
#if DEBUG>=2
   debug_header(std::cerr);
   std::cerr << " copy to host object" << std::endl;
#endif
   // allocate place for result
   libbase::vector<T> x(length);
   // copy data from device to host if necessary
   if (data != NULL)
      {
      cudaSafeMemcpy(&x(0), data, length, cudaMemcpyDeviceToHost);
      }
   return x;
   }
#endif

// Prior definition of matrix class

template <class T>
class matrix;

/*!
 * \brief   A reference to a one-dimensional array in device memory.
 * \author  Johann Briffa
 *
 * A vector reference is a vector that does not own its allocated memory.
 * Consequently, all operations that require a resize are forbidden.
 * The data set is really just a reference to (part of) a regular vector.
 * When an indirect vector is destroyed, the actual allocated memory is not
 * released. This only happens when the referenced vector is destroyed.
 * There is always a risk that the referenced vector is destroyed before
 * the indirect references, in which case those references become stale.
 *
 * It is intended that for the user, the use of vector references should be
 * essentially transparent (in that they can mostly be used in place of a
 * normal vector). There is only one scenario where the user needs to create
 * one explicitly: when passing as an argument to a kernel, since these do
 * not take reference arguments in the usual way. Otherwise, creation should
 * happen only through a normal vector's methods.
 */
template <class T>
class vector_reference : public vector<T> {
private:
   // Class friends
   friend class vector<T> ;
   friend class matrix<T> ;
   // Shorthand for class hierarchy
   typedef vector<T> Base;
protected:
   /*! \name Test and debug functions */
   //! Outputs a standard debug header, identifying object type and address
   std::ostream& debug_header(std::ostream& sout) const
      {
      sout << "DEBUG (cuda::vector_reference<" << typeid(T).name() << "> at "
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
    * This method is disabled in vector references.
    */
   void init(const int n)
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
   vector_reference()
      {
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " new ";
      debug_trailer(std::cerr);
#endif
      }
   /*! \brief Automatic conversion from normal vector
    * \warning This allows modification of 'const' vectors
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   vector_reference(const vector<T>& x)
      {
      // do not invoke the base constructor, to avoid a deep copy
      // note: this operation requires this class to be a friend of vector
      Base::copyfrom(x);
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " copy construction from base object " << &x << " - new ";
      debug_trailer(std::cerr);
#endif
      }
   //! Unique constructor
#ifdef __CUDACC__
   __device__ __host__
#endif
   vector_reference(T* start, const int n)
      {
      // update base class by shallow copy, as needed
      if (n > 0)
         {
         Base::length = n;
         Base::data = start;
         }
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " new ";
      debug_trailer(std::cerr);
#endif
      }
   // @}
   /*! \brief Assignment from normal vector
    * \note Assignment is a shallow copy.
    * \warning This allows modification of 'const' vectors
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   vector_reference<T>& operator=(const vector<T>& x)
      {
      Base::copyfrom(x);
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " copy assignment from base object " << &x << " - new ";
      debug_trailer(std::cerr);
#endif
      return *this;
      }

   /*! \name Law of the Big Three */
   //! Destructor
#ifdef __CUDACC__
   __device__ __host__
#endif
   ~vector_reference()
      {
      // reset base class, in preparation for eventual destruction
      Base::reset();
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
   vector_reference(const vector_reference<T>& x)
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
   vector_reference<T>& operator=(const vector_reference<T>& x)
      {
      Base::copyfrom(x);
#if DEBUG>=2 && !defined(__CUDA_ARCH__)
      debug_header(std::cerr);
      std::cerr << " copy assignment from " << &x << " - new ";
      debug_trailer(std::cerr);
#endif
      return *this;
      }
   // @}

   /*! \name Conversion to/from equivalent host objects */
   //! copy from standard vector
   vector_reference<T>& operator=(const libbase::vector<T>& x);
   //! copy to standard vector
   operator libbase::vector<T>() const
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " copy to host object" << std::endl;
#endif
      return Base::operator libbase::vector<T>();
      }
   // @}
};

#ifdef __CUDACC__

template <class T>
inline vector_reference<T>& vector_reference<T>::operator=(const libbase::vector<T>& x)
   {
#if DEBUG>=2
   debug_header(std::cerr);
   std::cerr << " copy from host object " << &x << std::endl;
#endif
   assert(x.size() == Base::length);
   // copy data from host to device
   cudaSafeMemcpy(Base::data, &x(0), Base::length, cudaMemcpyHostToDevice);
   return *this;
   }

#endif

/*!
 * \brief   A one-dimensional array in device memory - automatic
 * \author  Johann Briffa
 *
 * The first automatic object creates an actual vector on the device.
 * Copies of this object (through copy construction) create shallow copies
 * (references to the same memory) on the device.
 */
template <class T>
class vector_auto : public vector<T> {
private:
   // Shorthand for class hierarchy
   typedef vector<T> Base;
private:
   bool isowner __attribute__((aligned(8)));
protected:
   /*! \name Test and debug functions */
   //! Outputs a standard debug header, identifying object type and address
   std::ostream& debug_header(std::ostream& sout) const
      {
      sout << "DEBUG (cuda::vector_auto<" << typeid(T).name() << "> at "
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
   void allocate(int n)
      {
      // this should only be called on an owned object
      assert(isowner);
      Base::allocate(n);
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
   ~vector_auto()
      {
#ifndef __CUDA_ARCH__ // Host code path
      // decide what to do before the base object is destroyed
      free();
#endif
      }
   /*! \brief Copy constructor
    * \note Copy construction is a shallow copy.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   vector_auto(const vector_auto<T>& x) :
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
   vector_auto<T>& operator=(const vector_auto<T>& x)
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
   vector_auto() :
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
   void init(const int n)
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " set size to " << n << " for ";
      debug_trailer(std::cerr);
#endif
      if (n == Base::length)
         return;
      free();
      allocate(n);
      }
   // @}

   /*! \name Conversion to/from equivalent host objects */
   /*! \brief Copy from standard vector
    *
    * This method re-allocates memory, taking ownership, if necessary
    */
   vector_auto<T>& operator=(const libbase::vector<T>& x)
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " copy from host object " << &x << std::endl;
#endif
      // (re-)allocate memory if needed
      init(x.size().length());
      Base::operator=(x);
      return *this;
      }
   //! copy to standard vector
   operator libbase::vector<T>() const
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " copy to host object" << std::endl;
#endif
      return Base::operator libbase::vector<T>();
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
