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

#ifndef __cuda_value_h
#define __cuda_value_h

#include "config.h"
#include "cuda-all.h"

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
class value_reference;

/*!
 * \brief   A single object in device memory
 * \author  Johann Briffa
 *
 * This class represents a 'single object in device memory'. It consists of two
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
class value {
private:
   // Class friends
   friend class value_reference<T> ;

protected:
   /*! \name Object representation */
   T* data __attribute__((aligned(8))); //!< Pointer to allocated memory in global device space
   // @}

protected:
   /*! \name Test and debug functions */
   //! Outputs a standard debug header, identifying object type and address
   std::ostream& debug_header(std::ostream& sout) const
      {
      sout << "DEBUG (cuda::value<" << typeid(T).name() << "> at " << this
            << "):";
      return sout;
      }
   //! Outputs a standard debug trailer, identifying object contents
   std::ostream& debug_trailer(std::ostream& sout) const
      {
      if (data == NULL)
         sout << "empty value" << std::endl;
      else
         sout << " element (size " << sizeof(T) << ") at " << data << std::endl;
      return sout;
      }
   // @}

   /*! \name Data setting functions */
   //! shallow copy from an equivalent object
#ifdef __CUDACC__
   __device__ __host__
#endif
   void copyfrom(const value<T>& x)
      {
      data = x.data;
      }
   //! reset to a null value
#ifdef __CUDACC__
   __device__ __host__
#endif
   void reset()
      {
      data = NULL;
      }
   // @}

   /*! \name Memory allocation functions */
   //! allocate requested number of elements
   void allocate();
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
   value() :
      data(NULL)
      {
      }
   // @}

   /*! \name Law of the Big Three */
   //! Destructor
#ifdef __CUDACC__
   __device__ __host__
#endif
   ~value()
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
   value(const value<T>& x);
   /*! \brief Copy assignment operator
    * \note Copy assignment on a host is a deep copy.
    * \note Copy assignment on a device is a shallow copy.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   value<T>& operator=(const value<T>& x);
   // @}

   /*! \name Memory operations */
   /*! \brief Initialize allocation
    *
    * This method leaves the object as it is if the memory was already
    * allocated, and allocates if necessary.
    */
   void init()
      {
      if (data != NULL)
         return;
      allocate();
      }
   /*! \brief Set device memory to the given byte value
    *
    * This method assumes the device object has been allocated.
    */
   void fill(const unsigned char value)
      {
      cudaSafeMemset(data, value, 1);
      }
   //! Returns the address in device memory where the data is held
   const T* get_address() const
      {
      return data;
      }
   // @}

   /*! \name Conversion to/from equivalent host objects */
   //! copy from standard value
   value<T>& operator=(const T& x);
   //! copy to standard value
   operator T() const;
   // @}

   // Methods for device code only
#ifdef __CUDACC__
   /*! \name Element access */
   /*! \brief Index operator (write-access)
    * \note Does not perform boundary checking.
    */
   __device__
   T& operator()()
      {
      return *data;
      }
   /*! \brief Index operator (read-only access)
    * \note Does not performs boundary checking.
    */
   __device__
   const T& operator()() const
      {
      return *data;
      }
   // @}
#endif
};

#ifdef __CUDACC__
template <class T>
inline void value<T>::allocate()
   {
   // only allocate on an empty matrix
   assert(data == NULL);
   // if there is something to allocate, do it
   data = cudaSafeMalloc<T>(1);
   }

template <class T>
inline void value<T>::free()
   {
   // if there is something allocated, free it
   if (data != NULL)
      {
      // free device memory
      cudaSafeFree(data);
      // reset variables
      reset();
      }
   }

template <class T>
inline value<T>::value(const value<T>& x) :
data(NULL)
   {
#ifdef __CUDA_ARCH__ // Device code path (for all compute capabilities)
   copyfrom(x);
#else // Host code path
   if (x.data)
      {
      // allocate memory
      allocate();
      // copy data from device to device
      cudaSafeMemcpy(data, x.data, 1, cudaMemcpyDeviceToDevice);
      }
#endif
   }

template <class T>
inline value<T>& value<T>::operator=(const value<T>& x)
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
      init();
      // copy data from device to device
      cudaSafeMemcpy(data, x.data, 1, cudaMemcpyDeviceToDevice);
      }
   return *this;
#endif
   }

template <class T>
inline value<T>& value<T>::operator=(const T& x)
   {
   // (re-)allocate memory if needed
   init();

   // copy data from host to device if necessary
   if (data != NULL)
      {
      cudaSafeMemcpy(data, &x, 1, cudaMemcpyHostToDevice);
      }

   return *this;
   }

template <class T>
inline value<T>::operator T() const
   {
   T x;

   // copy data from device to host if necessary
   if (data != NULL)
      {
      cudaSafeMemcpy(&x, data, 1, cudaMemcpyDeviceToHost);
      }

   return x;
   }
#endif

/*!
 * \brief   A reference to a single object in device memory.
 * \author  Johann Briffa
 *
 * A value reference is a value that does not own its allocated memory.
 * Consequently, all operations that require memory operations are forbidden.
 * The data set is really just a reference to a regular value.
 * When an indirect value is destroyed, the actual allocated memory is not
 * released. This only happens when the referenced value is destroyed.
 * There is always a risk that the referenced value is destroyed before
 * the indirect references, in which case those references become stale.
 *
 * It is intended that for the user, the use of value references should be
 * essentially transparent (in that they can mostly be used in place of a
 * normal value). There is only one scenario where the user needs to create
 * one explicitly: when passing as an argument to a kernel, since these do
 * not take reference arguments in the usual way. Otherwise, creation should
 * happen only through a normal value's methods.
 */
template <class T>
class value_reference : public value<T> {
private:
   // Class friends
   friend class value<T> ;
   // Shorthand for class hierarchy
   typedef value<T> Base;
protected:
   /*! \name Test and debug functions */
   //! Outputs a standard debug header, identifying object type and address
   std::ostream& debug_header(std::ostream& sout) const
      {
      sout << "DEBUG (cuda::value_reference<" << typeid(T).name() << "> at "
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
   /*! \brief Initialize allocation
    *
    * This method is disabled in value references.
    */
   void init()
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
   value_reference()
      {
      }
   /*! \brief Automatic conversion from normal value
    * \warning This allows modification of 'const' values
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   value_reference(const value<T>& x)
      {
      // do not invoke the base constructor, to avoid a deep copy
      // note: this operation requires this class to be a friend of value
      Base::copyfrom(x);
      }
   // @}
   /*! \brief Assignment from normal value
    * \note Assignment is a shallow copy.
    * \warning This allows modification of 'const' values
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   value_reference<T>& operator=(const value<T>& x)
      {
      Base::copyfrom(x);
      return *this;
      }

   /*! \name Law of the Big Three */
   //! Destructor
#ifdef __CUDACC__
   __device__ __host__
#endif
   ~value_reference()
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
   value_reference(const value_reference<T>& x)
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
   value_reference<T>& operator=(const value_reference<T>& x)
      {
      Base::copyfrom(x);
      return *this;
      }
   // @}

   /*! \name Conversion to/from equivalent host objects */
   //! copy from standard value
   value_reference<T>& operator=(const T& x);
   //! copy to standard value
   operator T() const
      {
      return Base::operator T();
      }
   // @}
};

#ifdef __CUDACC__

template <class T>
inline value_reference<T>& value_reference<T>::operator=(const T& x)
   {
   assert(Base::data != NULL);
   // copy data from host to device
   cudaSafeMemcpy(Base::data, &x, 1, cudaMemcpyHostToDevice);
   return *this;
   }

#endif

/*!
 * \brief   A single object in device memory - automatic
 * \author  Johann Briffa
 *
 * The first automatic object creates an actual value on the device.
 * Copies of this object (through copy construction) create shallow copies
 * (references to the same memory) on the device.
 */
template <class T>
class value_auto : public value<T> {
private:
   // Shorthand for class hierarchy
   typedef value<T> Base;
private:
   bool isowner __attribute__((aligned(8)));
protected:
   /*! \name Test and debug functions */
   //! Outputs a standard debug header, identifying object type and address
   std::ostream& debug_header(std::ostream& sout) const
      {
      sout << "DEBUG (cuda::value_auto<" << typeid(T).name() << "> at " << this
            << "):";
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
   void allocate()
      {
      // this should only be called on an owned object
      assert(isowner);
      Base::allocate();
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
   ~value_auto()
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
   value_auto(const value_auto<T>& x) :
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
   value_auto<T>& operator=(const value_auto<T>& x)
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
   value_auto() :
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
   /*! \brief Initialize allocation (host only)
    *
    * This method leaves the object as it is if the memory was already
    * allocated, and allocates if necessary.
    *
    * If re-allocation is necessary:
    * - the old memory is only freed if this object is not a reference.
    * - the object becomes the owner of the newly allocated memory, even if it
    *   was only a reference to the old memory.
    */
   void init()
      {
#if DEBUG>=2
      debug_header(std::cerr);
      std::cerr << " initialize for ";
      debug_trailer(std::cerr);
#endif
      if (Base::data != NULL)
         return;
      allocate();
      }
   // @}

   /*! \name Conversion to/from equivalent host objects */
   /*! \brief Copy from standard value
    *
    * This method re-allocates memory, taking ownership, if necessary
    */
   value_auto<T>& operator=(const T& x)
      {
      // (re-)allocate memory if needed
      init();
      Base::operator=(x);
      return *this;
      }
   //! copy to standard value
   operator T() const
      {
      return Base::operator T();
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
