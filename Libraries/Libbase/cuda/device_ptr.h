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

#ifndef __cuda_device_ptr_h
#define __cuda_device_ptr_h

#include "cuda/util.h"

namespace cuda
{

#ifdef __CUDACC__

/*! \brief Initializes a given pointer by calling constructor T(args...)
 *
 * This kernel is needed because __device__ instances are allocated with
 * cudaMalloc() which does not initialize them using a constructor. This is
 * problematic especially when the allocated class contains virtual methods,
 * as virtual table is not initialized.
 *
 * This kernel initializes a temporary automatic object using constructor
 * T(args...), and then copies this object byte by byte into an object allocated
 * with cudaMalloc. Virtual table should be copied correctly and so should any
 * fields which do not contain pointers.
 */
template <class T, class... Args>
__global__ void
init_ptr_kern(T* ptr, Args... args)
{
    // initialize an automatic object
    T tmp(args...);

    // Copy contents of automatic object into the ptr, byte by byte
    // This will copy the virtual table initialized in the automatic object,
    // and should also work correctly with fields
    // provided tmp does not contain any pointers
    for (int i = 0; i < sizeof(T); i++)
        ((char*)ptr)[i] = ((char*)&tmp)[i];
}

#endif

/*!
 * \brief   A smart pointer for device memory
 * \author  Mark Mizzi
 *
 * This class represents a single device variable, accessed through a pointer.
 */
template <class T>
class device_ptr
{
private:
    T* ptr;

public:
    /*! \brief Constructor that allocates memory for device instance of \p T and
     * initializes it with constructor T(args...).
     */
    template <class... Args>
#ifdef __CUDACC__
    __host__
#endif
    device_ptr(Args... args)
    {
#ifdef __CUDACC__
        ptr = cudaSafeMalloc<T>(static_cast<size_t>(1));
        init_ptr_kern<<<1, 1>>>(get(), args...);
#endif
    }

public:
#ifdef __CUDACC__
    __host__
#endif
    ~device_ptr()
    {
#ifdef __CUDACC__
        cudaSafeFree(ptr);
#endif
    }

    // delete copy constructors as device_ptr owns its pointer;
    // copying would result in double-frees, and so on.
    // Compare to unique_ptr.
    device_ptr(const device_ptr&) = delete;
    device_ptr& operator=(const device_ptr&) = delete;

    // ensure move constructors are defined
    // Compare to unique_ptr; device_ptr also owns its pointer,
    // so it can only be moved not copied.
    device_ptr(device_ptr&&) = default;
    device_ptr& operator=(device_ptr&&) = default;

#ifdef __CUDACC__
    __host__
#endif
    void to_host(T* val)
    {
#ifdef __CUDACC__
        cudaSafeMemcpy(
            val, this->ptr, static_cast<size_t>(1), cudaMemcpyDeviceToHost);
#endif
    }

#ifdef __CUDACC__
    __device__
#endif
    T& operator*() { return *this->ptr; }

#ifdef __CUDACC__
    __device__
#endif
    T* operator->() { return this->ptr; }

#ifdef __CUDACC__
    __host__
    __device__
#endif
    T* get() { return this->ptr; }
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG
#endif

} // namespace cuda

#endif // __cuda_device_ptr_h