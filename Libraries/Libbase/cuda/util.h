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

#ifndef __cuda_util_h
#define __cuda_util_h

#include "config.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <typeinfo>

namespace cuda {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of memory copies and sets
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// CUDA utility functions

int cudaGetCurrentDevice();
int cudaGetMultiprocessorCount(int device = -1);
int cudaGetMultiprocessorSize(int device = -1);
int cudaGetSharedMemPerBlock(int device = -1);
int cudaGetRegsPerBlock(int device = -1);
int cudaGetMaxThreadsPerBlock(int device = -1);
int cudaGetWarpSize(int device = -1);
double cudaGetClockRate(int device = -1);
std::string cudaGetDeviceName(int device = -1);
size_t cudaGetGlobalMem(int device = -1);
int cudaGetComputeCapability(int device = -1);

int cudaGetDeviceCount();
int cudaGetDriverVersion();
int cudaGetRuntimeVersion();
std::string cudaPrettyVersion(int version);

int cudaGetMaxGflopsDeviceId();
int cudaGetComputeModel();

void cudaInitialize(std::ostream& sout);
void cudaQueryDevices(std::ostream& sout);

size_t cudaGetSharedSize(const void* func);
size_t cudaGetLocalSize(const void* func);
int cudaGetNumRegsPerThread(const void* func);
int cudaGetMaxThreadsPerBlock(const void* func);

#ifdef __CUDACC__

// Disable printf() for devices of compute capability < 2.0
// [removed as it conflicts with stdio]
//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
//#endif

// error wrappers

#define cudaSafeCall(error) cuda::__cudaSafeCall(error, __FILE__, __LINE__)

inline void __cudaSafeCall(const cudaError_t error, const char *file,
      const int line)
   {
   if (error == cudaSuccess)
      return;
   std::cerr << "CUDA error in file <" << file << ">, line " << line << " : " << cudaGetErrorString(error) << "." << std::endl;
   cudaDeviceReset();
   exit(1);
   }

// wait for kernels in all streams to finish

#define cudaSafeDeviceSynchronize() cudaSafeCall(cudaDeviceSynchronize());

// wait for kernel in specified stream to finish

#define cudaSafeStreamSynchronize(stream) cudaSafeCall(cudaStreamSynchronize(stream));

// device functions for min/max and swap

template <class T>
__device__
inline const T& min(const T& a, const T&b)
   {
   return (a < b) ? a : b;
   }

template <class T>
__device__
inline const T& max(const T& a, const T&b)
   {
   return (a > b) ? a : b;
   }

template <class T>
__device__
inline void swap(T& a, T&b)
   {
   T tmp = a;
   a = b;
   b = tmp;
   }

// safe memory operations

inline std::string cudaGetDescription(enum cudaMemcpyKind kind)
   {
   if (kind == cudaMemcpyHostToDevice)
      {
      return "H-to-D";
      }
   else if (kind == cudaMemcpyDeviceToHost)
      {
      return "D-to-H";
      }
   else if (kind == cudaMemcpyDeviceToDevice)
      {
      return "D-to-D";
      }
   return "unrecognized";
   }

template <class T>
inline std::string getTypeInfo()
   {
   std::ostringstream sout;
   sout << "type " << typeid(T).name() << ", size " << sizeof(T);
   return sout.str();
   }

template <class T>
inline void cudaSafeMemcpy(T* dst, const T* src, size_t count, enum cudaMemcpyKind kind)
   {
#if DEBUG>=2
   std::cerr << "DEBUG (util): " << cudaGetDescription(kind) << " copy for " << count << " elements (" << getTypeInfo<T>() << ") from " << src << " to " << dst << std::endl;
#endif
   if(count > 0)
      {
      assert(dst != NULL);
      assert(src != NULL);
      cudaSafeCall(cudaMemcpy(dst, src, count * sizeof(T), kind));
      }
   }

template <class T>
inline void cudaSafeMemcpy2D(T* dst, size_t dpitch, const T* src, size_t spitch, size_t cols, size_t rows, enum cudaMemcpyKind kind)
   {
#if DEBUG>=2
   std::cerr << "DEBUG (util): " << cudaGetDescription(kind) << " copy for " << rows << "×" << cols << " elements (" << getTypeInfo<T>() << ") from " << src << " (pitch " << spitch << ") to " << dst << " (pitch " << dpitch << ")" << std::endl;
#endif
   assert((cols > 0 && rows > 0) || (cols == 0 && rows == 0));
   if(cols > 0 && rows > 0)
      {
      assert(dst != NULL);
      assert(dpitch >= cols * sizeof(T));
      assert(src != NULL);
      assert(spitch >= cols * sizeof(T));
      cudaSafeCall(cudaMemcpy2D(dst, dpitch, src, spitch, cols * sizeof(T), rows, kind));
      }
   }

template <class T>
inline void cudaSafeMemset(T *data, int value, size_t count)
   {
#if DEBUG>=2
   std::cerr << "DEBUG (util): memory set to " << value << " for " << count << " elements (" << getTypeInfo<T>() << ") at " << data << std::endl;
#endif
   assert(data != NULL);
   assert(count > 0);
   cudaSafeCall(cudaMemset(data, value, count * sizeof(T)));
   }

template <class T>
inline void cudaSafeMemset2D(T *data, size_t pitch, int value, size_t cols, size_t rows)
   {
#if DEBUG>=2
   std::cerr << "DEBUG (util): memory set to " << value << " for " << rows << "×" << cols << " elements (" << getTypeInfo<T>() << ") at " << data << " (pitch " << pitch << ")" << std::endl;
#endif
   assert(data != NULL);
   assert(rows > 0 && cols > 0);
   assert(pitch >= cols * sizeof(T));
   cudaSafeCall(cudaMemset2D(data, pitch, value, cols * sizeof(T), rows));
   }

template <class T>
inline T* cudaSafeMalloc(size_t count)
   {
   void *p;
   cudaSafeCall(cudaMalloc(&p, count * sizeof(T)));
#if DEBUG>=2
   std::cerr << "DEBUG (util): allocated " << count << " elements (" << getTypeInfo<T>() << ") at " << p << std::endl;
#endif
   return (T*) p;
   }

template <class T>
inline T* cudaSafeMalloc2D(size_t *pitch_ptr, size_t cols, size_t rows)
   {
   void *p;
   cudaSafeCall(cudaMallocPitch(&p, pitch_ptr, cols * sizeof(T), rows));
#if DEBUG>=2
   std::cerr << "DEBUG (util): allocated " << rows << "×" << cols << " elements (" << getTypeInfo<T>() << ") at " << p << " (pitch " << *pitch_ptr << ")" << std::endl;
#endif
   return (T*) p;
   }

template <class T>
inline void cudaSafeFree(T *data)
   {
#if DEBUG>=2
   std::cerr << "DEBUG (util): freeing data at " << data << std::endl;
#endif
   cudaSafeCall(cudaFree(data));
   }

template <class T>
inline T* cudaSafeGetSymbolAddress(const T& symbol)
   {
   void *p;
   cudaSafeCall(cudaGetSymbolAddress(&p, symbol));
#if DEBUG>=2
   std::cerr << "DEBUG (util): symbol is at " << p << std::endl;
#endif
   return (T*)p;
   }

template <class T>
inline size_t cudaSafeGetSymbolSize(const T& symbol)
   {
   size_t n;
   cudaSafeCall(cudaGetSymbolSize(&n, symbol));
#if DEBUG>=2
   std::cerr << "DEBUG (util): symbol has size " << n << std::endl;
#endif
   return n;
   }

inline std::ostream& operator<<(std::ostream& sout, const dim3& size)
   {
   sout << "[" << size.x;
   if (size.y > 1 || size.z > 1)
      sout << "×" << size.y;
   if (size.z > 1)
      sout << "×" << size.z;
   sout << "]";
   return sout;
   }

inline int count(const dim3& size)
   {
   return size.x * size.y * size.z;
   }

#endif

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
