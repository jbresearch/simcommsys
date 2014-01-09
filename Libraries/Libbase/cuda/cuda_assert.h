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

#ifndef __cuda_assert_h
#define __cuda_assert_h

#include "config.h"

#include <cassert>
#include <cstdio>

// *** Global namespace ***

// An assertion that is implemented even in release builds

#define cuda_assertalways(_Expression) (void)( (!!(_Expression)) || (cuda::reportassertionandfail(#_Expression, __FILE__, __LINE__), 0) )

// An assertion that is valid for host and gpu code

#ifndef NDEBUG // Debug build
#  ifndef __CUDA_ARCH__ // Host code
#    define cuda_assert(_Expression) assert(_Expression)
#  elif __CUDA_ARCH__ < 200 // pre-Fermi
#    define cuda_assert(_Expression) (void)0
#  else // Fermi
#    define cuda_assert(_Expression) assert(_Expression)
#  endif
#else // Release build
#  define cuda_assert(_Expression) (void)0
#endif

// Fail with error

#define cuda_failwith(_String) cuda::reporterrorandfail(_String, __FILE__, __LINE__)

// *** Within library namespace ***

namespace cuda {

// Debugging tools

#ifdef __CUDACC__
__device__ __host__
#endif
inline void reportassertionandfail(const char *expression, const char *file, int line)
   {
#ifndef __CUDA_ARCH__
   libbase::reportassertionandfail(expression, file, line);
#elif __CUDA_ARCH__ >= 200
   printf("CUDA ERROR (%s line %d): assertion %s failed.\n", file, line, expression);
   //TODO: we really want to stop everything here, but there is no function to do so.
   assert(0);
#endif
   }

#ifdef __CUDACC__
__device__ __host__
#endif
inline void reporterrorandfail(const char *expression, const char *file, int line)
   {
#ifndef __CUDA_ARCH__
   libbase::reporterrorandfail(expression, file, line);
#elif __CUDA_ARCH__ >= 200
   printf("CUDA ERROR (%s line %d): %s\n", file, line, expression);
   //TODO: we really want to stop everything here, but there is no function to do so.
   assert(0);
#endif
   }

} // end namespace

#endif
