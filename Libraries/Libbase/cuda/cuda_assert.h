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
#  ifdef __CUDA_ARCH__ // GPU code
#    define cuda_assert(_Expression) (void)( (!!(_Expression)) || (cuda::reportassertionandfail(#_Expression, __FILE__, __LINE__), 0) )
#  else // Host code
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
   //exit(1);
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
   //exit(1);
#endif
   }

} // end namespace

#endif
