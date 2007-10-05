#ifndef __config_h
#define __config_h

typedef unsigned char		int8u;
typedef unsigned short		int16u;
typedef unsigned long		int32u;
typedef unsigned long long	int64u;

typedef signed char		int8s;
typedef signed short		int16s;
typedef signed long			int32s;
typedef signed long long		int64s;

template <class T> inline T min(const T a, const T b) { return( a<b ? a : b); };
template <class T> inline T max(const T a, const T b) { return( a>b ? a : b); };

template <class T> inline void swap(T& a, T& b)
   {
   T temp = b;
   b = a;
   a = temp;
   }

// Machine- and compiler-dependent section

#ifndef __GNUC__
#define bool int
#define false 0
#define true 1
#endif //ifndef gnu

#ifdef sparc
extern "C" int getrusage(int who, struct rusage *usage);

#include <ieeefp.h>
inline int isinf(double value)
   {
   switch(fpclass(value))
      {
      case FP_NINF:
         return -1;
      case FP_PINF:
         return +1;
      default:
         return 0;
      }
   }
#endif // ifdef sparc

#endif
