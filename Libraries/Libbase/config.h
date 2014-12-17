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

#ifndef __config_h
#define __config_h

/*!
 * \file
 * \brief   Main Configuration.
 * \author  Johann Briffa
 */

// Global compilation settings / options (pre-deployment only)

// Uncoment to include definitions and testing of 128-bit integer types
//#define USE_128BIT_INT

// Enable secure function overload for CRT in Win32

#ifdef _WIN32
#  if defined(_CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES)
#     undef _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES
#     undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#     undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_COUNT
#     undef _CRT_SECURE_NO_DEPRECATE
#  endif
#  define _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES 1
#  define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#  define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_COUNT 1
#  define _CRT_SECURE_NO_DEPRECATE 1
#endif

// Disable checked-iterator warning

#ifdef _WIN32
#  define _SCL_SECURE_NO_WARNINGS
#endif

// Disable min/max macros

#ifdef _WIN32
#  define NOMINMAX
#endif

// Disable specific warnings

#ifdef _WIN32
// TODO: consider each of these and decide whether to remove the pragma and fix the code
//#  pragma warning( disable : 4250 ) // dominance warning
#  pragma warning( disable : 4800 ) // forcing int to bool
#  pragma warning( disable : 4804 ) // '>=': unsafe use of type 'bool' in operation
#  pragma warning( disable : 4244 ) // 'initializing' : conversion from 'std::streamsize' to 'const int', possible loss of data
#  pragma warning( disable : 4267 ) // 'initializing' : conversion from 'size_t' to 'const int', possible loss of data	
#  pragma warning( disable : 4090 ) // 'initializing' : different '__unaligned' qualifiers

#endif

// system include files - all architectures

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <stdint.h>

// system include files - specific architectures

#ifdef _WIN32
#  include <basetsd.h>
#endif

// module include files

#include "assertalways.h"


// *** Global namespace ***

// Implemented log2, round, and sgn if these are not already available

#ifdef _WIN32
inline double log2(double x)
   {
   return log(x)/log(double(2));
   }
inline double round(double x)
   {
   return (floor(x + 0.5));
   }
#endif
inline double round(double x, double r)
   {
   return round(x / r) * r;
   }
inline double sign(double x)
   {
   return (x > 0) ? +1 : ((x < 0) ? -1 : 0);
   }

// Automatic upgrading of various math functions with int parameter

//inline double sqrt(int x)
//   {
//   return sqrt(double(x));
//   }

#ifdef _WIN32
inline double log(int x)
   {
   return log(double(x));
   }

inline double pow(int x, int y)
   {
   return pow(double(x), y);
   }
#endif

// Define a function that returns the square of the input

template <class T>
inline T square(const T x)
   {
   return x * x;
   }

// Define signed size type

#ifdef _WIN32
typedef SSIZE_T ssize_t;
#endif

// Define math functions to identify NaN and Inf values

#ifdef _WIN32
inline int isnan(double value)
   {
   return _isnan(value);
   }

inline int isinf(double value)
   {
   switch(_fpclass(value))
      {
      case _FPCLASS_NINF:
      return -1;
      case _FPCLASS_PINF:
      return +1;
      default:
      return 0;
      }
   }
#endif //ifdef _WIN32

// C99 Names for integer types - only on Windows prior to MSVC++ 10.0 (VS 2010)
#if defined(_WIN32) && (_MSC_VER < 1600)
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif

// Non-standard 128-bit integer types

#if defined(USE_128BIT_INT)
#if defined(_WIN32)
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;
#else
typedef __int128_t int128_t;
typedef __uint128_t uint128_t;
#endif
#endif

// *** Within standard library namespace ***

namespace std {

// Define math functions to identify NaN and Inf values

#ifdef _WIN32
inline bool isfinite(double value)
   {
   switch(_fpclass(value))
      {
      case _FPCLASS_SNAN:
      case _FPCLASS_QNAN:
      case _FPCLASS_NINF:
      case _FPCLASS_PINF:
      return false;
      default:
      return true;
      }
   }
#endif //ifdef _WIN32

//! Operator to concatenate STL vectors
template <class T>
void operator+=(std::vector<T>& a, const std::vector<T>& b)
   {
   a.insert(a.end(), b.begin(), b.end());
   }

} // end namespace

// *** Within library namespace ***

namespace libbase {

// Debugging tools

extern std::ostream trace;

// Names for integer types

typedef uint8_t int8u;
typedef uint16_t int16u;
typedef uint32_t int32u;
typedef uint64_t int64u;
typedef int8_t int8s;
typedef int16_t int16s;
typedef int32_t int32s;
typedef int64_t int64s;

#if defined(USE_128BIT_INT)
typedef uint128_t int128u;
typedef int128_t int128s;
#endif

// Names for floating-point types

typedef float float32;
typedef double float64;
typedef long double float80;
//typedef __float128 float128;

// Friendly names for high-precision floating-point types

typedef float80 extended;
//typedef float128 quadruple;

// Constants

extern const double PI;
extern const char DIR_SEPARATOR;
extern const int ALIGNMENT;

// Interactive keyboard handling
int keypressed(void);
int readkey(void);

// Interrupt-signal handling function
bool interrupted(void);

// System error message reporting
std::string getlasterror();

// Functions to skip over whitespace and comments
std::istream& eatwhite(std::istream& is);
std::istream& eatcomments(std::istream& is);

// Exception class for stream load errors
class load_error : public std::runtime_error {
public:
   explicit load_error(const std::string& what_arg) :
      std::runtime_error(what_arg)
      {
      }
};

// Stream data loading verification functions
void check_failedload(std::istream &is);
void check_incompleteload(std::istream &is);
std::istream& verify(std::istream& is);
std::istream& verifycomplete(std::istream& is);

// Check for alignment
inline bool isaligned(const void *buf, int bytes)
   {
   return ((long) buf & (bytes - 1)) == 0;
   }

} // end namespace

#endif
