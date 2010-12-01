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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __config_h
#define __config_h

/*!
 * \file
 * \brief   Main Configuration.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

// Enable secure function overload for CRT in Win32

#ifdef WIN32
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

#ifdef WIN32
#  define _SCL_SECURE_NO_WARNINGS
#endif

// Disable min/max macros

#ifdef WIN32
#  define NOMINMAX
#endif

// Disable dominance warning

//#ifdef WIN32
//#  pragma warning( disable : 4250 )
//#endif

// system include files

#include <cstdlib>
#include <iostream>
#include <string>
#include <math.h>
#ifdef WIN32
#  include <float.h>
#  include <basetsd.h>
#else
#  include <stdint.h>
#endif

// module include files

#include "assertalways.h"


// *** Global namespace ***

// Dummy entries for version-control macros, currently useful only in Windows
// builds; UNIX builds get the values automatically determined on build.

#ifndef __WCURL__
#  define __WCURL__ "undefined"
#  define __WCVER__ "undefined"
#endif

// Implemented log2, round, and sgn if these are not already available

#ifdef WIN32
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

inline double sqrt(int x)
   {
   return sqrt(double(x));
   }

inline double log(int x)
   {
   return log(double(x));
   }

#ifdef WIN32
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

#ifdef WIN32
typedef SSIZE_T ssize_t;
#endif

// Define math functions to identify NaN and Inf values

#ifdef WIN32
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
#endif //ifdef WIN32
// C99 Names for integer types

#ifdef WIN32
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif

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

// Stream data loading verification functions
bool isfailedload(std::istream &is);
bool isincompleteload(std::istream &is);
void verifycompleteload(std::istream& is);

// Check for alignment
inline bool isaligned(const void *buf, int bytes)
   {
   return ((long) buf & (bytes - 1)) == 0;
   }

} // end namespace

#endif
