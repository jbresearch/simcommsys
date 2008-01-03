#ifndef __config_h
#define __config_h

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

// include files

#include <assert.h>
#include <iostream>
#include <string>
#include <math.h>
#ifdef WIN32
#  include <float.h>
#  include <basetsd.h>
#else
#  include <stdint.h>
#endif

/*!
   \file    Main Configuration.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 2.01 (11 Jun 2002)
   - removed the definition of swap() from this file for gcc (before was only removed for
   Win32), since this is defined in <algorithm>
   - also removed the inclusion of <ios> since this should not be necessary anyway.

   \version 2.02 (17 Jul 2006)
   - changed __int64 to long long for non-Win32 systems
   - removed definition of min/max for GCC systems

   \version 2.03 (18 Jul 2006)
   - added GCCONLY()

   \version 2.04 (25 Jul 2006)
   - added keypressed() and readkey(), with definitions for POSIX and WIN32

   \version 2.05 (28 Jul 2006)
   - changed the definition of the 32-bit integers to 'int' instead of 'long'; this
   should only matter on 64-bit systems, given that this code never runs on 16-bit
   machines anyway.

   \version 2.06 (9 Aug 2006)
   - added interrupted() function and related handlers - this is meant to catch Ctrl-C
   during execution, to be used in the same way as keypressed(), allowing pre-mature
   interruption of running MPI processes (which can't handle keypressed() events).
   - In reality, while this has been set to catch SIGINT, mpirun will not really
   propagate this signal to the node processes. Thus, to get this to work, the root
   process must be sent the INT signal directly - the root can be identified as being
   the one that is not nice'd on UNIX. The interrupt handler does not work on Win32
   anyway.
   - Note that the signal handler is set the first time that interrupted() is called -
   this means that the mechanism is not activated until the first time it is called,
   which generally works fine as this function is meant to be used within a loop as
   part of the condition statement.

   \version 2.07 (6 Oct 2006)
   - added definition to overload CRT functions with secure versions in Win32;
   this has been done for compatibility with VS .NET 2005, since the original
   functions are now deprecated.
   - removed definition of min/max template functions for Win32 platform on recent
   compilers.
   - renamed GCCONLY to STRICT, reflecting that this is applied only to compilers that
   follow the strict declaration of templated friend functions; definition has been
   modified so that it also applies to recent Win32 compilers.

   \version 2.08 (7 Oct 2006)
   - modified CRT settings to allow redefinition
   - renamed STRICT to TPLFRIEND to avoid name collision

   \version 3.00 (13 Oct 2006)
   - abandoned support for Sparc and Alpha architectures
   - abandoned support for Visual Studio 6 - among other things this allows us to ignore
      - scope problems for variables defined in for() loop initialization, which had
        to be solved in VS6 by creating a block around the for loop.
      - template friend definition problems (VS6 did not allow the required <> before
        the parameter list).
      - the "identifier truncated" warning shown in VS6
      - definition of max/min/swap inline template functions (already present)
      - definition of bool type
   - changed directory separator character from a macro to a const variable.
   - added definitions of various math functions (sqrt, log, pow) with integer parameters,
    so that they automatically upgrade to type double.

   \version 3.10 (26 Oct 2006)
   - defined class and associated data within "libbase" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 3.20 (9 Nov 2006)
   - re-inserted definition of max/min inline template functions in global namespace,
   undefining any macros with that name.

   \version 3.21 (18 Apr 2007)
   updated pow() function so that both parameters are called as double
   created new pow(double,int) function, forcing promotion to double,double

   \version 3.22 (8 May 2007)
   - converted pow(int,int) back to pow(double,int)
   - updated pow(double,int) upgrade to pow(double,double) to be active only in gcc,
    as .NET2005 already has this defined.
   - added typedef for ssize_t
   - TODO: change fixed integer types to use #include <inttypes.h>

   \version 3.23 (17 Jul 2007)
   - moved pow(double,int) upgrade to pow(double,double) to be before pow(int,int)
        upgrade to pow(double,int).
   - moved isinf() and isnan() back to global namespace.

   \version 3.30 (7 Nov 2007)
   - added pacifier() function, which returns a string according to an input
    percentage value.

   \version 3.31 (19 Nov 2007)
   - added error report when interrupt signal is caught.

   \version 3.32 (20 Nov 2007)
   - added getlasterror() function, for use in POSIX and Win32

   \version 3.33 (18 Dec 2007)
   - Modified integer type names to use compiler-specific versions in Win32 and
     C99 versions in anything else. Eventually, all references will be replaced
     by the C99 equivalent.

   \version 3.34 (3 Jan 2008)
   - moved log2() and round() from itfunc file.
   - defining these in global namespace, in order to use platform implementation
     when compiling with gcc. Also, these functions aren't really IT-related.
*/

// *** Global namespace ***

// Implemented log2 and round if these are not already available

#ifdef WIN32
inline double log2(double x) { return log(x)/log(double(2)); }
inline double round(double x) { return (floor(x + 0.5)); }
#endif
inline double round(double x, double r) { return round(x/r)*r; }

// Automatic upgrade of various math functions from int to double parameter

inline double sqrt(int x) { return sqrt(double(x)); };
inline double log(int x) { return log(double(x)); };
#ifndef WIN32
inline double pow(double x, int y) { return pow(x,double(y)); };
#endif //ifdef WIN32
inline double pow(int x, int y) { return pow(double(x),y); };

// Remove any macros for min/max and define as inline templates

#ifdef min
#  undef min
#  undef max
#endif
template <class T> inline T min(const T a, const T b) { return( a<b ? a : b); };
template <class T> inline T max(const T a, const T b) { return( a>b ? a : b); };

// Define signed size type

#ifdef WIN32
typedef SSIZE_T ssize_t;
#endif

// Define math functions to identify NaN and Inf values

#ifdef WIN32
inline int isnan(double value) { return _isnan(value); };

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
typedef __int8             int8_t;
typedef __int16            int16_t;
typedef __int32            int32_t;
typedef __int64            int64_t;
typedef unsigned __int8    uint8_t;
typedef unsigned __int16   uint16_t;
typedef unsigned __int32   uint32_t;
typedef unsigned __int64   uint64_t;
#endif


// *** Within library namespace ***

namespace libbase {

// Debugging tools (assert and trace; also includes standard streams)

extern std::ostream trace;

// Names for integer types

typedef uint8_t   int8u;
typedef uint16_t  int16u;
typedef uint32_t  int32u;
typedef uint64_t  int64u;
typedef int8_t    int8s;
typedef int16_t   int16s;
typedef int32_t   int32s;
typedef int64_t   int64s;

// Constants

extern const double PI;
extern const char DIR_SEPARATOR;

// Define interactive keyboard-handling functions
// Checks if a key has been pressed and returns true if this has happened.
int keypressed(void);
// Waits for the user to hit a key and returns its value.
// The user's response is not shown on screen.
int readkey(void);

// Interrupt-signal handling function
bool interrupted(void);

// Pacifier output
std::string pacifier(int percent);

// System error message reporting
std::string getlasterror();

}; // end namespace

#endif
