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
   \file
   \brief   Main Configuration.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

// *** Global namespace ***

// Dummy entries for version-control macros, currently useful only in Windows
// builds; UNIX builds get the values automatically determined on build.

#ifndef __WCURL__
#  define __WCURL__ "undefined"
#  define __WCVER__ "undefined"
#endif

// An assertion that is implemented even in release builds

#ifdef NDEBUG
#  define assertalways(_Expression) (void)( (!!(_Expression)) || (libbase::fail(#_Expression, __FILE__, __LINE__), 0) )
#else
#  define assertalways(_Expression) assert(_Expression)
#endif

// Implemented log2, round, and sgn if these are not already available

#ifdef WIN32
inline double log2(double x) { return log(x)/log(double(2)); }
inline double round(double x) { return (floor(x + 0.5)); }
#endif
inline double round(double x, double r) { return round(x/r)*r; }
inline double sign(double x) {return (x > 0) ? +1 : ((x < 0) ? -1 : 0); }

// Automatic upgrade of various math functions from int to double parameter

inline double sqrt(int x) { return sqrt(double(x)); };
inline double log(int x) { return log(double(x)); };
#ifdef WIN32
inline double pow(int x, int y) { return pow(double(x),y); };
#endif //ifdef WIN32

// Define a function that returns the square of the input

template <class T>
inline T square(const T x) { return x*x; };

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

// Debugging tools

void fail(const char *expression, const char *file, int line);
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

/*! \brief Interrupt-signal handling function
   This function is meant to catch Ctrl-C during execution, to be used in the
   same way as keypressed(), allowing pre-mature interruption of running MPI
   processes (which can't handle keypressed() events).
   
   \note The signal handler is set the first time that interrupted() is
         called; this means that the mechanism is not activated until the
         first time it is called, which generally works fine as this function
         is meant to be used within a loop as part of the condition statement.
*/
bool interrupted(void);

// System error message reporting
std::string getlasterror();

// Function to skip over whitespace
std::istream& eatwhite(std::istream& is);

// Stream data loading verification functions
bool isfailedload(std::istream &is);
bool isincompleteload(std::istream &is);
void verifycompleteload(std::istream& is);

}; // end namespace

#endif
