#ifndef __itfunc_h
#define __itfunc_h

#include "config.h"
#include "vcs.h"
#include <math.h>
#include <stdlib.h>
#include <string>

/*!
   \file    Information Theory Functions.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.01 (6 Mar 2002)
  changed use of iostream from global to std namespace.

  Version 1.02 (17 Mar 2002)
  added function for rounding numbers.

  Version 1.03 (21 Apr 2002)
  added function for rounding numbers to specified resolution.
  also added alternative error function routines based on Chebychev fitting to an
  inspired guess as to the functional form (cf. Numerical Recipes in C, p.221).

  Version 1.04 (12 Jun 2002)
  modified all functions so that parameters are non-const; since we are not passing
  the arguments by reference, this does not make any difference and allows us
  greater flexibility in using the functions.

  Version 1.05 (16 Nov 2002)
  added function for limiting numbers between a high and low limit.

  Version 1.06 (5 Jul 2003)
  added functions for converting between a string and its hex representation

  Version 1.07 (17 Jul 2006)
  changed int round(double x) to double round(double x) to conform with gcc's version.
  changed the loop variable type from int to size_t in hexify and dehexify, to avoid
  a gcc warning about comparisons between signed and unsigned types.

  Version 1.10 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libbase {

extern const vcs itfunc_version;

// gamma functions (cf. Numerical Recipes in C, p.216-219)
double gammln(double xx);
double gammser(double a, double x);
double gammcf(double a, double x);
double gammp(double a, double x);

/*
   error function (cf. Numerical Recipes in C, p.220)
   erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t^2) dt
   erfc(x) = 1-erf(x) = 2/sqrt(pi) * integral from x to inf of exp(-t^2) dt
*/
// based on Chebychev fitting - fractional error is everywhere less than 1.2E-7
double cerf(double x);
double cerfc(double x);
// based on incomplete Gamma functions - more accurate but much slower
inline double erff(double x) { return x < 0.0 ? -gammp(0.5,x*x) : gammp(0.5,x*x); }
inline double erffc(double x) { return x < 0.0 ? 1.0+gammp(0.5,x*x) : 1.0-gammp(0.5,x*x); }

inline double Q(double x) { return 0.5 * erffc(x / sqrt(2.0)); }
inline double gauss(double x) { return exp(-0.5 * x * x)/sqrt(2.0 * PI); }

inline double log2(double x) { return log(x)/log(double(2)); }
inline double round(double x) { return (floor(x + 0.5)); }
inline double round(double x, double r) { return round(x/r)*r; }

template <class T> inline T limit(const T x, const T lo, const T hi) { return max(lo, min(hi, x)); };

int weight(int cw);

int factorial(int x);
int permutations(int n, int r);
inline int combinations(int n, int r) { return permutations(n,r)/factorial(r); }

double factoriald(int x);
double permutationsd(int n, int r);
inline double combinationsd(int n, int r) { return permutationsd(n,r)/factoriald(r); }

// string <-> hex functions
std::string hexify(const std::string input);
std::string dehexify(const std::string input);

}; // end namespace

#endif
