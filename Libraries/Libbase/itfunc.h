#ifndef __itfunc_h
#define __itfunc_h

#include "config.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string>

/*!
 * \file
 * \brief   Information Theory Functions.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 */

namespace libbase {

/*! \name Gamma functions */
//! \sa Numerical Recipes in C, p.216-219
double gammln(double xx);
double gammser(double a, double x);
double gammcf(double a, double x);
double gammp(double a, double x);
// @}

/*! \name Error function */
double cerf(double x);
double cerfc(double x);
/*! \brief Error function based on incomplete Gamma functions
 * More accurate but much slower than Chebychev fitting
 */
inline double erff(double x)
   {
   return x < 0.0 ? -gammp(0.5, x * x) : gammp(0.5, x * x);
   }
/*! \brief Complementary rrror function based on incomplete Gamma functions
 * More accurate but much slower than Chebychev fitting
 */
inline double erffc(double x)
   {
   return x < 0.0 ? 1.0 + gammp(0.5, x * x) : 1.0 - gammp(0.5, x * x);
   }
// @}

inline double Q(double x)
   {
   return 0.5 * erffc(x / sqrt(2.0));
   }
inline double gauss(double x)
   {
   return exp(-0.5 * x * x) / sqrt(2.0 * PI);
   }

//! Limits numbers between a high and low limit
template <class T>
inline T limit(const T x, const T lo, const T hi)
   {
   return std::max(lo, std::min(hi, x));
   }
;

int weight(int cw);

/*! \brief Binary Hamming weight
 */
template <class T>
int weight(const matrix<T>& m)
   {
   matrix<int> t;
   t = m;
   t.apply(weight);
   return t.sum();
   }

//! Gray code
inline int32u gray(int32u n)
   {
   return n ^ (n >> 1);
   }
;
int32u igray(int32u n);

int gcd(int a, int b);

int factorial(int x);
int permutations(int n, int r);
inline int combinations(int n, int r)
   {
   return permutations(n, r) / factorial(r);
   }

double factoriald(int x);
double permutationsd(int n, int r);
inline double combinationsd(int n, int r)
   {
   return permutationsd(n, r) / factoriald(r);
   }

std::string hexify(const std::string input);
std::string dehexify(const std::string input);

} // end namespace

#endif
