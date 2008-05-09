#ifndef __fastsecant_h
#define __fastsecant_h

#include "config.h"
#include "secant.h"
#include "vector.h"

namespace libbase {

/*!
   \brief   Semi-cached root-finding by Secant method.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (30 Nov 2001)
  speeded-up version of the secant method module - we build a cache on seeding
  which we then use to initialise the starting points for the algorithm.

   \version 1.01 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.

   \version 1.02 (12 Jun 2002)
  modified the definition of 'f' so that its parameter is non-const, in conformance
  with secant 1.02.

   \version 1.10 (26 Oct 2006)
   - defined class and associated data within "libbase" namespace.
*/

class fastsecant : public secant {
   vector<double> m_vdCache;
   double   m_dMin, m_dMax, m_dStep;
public:
   fastsecant(double (*func)(double) = NULL);
   void init(const double x1, const double x2, const int n);
   double solve(const double y);
   double operator()(const double y) { return solve(y); };
};

}; // end namespace

#endif

