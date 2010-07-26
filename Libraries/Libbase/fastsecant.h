#ifndef __fastsecant_h
#define __fastsecant_h

#include "config.h"
#include "secant.h"
#include "vector.h"

namespace libbase {

/*!
 * \brief   Semi-cached root-finding by Secant method.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Speeded-up version of the secant method module - we build a cache on seeding
 * which we then use to initialise the starting points for the algorithm.
 */

class fastsecant : public secant {
   vector<double> m_vdCache;
   double m_dMin, m_dMax, m_dStep;
public:
   fastsecant(double(*func)(double) = NULL);
   void init(const double x1, const double x2, const int n);
   double solve(const double y);
   double operator()(const double y)
      {
      return solve(y);
      }
};

} // end namespace

#endif

