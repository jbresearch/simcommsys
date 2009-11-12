/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "rvstatistics.h"
#include <cfloat>

namespace libbase {

void rvstatistics::reset()
   {
   m_n = 0;
   m_sum = m_sumsq = 0;
   m_hi = -DBL_MAX;
   m_lo = DBL_MAX;
   }

// insertion functions

void rvstatistics::insert(const double x)
   {
   m_n++;
   m_sum += x;
   m_sumsq += x * x;
   if (x > m_hi)
      m_hi = x;
   if (x < m_lo)
      m_lo = x;
   }

void rvstatistics::insert(const vector<double>& x)
   {
   for (int i = 0; i < x.size(); i++)
      insert(x(i));
   }

void rvstatistics::insert(const matrix<double>& x)
   {
   for (int i = 0; i < x.size().rows(); i++)
      for (int j = 0; j < x.size().cols(); j++)
         insert(x(i, j));
   }

} // end namespace
