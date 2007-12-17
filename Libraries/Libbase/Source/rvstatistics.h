#ifndef __rvstatistics_h
#define __rvstatistics_h

#include "config.h"
#include "vcs.h"
#include "vector.h"
#include "matrix.h"
#include <math.h>

namespace libbase {

/*!
   \brief   Random Variable Statistics.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.00
  a class which gathers statistics about a random variable

  Version 1.01 (17 Oct 2001)
  solved a bug where the variance returned negative due to
  floating-point resolution - now this returns zero instead.
  This used to affect the sigma() which had a domain error.

  Version 1.02 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.03 (6 Apr 2002)
  made the destructor virtual and inline; added a public reset() function which
  resets the statistics, in order to start afresh on a new set of data. The
  default constructor has been modified to make use of this function.

  Version 1.04 (23 Apr 2002)
  added insert() functions for vectors and matrices - these avoid the loops that
  are otherwise necessary.

  Version 1.80 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
*/

class rvstatistics {
   static const vcs version;
   double m_sum, m_sumsq;
   double m_hi, m_lo;
   int m_n;
public:
   rvstatistics() { reset(); };
   virtual ~rvstatistics() {};

   void reset();
   void insert(const double x);
   void insert(const vector<double>& x);
   void insert(const matrix<double>& x);

   int count() const { return m_n; };
   double hi() const { return m_hi; };
   double lo() const { return m_lo; };
   double mean() const { return m_sum/double(m_n); };
   double var() const;
   double sigma() const { return sqrt(var()); };
};

// inline functions

inline double rvstatistics::var() const
   {
   double _mean = mean();
   double _var = m_sumsq/double(m_n) - _mean*_mean;
   return (_var > 0) ? _var : 0;
   }

}; // end namespace

#endif
