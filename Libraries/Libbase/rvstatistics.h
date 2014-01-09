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

#ifndef __rvstatistics_h
#define __rvstatistics_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include <cmath>
#include <cfloat>

namespace libbase {

/*!
 * \brief   Random Variable Statistics.
 * \author  Johann Briffa
 *
 * A class which gathers statistics about a random variable
 */

class rvstatistics {
   int64u m_n;
   double m_sum, m_sumsq;
   double m_hi, m_lo;
public:
   rvstatistics()
      {
      reset();
      }
   virtual ~rvstatistics()
      {
      }

   // reset gathered statistics (to start accumulating for a new set of data)
   void reset()
      {
      m_n = 0;
      m_sum = m_sumsq = 0;
      m_hi = -DBL_MAX;
      m_lo = DBL_MAX;
      }
   // inserts a single value
   void insert(const double x)
      {
      m_n++;
      m_sum += x;
      m_sumsq += x * x;
      if (x > m_hi)
         m_hi = x;
      if (x < m_lo)
         m_lo = x;
      }
   // inserts all items in a vector
   void insert(const vector<double>& x)
      {
      for (int i = 0; i < x.size(); i++)
         insert(x(i));
      }
   // inserts all items in a matrix
   void insert(const matrix<double>& x)
      {
      for (int i = 0; i < x.size().rows(); i++)
         for (int j = 0; j < x.size().cols(); j++)
            insert(x(i, j));
      }

   int64u count() const
      {
      return m_n;
      }
   double hi() const
      {
      return m_hi;
      }
   double lo() const
      {
      return m_lo;
      }
   double sum() const
      {
      return m_sum;
      }
   double mean() const
      {
      return m_sum / double(m_n);
      }
   double var() const
      {
      double _mean = mean();
      double _var = m_sumsq / double(m_n) - _mean * _mean;
      // avoid negative values due to FP resolution - return zero instead.
      return (_var > 0) ? _var : 0;
      }
   double sigma() const
      {
      return sqrt(var());
      }
};

} // end namespace

#endif
