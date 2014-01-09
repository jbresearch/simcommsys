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

#ifndef __awfilter_h
#define __awfilter_h

#include "filter.h"
#include "rvstatistics.h"

namespace libimage {

/*
 * \brief   Adaptive Wiener Filter
 * \author  Johann Briffa
 *
 * This filter implements Lee's algorithm (Lee, 1980), as used in Matlab's
 * wiener2 filter function (in the image processing library).
 * Note that Matlab provides a way for the function to estimate the noise
 * variance itself - this is actually computed as the mean value of the image
 * local variance. This class allows this to be done by using the appropriate
 * constructor. The estimator function is also publicly available.
 */

template <class T>
class awfilter : public filter<T> {
protected:
   // user-supplied settings
   int m_d; //!< greatest distance from current pixel in neighbourhood
   bool m_autoestimate; //!< flag for automatic estimation of noise energy
   double m_noise; //!< estimate of noise energy to remove
   // internal variables
   libbase::rvstatistics rvglobal;
public:
   awfilter(const int d, const double noise)
      {
      init(d, noise);
      }
   awfilter(const int d)
      {
      init(d);
      }
   // initialization
   void init(const int d, const double noise);
   void init(const int d);
   // progress display
   void display_progress(const int done, const int total) const
      {
      }
   // parameter estimation (updates internal statistics)
   void reset();
   void update(const libbase::matrix<T>& in);
   void estimate();
   double get_estimate() const
      {
      return m_noise;
      }
   // filter process loop (only updates output matrix)
   void process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const;
};

} // end namespace

#endif
