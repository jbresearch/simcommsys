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

#ifndef __histogram_nd_flat_h
#define __histogram_nd_flat_h

#include "config.h"
#include "vector.h"

namespace libbase {

/*!
 * \brief   Histogram of multi-variate sequence.
 * \author  Johann Briffa
 *
 * Computes the N-dimensional histogram of a vector of N-D coordinates with
 * fixed bin widths along each dimension. For each dimension, the range of
 * values and number of bins need to be explicitly stated by the user.
 *
 * \tparam dims The number of dimensions to use
 */

class histogram_nd_flat {
private:
   /*! \name User-defined parameters */
   double min; //!< The left edge of the first bin in each dimension
   double max; //!< The right edge of the last bin in each dimension
   int bins; //!< The number of bins in each dimension
   // @}
   /*! \name Internal state */
   vector<int> count; //!< The occurrence count for each bin as a flat array
   int N; //! The total number of occurrences over all bins
   // @}

private:
   double get_step() const
      {
      return (max - min) / double(bins);
      }

public:
   //! Principal constructor
   histogram_nd_flat(const vector<vector<double> >& a, const int dims,
         const double min, const double max, const int bins);
   //! Returns the absolute frequency count for each bin
   const vector<int>& get_frequency()
      {
      return count;
      }
   //! Returns the relative frequency (empirical probability) for each bin
   const vector<double> get_probability()
      {
      vector<double> result(count);
      result /= double(N);
      return result;
      }
   const vector<double> get_bin_edges();
   const vector<double> get_bin_centres();
};

} // end namespace

#endif

