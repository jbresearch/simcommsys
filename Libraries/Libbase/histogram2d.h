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

#ifndef __histogram2d_h
#define __histogram2d_h

#include "config.h"
#include "vector.h"
#include "matrix.h"

namespace libbase {

/*!
 * \brief   Histogram of bivariate sequence.
 * \author  Johann Briffa
 *
 * Computes the 2D histogram of a vector of coordinate pairs with fixed bin
 * widths along each dimension. For each dimension, the range of values and
 * number of bins need to be explicitly stated by the user.
 */

class histogram2d {
   /*! \name User-defined parameters */
   double min_x; //!< The left edge of the first bin in first dimension
   double max_x; //!< The right edge of the last bin in first dimension
   int bins_x; //!< The number of bins in first dimension
   double min_y; //!< The left edge of the first bin in second dimension
   double max_y; //!< The right edge of the last bin in second dimension
   int bins_y; //!< The number of bins in second dimension
   // @}
   /*! \name Internal state */
   matrix<int> count; //!< The occurrence count for each bin
   int N; //! The total number of occurrences over all bins
   // @}

private:
   double get_step_x() const
      {
      return (max_x - min_x) / double(bins_x);
      }
   double get_step_y() const
      {
      return (max_y - min_y) / double(bins_y);
      }

public:
   //! Principal constructor
   histogram2d(const vector<vector<double> >& a, const double min_x,
         const double max_x, const int bins_x, const double min_y,
         const double max_y, const int bins_y);
   //! Returns the absolute frequency count for each bin
   const matrix<int>& get_frequency()
      {
      return count;
      }
   //! Returns the relative frequency (empirical probability) for each bin
   const matrix<double> get_probability()
      {
      matrix<double> result(count);
      result /= double(N);
      return result;
      }
   const vector<double> get_bin_edges_x();
   const vector<double> get_bin_centres_x();
   const vector<double> get_bin_edges_y();
   const vector<double> get_bin_centres_y();
};

} // end namespace

#endif

