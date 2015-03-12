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

#ifndef __histogram_h
#define __histogram_h

#include "config.h"
#include "vector.h"
#include "matrix.h"

namespace libbase {

/*!
 * \brief   Histogram.
 * \author  Johann Briffa
 *
 * Computes the histogram of the values in a vector with a fixed bin width.
 * The range of values and number of bins need to be explicitly stated by the
 * user.
 */

class histogram {
   double min_val; //!< The left edge of the first bin
   double max_val; //!< The right edge of the last bin
   int bins; //!< The number of bins
   vector<int> count; //!< The occurrence count for each bin
private:
   const double get_step()
      {
      return (max_val - min_val) / double(bins);
      }
public:
   //! Principal constructor
   histogram(const vector<double>& a, const double min_val,
         const double max_val, const int bins);
   const vector<int>& get_count()
      {
      return count;
      }
   const vector<double> get_bin_edges();
   const vector<double> get_bin_centres();
};

} // end namespace

#endif

