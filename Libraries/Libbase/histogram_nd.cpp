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

#include "histogram_nd.h"
#include "itfunc.h"

namespace libbase {

template <int dims>
histogram_nd<dims>::histogram_nd(const vector<vector<double> >& a,
      const double min, const double max, const int bins)
   {
   // sanity checks
   assert(max > min);
   assert(bins > 0);
   // initialize representation
   this->min = min;
   this->max = max;
   this->bins = bins;
   // initialize bin array
   boost::array < typename array_ni::index, dims > my_extents;
   for (int j = 0; j < dims; j++)
      my_extents[j] = bins;
   count.resize(my_extents);
   count = 0;
   // compute the histogram_nd
   const double step = get_step();
   for (int i = 0; i < a.size(); i++)
      {
      assert(a(i).size() == dims);
      boost::array < typename array_ni::index, dims > idx;
      for (int j = 0; j < dims; j++)
         idx[j] = limit<int>(int(floor((a(i)(j) - min) / step)), 0, bins - 1);
      count(idx)++;
      }
   }

template <int dims>
const vector<double> histogram_nd<dims>::get_bin_edges()
   {
   const double step = get_step();
   vector<double> edges(bins + 1);
   for (int i = 0; i <= bins; i++)
      edges(i) = min + i * step;
   return edges;
   }

template <int dims>
const vector<double> histogram_nd<dims>::get_bin_centres()
   {
   const double step = get_step();
   vector<double> centres(bins);
   for (int i = 0; i < bins; i++)
      centres(i) = min + i * step + step / 2;
   return centres;
   }

} // end namespace
