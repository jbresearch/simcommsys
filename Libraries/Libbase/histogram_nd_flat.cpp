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

#include "histogram_nd_flat.h"
#include "itfunc.h"

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show memory allocation requirements
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 2
#endif

histogram_nd_flat::histogram_nd_flat(const vector<vector<double> >& a,
      const int dims, const double min, const double max, const int bins)
   {
   // sanity checks
   assert(max > min);
   assert(bins > 0);
   // initialize representation
   this->min = min;
   this->max = max;
   this->bins = bins;
   // initialize bin array
   int size = 1;
   for (int j = 0; j < dims; j++)
      size *= bins;
#if DEBUG>=2
   std::cerr << "DEBUG (histogram_nd_flat): memory required = "
         << (size * sizeof(int) >> 20) << "MiB" << std::endl;
#endif
   count.init(size);
   count = 0;
   N = 0;
   // compute the histogram_nd_flat
   const double step = get_step();
   for (int i = 0; i < a.size(); i++)
      {
      assert(a(i).size() == dims);
      int idx = 0;
      for (int j = 0; j < dims; j++)
         {
         const int idx_j = int(floor((a(i)(j) - min) / step));
         idx = idx * bins + limit<int>(idx_j, 0, bins - 1);
         }
      count(idx)++;
      N++;
      }
   }

const vector<double> histogram_nd_flat::get_bin_edges()
   {
   const double step = get_step();
   vector<double> edges(bins + 1);
   for (int i = 0; i <= bins; i++)
      edges(i) = min + i * step;
   return edges;
   }

const vector<double> histogram_nd_flat::get_bin_centres()
   {
   const double step = get_step();
   vector<double> centres(bins);
   for (int i = 0; i < bins; i++)
      centres(i) = min + i * step + step / 2;
   return centres;
   }

} // end namespace
