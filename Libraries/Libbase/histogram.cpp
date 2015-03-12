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

#include "histogram.h"
#include "itfunc.h"

namespace libbase {

histogram::histogram(const vector<double>& a, const double min_val,
      const double max_val, const int bins)
   {
   // sanity checks
   assert(max_val > min_val);
   assert(bins > 0);
   // initialize representation
   this->min_val = min_val;
   this->max_val = max_val;
   this->bins = bins;
   count.init(bins);
   // compute the histogram
   const double step = get_step();
   for (int i = 0; i < a.size(); i++)
      {
      const int j = int(floor((a(i) - min_val) / step));
      count(limit<int>(j, 0, bins - 1))++;
      }
   }

const vector<double> histogram::get_bin_edges()
   {
   const double step = get_step();
   vector<double> edges(bins + 1);
   for (int i = 0; i <= bins; i++)
      edges(i) = min_val + i * step;
   return edges;
   }

const vector<double> histogram::get_bin_centres()
   {
   const double step = get_step();
   vector<double> centres(bins);
   for (int i = 0; i < bins; i++)
      centres(i) = min_val + i * step + step / 2;
   return centres;
   }

} // end namespace
