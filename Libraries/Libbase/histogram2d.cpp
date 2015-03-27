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

#include "histogram2d.h"
#include "itfunc.h"

namespace libbase {

histogram2d::histogram2d(const vector<vector<double> >& a, const double min_x,
      const double max_x, const int bins_x, const double min_y,
      const double max_y, const int bins_y)
   {
   // sanity checks
   assert(max_x > min_x);
   assert(bins_x > 0);
   assert(max_y > min_y);
   assert(bins_y > 0);
   // initialize representation
   this->min_x = min_x;
   this->max_x = max_x;
   this->bins_x = bins_x;
   this->min_y = min_y;
   this->max_y = max_y;
   this->bins_y = bins_y;
   count.init(bins_x, bins_y);
   count = 0;
   N = 0;
   // compute the histogram2d
   const double step_x = get_step_x();
   const double step_y = get_step_y();
   for (int i = 0; i < a.size(); i++)
      {
      assert(a(i).size() == 2);
      const int x = int(floor((a(i)(0) - min_x) / step_x));
      const int y = int(floor((a(i)(1) - min_y) / step_y));
      count(limit<int>(x, 0, bins_x - 1), limit<int>(y, 0, bins_y - 1))++;
      N++;
      }
   }

const vector<double> histogram2d::get_bin_edges_x()
   {
   const double step = get_step_x();
   vector<double> edges(bins_x + 1);
   for (int i = 0; i <= bins_x; i++)
      edges(i) = min_x + i * step;
   return edges;
   }

const vector<double> histogram2d::get_bin_centres_x()
   {
   const double step = get_step_x();
   vector<double> centres(bins_x);
   for (int i = 0; i < bins_x; i++)
      centres(i) = min_x + i * step + step / 2;
   return centres;
   }

const vector<double> histogram2d::get_bin_edges_y()
   {
   const double step = get_step_y();
   vector<double> edges(bins_y + 1);
   for (int i = 0; i <= bins_y; i++)
      edges(i) = min_y + i * step;
   return edges;
   }

const vector<double> histogram2d::get_bin_centres_y()
   {
   const double step = get_step_y();
   vector<double> centres(bins_y);
   for (int i = 0; i < bins_y; i++)
      centres(i) = min_y + i * step + step / 2;
   return centres;
   }

} // end namespace
