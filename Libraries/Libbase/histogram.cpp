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

template <class number>
histogram<number>::histogram(const vector<number>& a, const number min_val,
      const number max_val, const int bins)
   {
   // sanity checks
   assert(max_val > min_val);
   assert(bins > 0);
   // initialize representation
   this->min_val = min_val;
   this->max_val = max_val;
   this->bins = bins;
   count.init(bins);
   count = 0;
   N = 0;
   // compute the histogram
   const number step = get_step();
   for (int i = 0; i < a.size(); i++)
      {
      const int j = int(floor((a(i) - min_val) / step));
      count(limit<int>(j, 0, bins - 1))++;
      N++;
      }
   }

template <class number>
const vector<number> histogram<number>::get_bin_edges()
   {
   const number step = get_step();
   vector<number> edges(bins + 1);
   for (int i = 0; i <= bins; i++)
      edges(i) = min_val + i * step;
   return edges;
   }

template <class number>
const vector<number> histogram<number>::get_bin_centres()
   {
   const number step = get_step();
   vector<number> centres(bins);
   for (int i = 0; i < bins; i++)
      centres(i) = min_val + i * step + step / 2;
   return centres;
   }

} // end namespace

namespace libbase {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>

#define NUMBER_TYPE_SEQ \
   (int)(float)(double)

#define INSTANTIATE(r, x, type) \
      template class histogram<type>;

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, NUMBER_TYPE_SEQ)

} // end namespace
