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

#include "experiment_normal.h"
#include <limits>

namespace libcomm {

// Normally-distributed sample experiment

void experiment_normal::derived_reset()
   {
   // Initialise running values only if space is allocated
   if (sum.size() > 0)
      sum = 0;
   if (sumsq.size() > 0)
      sumsq = 0;
   }

void experiment_normal::derived_accumulate(
      const libbase::vector<double>& result)
   {
   assert(result.size() > 0);
   // accumulate results
   libbase::vector<double> sample = result;
   safe_accumulate(sum, sample);
   sample.apply(square);
   safe_accumulate(sumsq, sample);
   }

void experiment_normal::accumulate_state(const libbase::vector<double>& state)
   {
   assert(state.size() > 0);
   // divide state into constituent components and accumulate
   const int n = state.size() / 2;
   assert(state.size() == 2 * n);
   safe_accumulate(sum, state.extract(0, n));
   safe_accumulate(sumsq, state.extract(n, n));
   }

void experiment_normal::get_state(libbase::vector<double>& state) const
   {
   assert(count() == sum.size());
   assert(count() == sumsq.size());
   state.init(2 * count());
   for (int i = 0; i < count(); i++)
      {
      state(i) = sum(i);
      state(count() + i) = sumsq(i);
      }
   }

void experiment_normal::estimate(libbase::vector<double>& estimate,
      libbase::vector<double>& stderror) const
   {
   assert(count() == sum.size());
   assert(count() == sumsq.size());
   // estimate is the mean value
   assert(get_samplecount() > 0);
   estimate = sum / double(get_samplecount());
   // standard error is sigma/sqrt(n)
   stderror.init(count());
   if (get_samplecount() > 1)
      for (int i = 0; i < count(); i++)
         stderror(i) = sqrt((sumsq(i) / double(get_samplecount()) - estimate(i)
               * estimate(i)) / double(get_samplecount() - 1));
   else
      stderror = std::numeric_limits<double>::max();
   }

} // end namespace
