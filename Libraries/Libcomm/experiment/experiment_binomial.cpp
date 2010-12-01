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
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "experiment_binomial.h"

namespace libcomm {

// Experiment for estimation of a binomial proportion

void experiment_binomial::derived_reset()
   {
   assert(count() > 0);
   // Initialise space for running values
   sum.init(count());
   // Initialise running values
   sum = 0;
   }

void experiment_binomial::derived_accumulate(
      const libbase::vector<double>& result)
   {
   assert(count() == result.size());
   assert(count() == sum.size());
   // accumulate results
   sum += result;
   }

void experiment_binomial::accumulate_state(const libbase::vector<double>& state)
   {
   assert(count() == sum.size());
   assert(count() == state.size());
   // accumulate results from saved state
   sum += state;
   }

void experiment_binomial::get_state(libbase::vector<double>& state) const
   {
   assert(count() == sum.size());
   state = sum;
   }

void experiment_binomial::estimate(libbase::vector<double>& estimate,
      libbase::vector<double>& stderror) const
   {
   assert(count() == sum.size());
   // initialize space for results
   estimate.init(count());
   stderror.init(count());
   // compute results
   assert(get_samplecount() > 0);
   for (int i = 0; i < count(); i++)
      {
      // estimate is the proportion
      estimate(i) = sum(i) / double(get_samplecount(i));
      // standard error is sqrt(p(1-p)/n)
      stderror(i) = sqrt((estimate(i) * (1 - estimate(i)))
            / double(get_samplecount(i)));
      }
   }

} // end namespace
