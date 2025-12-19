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

#include "experiment_binomial.h"

namespace libcomm
{

// Experiment for estimation of a binomial proportion

void
experiment_binomial::derived_reset()
{
    // Initialise running values only if space is allocated
    if (sum.size() > 0) {
        sum = 0;
    }
    if (count.size() > 0) {
        count = 0;
    }
}

void
experiment_binomial::derived_accumulate_result(
    const libbase::vector<double>& sample_result,
    const libbase::vector<uint64_t>& sample_count)
{
    assert(sample_result.size() > 0);
    // accumulate results
    safe_accumulate(sum, sample_result);
    safe_accumulate(count, sample_count);
}

void
experiment_binomial::derived_accumulate_state(
    const libbase::vector<double>& state)
{
    assert(state.size() > 0);
    // accumulate results from saved state
    safe_accumulate(sum, state);
}

void
experiment_binomial::get_state(libbase::vector<double>& state) const
{
    assert(result_count() == sum.size());
    state = sum;
}

void
experiment_binomial::estimate(libbase::vector<double>& estimate,
                              libbase::vector<double>& stderror) const
{
    assert(result_count() == sum.size());
    // initialize space for results
    estimate.init(result_count());
    stderror.init(result_count());
    // compute results
    for (int i = 0; i < result_count(); i++) {
#ifdef DEBUG
        if (count(i) == 0) {
            std::cout << "experiment_binomial: count(" << i
                      << ") = " << count(i) << std::endl;
        }
#endif
        // estimate is the proportion
        estimate(i) = sum(i) / count(i);
        // standard error is sqrt(p(1-p)/n)
        stderror(i) = sqrt((estimate(i) * (1 - estimate(i))) / count(i));
    }
}

} // namespace libcomm
