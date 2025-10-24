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

#include "fidelity_pos.h"
#include "experiment/binomial/commsys_simulator.h"

namespace libcomm
{

void
fidelity_pos::init(const queryable& system)
{
    symbolsperframe = std::any_cast<int>(
        system.get_value(commsys_simulator_base::SYMBOLS_PER_FRAME));
}

/*!
 * \brief Update result set
 * \param[out] result   Vector containing the set of results to be updated
 * \param[in]  act_drift  Actual drift positions
 * \param[in]  est_drift  Estimated drift positions
 *
 * Results are organized as (symbol,frame) error count, repeated for
 * every iteration that needs to be performed. Eventually these will be
 * divided by the respective multiplicity to get the average error rates.
 */
void
fidelity_pos::compute_result_and_accumulate(
    libbase::vector<double>& accumulated_result,
    libbase::vector<uint64_t>& accumulated_count,
    const libbase::vector<int>& act_drift,
    const libbase::vector<int>& est_drift) const
{
    const int N = result_count();
    assert(accumulated_result.size() == N);
    assert(act_drift.size() == N);
    assert(est_drift.size() == N);
    // Accumulate fidelity errors
    for (int t = 0; t < N; t++) {
        accumulated_result(t) += (act_drift(t) == est_drift(t)) ? 1 : 0;
        accumulated_count(t) += 1;
    }
}

// Serialisation interface

const libbase::serializer fidelity_pos::shelper("results_collector",
                                                "fidelity_pos",
                                                fidelity_pos::create);

std::ostream&
fidelity_pos::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
fidelity_pos::serialize(std::istream& sin)
{
    return sin;
}

} // namespace libcomm
