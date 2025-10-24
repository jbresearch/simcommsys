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

#include "prof_burst.h"
#include "fsm.h"
#include <sstream>

namespace libcomm
{

// commsys functions

void
prof_burst::compute_result_and_accumulate(
    libbase::vector<double>& accumulated_result,
    libbase::vector<uint64_t>& accumulated_count,
    const libbase::vector<int>& source,
    const libbase::vector<int>& decoded) const
{
    assert(source.size() == symbolsperblock);
    assert(decoded.size() == symbolsperblock);
    // Update the relevant count for every symbol in error
    // Check the first symbol first
    assert(source(0) != fsm::tail);
    if (source(0) != decoded(0)) {
        accumulated_result(0)++;
    }
    // Accumulate count in the first frame symbol (at most 1/frame)
    accumulated_count(0)++;

    // For each remaining symbol
    for (int t = 1; t < symbolsperblock; t++) {
        // Symbol errors in the prior symbol (required when applying Bayes' rule
        // to the above two counts)
        if (source(t - 1) != decoded(t - 1)) {
            accumulated_result(3)++;
        }
        // Accumulate count (at most #symbols/frame - 1/frame)
        accumulated_count(3) += symbolsperblock - 1;

        assert(source(t) != fsm::tail);
        if (source(t) != decoded(t)) {
            // Keep separate counts for errors in subsequent symbols, depending
            // on whether the previous symbol was in error
            if (source(t - 1) != decoded(t - 1)) {
                accumulated_result(2)++;
            } else {
                accumulated_result(1)++;
            }
        // Accumulate count (at most #symbols/frame - 1/frame)
        accumulated_count(1) += symbolsperblock - 1;
        accumulated_count(2) += symbolsperblock - 1;
        }
    }
}

// Serialisation interface

const libbase::serializer
    prof_burst::shelper("results_collector", "prof_burst", prof_burst::create);

std::ostream&
prof_burst::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
prof_burst::serialize(std::istream& sin)
{
    return sin;
}

} // namespace libcomm
