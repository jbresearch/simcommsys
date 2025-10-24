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

#include "prof_pos.h"
#include "fsm.h"
#include <sstream>

namespace libcomm
{

// commsys functions

void
prof_pos::compute_result_and_accumulate(
    libbase::vector<double>& accumulated_result,
    libbase::vector<uint64_t>& accumulated_count,
    const libbase::vector<int>& source,
    const libbase::vector<int>& decoded) const
{
    // Update the count for every symbol in error
    assert(source.size() == symbolsperblock);
    assert(decoded.size() == symbolsperblock);
    for (int t = 0; t < symbolsperblock; t++) {
        assert(source(t) != fsm::tail);
        if (source(t) != decoded(t)) {
            accumulated_result(t)++;
        }
    }
    // Update the cumulative maximum count
    accumulated_count += 1;
}

// Serialisation interface

const libbase::serializer
    prof_pos::shelper("results_collector", "prof_pos", prof_pos::create);

std::ostream&
prof_pos::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
prof_pos::serialize(std::istream& sin)
{
    return sin;
}

} // namespace libcomm
