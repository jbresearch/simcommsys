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

#include "hist_symerr.h"
#include "hamming.h"
#include <sstream>

namespace libcomm
{

// commsys functions

void
hist_symerr::compute_result_and_accumulate(
    libbase::vector<double>& accumulated_result,
    libbase::vector<uint64_t>& accumulated_count,
    const libbase::vector<int>& source,
    const libbase::vector<int>& decoded) const
{
    int symerrors = libbase::hamming(source, decoded);
    // Update the count for that number of symbol errors (may be zero)
    accumulated_result(symerrors)++;
    accumulated_count += 1;
}

// Serialisation interface

const libbase::serializer hist_symerr::shelper("results_collector",
                                               "hist_symerr",
                                               hist_symerr::create);

std::ostream&
hist_symerr::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
hist_symerr::serialize(std::istream& sin)
{
    return sin;
}

} // namespace libcomm
