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

#include "errors_levenshtein.h"
#include "hamming.h"
#include "levenshtein.h"

namespace libcomm
{

/*!
 * \copydoc results_collector::compute_result_and_accumulate()
 *
 * Results are organized as (symbol_hamming, symbol_levenshtein,frame)
 * error count, repeated for every iteration that needs to be performed.
 * Eventually these will be divided by the respective count to get the
 * average error rates.
 */
void
errors_levenshtein::compute_result_and_accumulate(
    libbase::vector<double>& accumulated_result,
    libbase::vector<uint64_t>& accumulated_count,
    const libbase::vector<int>& source,
    const libbase::vector<int>& decoded) const
{
    // Count errors
    const int hd = libbase::hamming(source, decoded);
    const int ld = libbase::levenshtein(source, decoded);
    // Estimate the SER, LD, FER
    accumulated_result(0) += hd;
    accumulated_count(0) += symbolsperblock;
    accumulated_result(1) += ld;
    accumulated_count(1) += symbolsperblock;
    accumulated_result(2) += hd ? 1 : 0;
    accumulated_count(2) += 1;
}

// Serialisation interface

const libbase::serializer errors_levenshtein::shelper(
    "results_collector", "errors_levenshtein", errors_levenshtein::create);

std::ostream&
errors_levenshtein::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
errors_levenshtein::serialize(std::istream& sin)
{
    return sin;
}

} // namespace libcomm
