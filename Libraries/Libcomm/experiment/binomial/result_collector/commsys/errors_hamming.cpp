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

#include "errors_hamming.h"
#include "experiment/binomial/commsys_simulator.h"
#include "hamming.h"

namespace libcomm
{

void
errors_hamming::init(const queryable& system)
{
    symbolsperblock = std::any_cast<int>(
        system.get_value(commsys_simulator_base::SYMBOLS_PER_BLOCK));
}

/*!
 * \copydoc results_collector::updateresults()
 * \param[out] result   Vector containing the set of results to be updated
 * \param[in]  source   Source data sequence
 * \param[in]  decoded  Decoded data sequence
 *
 * Results are organized as (symbol,frame) error count. Eventually these will be
 * divided by the respective multiplicity to get the average error rates.
 */
void
errors_hamming::updateresults(libbase::vector<double>& result,
                              const libbase::vector<int>& source,
                              const libbase::vector<int>& decoded) const
{
    assert(source.size() == symbolsperblock);
    assert(decoded.size() == symbolsperblock);
    // Count errors
    int symerrors = libbase::hamming(source, decoded);
    // Estimate the SER, FER
    result(0) += symerrors;
    result(1) += symerrors ? 1 : 0;
}

// Serialisation interface

const libbase::serializer errors_hamming::shelper("results_collector",
                                                  "errors_hamming",
                                                  errors_hamming::create);

std::ostream&
errors_hamming::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
errors_hamming::serialize(std::istream& sin)
{
    return sin;
}

} // namespace libcomm
