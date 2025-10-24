/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela.
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

#include "dv_qkd_errors_hamming.h"
#include "experiment/binomial/qkd_commsys_simulator.h"
#include "fsm.h"
#include "hamming.h"
#include "itfunc.h"

namespace libcomm
{

void
dv_qkd_errors_hamming::init(const queryable& system)
{
    source_length = std::any_cast<int>(
        system.get_value(qkd_commsys_simulator_base::SOURCE_LENGTH));

    secret_key_length = std::any_cast<int>(
        system.get_value(qkd_commsys_simulator_base::SECRET_KEY_LENGTH));

    // alphabet_size = std::any_cast<int>(
    //     system.get_value(qkd_commsys_simulator_base::ALPHABET_SIZE));
}

/*!
* \copydoc results_collector::compute_result_and_accumulate()
 *
 * Results are organized as (SKR,symbol,frame) error count. Eventually these
 * will be divided by the respective counts to get the average error
 * rates.
 */
void
dv_qkd_errors_hamming::compute_result_and_accumulate(libbase::vector<double>& accumulated_result,
                                     libbase::vector<uint64_t>& accumulated_count,
                                     libbase::vector<qubit> source,
                                     libbase::vector<bool>& key_KA,
                                     libbase::vector<bool>& key_KB) const
{
    assert(source.size() == source_length);
    assert(key_KA.size() == secret_key_length);
    accumulated_result(0) += key_KA.size(); // SKR = sum(len(KA)) / sum(len(source))
    accumulated_count(0) += source_length;
    // Count errors
    int symerrors = libbase::hamming(key_KA, key_KB);
    accumulated_result(1) += symerrors; // SER = sum(hamming(KA,KB)) / sum(len(KA))
    accumulated_count(1) += secret_key_length;
    accumulated_result(2) +=
        symerrors ? 1 : 0; // FER = sum(hamming(KA,KB)>0) / sum(samples)
    accumulated_count(2) += 1;
}

// Serialisation interface

const libbase::serializer dv_qkd_errors_hamming::shelper("results_collector",
                                                  "dv_qkd_errors_hamming",
                                                  dv_qkd_errors_hamming::create);

std::ostream&
dv_qkd_errors_hamming::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
dv_qkd_errors_hamming::serialize(std::istream& sin)
{
    return sin;
}


} // namespace libcomm
