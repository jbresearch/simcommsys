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

#include "qkd_errors_hamming.h"
#include "experiment/binomial/qkd_commsys_simulator.h"
#include "fsm.h"
#include "hamming.h"
#include "itfunc.h"

namespace libcomm
{

void
qkd_errors_hamming::init(const queryable& system)
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
qkd_errors_hamming::compute_result_and_accumulate(
    libbase::vector<double>& accumulated_result,
    libbase::vector<uint64_t>& accumulated_count,
    const libbase::vector<bool>& key_KA,
    const libbase::vector<bool>& key_KB) const
{
    // keys could be both length 0 or L
    assert(key_KA.size() == key_KB.size());
    // SKR = sum(len(KA)) / sum(len(source))
    accumulated_result(0) += key_KA.size();
    accumulated_count(0) += source_length;
    // AKR = sum(len(KA)>0) / sum(samples)
    accumulated_result(1) += key_KA.size() ? 1 : 0;
    accumulated_count(1) += 1;
    // AKL = sum(len(KA)) / sum(samples)
    accumulated_result(2) += key_KA.size();
    accumulated_count(2) += 1;
    // Count errors, if we actually have a non-zero length key
    if (key_KA.size() == 0) {
        return;
    }
    assert(key_KA.size() == secret_key_length);
    const int symerrors = libbase::hamming(key_KA, key_KB);
    // SER = sum(hamming(KA,KB)) / sum(len(KA))
    accumulated_result(3) += symerrors;
    accumulated_count(3) += secret_key_length;
    // FER = sum(hamming(KA,KB)>0) / sum(samples)
    accumulated_result(4) += symerrors ? 1 : 0;
    accumulated_count(4) += 1;
}

// Serialisation interface

const libbase::serializer qkd_errors_hamming::shelper(
    "results_collector", "qkd_errors_hamming", qkd_errors_hamming::create);

std::ostream&
qkd_errors_hamming::serialize(std::ostream& sout) const
{
    return sout;
}

std::istream&
qkd_errors_hamming::serialize(std::istream& sin)
{
    return sin;
}

} // namespace libcomm
