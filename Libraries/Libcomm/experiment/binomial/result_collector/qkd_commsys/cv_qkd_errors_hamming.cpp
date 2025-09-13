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

#include "cv_qkd_errors_hamming.h"
#include "fsm.h"
#include "hamming.h"
#include "itfunc.h"

namespace libcomm
{

/*!
 * \brief Update result set
 * \param[out] result   Vector containing the set of results to be updated
 * \param[in]  source   Source data sequence
 * \param[in]  decoded  Decoded data sequence
 *
 * Results are organized as (symbol,frame, SKR) error count. Eventually these will be
 * divided by the respective multiplicity to get the average error rates.
 */
void
cv_qkd_errors_hamming::updateresults(libbase::vector<double>& result, libbase::vector<gaussian_state> source,
    libbase::vector<bool> vector_s, libbase::vector<bool>& key_KA, libbase::vector<bool>& key_KB) const
{
    /* Results needed to calculate the Secret Key Rate (SKR) are results(0) and results(1).

    The SKR is calculated using: SKR = result(0)/result(1)

    TODO: Confirm with Johann where the division needs to happen.
    */

    result(0) += key_KA.size(); // Sum of Lengths of Key KA where the length will be the final length of the secret key after privacy amplification.

    result(1) += source.size(); // Sum of length of source which is the sequence of generated coherent states. This is equal to N because it is fixed. In qkd_commsys_simulator.cpp it is called the framesize.

    /* Results needed to calculate the SER: results(0) and results(2)

    SER = Sum of Hamming Differences of keys KA and KB / Sum of lengths of key KA.

    TODO: Again to confirm with Johann where the division needs to happen.
    */

    int symerrors = libbase::hamming(key_KA, key_KB);
    result(2) += symerrors;

    // FER
    result(3) += symerrors ? 1 : 0;
}

} // namespace libcomm
