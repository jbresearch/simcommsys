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
#include "fsm.h"
#include "itfunc.h"
#include "hamming.h"

namespace libcomm {

/*!
 * \brief Update result set
 * \param[out] result   Vector containing the set of results to be updated
 * \param[in]  source   Source data sequence
 * \param[in]  decoded  Decoded data sequence
 *
 * Results are organized as (symbol,frame) error count. Eventually these will be
 * divided by the respective multiplicity to get the average error rates.
 */
void errors_hamming::updateresults(libbase::vector<double>& result,
      const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   // Count errors
   int symerrors = libbase::hamming(source, decoded);
   // Estimate the BER, SER, FER
   result(0) += symerrors;
   result(1) += symerrors ? 1 : 0;
   }

} // end namespace
