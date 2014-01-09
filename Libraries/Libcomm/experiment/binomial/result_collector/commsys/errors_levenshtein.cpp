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
#include "fsm.h"
#include "itfunc.h"
#include "hamming.h"
#include "levenshtein.h"

namespace libcomm {

/*!
 * \brief Update result set
 * \param[out] result   Vector containing the set of results to be updated
 * \param[in]  source   Source data sequence
 * \param[in]  decoded  Decoded data sequence
 *
 * Results are organized as (symbol_hamming, symbol_levenshtein,frame)
 * error count, repeated for every iteration that needs to be performed.
 * Eventually these will be divided by the respective multiplicity to get the
 * average error rates.
 */
void errors_levenshtein::updateresults(libbase::vector<double>& result,
      const libbase::vector<int>& source, const libbase::vector<
            int>& decoded) const
   {
   // Count errors
   const int hd = libbase::hamming(source, decoded);
   const int ld = libbase::levenshtein(source, decoded);
   // Estimate the SER, LD, FER
   result(0) += hd;
   result(1) += ld;
   result(2) += hd ? 1 : 0;
   }

} // end namespace
