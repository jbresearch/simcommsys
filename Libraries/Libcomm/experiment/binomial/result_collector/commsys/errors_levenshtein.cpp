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
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "errors_levenshtein.h"
#include "fsm.h"
#include "itfunc.h"
#include "hamming.h"
#include "levenshtein.h"
#include <sstream>

namespace libcomm {

/*!
 * \brief Update result set
 * \param[out] result   Vector containing the set of results to be updated
 * \param[in]  i        Iteration just performed
 * \param[in]  source   Source data sequence
 * \param[in]  decoded  Decoded data sequence
 * 
 * Results are organized as (symbol_hamming, symbol_levenshtein,frame)
 * error count, repeated for every iteration that needs to be performed.
 * Eventually these will be divided by the respective multiplicity to get the
 * average error rates.
 */
void errors_levenshtein::updateresults(libbase::vector<double>& result,
      const int i, const libbase::vector<int>& source, const libbase::vector<
            int>& decoded) const
   {
   assert(i >= 0 && i < get_iter());
   // Count errors
   const int hd = libbase::hamming(source, decoded);
   const int ld = libbase::levenshtein(source, decoded);
   // Estimate the SER, LD, FER
   result(3 * i + 0) += hd;
   result(3 * i + 1) += ld;
   result(3 * i + 2) += hd ? 1 : 0;
   }

/*!
 * \copydoc experiment::get_multiplicity()
 * 
 * Since results are organized as (symbol_hamming, symbol_levenshtein,frame)
 * error count, repeated for every iteration, the multiplicity is respectively
 * the number of symbols (twice) and the number of frames (=1) per sample.
 *
 * \warning In the case of Levenshtein distance, it is not clear how the
 * multiplicity should be computed.
 */
int errors_levenshtein::get_multiplicity(int i) const
   {
   assert(i >= 0 && i < count());
   switch(i % 3)
      {
      case 0:
      case 1:
         return get_symbolsperblock();
      case 2:
         return 1;
      }
   return -1; // This should never happen
   }

/*!
 * \copydoc experiment::result_description()
 * 
 * The description is a string XX_Y, where 'XX' is SER,LD,FER to indicate
 * symbol error rate (Hamming distance), Levenshtein distance, or frame error
 * rate respectively. 'Y' is the iteration, starting at 1.
 */
std::string errors_levenshtein::result_description(int i) const
   {
   assert(i >= 0 && i < count());
   std::ostringstream sout;
   switch(i % 3)
      {
      case 0:
         sout << "SER_";
         break;
      case 1:
         sout << "LD_";
         break;
      case 2:
         sout << "FER_";
         break;
      }
   sout << (i / 3) + 1;
   return sout.str();
   }

} // end namespace
