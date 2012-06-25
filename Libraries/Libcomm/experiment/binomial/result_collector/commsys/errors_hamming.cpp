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

#include "errors_hamming.h"
#include "fsm.h"
#include "itfunc.h"
#include "hamming.h"
#include <sstream>

namespace libcomm {

/*!
 * \brief Update result set
 * \param[out] result   Vector containing the set of results to be updated
 * \param[in]  i        Iteration just performed
 * \param[in]  source   Source data sequence
 * \param[in]  decoded  Decoded data sequence
 * 
 * Results are organized as (symbol,frame) error count, repeated for
 * every iteration that needs to be performed. Eventually these will be
 * divided by the respective multiplicity to get the average error rates.
 */
void errors_hamming::updateresults(libbase::vector<double>& result,
      const int i, const libbase::vector<int>& source, const libbase::vector<
            int>& decoded) const
   {
   assert(i >= 0 && i < get_iter());
   // Count errors
   int symerrors = libbase::hamming(source, decoded);
   // Estimate the BER, SER, FER
   result(2 * i + 0) += symerrors;
   result(2 * i + 1) += symerrors ? 1 : 0;
   }

/*!
 * \copydoc experiment::get_multiplicity()
 * 
 * Since results are organized as (symbol,frame) error count, repeated for
 * every iteration, the multiplicity is respectively the number of symbols
 * and the number of frames (=1) per sample.
 */
int errors_hamming::get_multiplicity(int i) const
   {
   assert(i >= 0 && i < count());
   if (i % 2 == 0)
      return get_symbolsperblock();
   return 1;
   }

/*!
 * \copydoc experiment::result_description()
 * 
 * The description is a string XER_Y, where 'X' is S,F to indicate
 * symbol or frame error rates respectively. 'Y' is the iteration,
 * starting at 1.
 */
std::string errors_hamming::result_description(int i) const
   {
   assert(i >= 0 && i < count());
   std::ostringstream sout;
   if (i % 2 == 0)
      sout << "SER_";
   else
      sout << "FER_";
   sout << (i / 2) + 1;
   return sout.str();
   }

} // end namespace
