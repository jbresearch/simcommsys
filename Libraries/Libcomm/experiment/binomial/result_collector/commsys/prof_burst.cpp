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

#include "prof_burst.h"
#include "fsm.h"
#include <sstream>

namespace libcomm {

// commsys functions

void prof_burst::updateresults(libbase::vector<double>& result,
      const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   assert(source.size() == get_symbolsperblock());
   assert(decoded.size() == get_symbolsperblock());
   // Update the relevant count for every symbol in error
   // Check the first symbol first
   assert(source(0) != fsm::tail);
   if (source(0) != decoded(0))
      result(0)++;
   // For each remaining symbol
   for (int t = 1; t < get_symbolsperblock(); t++)
      {
      if (source(t - 1) != decoded(t - 1))
         result(3)++;
      assert(source(t) != fsm::tail);
      if (source(t) != decoded(t))
         {
         // Keep separate counts, depending on whether the previous symbol was in error
         if (source(t - 1) != decoded(t - 1))
            result(2)++;
         else
            result(1)++;
         }
      }
   }

} // end namespace
