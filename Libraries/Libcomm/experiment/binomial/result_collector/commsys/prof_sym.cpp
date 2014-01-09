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

#include "prof_sym.h"
#include "fsm.h"
#include <sstream>

namespace libcomm {

// commsys functions

void prof_sym::updateresults(libbase::vector<double>& result,
      const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   // Update the count for every bit in error
   assert(source.size() == get_symbolsperblock());
   assert(decoded.size() == get_symbolsperblock());
   for (int t = 0; t < get_symbolsperblock(); t++)
      {
      assert(source(t) != fsm::tail);
      if (source(t) != decoded(t))
         result(source(t))++;
      }
   }

} // end namespace
