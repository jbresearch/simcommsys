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

#include "prof_pos.h"
#include "fsm.h"
#include <sstream>

namespace libcomm {

// commsys functions

void prof_pos::updateresults(libbase::vector<double>& result,
      const int i, const libbase::vector<int>& source, const libbase::vector<
            int>& decoded) const
   {
   assert(i >= 0 && i < get_iter());
   const int skip = count() / get_iter();
   // Update the count for every symbol in error
   assert(source.size() == get_symbolsperblock());
   assert(decoded.size() == get_symbolsperblock());
   for (int t = 0; t < get_symbolsperblock(); t++)
      {
      assert(source(t) != fsm::tail);
      if (source(t) != decoded(t))
         result(skip * i + t)++;
      }
   }

/*!
 * \copydoc experiment::result_description()
 * 
 * The description is a string SER_X_Y, where 'X' is the symbol position
 * (starting at zero), and 'Y' is the iteration, starting at 1.
 */
std::string prof_pos::result_description(int i) const
   {
   assert(i >= 0 && i < count());
   std::ostringstream sout;
   const int x = i % get_symbolsperblock();
   const int y = (i / get_symbolsperblock()) + 1;
   sout << "SER_" << x << "_" << y;
   return sout.str();
   }

} // end namespace
