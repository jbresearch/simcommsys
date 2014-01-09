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

#ifndef __hist_symerr_h
#define __hist_symerr_h

#include "config.h"
#include "errors_hamming.h"
#include <sstream>

namespace libcomm {

/*!
 * \brief   CommSys Results - Symbol-Error per Frame Histogram.
 * \author  Johann Briffa
 *
 * Computes histogram of symbol error count for each block simulated.
 */

class hist_symerr : public errors_hamming {
public:
   // Public interface
   void updateresults(libbase::vector<double>& result, const libbase::vector<
         int>& source, const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
    * We count the frequency of each possible symbol-error count, including zero
    */
   int count() const
      {
      return get_symbolsperblock() + 1;
      }
   /*! \copydoc experiment::get_multiplicity()
    * Only one result can be incremented for every frame.
    */
   int get_multiplicity(int i) const
      {
      return 1;
      }
   /*! \copydoc experiment::result_description()
    *
    * The description is a string ER_X, where 'X' is the symbol-error
    * count (starting at zero).
    */
   std::string result_description(int i) const
      {
      assert(i >= 0 && i < count());
      std::ostringstream sout;
      sout << "ER_" << i;
      return sout.str();
      }
};

} // end namespace

#endif
