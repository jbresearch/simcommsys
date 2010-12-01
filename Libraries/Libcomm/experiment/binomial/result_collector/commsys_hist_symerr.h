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

#ifndef __commsys_hist_symerr_h
#define __commsys_hist_symerr_h

#include "config.h"
#include "commsys_errorrates.h"

namespace libcomm {

/*!
 * \brief   CommSys Results - Symbol-Error per Frame Histogram.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Computes histogram of symbol error count for each block simulated.
 */

class commsys_hist_symerr : public commsys_errorrates {
public:
   // Public interface
   void updateresults(libbase::vector<double>& result, const int i,
         const libbase::vector<int>& source,
         const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
    * For each iteration, we count the frequency of each possible
    * symbol-error count, including zero
    */
   int count() const
      {
      return (get_symbolsperblock() + 1) * get_iter();
      }
   /*! \copydoc experiment::get_multiplicity()
    * Only one result can be incremented for every frame.
    */
   int get_multiplicity(int i) const
      {
      return 1;
      }
   std::string result_description(int i) const;
};

} // end namespace

#endif
