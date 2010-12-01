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

#ifndef __commsys_prof_burst_h
#define __commsys_prof_burst_h

#include "config.h"
#include "commsys_errorrates.h"

namespace libcomm {

/*!
 * \brief   CommSys Results - Error Burstiness Profile.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Determines separately the error probabilities for:
 * the first symbol in a frame
 * a symbol following a correctly-decoded one
 * a symbol following an incorrectly-decoded one
 */

class commsys_prof_burst : public commsys_errorrates {
public:
   // Public interface
   void updateresults(libbase::vector<double>& result, const int i,
         const libbase::vector<int>& source,
         const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
    * For each iteration, we count respectively the number symbol errors:
    * - in the first frame symbol
    * - in subsequent symbols:
    * - if the prior symbol was correct (ie. joint probability)
    * - if the prior symbol was in error
    * - in the prior symbol (required when applying Bayes' rule
    * to the above two counts)
    */
   int count() const
      {
      return 4 * get_iter();
      }
   int get_multiplicity(int i) const;
   std::string result_description(int i) const;
};

} // end namespace

#endif
