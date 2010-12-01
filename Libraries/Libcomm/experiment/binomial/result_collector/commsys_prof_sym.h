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

#ifndef __commsys_prof_sym_h
#define __commsys_prof_sym_h

#include "config.h"
#include "commsys_errorrates.h"

namespace libcomm {

/*!
 * \brief   CommSys Results - Symbol-Value Error Profile.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Computes symbol-error histogram as dependent on source symbol value.
 */

class commsys_prof_sym : public commsys_errorrates {
public:
   // Public interface
   void updateresults(libbase::vector<double>& result, const int i,
         const libbase::vector<int>& source,
         const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
    * For each iteration, we count the number of symbol errors for
    * every input alphabet symbol value.
    */
   int count() const
      {
      return get_alphabetsize() * get_iter();
      }
   /*! \copydoc experiment::get_multiplicity()
    * A total equal to the number of symbols/frame may be incremented
    * in every sample.
    */
   int get_multiplicity(int i) const
      {
      return get_symbolsperblock();
      }
   std::string result_description(int i) const;
};

} // end namespace

#endif
