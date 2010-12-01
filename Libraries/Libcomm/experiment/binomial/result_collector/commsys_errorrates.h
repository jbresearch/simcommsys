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

#ifndef __commsys_errorrates_h
#define __commsys_errorrates_h

#include "config.h"
#include "vector.h"
#include <string>

namespace libcomm {

/*!
 * \brief   CommSys Results - Symbol/Frame Error Rates.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements standard error rate calculators.
 */
class commsys_errorrates {
protected:
   /*! \name System Interface */
   //! The number of decoding iterations performed
   virtual int get_iter() const = 0;
   //! The number of information symbols per block
   virtual int get_symbolsperblock() const = 0;
   //! The information symbol alphabet size
   virtual int get_alphabetsize() const = 0;
   // @}
   /*! \name Helper functions */
   int countbiterrors(const libbase::vector<int>& source,
         const libbase::vector<int>& decoded) const;
   int countsymerrors(const libbase::vector<int>& source,
         const libbase::vector<int>& decoded) const;
   // @}
public:
   virtual ~commsys_errorrates()
      {
      }
   /*! \name Public interface */
   void updateresults(libbase::vector<double>& result, const int i,
         const libbase::vector<int>& source,
         const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
    * For each iteration, we count the number of symbol and frame errors
    */
   int count() const
      {
      return 2 * get_iter();
      }
   int get_multiplicity(int i) const;
   std::string result_description(int i) const;
   // @}
};

} // end namespace

#endif
