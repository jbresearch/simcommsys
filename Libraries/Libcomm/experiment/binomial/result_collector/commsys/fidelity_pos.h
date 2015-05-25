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

#ifndef __fidelity_pos_h
#define __fidelity_pos_h

#include "config.h"
#include "vector.h"
#include <string>
#include <sstream>

namespace libcomm {

/*!
 * \brief   CommSys Results - Codeword Boundary Fidelity.
 * \author  Johann Briffa
 *
 * Implements computation of the fidelity metric at frame and codeword
 * boundary positions.
 */
class fidelity_pos {
protected:
   /*! \name System Interface */
   //! The number of information symbols per frame (ie modem input)
   virtual int get_symbolsperframe() const = 0;
   // @}
public:
   virtual ~fidelity_pos()
      {
      }
   /*! \name Public interface */
   void updateresults(libbase::vector<double>& result, const libbase::vector<
         int>& act_drift, const libbase::vector<int>& est_drift) const;
   /*! \copydoc experiment::count()
    * For each iteration, we count the fidelity at codeword boundary positions.
    * \warning This assumes that the codec and modem output sizes are the same!
    */
   int count() const
      {
      return get_symbolsperframe() + 1;
      }
   /*! \copydoc experiment::get_multiplicity()
    * Only one result can be incremented for every position.
    */
   int get_multiplicity(int i) const
      {
      return 1;
      }
   /*! \copydoc experiment::result_description()
    *
    * The description is a string FID_X, where 'X' is the symbol position
    * (starting at zero), denoting the fidelity at the start of the
    * corresponding symbol.
    */
   std::string result_description(int i) const
      {
      assert(i >= 0 && i < count());
      std::ostringstream sout;
      sout << "FID_" << i;
      return sout.str();
      }
   // @}
};

} // end namespace

#endif
