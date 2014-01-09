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

#ifndef __gnrcc_h
#define __gnrcc_h

#include "ccfsm.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Generalized Non-Recursive Convolutional Code.
 * \author  Johann Briffa
 *
 * \version 1.00 (13 Dec 2007)
 * - Initial version; implements NRCC where polynomial coefficients are elements
 * of a finite field.
 * - Derived from gnrcc 1.00 and nrcc 1.70
 * - The finite field is specified as a template parameter.
 *
 * \version 1.01 (4 Jan 2008)
 * - removed serialization functions, which were redundant
 * - removed resetcircular(), which is now implemented in fsm()
 */

template <class G>
class gnrcc : public ccfsm<G> {
protected:
   /*! \name FSM helper operations */
   libbase::vector<int> determineinput(const libbase::vector<int>& input) const;
   libbase::vector<G> determinefeedin(const libbase::vector<int>& input) const;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   gnrcc()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   gnrcc(const libbase::matrix<libbase::vector<G> >& generator) :
      ccfsm<G> (generator)
      {
      }
   // @}

   // FSM state operations (getting and resetting)
   void resetcircular(const libbase::vector<int>& zerostate, int n);

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(gnrcc)
};

} // end namespace

#endif

