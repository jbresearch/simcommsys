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

#ifndef __grscc_h
#define __grscc_h

#include "ccfsm.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Generalized Recursive Systematic Convolutional Code.
 * \author  Johann Briffa
 *
 * Implements RSCC where polynomial coefficients are elements of a finite
 * field, which is specified as a template parameter.
 */

template <class G>
class grscc : public ccfsm<G> {
private:
   /*! \name Object representation */
   libbase::matrix<int> csct; //!< Circulation state correspondence table
   // @}
   /*! \name Internal functions */
   libbase::matrix<G> getstategen() const;
   // TODO: Separate circulation state stuff from this class
   // (not all RSC codes are suitable)
   void initcsct();
   // @}
protected:
   /*! \name FSM helper operations */
   libbase::vector<int> determineinput(const libbase::vector<int>& input) const;
   libbase::vector<G> determinefeedin(const libbase::vector<int>& input) const;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   grscc()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   grscc(const libbase::matrix<libbase::vector<G> >& generator) :
      ccfsm<G> (generator)
      {
      initcsct();
      }
   // @}

   // FSM state operations (getting and resetting)
   void resetcircular(const libbase::vector<int>& zerostate, int n);

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(grscc)
};

} // end namespace

#endif

