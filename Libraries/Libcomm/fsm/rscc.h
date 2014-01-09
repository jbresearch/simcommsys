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

#ifndef __rscc_h
#define __rscc_h

#include "ccbfsm.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Recursive Systematic Convolutional Coder.
 * \author  Johann Briffa
 */

class rscc : public ccbfsm {
protected:
   libbase::vector<int> determineinput(const libbase::vector<int>& input) const;
   libbase::bitfield determinefeedin(const libbase::vector<int>& input) const;
   rscc()
      {
      }
public:
   /*! \name Constructors / Destructors */
   rscc(libbase::matrix<libbase::bitfield> const &generator) :
      ccbfsm(generator)
      {
      }
   // @}

   // FSM state operations (getting and resetting)
   void resetcircular(const libbase::vector<int>& zerostate, int n);

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(rscc)
};

} // end namespace

#endif

