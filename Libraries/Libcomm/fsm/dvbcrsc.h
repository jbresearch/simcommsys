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

#ifndef __dvbcrsc_h
#define __dvbcrsc_h

#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   DVB-Standard Circular Recursive Systematic Convolutional Coder.
 * \author  Johann Briffa
 *
 */

class dvbcrsc : public fsm {
   /*! \name Object representation */
   static const int csct[7][8]; //!< Circulation state correspondence table
   static const int k, n; //!< Number of input and output bits, respectively
   static const int nu; //!< Number of memory elements (constraint length)
   libbase::bitfield reg; //!< Present state (shift register)
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   dvbcrsc();
   // @}

   // FSM state operations (getting and resetting)
   libbase::vector<int> state() const;
   void reset();
   void reset(const libbase::vector<int>& state);
   void resetcircular(const libbase::vector<int>& zerostate, int n);
   // FSM operations (advance/output/step)
   void advance(libbase::vector<int>& input);
   libbase::vector<int> output(const libbase::vector<int>& input) const;

   // FSM information functions
   int mem_order() const
      {
      return nu;
      }
   int mem_elements() const
      {
      return nu;
      }
   int num_inputs() const
      {
      return k;
      }
   int num_outputs() const
      {
      return n;
      }
   int num_symbols() const
      {
      return 2;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(dvbcrsc)
};

} // end namespace

#endif

