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

#ifndef __ccbfsm_h
#define __ccbfsm_h

#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"
#include "vector.h"

namespace libcomm {

/*!
 * \brief   Controller-Canonical Binary Finite State Machine.
 * \author  Johann Briffa
 *
 * Implements common elements of a controller-canonical binary fsm.
 * The generator matrix is held such that the least-significant bit is
 * left-most; this is according to the convention described by Alex, such that
 * from state 0, input 1, we always get state 1, no matter how long the state
 * register is. This is also the same as the format used in Benedetto et al.'s
 * 1998 paper "A Search for Good Convolutional Codes to be Used in the
 * Construction of Turbo Codes". This is different from the assumed convention
 * in Lin & Costello (ie. high-order bits are left-most, and closest to the
 * input junction).
 */

class ccbfsm : public fsm {
protected:
   /*! \name Object representation */
   int k; //!< Number of input bits
   int n; //!< Number of output bits
   int nu; //!< Number of memory elements (constraint length)
   int m; //!< Memory order (longest input register)
   /*! \brief Shift registers (one for each input bit);
    * not all registers need be the same length
    */
   libbase::vector<libbase::bitfield> reg;
   libbase::matrix<libbase::bitfield> gen; //!< Generator sequence
   // @}
private:
   /*! \name Internal functions */
   void init();
   // @}
protected:
   /*! \name FSM helper operations */
   //! Computes the actual input to be applied (works out tail)
   virtual libbase::vector<int>
   determineinput(const libbase::vector<int>& input) const = 0;
   //! Computes the memory register input
   virtual libbase::bitfield
   determinefeedin(const libbase::vector<int>& input) const = 0;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   ccbfsm()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   ccbfsm(const libbase::matrix<libbase::bitfield>& generator);
   // @}

   // FSM state operations (getting and resetting)
   libbase::vector<int> state() const;
   void reset();
   void reset(const libbase::vector<int>& state);
   // FSM operations (advance/output/step)
   void advance(libbase::vector<int>& input);
   libbase::vector<int> output(const libbase::vector<int>& input) const;

   // FSM information functions
   int mem_order() const
      {
      return m;
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

   // Description & Serialization
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

} // end namespace

#endif

