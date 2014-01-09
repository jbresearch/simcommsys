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

#ifndef __zsm_h
#define __zsm_h

#include "fsm.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Zero State Machine.
 * \author  Johann Briffa
 *
 * Implements a zero-state machine, or in other words a repeater.
 * The symbol type is specified as a template parameter; this must support:
 * - alphabet size through a static elements() method
 * - assignment
 * - conversion to/from int
 */

template <class S>
class zsm : public fsm {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   // @}
protected:
   /*! \name Object representation */
   int r; //!< Repetition count
   // @}
protected:
   /*! \name Constructors / Destructors */
   //! Default constructor
   zsm()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   /*!
    * \brief Principal constructor
    */
   zsm(const int r) :
         r(r)
      {
      assert(r >= 1);
      }
   // @}

   // FSM state operations (getting and resetting)
   array1i_t state() const
      {
      return array1i_t();
      }
   void resetcircular(const array1i_t& zerostate, int n)
      {
      }
   // FSM operations (advance/output/step)
   array1i_t output(const array1i_t& input) const;

   // FSM information functions
   int mem_order() const
      {
      return 0;
      }
   int mem_elements() const
      {
      return 0;
      }
   int num_inputs() const
      {
      return 1;
      }
   int num_outputs() const
      {
      return r;
      }
   int num_symbols() const
      {
      return S::elements();
      }

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(zsm)
};

} // end namespace

#endif
