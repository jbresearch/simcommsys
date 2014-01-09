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

#ifndef CACHED_FSM_H_
#define CACHED_FSM_H_

#include "fsm.h"
#include "matrix.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of construction
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Cached FSM.
 * \author  Johann Briffa
 *
 * Class representing an FSM held internally by its state table. This can
 * be constructed from any other FSM class, memory permitting, by iteratively
 * obtaining the state table. Construction is expensive, but state operations
 * are very fast.
 *
 * \todo Fix class serialization and descriptions
 */

class cached_fsm : public fsm {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::matrix<int> array2i_t;
   typedef libbase::matrix<array1i_t> array2vi_t;
   // @}
private:
   /*! \name Internal variables */
   int ps; //!< Present state
   // @}

   /*! \name Internal details about state machine */
   int k; //!< Number of input lines
   int n; //!< Number of output lines
   int nu; //!< Total number of memory elements (constraint length)
   int m; //!< Memory order (longest input register)
   int q; //!< Alphabet size of input/output symbols
   std::string base_description; //!< Description string of base FSM
   std::string base_serialization; //!< Serialization string of base FSM
   // @}

   /*! \name Internal representation of state table */
   //! lut_X(m,i) = encoder output, given state 'm' and input 'i'
   array2i_t lut_X;
   //! lut_m(m,i) = next state, given state 'm' and input 'i'
   array2i_t lut_m;
   //! lut_Xv(m,i) = encoder output (as vector), given state 'm' and input 'i'
   array2vi_t lut_Xv;
   // @}

protected:
   /*! \name Helper functions */
   //! Main initialization routine
   void init(fsm& encoder);
   // @}

public:
   /*! \name Constructor & destructor */
   //! Principal constructor
   explicit cached_fsm(fsm& encoder)
      {
#if DEBUG>=2
      std::cerr << "DEBUG: cached_fsm(fsm&) constructor" << std::endl;
#endif
      init(encoder);
      reset(encoder.state());
      }
   //! Principal constructor (for const argument)
   explicit cached_fsm(const fsm& encoder)
      {
#if DEBUG>=2
      std::cerr << "DEBUG: cached_fsm(const fsm&) constructor" << std::endl;
#endif
      fsm *encoder_copy = dynamic_cast<fsm*> (encoder.clone());
      init(*encoder_copy);
      delete encoder_copy;
      reset(encoder.state());
      }
   //! Default constructor
   cached_fsm()
      {
      }
   // @}

   // FSM state operations (getting and resetting)
   array1i_t state() const
      {
      return fsm::convert_state(ps);
      }
   void reset()
      {
      ps = 0;
      }
   void reset(const array1i_t& state)
      {
      ps = fsm::convert_state(state);
      }
   void resetcircular(const array1i_t& zerostate, int n)
      {
      failwith("Function not implemented");
      }

   // FSM operations (advance/output/step)
   void advance(array1i_t& input)
      {
      // NOTE: this won't work with tail bits
      const int i = fsm::convert_input(input);
      ps = lut_m(ps, i);
      }
   array1i_t output(const array1i_t& input) const
      {
      // NOTE: this won't work with tail bits
      const int i = fsm::convert_input(input);
      return lut_Xv(ps, i);
      }
   array1i_t step(array1i_t& input)
      {
      // NOTE: this won't work with tail bits
      const int i = fsm::convert_input(input);
      array1i_t op = lut_Xv(ps, i);
      ps = lut_m(ps, i);
      return op;
      }

   // FSM information functions - fundamental
   //! Memory order (length of tail)
   int mem_order() const
      {
      return m;
      }
   //! Number of memory elements
   int mem_elements() const
      {
      return nu;
      }
   //! Number of input lines
   int num_inputs() const
      {
      return k;
      }
   //! Number of output lines
   int num_outputs() const
      {
      return n;
      }
   //! Alphabet size of input/output symbols
   int num_symbols() const
      {
      return q;
      }

   // Description
   std::string description() const
      {
      return "Cached " + base_description;
      }

   // Serialization Support
DECLARE_SERIALIZER(cached_fsm)
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif /* CACHED_FSM_H_ */
