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

#ifndef __fba_h
#define __fba_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "matrix3.h"
#include "fsm.h"
#include "multi_array.h"

#include <cmath>
#include <iostream>
#include <fstream>

namespace libcomm {

/*!
 * \brief   Bit-Level Forward-Backward Algorithm (for Davey-MacKay codes).
 * \author  Johann Briffa
 *
 * Implements a Forward-Backward Algorithm for a HMM. This is based on the
 * paper by Davey & McKay, "Watermark Codes: Reliable communication over
 * Insertion/Deletion channels", Trans. IT, 47(2), Feb 2001.
 *
 * Algorithm is implemented on a single block; in the case of Davey's
 * Watermark codes, each block is N elements of n-bit in length, and is the
 * size of the sparsifier's output for a single LDPC codeword. Typical values
 * of n used by Davey were 5,6,7. With watermark codes, N was typically in the
 * range 500-1000. For other examples of LDPC codes, Davey used N up to about
 * 16000.
 *
 * FBA operates at bit-level only, without knowledge of the watermark code
 * and therefore without the possibility of doing the final decode stage.
 *
 * \todo Confirm correctness of the backward matrix computation referring to
 * bit j instead of j+1.
 *
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 */

template <class sig, class real>
class fba {
public:
   /*! \name Type definitions */
   typedef libbase::vector<sig> array1s_t;
   typedef boost::assignable_multi_array<real, 2> array2r_t;
   // @}
private:
   /*! \name User-defined parameters */
   int tau; //!< The (transmitted) block size in channel symbols
   int mtau_min; //!< The largest negative drift within a whole frame is \f$ m_\tau^{-} \f$
   int mtau_max; //!< The largest positive drift within a whole frame is \f$ m_\tau^{+} \f$
   int m1_min; //!< The largest negative drift over a single channel symbol is \f$ m_1^{-} \f$
   int m1_max; //!< The largest positive drift over a single channel symbol is \f$ m_1^{+} \f$
   real th_inner; //!< Threshold factor for inner cycle
   bool norm; //!< Flag to indicate if metrics should be normalized between time-steps
   // @}
   /*! \name Internally-used objects */
   bool initialised; //!< Flag to indicate when memory is allocated
   array2r_t F; //!< Forward recursion metric
   array2r_t B; //!< Backward recursion metric
   // @}
private:
   /*! \name Internal functions */
   void allocate();
   void free();
   // @}
protected:
   /*! \name Internal functions */
   // handles for channel-specific metrics - to be implemented by derived classes
   virtual real R(const int i, const array1s_t& r) = 0;
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba() :
         initialised(false)
      {
      }
   virtual ~fba()
      {
      }
   // @}

   // main initialization routine
   void init(int tau, int mtau_min, int mtau_max, int m1_min, int m1_max, double th_inner, bool norm);
   // getters for forward and backward metrics
   real getF(const int j, const int y) const
      {
      return F[j][y];
      }
   real getB(const int j, const int y) const
      {
      return B[j][y];
      }
   // decode functions
   void work_forward(const array1s_t& r);
   void work_backward(const array1s_t& r);
   void prepare(const array1s_t& r);
};

} // end namespace

#endif
