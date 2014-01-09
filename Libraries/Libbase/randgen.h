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

#ifndef __randgen_h
#define __randgen_h

#include "config.h"
#include "random.h"

namespace libbase {

/*!
 * \brief   Knuth's Subtractive Random Generator.
 * \author  Johann Briffa
 *
 * A pseudo-random generator using the subtractive technique due to
 * Knuth. This algorithm was found to give very good results in the
 * communications lab during the third year.
 *
 * \note
 * - The subtractive algorithm has a very long period (necessary for low
 * bit error rates in the tested data stream)
 * - It also does not suffer from low-order correlations (facilitating its
 * use with a variable number of bits/code in the data stream)
 */

class randgen : public random {
private:
   /*! \name Object representation */
   static const int32s mbig;
   static const int32s mseed;
   int32s next, nextp;
   int32s ma[56], mj;
   // @}

protected:
   // Interface with random
   void init(int32u s);
   void advance();
   int32u get_value() const
      {
      return mj;
      }
   int32u get_max() const
      {
      return mbig;
      }
};

} // end namespace

#endif
