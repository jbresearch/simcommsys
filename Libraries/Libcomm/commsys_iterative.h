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

#ifndef __commsys_iterative_h
#define __commsys_iterative_h

#include "commsys.h"

namespace libcomm {

/*!
 * \brief   Communication System with Iterative Demodulation.
 * \author  Johann Briffa
 *
 * Communication system with iterative demodulation (and possibly iterative
 * decoding); result set is for the given number of iterative demodulations
 * (say N) followed by iterative decoding. That is, result 1 will be for N
 * demodulations + 1 decoding iteration; result 2 is for N demodulations +
 * 2 decoding iter, etc.
 *
 * \todo Update interface to cater for various iterative modes between codec
 * and modem.
 *
 * \todo Move iterative codec nature into this class (requires codec split).
 */

template <class S, template <class > class C = libbase::vector>
class commsys_iterative : public commsys<S, C> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   // @}

private:
   /*! \name User parameters */
   int iter; //!< Number of demodulation iterations
   // @}
public:
   // Communication System Interface
   void receive_path(const C<S>& received);

   // Description
   std::string description() const;
   // Serialization Support
DECLARE_SERIALIZER(commsys_iterative)
};

} // end namespace

#endif
