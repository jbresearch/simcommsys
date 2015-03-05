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

#ifndef __channel_stream_h
#define __channel_stream_h

#include "channel_insdel.h"

namespace libcomm {

/*!
 * \brief   Stream-Oriented Channel Interface.
 * \author  Johann Briffa
 *
 * Defines the additional interface methods for stream-oriented channels.
 *
 * \tparam S Channel symbol type
 * \tparam real Floating-point type for metric computer interface
 */

template <class S, class real>
class channel_stream : public channel_insdel<S, real> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   // @}
public:
   /*! \name Stream-oriented channel characteristics */
   /*!
    * \brief Get the expected drift distribution after transmitting 'tau'
    * symbols, assuming the start-of-frame drift is zero.
    *
    * For systems with a variable-size state space, this method determines the
    * required limit, and computes the end-of-frame distribution for this range.
    * It returns the necessary offset accordingly.
    */
   virtual void get_drift_pdf(int tau, double Pr, array1d_t& eof_pdf, libbase::size_type<
         libbase::vector>& offset) const = 0;
   /*!
    * \brief Get the expected drift distribution after transmitting 'tau'
    * symbols, assuming the start-of-frame distribution is as given.
    *
    * For systems with a variable-size state space, this method determines an
    * updated limit, and computes the end-of-frame distribution for this range.
    * It also resizes the start-of-frame pdf accordingly and updates the given
    * offset.
    */
   virtual void get_drift_pdf(int tau, double Pr, array1d_t& sof_pdf, array1d_t& eof_pdf,
         libbase::size_type<libbase::vector>& offset) const = 0;
   // @}
};

} // end namespace

#endif
