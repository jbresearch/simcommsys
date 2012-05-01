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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __channel_stream_h
#define __channel_stream_h

#include "channel.h"

namespace libcomm {

/*!
 * \brief   Stream-Oriented Channel Interface.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Defines the additional interface methods for stream-oriented channels.
 */

template <class S>
class channel_stream : public channel<S> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   // @}
public:
   /*! \name Stream-oriented channel characteristics */
   virtual void get_drift_pdf(int tau, array1d_t& eof_pdf, libbase::size_type<
         libbase::vector>& offset) const = 0;
   virtual void get_drift_pdf(int tau, array1d_t& sof_pdf, array1d_t& eof_pdf,
         libbase::size_type<libbase::vector>& offset) const = 0;
   // @}
};

} // end namespace

#endif
