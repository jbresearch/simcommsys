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

/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "codec_reshaped.h"

namespace libcomm {

// object serialization - saving

template <class base_codec>
std::ostream& codec_reshaped<base_codec>::serialize(std::ostream& sout) const
   {
   return base.serialize(sout);
   }

// object serialization - loading

template <class base_codec>
std::istream& codec_reshaped<base_codec>::serialize(std::istream& sin)
   {
   return base.serialize(sin);
   }

} // end namespace

// Explicit Realizations

#include "turbo.h"
#include "uncoded.h"

namespace libcomm {

using libbase::serializer;

/*** Turbo codes ***/

template class codec_reshaped<turbo<double> > ;
template <>
const serializer codec_reshaped<turbo<double> >::shelper = serializer("codec",
      "codec_reshaped<turbo<double>>", codec_reshaped<turbo<double> >::create);

/*** Uncoded/repetition codes ***/

template class codec_reshaped<uncoded<double> > ;
template <>
const serializer codec_reshaped<uncoded<double> >::shelper = serializer(
      "codec", "codec_reshaped<uncoded<double>>", codec_reshaped<
            uncoded<double> >::create);

} // end namespace
