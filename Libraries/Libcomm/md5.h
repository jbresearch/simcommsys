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

#ifndef __md5_h
#define __md5_h

#include "config.h"
#include "vector.h"
#include "digest32.h"

#include <string>
#include <iostream>

namespace libcomm {

/*!
 * \brief   Message Digest MD5 Algorithm.
 * \author  Johann Briffa
 *
 * Implements Message Digest MD5, as specified in Schneier, "Applied
 * Cryptography", 1996, pp.436-441.
 * Class performs self-testing on creation of the first object.
 *
 * \note there are bugs in Schneier's descriptions of MD5:
 * - chaining variables should be initialised like SHA's
 * - message length is low-order byte first
 * - message is encoded into 32-bit words in low-order byte first
 */

class md5 : public digest32 {
   /*! \name Class-wide constants */
   static bool tested; //!< Flag to indicate self-test has been done
   static libbase::vector<libbase::int32u> t; //!< Additive constants
   static const int s[]; //!< Rotational constants
   static const int ndx[]; //!< Message index constants
   // @}
protected:
   /*! \name Internal functions */
   // self-test function
   static void selftest();
   // verification function
   static bool verify(const std::string message, const std::string hash);
   // circular shift
   static libbase::int32u cshift(const libbase::int32u x, const int s);
   // nonlinear functions
   static libbase::int32u f(const int i, const libbase::int32u X,
         const libbase::int32u Y, const libbase::int32u Z);
   // step operation
   static libbase::int32u op(const int i, const libbase::int32u a,
         const libbase::int32u b, const libbase::int32u c,
         const libbase::int32u d, const libbase::vector<libbase::int32u>& M);
   // @}
   /*! \name Digest-specific functions */
   void derived_reset();
   void process_block(const libbase::vector<libbase::int32u>& M);
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   md5();
   // @}
};

} // end namespace

#endif
