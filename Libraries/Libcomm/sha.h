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

#ifndef __sha_h
#define __sha_h

#include "config.h"
#include "vector.h"
#include "digest32.h"

#include <string>
#include <iostream>

namespace libcomm {

/*!
 * \brief   Secure Hash Algorithm.
 * \author  Johann Briffa
 *
 * Implements Secure Hash Algorithm SHA-1 (160-bit), as specified in
 * Schneier, "Applied Cryptography", 1996, pp.442-445.
 */

class sha : public digest32 {
   /*! \name Class-wide constants */
   static bool tested; //!< Flag to indicate self-test has been done
   static const libbase::int32u K[]; //!< Additive constants
   // @}
protected:
   /*! \name Internal functions */
   // self-test function
   static void selftest();
   // verification function
   static bool verify(const std::string message, const std::string hash);
   // Nonlinear functions
   static libbase::int32u f(const int t, const libbase::int32u X,
         const libbase::int32u Y, const libbase::int32u Z);
   // Circular shift
   static libbase::int32u cshift(const libbase::int32u x, const int s);
   // Message expander
   static void expand(const libbase::vector<libbase::int32u>& M,
         libbase::vector<libbase::int32u>& W);
   // @}
   /*! \name Digest-specific functions */
   void derived_reset();
   void process_block(const libbase::vector<libbase::int32u>& M);
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   sha();
   // @}
};

} // end namespace

#endif
