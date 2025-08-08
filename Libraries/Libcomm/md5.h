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
#include "digest32.h"
#include "vector.h"

#include <iostream>
#include <string>

namespace libcomm
{

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

class md5 : public digest32
{
    /*! \name Class-wide constants */
    static bool tested; //!< Flag to indicate self-test has been done
    static libbase::vector<uint32_t> t; //!< Additive constants
    static const int s[];                      //!< Rotational constants
    static const int ndx[];                    //!< Message index constants
                                               // @}
protected:
    /*! \name Internal functions */
    // self-test function
    static void selftest();
    // verification function
    static bool verify(const std::string message, const std::string hash);
    // circular shift
    static uint32_t cshift(const uint32_t x, const int s);
    // nonlinear functions
    static uint32_t f(const int i,
                             const uint32_t X,
                             const uint32_t Y,
                             const uint32_t Z);
    // step operation
    static uint32_t op(const int i,
                              const uint32_t a,
                              const uint32_t b,
                              const uint32_t c,
                              const uint32_t d,
                              const libbase::vector<uint32_t>& M);
    // @}
    /*! \name Digest-specific functions */
    void derived_reset();
    void process_block(const libbase::vector<uint32_t>& M);
    // @}
public:
    /*! \name Constructors / Destructors */
    //! Default constructor
    md5();
    // @}
};

} // namespace libcomm

#endif
