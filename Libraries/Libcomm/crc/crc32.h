
/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
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

#ifndef __crc32_h
#define __crc32_h

#include "crc.h"
#include "serializer.h"

namespace libcomm
{

// Standard IEEE CRC-32
template <template<class> class C = libbase::vector>
class crc32_ieee : public crc_base<32, C>, public libbase::serializable {
public:
    using base_t     = crc_base<32, C>;
    using value_type = typename base_t::value_type;

    /* References: [1] https://boost.org.cpp.al/doc/libs/master/doc/html/crc/reference.html? [2] https://www.ieee802.org/3/as/public/0503/3d0_1_CMP.pdf?

    IEEE Standard CRC-32 (IEEE 802.3) polynomial: G(x) = x32 + x26 + x23 + x22 + x16 + x12 + x11 + x10 + x8 + x7 + x5 + x4 + x2 + x + 1

    Compare obtained answers to this online calculator using CRC-32/MPEG-2: https://crccalc.com/?crc=123456789&method=&datatype=ascii&outtype=hex*/

    crc32_ieee()
        : base_t(0x04C11DB7u, 0xFFFFFFFFu, 0x00000000, false, false) {} // CRC-32 MPE|G-2
        // : base_t(0x04C11DB7u, 0xFFFFFFFFu, 0xFFFFFFFFu, false, false) {} // IEEE standard 802.3

    // Static convenience (works with C<bool>, C<int>, ...)
    template <class T>
    static value_type compute(const C<T>& bits01) {
        crc32_ieee<C> c;
        c.process_bits(bits01);
        return c.checksum();
    }

    // Serialization Support
    DECLARE_SERIALIZER(crc32_ieee<C>)
};

}
#endif