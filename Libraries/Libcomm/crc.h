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

#ifndef __crc_h
#define __crc_h

#include "config.h"
#include "serializer.h"
#include "vector.h"
#include <boost/crc.hpp>
#include <cstdint>
#include <type_traits>

namespace libcomm
{

// Generic bitwise CRC base using Boost
// and the container template (defaults to libbase::vector). The Bits template
// parameter is the CRC width.
template <unsigned Bits, template <class> class C = libbase::vector>
class crc_base
{
protected:
    boost::crc_basic<Bits> crc_;

public:
    // value_type is an unsigned integer type. E.g. for CRC-32 the value_type is
    // std::uint32_t
    using value_type = typename boost::crc_basic<Bits>::value_type;

    /* Reference: [1]
     * https://boost.org.cpp.al/doc/libs/master/doc/html/crc/reference.html?*/

    explicit crc_base(value_type truncated_polynomial,
                      value_type initial_remainder,
                      value_type final_xor_value,
                      bool reflect_input,
                      bool reflect_remainder)
        : crc_(truncated_polynomial,
               initial_remainder,
               final_xor_value,
               reflect_input,
               reflect_remainder)
    {
    }

    void reset() { crc_.reset(); }

    // Per bit. Turns b into 0/1 and pushes that one bit into the CRC
    // calculator.
    void process_bit(bool b) { crc_.process_bit(b ? 1u : 0u); }

    // Per sequence.
    template <class T>
    void process_bits(const C<T>& bits01)
    {
        static_assert(std::is_integral<T>::value,
                      "Element type must be integral (e.g., bool, int).");
        const std::size_t n = bits01.size();
        for (std::size_t i = 0; i < n; ++i) {
            process_bit(bits01(i) != 0);
        }
    }

    value_type checksum() const { return crc_.checksum(); }

    // Computes for any integral element type (bool/int/...)
    template <class T>
    value_type compute(const C<T>& bits01)
    {
        reset();
        process_bits(bits01);
        return checksum();
    }

    // --- Serialization support for base class ---
    using crc = crc_base<Bits, C>;

    // Serialization Support
    DECLARE_BASE_SERIALIZER(crc)
};

} // namespace libcomm

#endif