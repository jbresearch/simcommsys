/*!
 * \file
 *
 * Copyright (c) 2022 Noel Farrugia
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
#include "channel/selective.h"

#include "gf.h"

#include <sstream>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

namespace libcomm
{

template <class Symbol>
selective<Symbol>::selective(const std::string& bitstring)
    : selective(bitstring, nullptr)
{
}

template <class Symbol>
selective<Symbol>::selective(const std::string& bitstring,
                             std::shared_ptr<channel<Symbol>> channel)
    : m_bitmask(create_bitmask_from_bistring(bitstring)), m_channel(channel)
{
}

template <class Symbol>
std::string
selective<Symbol>::get_bitmask() const
{
    std::stringstream ss;

    for (const auto& bit_value : m_bitmask) {
        ss << (bit_value == true ? "1" : "0");
    }

    return ss.str();
}

template <class Symbol>
void
selective<Symbol>::transmit(const libbase::vector<Symbol>& tx,
                            libbase::vector<Symbol>& rx)
{
    if (static_cast<std::size_t>(tx.size()) != m_bitmask.size()) {
        std::stringstream ss;
        ss << "Mismatch between length of bitmask (" << m_bitmask.size()
           << ") and transmission sequence (" << tx.size() << ")";

        throw std::runtime_error(ss.str());
    }

    rx.init(tx.size());

    for (auto i = 0; i < tx.size(); ++i) {
        if (true == m_bitmask.at(i)) {
            rx(i) = m_channel->corrupt(tx(i));
        } else {
            rx(i) = tx(i);
        }
    }
}

template <class Symbol>
std::vector<bool>
selective<Symbol>::create_bitmask_from_bistring(const std::string& bitstring)
{
    validate_bitstring(bitstring);

    auto bitmask = std::vector<bool>();
    bitmask.reserve(bitstring.length());

    for (const char& bit_value : bitstring) {
        bitmask.push_back('1' == bit_value ? 1 : 0);
    }

    return bitmask;
}

template <class Symbol>
Symbol
selective<Symbol>::corrupt(const Symbol& s)
{
    failwith("selective_channel::corrupt MUST never be called");
    return s;
}

template <class Symbol>
double
selective<Symbol>::pdf(const Symbol& tx, const Symbol& rx) const
{
    failwith("selective_channel::pdf MUST never be called");
    return 0.0;
}

template <class Symbol>
void
selective<Symbol>::set_parameter(const double x)
{
}

template <class Symbol>
double
selective<Symbol>::get_parameter() const
{
    return 0.0;
}

template <class Symbol>
std::string
selective<Symbol>::description() const
{
    return "";
}

template <class Symbol>
void
selective<Symbol>::validate_bitstring(const std::string& bitstring)
{
    if (bitstring.find_first_not_of("01") != std::string::npos) {
        throw libbase::load_error("Bitstring can only contain '1' or "
                                  "'0' characters");
    }
}

template <class G>
std::ostream&
selective<G>::serialize(std::ostream& sout) const
{
    return sout;
}

template <class G>
std::istream&
selective<G>::serialize(std::istream& sin)
{
    return sin;
}

// Explicit Realizations
using libbase::serializer;

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ

/* Serialization string: selective<type>
 * where:
 *      type = bool | gf2 | gf4 ...
 */
#define INSTANTIATE(r, x, type) \
   template class selective<type>; \
   template <> \
   const serializer selective<type>::shelper( \
         "channel", \
         "selective<" BOOST_PP_STRINGIZE(type) ">", \
         selective<type>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // namespace libcomm
