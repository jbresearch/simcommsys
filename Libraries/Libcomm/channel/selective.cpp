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

template <class S>
selective<S>::selective(const std::string& bitstring)
    : selective(bitstring, nullptr, nullptr, 0.0)
{
}

template <class S>
selective<S>::selective(const std::string& bitstring,
                        std::shared_ptr<channel<S>> primary_channel,
                        std::shared_ptr<channel<S>> secondary_channel,
                        const double secondary_channel_parameter)
    : m_bitmask(create_bitmask_from_bistring(bitstring)),
      m_primary_channel(primary_channel), m_secondary_channel(secondary_channel)
{
    if (nullptr != secondary_channel) {
        m_secondary_channel->set_parameter(secondary_channel_parameter);
    }
}

template <class S>
std::string
selective<S>::get_bitmask() const
{
    std::stringstream ss;

    for (const auto& bit_value : m_bitmask) {
        ss << (bit_value == true ? "1" : "0");
    }

    return ss.str();
}

template <class S>
void
selective<S>::transmit(const libbase::vector<S>& tx, libbase::vector<S>& rx)
{
    validate_sequence_size(tx);

    auto separate_sequences = separate(tx);

    auto primary_tx_sequence = separate_sequences.first;
    auto primary_rx_sequence = libbase::vector<S>();
    m_primary_channel->transmit(primary_tx_sequence, primary_rx_sequence);

    auto secondary_tx_sequence = separate_sequences.second;
    auto secondary_rx_sequence = libbase::vector<S>();
    m_secondary_channel->transmit(secondary_tx_sequence, secondary_rx_sequence);

    merge(primary_rx_sequence, secondary_rx_sequence, rx);
}

template <class S>
std::pair<libbase::vector<S>, libbase::vector<S>>
selective<S>::separate(const libbase::vector<S>& bit_sequence) const
{
    auto primary_sequence = std::vector<S>();
    auto secondary_sequence = std::vector<S>();

    primary_sequence.reserve(bit_sequence.size());
    secondary_sequence.reserve(bit_sequence.size());

    for (auto i = 0ul; i < m_bitmask.size(); ++i) {
        if (true == m_bitmask.at(i)) {
            primary_sequence.push_back(bit_sequence(i));
        } else {
            secondary_sequence.push_back(bit_sequence(i));
        }
    }

    return std::make_pair(libbase::vector<S>(primary_sequence),
                          libbase::vector<S>(secondary_sequence));
}

template <class S>
void
selective<S>::merge(const libbase::vector<S>& primary,
                    const libbase::vector<S>& secondary,
                    libbase::vector<S>& merged) const
{
    assertalways((primary.size() + secondary.size()) == (int)m_bitmask.size());

    merged.init(m_bitmask.size());

    auto primary_idx = 0;
    auto secondary_idx = 0;

    for (auto i = 0ul; i < m_bitmask.size(); ++i) {
        if (true == m_bitmask.at(i)) {
            merged(i) = primary(primary_idx);
            ++primary_idx;
        } else {
            merged(i) = secondary(secondary_idx);
            ++secondary_idx;
        }
    }
}

template <class S>
std::vector<bool>
selective<S>::create_bitmask_from_bistring(const std::string& bitstring)
{
    validate_bitstring(bitstring);

    auto bitmask = std::vector<bool>();
    bitmask.reserve(bitstring.length());

    for (const char& bit_value : bitstring) {
        bitmask.push_back('1' == bit_value ? 1 : 0);
    }

    return bitmask;
}

template <class S>
S
selective<S>::corrupt(const S& s)
{
    failwith("selective_channel::corrupt MUST never be called");
    return s;
}

template <class S>
double
selective<S>::pdf(const S& tx, const S& rx) const
{
    failwith("selective_channel::pdf MUST never be called");
    return 0.0;
}

template <class S>
void
selective<S>::set_parameter(const double x)
{
}

template <class S>
double
selective<S>::get_parameter() const
{
    return 0.0;
}

template <class S>
std::string
selective<S>::description() const
{
    return "";
}

template <class S>
void
selective<S>::validate_bitstring(const std::string& bitstring)
{
    if (bitstring.find_first_not_of("01") != std::string::npos) {
        throw libbase::load_error("Bitstring can only contain '1' or "
                                  "'0' characters");
    }
}

template <class S>
void
selective<S>::validate_sequence_size(const libbase::vector<S>& sequence) const
{
    if (static_cast<std::size_t>(sequence.size()) != m_bitmask.size()) {
        std::stringstream ss;
        ss << "Mismatch between length of bitmask (" << m_bitmask.size()
           << ") and given sequence (" << sequence.size() << ")";

        throw std::runtime_error(ss.str());
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
