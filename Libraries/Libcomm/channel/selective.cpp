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
selective<S>::selective(const std::string& bitmask)
    : selective(bitmask, nullptr, nullptr, 0.0)
{
}

template <class S>
selective<S>::selective(const std::string& bitmask,
                        std::shared_ptr<channel<S>> primary_channel,
                        std::shared_ptr<channel<S>> secondary_channel,
                        const double secondary_channel_parameter)
    : m_primary_channel(primary_channel), m_secondary_channel(secondary_channel)
{
    init(bitmask, secondary_channel_parameter);
}

template <class S>
void
selective<S>::set_parameter(const double x)
{
    assertalways(m_primary_channel != nullptr);
    m_primary_channel->set_parameter(x);
}

template <class S>
double
selective<S>::get_parameter() const
{
    assertalways(m_primary_channel != nullptr);
    return m_primary_channel->get_parameter();
}

template <class S>
void
selective<S>::set_bitmask(const std::string& bitmask)
{
    validate_bitmask(bitmask);

    m_bitmask.clear();
    m_bitmask.reserve(bitmask.length());

    for (const char& bit_value : bitmask) {
        m_bitmask.push_back('1' == bit_value ? 1 : 0);
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
const channel<S>&
selective<S>::get_primary_channel() const
{
    return *m_primary_channel;
}

template <class S>
const channel<S>&
selective<S>::get_secondary_channel() const
{
    return *m_secondary_channel;
}

template <class S>
void
selective<S>::transmit(const libbase::vector<S>& tx, libbase::vector<S>& rx)
{
    validate_sequence_size(tx);

    auto split_sequences = split_sequence(tx);

    auto primary_tx_sequence = split_sequences.first;
    auto primary_rx_sequence = libbase::vector<S>();
    m_primary_channel->transmit(primary_tx_sequence, primary_rx_sequence);

    auto secondary_tx_sequence = split_sequences.second;
    auto secondary_rx_sequence = libbase::vector<S>();
    m_secondary_channel->transmit(secondary_tx_sequence, secondary_rx_sequence);

    merge_sequences(primary_rx_sequence, secondary_rx_sequence, rx);
}

template <class S>
void
selective<S>::receive(const libbase::vector<S>& possible_tx_symbols,
                      const libbase::vector<S>& rx,
                      libbase::vector<libbase::vector<double>>& ptable) const
{
    auto split_sequences = split_sequence(rx);

    auto primary_rx_sequence = split_sequences.first;
    auto primary_ptable = libbase::vector<libbase::vector<double>>();
    m_primary_channel->receive(
        possible_tx_symbols, primary_rx_sequence, primary_ptable);

    auto secondary_rx_sequence = split_sequences.second;
    auto secondary_ptable = libbase::vector<libbase::vector<double>>();
    m_secondary_channel->receive(
        possible_tx_symbols, secondary_rx_sequence, secondary_ptable);

    libbase::allocate(ptable, rx.size(), possible_tx_symbols.size());
    merge_ptables(primary_ptable, secondary_ptable, ptable);
}

template <class S>
std::string
selective<S>::description() const
{
    assertalways(m_primary_channel != nullptr);
    assertalways(m_secondary_channel != nullptr);

    std::stringstream ss;
    ss << "Selective channel. "
       << "Primary Channel: [" << m_primary_channel->description() << "]"
       << "Secondary Channel: [" << m_secondary_channel->description() << "]";

    return ss.str();
}

template <class S>
void
selective<S>::init(const std::string& bitmask,
                   const double secondary_channel_parameter)
{
    set_bitmask(bitmask);

    if (nullptr != m_secondary_channel) {
        m_secondary_channel->set_parameter(secondary_channel_parameter);
    }
}

template <class S>
std::pair<libbase::vector<S>, libbase::vector<S>>
selective<S>::split_sequence(const libbase::vector<S>& bit_sequence) const
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
selective<S>::merge_sequences(const libbase::vector<S>& primary,
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
void
selective<S>::merge_ptables(
    const libbase::vector<libbase::vector<double>>& primary_ptable,
    const libbase::vector<libbase::vector<double>>& secondary_ptable,
    libbase::vector<libbase::vector<double>>& ptable) const
{
    auto primary_idx = 0;
    auto secondary_idx = 0;

    for (auto i = 0ul; i < m_bitmask.size(); ++i) {
        if (true == m_bitmask.at(i)) {
            ptable(i).copyfrom(primary_ptable(primary_idx));
            ++primary_idx;
        } else {
            ptable(i).copyfrom(secondary_ptable(secondary_idx));
            ++secondary_idx;
        }
    }
}

template <class S>
void
selective<S>::validate_bitmask(const std::string& bitmask)
{
    if (bitmask.find_first_not_of("01") != std::string::npos) {
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

template <class G>
std::ostream&
selective<G>::serialize(std::ostream& sout) const
{
    assertalways(sout.good());

    // clang-format off
    sout << "# Bitmask\n"
         << get_bitmask() << "\n"
         << "# Primary channel\n"
         << m_primary_channel
         << "# Secondary channel\n"
         << m_secondary_channel
         << "# # Secondary channel fixed parameter value\n"
         << m_secondary_channel->get_parameter();
    // clang-format on

    return sout;
}

template <class G>
std::istream&
selective<G>::serialize(std::istream& sin)
{
    assertalways(sin.good());

    std::string bitmask;
    double secondary_channel_param;

    sin >> libbase::eatcomments >> bitmask >> libbase::verify;
    sin >> libbase::eatcomments >> m_primary_channel >> libbase::verify;
    sin >> libbase::eatcomments >> m_secondary_channel >> libbase::verify;
    sin >> libbase::eatcomments >> secondary_channel_param >> libbase::verify;

    init(bitmask, secondary_channel_param);

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
