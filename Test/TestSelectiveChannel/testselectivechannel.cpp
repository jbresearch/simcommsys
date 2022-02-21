/*!
 * \file
 *
 * Copyright (c) 2021 Noel Farrugia
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

#include <iostream>

#include <boost/numeric/conversion/cast.hpp>
#define BOOST_TEST_MODULE SelectiveChannel
#include <boost/test/included/unit_test.hpp>

#include "channel.h"
#include "channel/qsc.h"
#include "config.h"
#include "gf.h"
#include "serializer.h"
#include <boost/numeric/conversion/cast.hpp>

namespace libcomm
{
template <class Symbol>
class mock_channel : public libcomm::qsc<Symbol>
{
public:
    Symbol corrupt(const Symbol& s) { return mock_corrupt(s); }

    std::function<Symbol(Symbol)> mock_corrupt;
};

template <class Symbol>
class selective_channel : public channel<Symbol>
{
public:
    selective_channel() = default;

    selective_channel(const std::string& bitstring)
        : selective_channel(bitstring, nullptr)
    {
    }

    selective_channel(const std::string& bitstring,
                      std::shared_ptr<channel<Symbol>> channel)
        : m_bitmask(create_bitmask_from_bistring(bitstring)), m_channel(channel)
    {
    }

    const std::string get_bitmask()
    {
        std::stringstream ss;

        for (const auto& bit_value : m_bitmask) {
            ss << (bit_value == true ? "1" : "0");
        }

        return ss.str();
    }

    void transmit(const libbase::vector<Symbol>& tx,
                  libbase::vector<Symbol>& rx) override
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

private:
    std::vector<bool> create_bitmask_from_bistring(const std::string& bitstring)
    {
        validate_bitstring(bitstring);

        auto bitmask = std::vector<bool>();
        bitmask.reserve(bitstring.length());

        for (const char& bit_value : bitstring) {
            bitmask.push_back('1' == bit_value ? 1 : 0);
        }

        return bitmask;
    }

    Symbol corrupt(const Symbol& s)
    {
        failwith("selective_channel::corrupt MUST never be called");
        return s;
    }

    double pdf(const Symbol& tx, const Symbol& rx) const
    {
        failwith("selective_channel::pdf MUST never be called");
        return 0.0;
    }

    void set_parameter(const double x) {}
    double get_parameter() const { return 0.0; }

    std::string description() const { return ""; }

    // Serialization Support
    DECLARE_SERIALIZER(selective_channel);

    void validate_bitstring(const std::string& bitstring)
    {
        if (bitstring.find_first_not_of("01") != std::string::npos) {
            throw libbase::load_error("Bitstring can only contain '1' or "
                                      "'0' characters");
        }
    }

    std::vector<bool> m_bitmask;
    std::shared_ptr<channel<Symbol>> m_channel;
};

template <class G>
std::ostream&
selective_channel<G>::serialize(std::ostream& sout) const
{
    return sout;
}

template <class G>
std::istream&
selective_channel<G>::serialize(std::istream& sin)
{
    return sin;
}

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ

/* Serialization string: selective_channel<type>
 * where:
 *      type = bool | gf2 | gf4 ...
 */
#define INSTANTIATE(r, x, type) \
   template class selective_channel<type>; \
   template <> \
   const serializer selective_channel<type>::shelper( \
         "channel", \
         "selective_channel<" BOOST_PP_STRINGIZE(type) ">", \
         selective_channel<type>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // namespace libcomm

using namespace libcomm;

static const std::string TEST_BITMASK = "1110011";

BOOST_AUTO_TEST_CASE(initialise_bitmask_from_constructor)
{
    auto channel = selective_channel<libbase::gf2>(TEST_BITMASK);

    BOOST_CHECK_EQUAL(TEST_BITMASK, channel.get_bitmask());
}

BOOST_AUTO_TEST_CASE(validate_bitmask)
{
    BOOST_CHECK_THROW(selective_channel<libbase::gf2>("1210011"),
                      libbase::load_error);
    BOOST_CHECK_THROW(selective_channel<libbase::gf2>("1100z11"),
                      libbase::load_error);
}

BOOST_AUTO_TEST_CASE(throw_exception_if_bitmask_and_tx_sequence_mismatch)
{
    auto channel = selective_channel<libbase::gf2>(TEST_BITMASK);

    const auto tx_vector = std::vector<int>{1, 0, 1, 0};
    const auto tx_sequence = libbase::vector<libbase::gf2>(tx_vector);
    auto rx_sequence = libbase::vector<libbase::gf2>();

    BOOST_CHECK_THROW(channel.transmit(tx_sequence, rx_sequence),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(corrupt_only_selected_bits)
{
    const auto tx_vector = std::vector<int>{1, 1, 1, 1, 1, 1, 1};
    const auto expected = std::vector<int>{0, 0, 0, 1, 1, 0, 0};

    const auto tx_sequence = libbase::vector<libbase::gf2>(tx_vector);
    const auto expected_rx_sequence = libbase::vector<libbase::gf2>(expected);
    auto rx_sequence = libbase::vector<libbase::gf2>();

    auto test_channel = std::make_shared<mock_channel<libbase::gf2>>();
    test_channel->mock_corrupt = [](libbase::gf2 symbol) {
        return symbol += 1;
    };

    auto my_channel =
        selective_channel<libbase::gf2>(TEST_BITMASK, test_channel);

    my_channel.transmit(tx_sequence, rx_sequence);

    BOOST_CHECK(rx_sequence.isequalto(expected_rx_sequence));
}
