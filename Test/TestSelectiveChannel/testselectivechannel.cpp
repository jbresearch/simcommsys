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
#include "channel/selective.h"

#include "channel/qsc.h"
#include "gf.h"

#define BOOST_TEST_MODULE SelectiveChannel
#include <boost/test/included/unit_test.hpp>

namespace libcomm
{
template <class Symbol>
class mock_channel : public libcomm::qsc<Symbol>
{
public:
    Symbol corrupt(const Symbol& s) { return mock_corrupt(s); }

    std::function<Symbol(Symbol)> mock_corrupt;
};
} // namespace libcomm

using namespace libcomm;

static const std::string TEST_BITMASK = "1110011";

BOOST_AUTO_TEST_CASE(initialise_bitmask_from_constructor)
{
    auto channel = selective<libbase::gf2>(TEST_BITMASK);

    BOOST_CHECK_EQUAL(TEST_BITMASK, channel.get_bitmask());
}

BOOST_AUTO_TEST_CASE(validate_bitmask)
{
    BOOST_CHECK_THROW(selective<libbase::gf2>("1210011"), libbase::load_error);
    BOOST_CHECK_THROW(selective<libbase::gf2>("1100z11"), libbase::load_error);
}

BOOST_AUTO_TEST_CASE(throw_exception_if_bitmask_and_tx_sequence_mismatch)
{
    auto channel = selective<libbase::gf2>(TEST_BITMASK);

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

    auto my_channel = selective<libbase::gf2>(TEST_BITMASK, test_channel);

    my_channel.transmit(tx_sequence, rx_sequence);

    BOOST_CHECK(rx_sequence.isequalto(expected_rx_sequence));
}
