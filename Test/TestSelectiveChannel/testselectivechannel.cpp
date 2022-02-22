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

using namespace libcomm;

using symbol = libbase::gf2;

static const std::string TEST_BITMASK = "1110011";

BOOST_AUTO_TEST_CASE(initialise_bitmask_from_constructor)
{
    auto channel = selective<symbol>(TEST_BITMASK);

    BOOST_CHECK_EQUAL(TEST_BITMASK, channel.get_bitmask());
}

BOOST_AUTO_TEST_CASE(validate_bitmask)
{
    BOOST_CHECK_THROW(selective<symbol>("1210011"), libbase::load_error);
    BOOST_CHECK_THROW(selective<symbol>("1100z11"), libbase::load_error);
}

BOOST_AUTO_TEST_CASE(set_secondary_channel_parameter)
{
    auto primary_channel = std::make_shared<libcomm::qsc<symbol>>();
    auto secondary_channel = std::make_shared<libcomm::qsc<symbol>>();
    const auto secondary_channel_parameter = double(0.1);

    auto channel = selective<symbol>(TEST_BITMASK,
                                     primary_channel,
                                     secondary_channel,
                                     secondary_channel_parameter);

    BOOST_TEST(secondary_channel->get_parameter() ==
                   secondary_channel_parameter,
               boost::test_tools::tolerance(0.0001));
}

BOOST_AUTO_TEST_CASE(throw_exception_if_bitmask_and_tx_sequence_mismatch)
{
    auto channel = selective<symbol>(TEST_BITMASK);

    const auto tx_vector = std::vector<int>{1, 0, 1, 0};
    const auto tx_sequence = libbase::vector<symbol>(tx_vector);
    auto rx_sequence = libbase::vector<symbol>();

    BOOST_CHECK_THROW(channel.transmit(tx_sequence, rx_sequence),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(split_sequence_based_on_bitmask)
{
    const auto sequence =
        libbase::vector<symbol>(std::vector<int>{1, 0, 1, 0, 1, 0, 1});

    const auto expected_primary =
        libbase::vector<symbol>(std::vector<int>{1, 0, 1, 0, 1});

    const auto expected_secondary =
        libbase::vector<symbol>(std::vector<int>{0, 1});

    auto my_channel = selective<symbol>(TEST_BITMASK);

    auto split_sequence = my_channel.split_based_on_bitmask(sequence);

    BOOST_CHECK(expected_primary.isequalto(split_sequence.first));
    BOOST_CHECK(expected_secondary.isequalto(split_sequence.second));
}

BOOST_AUTO_TEST_CASE(merge_sequence_based_on_bitmask)
{
    const auto sequence =
        libbase::vector<symbol>(std::vector<int>{1, 1, 1, 0, 0, 1, 1});

    const auto primary =
        libbase::vector<symbol>(std::vector<int>{1, 1, 1, 1, 1});

    const auto secondary = libbase::vector<symbol>(std::vector<int>{0, 0});

    auto my_channel = selective<symbol>(TEST_BITMASK);

    auto merged_sequence = libbase::vector<symbol>();

    my_channel.merge_based_on_bitmask(primary, secondary, merged_sequence);

    BOOST_CHECK(sequence.isequalto(merged_sequence));
}

BOOST_AUTO_TEST_CASE(test_transmission)
{
    class mock_channel : public libcomm::qsc<symbol>
    {
    public:
        symbol corrupt(const symbol& s) override
        {
            ++num_times_corrupt_called;
            corrupt_symbol_call_sequence.push_back(s);
            auto new_symbol = s;
            return new_symbol += 1;
        }

        int num_times_corrupt_called = 0;
        std::vector<symbol> corrupt_symbol_call_sequence;
    };

    const auto tx_sequence =
        libbase::vector<symbol>(std::vector<int>{1, 0, 1, 1, 1, 1, 1});
    const auto expected_rx_sequence =
        libbase::vector<symbol>(std::vector<int>{0, 1, 0, 1, 1, 0, 0});

    auto primary_channel = std::make_shared<mock_channel>();

    auto secondary_channel = std::make_shared<libcomm::qsc<symbol>>();
    libbase::randgen r;
    r.seed(0);
    secondary_channel->seedfrom(r);

    auto selective_channel = selective<symbol>(
        TEST_BITMASK, primary_channel, secondary_channel, 0.0);

    auto rx_sequence = libbase::vector<symbol>();
    selective_channel.transmit(tx_sequence, rx_sequence);

    BOOST_CHECK(rx_sequence.isequalto(expected_rx_sequence));
    BOOST_CHECK_EQUAL(primary_channel->num_times_corrupt_called, 5);
    BOOST_CHECK(primary_channel->corrupt_symbol_call_sequence ==
                (std::vector<symbol>{1, 0, 1, 1, 1}));
}
