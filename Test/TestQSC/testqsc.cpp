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
#define BOOST_TEST_MODULE qsc
#include <boost/test/included/unit_test.hpp>

#include "channel/qsc.h"
#include "gf.h"
#include "randgen.h"
#include <iostream>

// NOTE: This test class is needed because the qsc class has the pdf() function
// set to protected and we want to access it to test its functionality.
template <class G>
class TestQsc : public libcomm::qsc<G>
{
public:
    double pdf(const G& tx, const G& rx) const
    {
        return libcomm::qsc<G>::pdf(tx, rx);
    }
};

template <class gf>
void
validate_pdf_sums_to_one(const TestQsc<gf>& channel)
{
    namespace tt = boost::test_tools;

    for (auto i = 0; i < gf::elements(); ++i) {
        gf transmit_symbol = i;
        double sum_pdf = 0.0;
        for (auto j = 0; j < gf::elements(); ++j) {
            gf receive_symbol = j;
            sum_pdf += channel.pdf(transmit_symbol, receive_symbol);
        }

        BOOST_TEST(sum_pdf == 1.0, tt::tolerance(0.0001));
    }
}

template <class G>
void
ShowHistogram(libbase::vector<G>& x)
{
    const int N = x.size();
    const int q = G::elements();
    libbase::vector<int> f(q);
    f = 0;
    for (int i = 0; i < N; i++) {
        f(x(i))++;
    }
    assertalways(f.sum() == N);
    const double E = double(N) / double(q);
    for (int i = 0; i < q; i++) {
        std::cout << i << "\t" << f(i) << "\t[" << 100.0 * (f(i) - E) / E
                  << "%]" << std::endl;
    }
}

template <class G>
void
TestChannel(libcomm::channel<G>& chan, double p)
{
    using libbase::randgen;

    const int N = 100000;
    const int q = G::elements();
    std::cout << std::endl << chan.description() << std::endl;
    randgen r;
    r.seed(0);
    libbase::vector<G> tx(N);
    for (int i = 0; i < N; i++) {
        tx(i) = r.ival(q);
    }
    std::cout << "Tx:" << std::endl;
    ShowHistogram(tx);
    libbase::vector<G> rx(N);
    chan.seedfrom(r);
    chan.set_parameter(p);
    chan.transmit(tx, rx);
    std::cout << "Rx:" << std::endl;
    ShowHistogram(rx);
}

template <int m, int poly>
void
TestQSC()
{
    using libbase::gf;

    libcomm::qsc<gf<m, poly>> chan;
    TestChannel(chan, 0.1);
}

/*!
 * \brief   Test program for q-ary symmetric channel
 * \author  Johann Briffa
 */

BOOST_AUTO_TEST_CASE(test_qsc_channel)
{
    // TestQSC<1,0x3>();
    // TestQSC<2,0x7>();
    TestQSC<3, 0xB>();
    TestQSC<4, 0x13>();
    // TestQSC<5,0x25>();
    // TestQSC<6,0x43>();
    // TestQSC<7,0x89>();
    // TestQSC<8,0x11D>();
    // TestQSC<9,0x211>();
    // TestQSC<10,0x409>();
}

BOOST_AUTO_TEST_CASE(test_pdf_all_sum_to_one)
{
    const auto p_s = 0.1;

    { // GF2
        TestQsc<libbase::gf2> test_channel;
        test_channel.set_parameter(p_s);
        validate_pdf_sums_to_one(test_channel);
    }

    { // GF4
        TestQsc<libbase::gf4> test_channel;
        test_channel.set_parameter(p_s);
        validate_pdf_sums_to_one(test_channel);
    }
}
