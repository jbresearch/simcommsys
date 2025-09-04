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

#include "cputimer.h"
#include "modem/dminner.h"
#include "modem/tvb.h"
#include "randgen.h"
#include "version.h"

#include <boost/program_options.hpp>

#include <iostream>
#include <memory>
#include <string>

using std::cerr;
using std::cout;

using libbase::cputimer;
using libbase::matrix;
using libbase::vector;

using libcomm::blockmodem;
using libcomm::channel;
using libcomm::dminner;
using libcomm::tvb;

typedef std::unique_ptr<blockmodem<bool>> modem_ptr;
typedef std::unique_ptr<channel<bool>> channel_ptr;

modem_ptr
create_modem(bool tvb,
             bool use_float,
             bool deep,
             int tau,
             int n,
             int k,
             libbase::random& r)
{
    // thresholds for 'deep' trellis following
    const double th_inner = 0;
    const double th_outer = 0;
    modem_ptr mdm;
    if (tvb) {
        if (use_float) {
            if (deep) {
                mdm = modem_ptr(new tvb<bool, float, float>(
                    n, k, th_inner, th_outer));
            } else {
                mdm = modem_ptr(new tvb<bool, float, float>(n, k));
            }
        } else {
            if (deep) {
                mdm = modem_ptr(new tvb<bool, double, float>(
                    n, k, th_inner, th_outer));
            } else {
                mdm = modem_ptr(new tvb<bool, double, float>(n, k));
            }
        }
    } else {
        if (use_float) {
            if (deep) {
                mdm = modem_ptr(new dminner<float>(n, k, th_inner, th_outer));
            } else {
                mdm = modem_ptr(new dminner<float>(n, k));
            }
        } else {
            if (deep) {
                mdm = modem_ptr(new dminner<double>(n, k, th_inner, th_outer));
            } else {
                mdm = modem_ptr(new dminner<double>(n, k));
            }
        }
    }
    mdm->seedfrom(r);
    mdm->set_blocksize(libbase::size_type<libbase::vector>(tau));
    return mdm;
}

channel_ptr
create_channel(double Pe, libbase::random& r)
{
    channel_ptr chan = channel_ptr(new libcomm::qids<bool, float>);
    chan->seedfrom(r);
    chan->set_parameter(Pe);
    return chan;
}

vector<int>
create_encoded(int k, int tau, bool display = true)
{
    vector<int> encoded(tau);
    for (int i = 0; i < tau; i++) {
        encoded(i) = (i % (1 << k));
    }

    if (display) {
        cout << "Encoded: " << encoded << std::endl;
    }

    return encoded;
}

void
print_signal(const std::string desc, int n, vector<bool> tx)
{
    cout << desc << ":" << std::endl;

    for (int i = 0; i < tx.size(); i++) {
        cout << tx(i);
        if (i % n == n - 1 || i == tx.size() - 1) {
            cout << std::endl;
        } else {
            cout << "\t";
        }
    }

    cout << std::flush;
}

vector<bool>
modulate_encoded(int k,
                 int n,
                 blockmodem<bool>& mdm,
                 vector<int>& encoded,
                 bool display = true)
{
    vector<bool> tx;
    mdm.modulate(1 << k, encoded, tx);

    if (display) {
        print_signal("Tx", n, tx);
    }

    return tx;
}

vector<bool>
transmit_modulated(int n,
                   channel<bool>& chan,
                   const vector<bool>& tx,
                   bool display = true)
{
    vector<bool> rx;
    chan.transmit(tx, rx);

    if (display) {
        print_signal("Rx", n, rx);
    }

    return rx;
}

vector<vector<double>>
demodulate_encoded(channel<bool>& chan,
                   blockmodem<bool>& mdm,
                   const vector<bool>& rx,
                   bool display = true)
{
    // demodulate received signal
    vector<vector<double>> ptable;
    cputimer t;
    mdm.demodulate(chan, rx, ptable);
    t.stop();

    if (display) {
        cout << "Ptable: " << ptable << std::endl;
    }

    cout << "Time taken: " << t << std::endl;
    return ptable;
}

void
count_errors(const vector<int>& encoded, const vector<vector<double>>& ptable)
{
    const int tau = ptable.size();
    assert(tau > 0);
    assert(encoded.size() == tau);
    const int n = ptable(0).size();
    int count = 0;

    for (int i = 0; i < tau; i++) {
        assert(ptable(i).size() == n);
        // find the most likely candidate
        int d = 0;
        for (int j = 1; j < n; j++) {
            if (ptable(i)(j) > ptable(i)(d)) {
                d = j;
            }
        }
        // see if there is an error
        if (d != encoded(i)) {
            count++;
        }
    }
    if (count > 0) {
        cout << "Symbol errors: " << count << " ("
             << int(100 * count / double(tau)) << "%)" << std::endl;
    }
}

void
testcycle(bool tvb,
          bool use_float,
          bool deep,
          int seed,
          int n,
          int k,
          int tau,
          double Pe = 0,
          bool display = true)
{
    // create prng for seeding systems
    libbase::randgen prng;
    prng.seed(seed);
    // create modem and channel
    modem_ptr mdm = create_modem(tvb, use_float, deep, tau, n, k, prng);
    channel_ptr chan = create_channel(Pe, prng);
    cout << std::endl;
    cout << mdm->description() << std::endl;
    cout << chan->description() << std::endl;
    cout << "Block size: N = " << tau << std::endl;

    // define an alternating encoded sequence
    vector<int> encoded = create_encoded(k, tau, display);
    // modulate it using the previously created inner code
    vector<bool> tx = modulate_encoded(k, n, *mdm, encoded, display);
    // pass it through the channel
    vector<bool> rx = transmit_modulated(n, *chan, tx, display);
    // demodulate received signal
    vector<vector<double>> ptable =
        demodulate_encoded(*chan, *mdm, rx, display);
    // count errors
    count_errors(encoded, ptable);
}

/*!
 * \brief   Test program for DM inner decoder
 * \author  Johann Briffa
 */

int
main(int argc, char* argv[])
{
    // error probabilities corresponding to SNR = 12dB and 1dB respectively
    const double Plo = 9.00601e-09;
    const double Phi = 0.056282;

    // Set up user parameters
    bool tvb, use_float, deep;
    int type, seed, k, n, N;
    double Pe;
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "print this help message");
    desc.add_options()("type,t", po::value<int>(&type)->default_value(0), "test to run (0: single-cycle, 1: multiple-cycle)");
    desc.add_options()("tvb", po::bool_switch(&tvb), "use alternative (rather than classic) decoder");
    desc.add_options()("float", po::bool_switch(&use_float), "use 32-bit float (rather than 64-bit double)");
    desc.add_options()("deep", po::bool_switch(&deep), "do not perform any path truncation");
    desc.add_options()("seed", po::value<int>(&seed)->default_value(0), "seed for random generator");
    desc.add_options()("k", po::value<int>(&k)->default_value(4), "number of bits in message (input) symbol");
    desc.add_options()("n", po::value<int>(&n)->default_value(15), "number of bits in sparse (output) symbol");
    desc.add_options()("N", po::value<int>(&N)->default_value(5), "block size in symbols at encoder output (single-cycle test only)");
    desc.add_options()("Pe", po::value<double>(&Pe)->default_value(Plo), "block size in symbols at encoder output (single-cycle test only)");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Validate user parameters
    if (vm.count("help")) {
        cout << desc << std::endl;
        return 0;
    }

    // show revision information
    cout << "Build: " << SIMCOMMSYS_BUILD << std::endl;
    cout << "Version: " << SIMCOMMSYS_VERSION << std::endl;

    // do what the user asked for
    switch (type) {
    case 0:
        testcycle(tvb, use_float, deep, seed, n, k, N, Pe);
        break;

    case 1:
        // try short,medium,large codes for benchmarking at low error
        // probability
        testcycle(tvb, use_float, deep, seed, n, k, 10, Plo, false);
        testcycle(tvb, use_float, deep, seed, n, k, 100, Plo, false);
        testcycle(tvb, use_float, deep, seed, n, k, 1000, Plo, false);
        // try short,medium codes for benchmarking at high error probability
        testcycle(tvb, use_float, deep, seed, n, k, 10, Phi, false);
        testcycle(tvb, use_float, deep, seed, n, k, 100, Phi, false);
        break;

    default:
        cout << "Unknown type = " << type << std::endl;
        break;
    }

    return 0;
}
