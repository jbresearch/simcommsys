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
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "logrealfast.h"
#include "modem/dminner.h"
#include "channel/bsid.h"
#include "randgen.h"
#include "rvstatistics.h"
#include "itfunc.h"

#include <boost/program_options.hpp>

#include <iostream>
#include <exception>

namespace testbsid {

using std::cout;
using std::cerr;
using libbase::vector;
using libbase::randgen;
using libcomm::bsid;
using libcomm::sigspace;
namespace po = boost::program_options;

void visualtest()
   {
   // define an alternating input sequence
   const int tau = 5;
   vector<bool> tx(tau);
   for (int i = 0; i < tau; i++)
      tx(i) = (i & 1);
   cout << "Tx: " << tx << std::endl;
   // pass that through the channel
   vector<bool> rx1, rx2;
   // probability of error corresponding to SNR=12
   const double p = 9.00601e-09;
   // seed generator
   randgen prng;
   prng.seed(0);
   // channel1 is a substitution-only channel
   bsid channel1(tau);
   channel1.seedfrom(prng);
   channel1.set_parameter(p);
   channel1.set_ps(0.3);
   channel1.transmit(tx, rx1);
   cout << "Rx1: " << rx1 << std::endl;
   // channel1 is an insdel-only channel
   bsid channel2(tau);
   channel2.seedfrom(prng);
   channel2.set_parameter(p);
   channel2.set_pi(0.3);
   channel2.set_pd(0.3);
   channel2.transmit(tx, rx2);
   cout << "Rx2: " << rx2 << std::endl;
   }

void testtransmission(int tau, double p, bool ins, bool del, bool sub, bool src)
   {
   // define channel according to specifications
   bsid channel(sub, del, ins);
   randgen prng;
   prng.seed(0);
   channel.seedfrom(prng);
   channel.set_parameter(p);
   // run a number of transmissions with an all-zero source
   cout << "Testing on an all-" << (src ? "one" : "zero") << " source:"
         << std::endl;
   cout << "   type:\t";
   if (ins)
      cout << "insertions ";
   if (del)
      cout << "deletions ";
   if (sub)
      cout << "substitutions";
   cout << std::endl;
   cout << "      N:\t" << tau << std::endl;
   cout << "      p:\t" << p << std::endl;
   // show xmax,I for well-behaved channels
   if ((ins && del) || tau <= 100)
      {
      cout << "   xmax:\t" << channel.compute_xmax(tau) << std::endl;
      cout << "      I:\t" << channel.compute_I(tau) << std::endl;
      }
   // define input sequence
   vector<bool> tx(tau);
   tx = src;
   // run simulation
   vector<bool> rx;
   libbase::rvstatistics drift, zeros, ones;
   for (int i = 0; i < 1000; i++)
      {
      channel.transmit(tx, rx);
      drift.insert(rx.size() - tau);
      int count = 0;
      for (int j = 0; j < rx.size(); j++)
         count += rx(j);
      ones.insert(count);
      zeros.insert(rx.size() - count);
      }
   // show results
   cout << "   Value\tMean\tSigma" << std::endl;
   cout << "  Drift:\t" << drift.mean() << "\t" << drift.sigma() << std::endl;
   cout << "  Zeros:\t" << zeros.mean() << "\t" << zeros.sigma() << std::endl;
   cout << "   Ones:\t" << ones.mean() << "\t" << ones.sigma() << std::endl;
   cout << std::endl;
   }

double estimate_drift_sd(int tau, double Pi, double Pd)
   {
   // define channel according to specifications
   bsid channel;
   randgen prng;
   prng.seed(0);
   channel.seedfrom(prng);
   channel.set_pi(Pi);
   channel.set_pd(Pd);
   // define input sequence
   vector<bool> tx(tau);
   tx = 0;
   // run simulation
   vector<bool> rx;
   libbase::rvstatistics drift;
   for (int i = 0; i < 1000; i++)
      {
      channel.transmit(tx, rx);
      drift.insert(rx.size() - tau);
      }
   // return result
   return drift.sigma();
   }

int estimate_xmax(int tau, double Pi, double Pd)
   {
   // determine required multiplier
   const double factor = libbase::Qinv(bsid::metric_computer::Pr / 2.0);
   // main computation
   int xmax = int(ceil(factor * estimate_drift_sd(tau, Pi, Pd)));
   // return result
   return xmax;
   }

void compute_parameters(int tau, double p, bool ins, bool del, bool sim)
   {
   const double Pi = ins ? p : 0;
   const double Pd = del ? p : 0;
   cout << p;
   const int I = bsid::metric_computer::compute_I(tau, Pi, 0);
   cout << "\t" << I;
   const int xmax_auto = bsid::metric_computer::compute_xmax(tau, Pi, Pd);
   cout << "\t" << xmax_auto;
   const int xmax_davey = bsid::metric_computer::compute_xmax_with(
         &bsid::metric_computer::compute_drift_prob_davey, tau, Pi, Pd);
   cout << "\t" << xmax_davey;
   try
      {
      const int xmax_exact = bsid::metric_computer::compute_xmax_with(
            &bsid::metric_computer::compute_drift_prob_exact, tau, Pi, Pd);
      cout << "\t" << xmax_exact;
      }
   catch (std::exception& e)
      {
      cout << "\tEXC (" << e.what() << ")";
      }
   if (sim)
      {
      const int xmax_est = estimate_xmax(tau, Pi, Pd);
      cout << "\t" << xmax_est;
      }
   cout << std::endl;
   }

void compute_parameters(int tau, bool ins, bool del, bool sim)
   {
   const double pstart = 1e-4;
   const double pstop = 0.5;
   const double pmul = pow(10, 1.0 / 10);
   cout << "p\tI\txmax_auto\txmax_davey\txmax_exact\txmax_est" << std::endl;
   for (double p = pstart; p < pstop; p *= pmul)
      compute_parameters(tau, p, ins, del, sim);
   }

/*!
 * \brief   Test program for BSID channel
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

int main(int argc, char *argv[])
   {
   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("ins,i", po::value<bool>()->default_value(true),
         "allow insertions");
   desc.add_options()("del,d", po::value<bool>()->default_value(true),
         "allow deletions");
   desc.add_options()("sub,s", po::value<bool>()->default_value(false),
         "allow substitutions");
   desc.add_options()("blocksize,N", po::value<int>(), "block size in bits");
   desc.add_options()("parameter,r", po::value<double>(), "channel parameter");
   desc.add_options()("simulate", po::bool_switch(),
         "perform simulation to estimate xmax");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      cout << desc << std::endl;
      return 0;
      }

   // if specific settings given by user, run that test
   if (vm.count("blocksize"))
      {
      const int N = vm["blocksize"].as<int> ();
      const bool ins = vm["ins"].as<bool> ();
      const bool del = vm["del"].as<bool> ();
      const bool sub = vm["sub"].as<bool> ();
      // if a specific parameter is given, just run a sim at that point
      if (vm.count("parameter"))
         {
         const double p = vm["parameter"].as<double> ();
         testtransmission(N, p, ins, del, sub, 0);
         }
      else // otherwise show parameter computation for range of parameter
         {
         const bool sim = vm["simulate"].as<bool> ();
         compute_parameters(N, ins, del, sim);
         }
      }
   else // otherwise run the default set
      {
      // create a test sequence and test BSID transmission
      visualtest();
      // test insertion-only channels
      testtransmission(1000, 0.01, true, false, false, 0);
      testtransmission(1000, 0.25, true, false, false, 0);
      // test deletion-only channels
      testtransmission(1000, 0.01, false, true, false, 0);
      testtransmission(1000, 0.25, false, true, false, 0);
      // test substitution-only channels
      testtransmission(1000, 0.1, false, false, true, 0);
      testtransmission(1000, 0.1, false, false, true, 1);
      // test insertion-deletion channels
      testtransmission(1000, 0.01, true, true, false, 0);
      testtransmission(1000, 0.25, true, true, false, 0);
      }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testbsid::main(argc, argv);
   }
