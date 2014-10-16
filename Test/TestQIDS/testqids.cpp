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

#include "logrealfast.h"
#include "mpreal.h"
#include "modem/dminner.h"
#include "channel/qids.h"
#include "randgen.h"
#include "rvstatistics.h"
#include "itfunc.h"
#include "cputimer.h"
#include "field_utils.h"

#include <boost/program_options.hpp>

#include <iostream>
#include <exception>

namespace testqids {

using std::cout;
using std::cerr;
using libbase::vector;
using libbase::randgen;
using libbase::logrealfast;
using libbase::mpreal;
using libcomm::sigspace;
namespace po = boost::program_options;

template <class S>
void createsource(vector<S>& tx, randgen& r, const int N)
   {
   assert(tx.size() == N);
   const int q = libcomm::field_utils<S>::elements();
   for (int i = 0; i < N; i++)
      tx(i) = r.ival(q);
   }

void test_visual()
   {
   // define an alternating input sequence
   const int tau = 5;
   vector<bool> tx(tau);
   for (int i = 0; i < tau; i++)
      tx(i) = (i & 1);
   cout << "Tx: " << tx << std::endl;
   // pass that through the channel
   vector<bool> rx1, rx2;
   // seed generator
   randgen prng;
   prng.seed(0);
   // channel1 is a substitution-only channel
   libcomm::qids<bool, float> channel1;
   channel1.seedfrom(prng);
   channel1.set_parameter(0);
   channel1.set_ps(0.3);
   channel1.set_blocksize(tau);
   channel1.transmit(tx, rx1);
   cout << "Rx1: " << rx1 << std::endl;
   // channel1 is an insdel-only channel
   libcomm::qids<bool, float> channel2;
   channel2.seedfrom(prng);
   channel2.set_parameter(0);
   channel2.set_pi(0.3);
   channel2.set_pd(0.3);
   channel1.set_blocksize(tau);
   channel2.transmit(tx, rx2);
   cout << "Rx2: " << rx2 << std::endl;
   }

void test_transmission(int tau, double p, bool ins, bool del, bool sub,
      bool src, double Pr)
   {
   // define channel according to specifications
   libcomm::qids<bool, float> channel(sub, del, ins);
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
      cout << "   xmax:\t" << channel.compute_xmax(tau, Pr) << std::endl;
      cout << "      I:\t" << channel.compute_I(tau, Pr) << std::endl;
      }
   // show end-of-frame priors
   libbase::vector<double> eof_pdf;
   libbase::size_type<libbase::vector> offset;
   channel.get_drift_pdf(tau, Pr, eof_pdf, offset);
   cout << "eof prior pdf:" << std::endl;
   cout << "\ti\tPr(i)" << std::endl;
   for (int i = 0; i < eof_pdf.size(); i++)
      cout << "\t" << i - offset << "\t" << eof_pdf(i) << std::endl;
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
   libcomm::qids<bool, float> channel;
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

int estimate_xmax(int tau, double Pi, double Pd, double Pr)
   {
   // determine required multiplier
   const double factor = libbase::Qinv(Pr / 2.0);
   // main computation
   int xmax = int(ceil(factor * estimate_drift_sd(tau, Pi, Pd)));
   // return result
   return xmax;
   }

void compute_statespace(int tau, int n, double p, bool ins, bool del, bool sim,
      double Pr)
   {
   using libcomm::qids_utils;
   const double Pi = ins ? p : 0;
   const double Pd = del ? p : 0;
   cout << p;

   // Davey-MacKay Gaussian approximation
   const int I_dm = 2;
   cout << "\t" << I_dm;
   const double error_I_dm = qids_utils::multiply_error_probability(
         qids_utils::compute_outofbounds_with(
               qids_utils::compute_drift_prob_exact, 1, Pi, Pd, I_dm, -1), tau);
   cout << "\t" << error_I_dm;
   if (Pi == Pd)
      {
      const int xmax_dm = qids_utils::compute_xmax_with(
            qids_utils::compute_drift_prob_davey, tau, Pi, Pd, Pr);
      cout << "\t" << xmax_dm;
      const double error_xmax_dm = qids_utils::compute_outofbounds_with(
            qids_utils::compute_drift_prob_exact, tau, Pi, Pd, xmax_dm,
            -xmax_dm);
      cout << "\t" << error_xmax_dm;
      }
   else
      {
      cout << "\tN/A\tN/A";
      }

   // Exact computation - I/xmax method
   const int I_exact = qids_utils::compute_I(tau, Pi, Pr);
   cout << "\t" << I_exact;
   const double error_I_exact = qids_utils::multiply_error_probability(
         qids_utils::compute_outofbounds_with(
               qids_utils::compute_drift_prob_exact, 1, Pi, Pd, I_exact, -1),
         tau);
   cout << "\t" << error_I_exact;
   const int xmax_exact = qids_utils::compute_xmax(tau, Pi, Pd, Pr);
   cout << "\t" << xmax_exact;
   const double error_xmax_exact = qids_utils::compute_outofbounds_with(
         qids_utils::compute_drift_prob_exact, tau, Pi, Pd, xmax_exact,
         -xmax_exact);
   cout << "\t" << error_xmax_exact;

   // Exact computation - m1/mtau method
   int m1_min, m1_max;
   qids_utils::compute_limits_with(qids_utils::compute_drift_prob_exact, 1, Pi,
         Pd, qids_utils::divide_error_probability(Pr, tau), m1_min, m1_max);
   cout << "\t" << m1_min;
   cout << "\t" << m1_max;
   const double error_m1 = qids_utils::multiply_error_probability(
         qids_utils::compute_outofbounds_with(
               qids_utils::compute_drift_prob_exact, 1, Pi, Pd, m1_max,
               m1_min), tau);
   cout << "\t" << error_m1;

   const int N = tau / n;
   assertalways(tau == N * n);
   int mn_min, mn_max;
   qids_utils::compute_limits_with(qids_utils::compute_drift_prob_exact, n, Pi,
         Pd, qids_utils::divide_error_probability(Pr, N), mn_min, mn_max);
   cout << "\t" << mn_min;
   cout << "\t" << mn_max;
   const double error_mn = qids_utils::multiply_error_probability(
         qids_utils::compute_outofbounds_with(
               qids_utils::compute_drift_prob_exact, n, Pi, Pd, mn_max,
               mn_min), N);
   cout << "\t" << error_mn;

   int mtau_min, mtau_max;
   qids_utils::compute_limits_with(qids_utils::compute_drift_prob_exact, tau,
         Pi, Pd, Pr, mtau_min, mtau_max);
   cout << "\t" << mtau_min;
   cout << "\t" << mtau_max;
   const double error_mtau = qids_utils::compute_outofbounds_with(
         qids_utils::compute_drift_prob_exact, tau, Pi, Pd, mtau_max, mtau_min);
   cout << "\t" << error_mtau;

   // determine xmax empirically, if requested
   if (sim)
      {
      const int xmax_est = estimate_xmax(tau, Pi, Pd, Pr);
      cout << "\t" << xmax_est;
      }

   cout << std::endl;
   }

void compute_drift_pdf(int tau, double p, bool ins, bool del, double Pr)
   {
   using libcomm::qids_utils;
   const double Pi = ins ? p : 0;
   const double Pd = del ? p : 0;
   cout << "p = " << p << std::endl;
   int mtau_min, mtau_max;
   qids_utils::compute_limits(tau, Pi, Pd, Pr, mtau_min, mtau_max);
   cout << "m\tP(m)" << std::endl;
   for (int m = mtau_min; m <= mtau_max; m++)
      {
      const double prob = qids_utils::compute_drift_prob_exact(m, tau, Pi,
            Pd);
      cout << m << "\t" << prob << std::endl;
      }
   }

template <class real>
void test_receiver(int tau, double p, bool ins, bool del, bool sub, double Pr)
   {
   // define channel according to specifications
   typedef libcomm::qids<bool, real> channel_t;
   channel_t channel(sub, del, ins);
   randgen prng;
   prng.seed(0);
   channel.seedfrom(prng);
   channel.set_parameter(p);
   channel.set_blocksize(tau);
   // determine state-space parameters
   int mtau_min, mtau_max;
   channel.compute_limits(tau, Pr, mtau_min, mtau_max);
   // define input sequences and output tables
   vector<bool> tx(tau);
   vector<bool> rx(tau + mtau_max);
   tx = 0;
   rx = 0;
   vector<real> ptable_trellis(mtau_max - mtau_min + 1);
   vector<real> ptable_lattice(mtau_max - mtau_min + 1);
   vector<real> ptable_corridor(mtau_max - mtau_min + 1);
   // compare results
   const int count = 1000;
   libbase::rvstatistics mse_trellis, mse_corridor;
   for (int i = 0; i < count; i++)
      {
      // generate tx/rx vectors
      createsource(tx, prng, tau);
      vector<bool> temp;
      channel.transmit(tx, temp);
      rx = 0;
      rx.segment(0, temp.size()) = temp;
      // determine results with all receivers
      const typename channel_t::metric_computer& computer =
            dynamic_cast<const typename channel_t::metric_computer&>(channel.get_computer());
      computer.receive_trellis(tx, rx, ptable_trellis);
      computer.receive_lattice(tx, rx, ptable_lattice);
      computer.receive_lattice_corridor(tx, rx, ptable_corridor);
      // compare results
      mse_trellis.insert(
            (ptable_trellis - ptable_lattice).sumsq() / (mtau_max - mtau_min + 1));
      mse_corridor.insert(
            (ptable_corridor - ptable_lattice).sumsq() / (mtau_max - mtau_min + 1));
      }
   // show results
   cout << p << "\t";
   cout << mse_trellis.mean() << "\t" << mse_trellis.sigma() << "\t";
   cout << mse_corridor.mean() << "\t" << mse_corridor.sigma() << std::endl;
   }

template <class real, class reference>
void test_precision(int tau, double p, bool ins, bool del, bool sub, double Pr)
   {
   // define two identical channels according to specifications
   typedef libcomm::qids<bool, real> channel_real_t;
   typedef libcomm::qids<bool, reference> channel_reference_t;
   channel_real_t channel_real(sub, del, ins);
   channel_reference_t channel_reference(sub, del, ins);
   randgen prng;
   prng.seed(0);
   channel_real.seedfrom(prng);
   channel_real.set_parameter(p);
   channel_real.set_blocksize(tau);
   prng.seed(0);
   channel_reference.seedfrom(prng);
   channel_reference.set_parameter(p);
   channel_reference.set_blocksize(tau);
   // determine state-space parameters
   int mtau_min, mtau_max;
   channel_real.compute_limits(tau, Pr, mtau_min, mtau_max);
   // define input sequences and output tables
   vector<bool> tx(tau);
   vector<bool> rx(tau + mtau_max);
   tx = 0;
   rx = 0;
   vector<real> ptable_trellis_real(mtau_max - mtau_min + 1);
   vector<real> ptable_corridor_real(mtau_max - mtau_min + 1);
   vector<reference> ptable_trellis_reference(mtau_max - mtau_min + 1);
   vector<reference> ptable_corridor_reference(mtau_max - mtau_min + 1);
   // compare results
   const int count = 1000;
   libbase::rvstatistics mse_trellis, mse_corridor;
   for (int i = 0; i < count; i++)
      {
      // generate tx/rx vectors
      createsource(tx, prng, tau);
      vector<bool> temp;
      channel_real.transmit(tx, temp);
      rx = 0;
      rx.segment(0, temp.size()) = temp;
      // determine results with two receivers - test system
      const typename channel_real_t::metric_computer& computer_real =
            dynamic_cast<const typename channel_real_t::metric_computer&>(channel_real.get_computer());
      computer_real.receive_trellis(tx, rx, ptable_trellis_real);
      computer_real.receive_lattice_corridor(tx, rx, ptable_corridor_real);
      // determine results with two receivers - reference system
      const typename channel_reference_t::metric_computer& computer_reference =
            dynamic_cast<const typename channel_reference_t::metric_computer&>(channel_reference.get_computer());
      computer_reference.receive_trellis(tx, rx, ptable_trellis_reference);
      computer_reference.receive_lattice_corridor(tx, rx,
            ptable_corridor_reference);
      // compare results
      mse_trellis.insert(
            (vector<reference>(ptable_trellis_real) - ptable_trellis_reference).sumsq()
                  / (mtau_max - mtau_min + 1));
      mse_corridor.insert(
            (vector<reference>(ptable_corridor_real) - ptable_corridor_reference).sumsq()
                  / (mtau_max - mtau_min + 1));
      }
   // show results
   cout << p << "\t";
   cout << mse_trellis.mean() << "\t" << mse_trellis.sigma() << "\t";
   cout << mse_corridor.mean() << "\t" << mse_corridor.sigma() << std::endl;
   }

template <class real>
void compute_timings(int tau, double p, bool ins, bool del, bool sub, double Pr)
   {
   typedef typename libcomm::qids<bool, real>::metric_computer metric_computer;
   using libcomm::qids_utils;
   const double Pi = ins ? p : 0;
   const double Pd = del ? p : 0;
   const double Ps = sub ? p : 0;
   // determine state-space parameters
   cout << p;
   int mtau_min, mtau_max;
   qids_utils::compute_limits(tau, Pi, Pd, Pr, mtau_min, mtau_max);
   cout << "\t" << mtau_min << "\t" << mtau_max;
   int m1_min, m1_max;
   qids_utils::compute_limits(1, Pi, Pd,
         qids_utils::divide_error_probability(Pr, tau), m1_min, m1_max);
   cout << "\t" << m1_min << "\t" << m1_max;
   // set up metric computer
   metric_computer computer;
   computer.init();
   computer.precompute(Ps, Pd, Pi, tau, mtau_min, mtau_max, m1_min, m1_max);
   // define input sequences and output table
   vector<bool> tx(tau);
   vector<bool> rx(tau + mtau_max);
   tx = 0;
   rx = 0;
   vector<real> ptable(mtau_max - mtau_min + 1);
   // determine timings - setup
   libbase::cputimer t;
   int j;
   const int count = 10000;
   const double cutoff = 0.01;

   // determine timings - trellis
   t.start();
   for (j = 0; t.elapsed() < cutoff; j++)
      for (int i = 0; i < count; i++)
         computer.receive_trellis(tx, rx, ptable);
   t.stop();
   cout << "\t" << t.elapsed() / (j * count);
   // determine timings - lattice
   t.start();
   for (j = 0; t.elapsed() < cutoff; j++)
      for (int i = 0; i < count; i++)
         computer.receive_lattice(tx, rx, ptable);
   t.stop();
   cout << "\t" << t.elapsed() / (j * count);
   // determine timings - lattice corridor
   t.start();
   for (j = 0; t.elapsed() < cutoff; j++)
      for (int i = 0; i < count; i++)
         computer.receive_lattice_corridor(tx, rx, ptable);
   t.stop();
   cout << "\t" << t.elapsed() / (j * count);

   // end
   cout << std::endl;
   }

/*!
 * \brief   Test program for QIDS channel
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   // test type
   desc.add_options()("default", "default test set");
   desc.add_options()("visual", "visual test");
   desc.add_options()("transmission", "simulate and validate transmission");
   desc.add_options()("receiver",
         "determine equivalence for trellis and lattice (full and corridor) "
               "receiver, in single precision");
   desc.add_options()("precision",
         "determine equivalence for trellis and lattice corridor receivers "
               "in single and double precision");
   desc.add_options()("timings",
         "determine timings for lattice and trellis receiver");
   desc.add_options()("state-space", "determine state-space limits");
   desc.add_options()("drift-pdf", "determine drift probability distribution");
   // test-specific parameters
   desc.add_options()("simulate", po::bool_switch(),
         "when determining state-space limits, also perform simulation to "
               "estimate xmax empirically");
   desc.add_options()("blocksize,N", po::value<int>(),
         "block size in channel symbols");
   desc.add_options()("codesize,n", po::value<int>(),
         "codeword size in channel symbols");
   desc.add_options()("parameter,r", po::value<double>(),
         "channel parameter to use; if not specified for timings and "
               "state-space, use pre-set range");
   desc.add_options()("ins,i", po::value<bool>()->default_value(true),
         "allow insertions");
   desc.add_options()("del,d", po::value<bool>()->default_value(true),
         "allow deletions");
   desc.add_options()("sub,s", po::value<bool>()->default_value(false),
         "allow substitutions");
   desc.add_options()("exclusion,e", po::value<double>()->default_value(1e-10),
         "probability of event outside range");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help")
         || (vm.count("default") + vm.count("visual") + vm.count("transmission")
               + vm.count("receiver") + vm.count("precision")
               + vm.count("timings") + vm.count("state-space")
               + vm.count("drift-pdf") != 1))
      {
      cout << desc << std::endl;
      return 0;
      }

   // default test set
   if (vm.count("default"))
      {
      const double Pr = vm["exclusion"].as<double>();
      // test insertion-only channels
      test_transmission(1000, 0.01, true, false, false, 0, Pr);
      test_transmission(1000, 0.25, true, false, false, 0, Pr);
      // test deletion-only channels
      test_transmission(1000, 0.01, false, true, false, 0, Pr);
      test_transmission(1000, 0.25, false, true, false, 0, Pr);
      // test substitution-only channels
      test_transmission(1000, 0.1, false, false, true, 0, Pr);
      test_transmission(1000, 0.1, false, false, true, 1, Pr);
      // test insertion-deletion channels
      test_transmission(1000, 0.01, true, true, false, 0, Pr);
      test_transmission(1000, 0.25, true, true, false, 0, Pr);
      }
   // visual test
   else if (vm.count("visual"))
      {
      // create a test sequence and test QIDS transmission
      test_visual();
      }
   // simulate and validate transmission
   else if (vm.count("transmission"))
      {
      const int N = vm["blocksize"].as<int>();
      const double p = vm["parameter"].as<double>();
      const bool ins = vm["ins"].as<bool>();
      const bool del = vm["del"].as<bool>();
      const bool sub = vm["sub"].as<bool>();
      const double Pr = vm["exclusion"].as<double>();
      test_transmission(N, p, ins, del, sub, 0, Pr);
      }
   // determine equivalence for trellis and lattice (full and corridor) receiver
   else if (vm.count("receiver"))
      {
      const int N = vm["blocksize"].as<int>();
      const bool ins = vm["ins"].as<bool>();
      const bool del = vm["del"].as<bool>();
      const bool sub = vm["sub"].as<bool>();
      const double Pr = vm["exclusion"].as<double>();
      // if a specific parameter is given, determine equivalence at that point
      // otherwise determine equivalence for pre-set range
      const double pstart =
            vm.count("parameter") ? vm["parameter"].as<double>() : 1e-4;
      const double pstop =
            vm.count("parameter") ? vm["parameter"].as<double>() : 0.5;
      const double pmul = pow(10.0, 1.0 / 10);
      cout << "Mean Square Error compared with lattice:" << std::endl;
      cout << "p\ttrellis (μ)\ttrellis (σ)\tcorridor (μ)\tcorridor (σ)"
            << std::endl;
      for (double p = pstart; p <= pstop; p *= pmul)
         test_receiver<float>(N, p, ins, del, sub, Pr);
      }
   // determine equivalence for receivers in single and double precision
   else if (vm.count("precision"))
      {
      const int N = vm["blocksize"].as<int>();
      const bool ins = vm["ins"].as<bool>();
      const bool del = vm["del"].as<bool>();
      const bool sub = vm["sub"].as<bool>();
      const double Pr = vm["exclusion"].as<double>();
      // if a specific parameter is given, determine equivalence at that point
      // otherwise determine equivalence for pre-set range
      const double pstart =
            vm.count("parameter") ? vm["parameter"].as<double>() : 1e-4;
      const double pstop =
            vm.count("parameter") ? vm["parameter"].as<double>() : 0.5;
      const double pmul = pow(10.0, 1.0 / 10);
      cout << "Mean Square Error for single precision (wrt double):"
            << std::endl;
      cout << "p\ttrellis (μ)\ttrellis (σ)\tcorridor (μ)\tcorridor (σ)"
            << std::endl;
      for (double p = pstart; p <= pstop; p *= pmul)
         test_precision<float, double>(N, p, ins, del, sub, Pr);
      }
   // determine timings for lattice and trellis receiver
   else if (vm.count("timings"))
      {
      const int N = vm["blocksize"].as<int>();
      const bool ins = vm["ins"].as<bool>();
      const bool del = vm["del"].as<bool>();
      const bool sub = vm["sub"].as<bool>();
      const double Pr = vm["exclusion"].as<double>();
      // if a specific parameter is given, determine equivalence at that point
      // otherwise determine equivalence for pre-set range
      const double pstart =
            vm.count("parameter") ? vm["parameter"].as<double>() : 1e-4;
      const double pstop =
            vm.count("parameter") ? vm["parameter"].as<double>() : 0.5;
      const double pmul = pow(10.0, 1.0 / 10);
      cout << "Timings (single precision):" << std::endl;
      cout << "p\tmtau_min\tmtau_max\tm1_min\tm1_max\ttrellis\tlattice\tcorridor" << std::endl;
      for (double p = pstart; p <= pstop; p *= pmul)
         compute_timings<float>(N, p, ins, del, sub, Pr);
      }
   // determine state-space limits
   else if (vm.count("state-space"))
      {
      const int N = vm["blocksize"].as<int>();
      const int n = vm["codesize"].as<int>();
      const bool ins = vm["ins"].as<bool>();
      const bool del = vm["del"].as<bool>();
      const bool sim = vm["simulate"].as<bool>();
      const double Pr = vm["exclusion"].as<double>();
      // if a specific parameter is given, determine equivalence at that point
      // otherwise determine equivalence for pre-set range
      const double pstart =
            vm.count("parameter") ? vm["parameter"].as<double>() : 1e-4;
      const double pstop =
            vm.count("parameter") ? vm["parameter"].as<double>() : 0.5;
      const double pmul = pow(10.0, 1.0 / 10);
      cout << "p" << "\tI_dm\tp(e)\txmax_dm\tp(e)"
            << "\tI_exact\tp(e)\txmax_exact\tp(e)" << "\tm1_min\tm1_max\tp(e)"
            << "\tmn_min\tmn_max\tp(e)" << "\tmtau_min\tmtau_max\tp(e)";
      if (sim)
         cout << "\txmax_est";
      cout << std::endl;
      for (double p = pstart; p <= pstop; p *= pmul)
         compute_statespace(N, n, p, ins, del, sim, Pr);
      }
   // determine drift probability distribution
   else if (vm.count("drift-pdf"))
      {
      const int N = vm["blocksize"].as<int>();
      const bool ins = vm["ins"].as<bool>();
      const bool del = vm["del"].as<bool>();
      const double p = vm["parameter"].as<double>();
      const double Pr = vm["exclusion"].as<double>();
      compute_drift_pdf(N, p, ins, del, Pr);
      }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testqids::main(argc, argv);
   }
