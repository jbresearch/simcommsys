#include "timer.h"
#include "randgen.h"
#include "logrealfast.h"
#include "dminner.h"
#include "dminner2.h"
#include "bsid.h"

#include <memory>
#include <iostream>
#include <string>

namespace testwmdemodulation {

using std::cout;
using std::cerr;

using libbase::timer;
using libbase::vector;
using libbase::matrix;
using libbase::logrealfast;

using libcomm::blockmodem;
using libcomm::channel;
using libcomm::dminner;
using libcomm::dminner2;

typedef std::auto_ptr<blockmodem<bool> > modem_ptr;
typedef std::auto_ptr<channel<bool> > channel_ptr;

modem_ptr create_modem(bool decoder, bool math, bool deep, int tau, int n,
      int k, libbase::random& r)
   {
   modem_ptr mdm;
   if (decoder)
      {
      if (math)
         {
         if (deep)
            mdm = modem_ptr(new dminner2<double, true> (n, k, 1e-21, 1e-12));
         else
            mdm = modem_ptr(new dminner2<double, true> (n, k));
         }
      else
         {
         if (deep)
            mdm = modem_ptr(new dminner2<logrealfast, false> (n, k, 1e-21,
                  1e-12));
         else
            mdm = modem_ptr(new dminner2<logrealfast, false> (n, k));
         }
      }
   else
      {
      if (math)
         {
         if (deep)
            mdm = modem_ptr(new dminner<double, true> (n, k, 1e-21, 1e-12));
         else
            mdm = modem_ptr(new dminner<double, true> (n, k));
         }
      else
         {
         if (deep)
            mdm = modem_ptr(
                  new dminner<logrealfast, false> (n, k, 1e-21, 1e-12));
         else
            mdm = modem_ptr(new dminner<logrealfast, false> (n, k));
         }
      }
   mdm->seedfrom(r);
   mdm->set_blocksize(libbase::size_type<libbase::vector>(tau));
   return mdm;
   }

channel_ptr create_channel(double Pe, libbase::random& r)
   {
   channel_ptr chan = channel_ptr(new libcomm::bsid);
   chan->seedfrom(r);
   chan->set_parameter(Pe);
   return chan;
   }

vector<int> create_encoded(int k, int tau, bool display = true)
   {
   vector<int> encoded(tau);
   for (int i = 0; i < tau; i++)
      encoded(i) = (i % (1 << k));
   if (display)
      cout << "Encoded: " << encoded << "\n" << std::flush;
   return encoded;
   }

void print_signal(const std::string desc, int n, vector<bool> tx)
   {
   cout << desc << ":\n";
   for (int i = 0; i < tx.size(); i++)
      cout << tx(i) << ((i % n == n - 1 || i == tx.size() - 1) ? "\n" : "\t");
   cout << std::flush;
   }

vector<bool> modulate_encoded(int k, int n, blockmodem<bool>& mdm,
      vector<int>& encoded, bool display = true)
   {
   vector<bool> tx;
   mdm.modulate(1 << k, encoded, tx);
   if (display)
      print_signal("Tx", n, tx);
   return tx;
   }

vector<bool> transmit_modulated(int n, channel<bool>& chan,
      const vector<bool>& tx, bool display = true)
   {
   vector<bool> rx;
   chan.transmit(tx, rx);
   if (display)
      print_signal("Rx", n, rx);
   return rx;
   }

vector<vector<double> > demodulate_encoded(channel<bool>& chan,
      blockmodem<bool>& mdm, const vector<bool>& rx, bool display = true)
   {
   // demodulate received signal
   vector<vector<double> > ptable;
   timer t;
   mdm.demodulate(chan, rx, ptable);
   t.stop();
   if (display)
      cout << "Ptable: " << ptable << "\n" << std::flush;
   cout << "Time taken: " << t << "\n" << std::flush;
   return ptable;
   }

void count_errors(const vector<int>& encoded,
      const vector<vector<double> >& ptable)
   {
   const int tau = ptable.size();
   assert(tau > 0);
   assert(encoded.size() == tau);
   const int n = ptable(0).size();
   int count = 0;
   for (int i = 0; i < tau; i++)
      {
      assert(ptable(i).size() == n);
      // find the most likely candidate
      int d = 0;
      for (int j = 1; j < n; j++)
         if (ptable(i)(j) > ptable(i)(d))
            d = j;
      // see if there is an error
      if (d != encoded(i))
         count++;
      }
   if (count > 0)
      cout << "Symbol errors: " << count << " (" << int(100 * count
            / double(tau)) << "%)\n" << std::flush;
   }

void testcycle(bool decoder, bool math, bool deep, int seed, int n, int k,
      int tau, double Pe = 0, bool display = true)
   {
   // create prng for seeding systems
   libbase::randgen prng;
   prng.seed(seed);
   // create modem and channel
   modem_ptr mdm = create_modem(decoder, math, deep, tau, n, k, prng);
   channel_ptr chan = create_channel(Pe, prng);
   cout << '\n';
   cout << mdm->description() << '\n';
   cout << chan->description() << '\n';
   cout << "Block size: N = " << tau << '\n';

   // define an alternating encoded sequence
   vector<int> encoded = create_encoded(k, tau, display);
   // modulate it using the previously created inner code
   vector<bool> tx = modulate_encoded(k, n, *mdm, encoded, display);
   // pass it through the channel
   vector<bool> rx = transmit_modulated(n, *chan, tx, display);
   // demodulate received signal
   vector<vector<double> > ptable =
         demodulate_encoded(*chan, *mdm, rx, display);
   // count errors
   count_errors(encoded, ptable);
   }

/*!
 \brief   Test program for DM inner decoder
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

int main(int argc, char *argv[])
   {
   // error probabilities corresponding to SNR = 12dB and 1dB respectively
   const double Plo = 9.00601e-09;
   const double Phi = 0.056282;

   // user-defined parameters
   if (argc < 3)
      {
      cout << "Usage: " << argv[0]
            << " <type> <decoder> [math [deep [seed [k [n [N [Pe]]]]]]]\n";
      cout << "Where: type = 1 for multiple-cycle test\n";
      cout << "       type = 2 for single-cycle test\n";
      cout << "       decoder = 0/1 for classic/alternative decoder\n";
      cout << "       math = 0/1 for logrealfast(default) / double\n";
      cout << "       deep = 0/1 for shallow(default) / deep path following\n";
      cout << "Code settings seed,n,k are used always;\n";
      cout << "   Defaults to seed 0, k/n = 4/15\n";
      cout << "Block size N and error probability Pe are for single-cycle.\n";
      exit(1);
      }

   int i = 0;
   const int type = atoi(argv[++i]);
   const bool decoder = atoi(argv[++i]) != 0;
   const bool math = ((argc > ++i) ? atoi(argv[i]) : 0) != 0;
   const bool deep = ((argc > ++i) ? atoi(argv[i]) : 0) != 0;
   const int seed = ((argc > ++i) ? atoi(argv[i]) : 0);
   const int k = ((argc > ++i) ? atoi(argv[i]) : 4);
   const int n = ((argc > ++i) ? atoi(argv[i]) : 15);
   const int N = ((argc > ++i) ? atoi(argv[i]) : 5);
   const double Pe = ((argc > ++i) ? atof(argv[i]) : Plo);

   // show revision information
   cout << "URL: " << __WCURL__ << "\n";
   cout << "Version: " << __WCVER__ << "\n";

   // do what the user asked for
   switch (type)
      {
      case 1:
         // try short,medium,large codes for benchmarking at low error probability
         testcycle(decoder, math, deep, seed, n, k, 10, Plo, false);
         testcycle(decoder, math, deep, seed, n, k, 100, Plo, false);
         testcycle(decoder, math, deep, seed, n, k, 1000, Plo, false);
         // try short,medium codes for benchmarking at high error probability
         testcycle(decoder, math, deep, seed, n, k, 10, Phi, false);
         testcycle(decoder, math, deep, seed, n, k, 100, Phi, false);
         break;

      case 2:
         testcycle(decoder, math, deep, seed, n, k, N, Pe);
         break;

      default:
         cout << "Unknown type = " << type << "\n";
         break;
      }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testwmdemodulation::main(argc, argv);
   }
