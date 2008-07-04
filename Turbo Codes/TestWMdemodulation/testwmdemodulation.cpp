/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "timer.h"
#include "randgen.h"
#include "logrealfast.h"
#include "dminner.h"
#include "dminner2.h"
#include "bsid.h"

#include <memory>
#include <iostream>
#include <string>

using std::cout;
using std::cerr;

using libbase::timer;
using libbase::vector;
using libbase::matrix;
using libbase::logrealfast;

using libcomm::modulator;
using libcomm::channel;
using libcomm::dminner;
using libcomm::dminner2;

typedef std::auto_ptr< modulator<bool> > modem_ptr;
typedef std::auto_ptr< channel<bool> > channel_ptr;

modem_ptr create_modem(int const type, int const n, int const k, bool deep, libbase::random& r)
   {
   modem_ptr modem;
   switch(type)
      {
      case 1:
         if(deep)
            modem = modem_ptr(new dminner<logrealfast,false>(n,k,1e-21,1e-12));
         else
            modem = modem_ptr(new dminner<logrealfast,false>(n,k));
         break;
      case 2:
         if(deep)
            modem = modem_ptr(new dminner2<logrealfast,false>(n,k,1e-21,1e-12));
         else
            modem = modem_ptr(new dminner2<logrealfast,false>(n,k));
         break;
      default:
         assertalways("Unknown decoder type.");
         break;
      }
   modem->seedfrom(r);
   return modem;
   }

channel_ptr create_channel(double Pe, libbase::random& r)
   {
   channel_ptr chan = channel_ptr(new libcomm::bsid);
   chan->seedfrom(r);
   chan->set_parameter(Pe);
   return chan;
   }

vector<int> create_encoded(int k, int tau, bool display=true)
   {
   vector<int> encoded(tau);
   for(int i=0; i<tau; i++)
      encoded(i) = (i%(1<<k));
   if(display)
      cout << "Encoded: " << encoded << "\n" << std::flush;
   return encoded;
   }

void print_signal(const std::string desc, int n, vector<bool> tx)
   {
   cout << desc << ":\n";
   for(int i=0; i<tx.size(); i++)
      cout << tx(i) << ((i%n == n-1 || i == tx.size()-1) ? "\n" : "\t");
   cout << std::flush;
   }

vector<bool> modulate_encoded(int k, int n, modulator<bool>& modem, vector<int>& encoded, bool display=true)
   {
   vector<bool> tx;
   modem.modulate(1<<k, encoded, tx);
   if(display)
      print_signal("Tx", n, tx);
   return tx;
   }

vector<bool> transmit_modulated(int n, channel<bool>& chan, const vector<bool>& tx, bool display=true)
   {
   vector<bool> rx;
   chan.transmit(tx, rx);
   if(display)
      print_signal("Rx", n, rx);
   return rx;
   }

matrix<double> demodulate_encoded(channel<bool>& chan, modulator<bool>& modem, const vector<bool>& rx, bool display=true)
   {
   // demodulate received signal
   matrix<double> ptable;
   timer t;
   modem.demodulate(chan, rx, ptable);
   t.stop();
   if(display)
      cout << "Ptable: " << ptable << "\n" << std::flush;
   cout << "Time taken: " << t << "\n" << std::flush;
   return ptable;
   }

void count_errors(const vector<int>& encoded, const matrix<double>& ptable)
   {
   const int tau = ptable.xsize();
   const int n = ptable.ysize();
   assert(encoded.size() == tau);
   int count = 0;
   for(int i=0; i<tau; i++)
      {
      // find the most likely candidate
      int d=0;
      for(int j=1; j<n; j++)
         if(ptable(i,j) > ptable(i,d))
            d = j;
      // see if there is an error
      if(d != encoded(i))
         count++;
      }
   if(count > 0)
      cout << "Symbol errors: " << count << " (" << int(100*count/double(tau)) << "%)\n" << std::flush;
   }

void testcycle(int const type, int const seed, int const n, int const k, int const tau, double Pe=0, bool deep=false, bool display=true)
   {
   // create prng for seeding systems
   libbase::randgen prng;
   prng.seed(seed);
   // create modem and channel
   modem_ptr modem = create_modem(type, n, k, deep, prng);
   channel_ptr chan = create_channel(Pe, prng);
   cout << '\n';
   cout << modem->description() << '\n';
   cout << chan->description() << '\n';
   cout << "Block size: N = " << tau << '\n';

   // define an alternating encoded sequence
   vector<int> encoded = create_encoded(k, tau, display);
   // modulate it using the previously created inner code
   vector<bool> tx = modulate_encoded(k, n, *modem, encoded, display);
   // pass it through the channel
   vector<bool> rx = transmit_modulated(n, *chan, tx, display);
   // demodulate received signal
   matrix<double> ptable = demodulate_encoded(*chan, *modem, rx, display);
   // count errors
   count_errors(encoded, ptable);
   }

int main(int argc, char *argv[])
   {
   // error probabilities corresponding to SNR = 12dB and 1dB respectively
   const double Plo = 9.00601e-09;
   const double Phi = 0.056282;

   // user-defined parameters
   if(argc == 1)
      {
      cout << "Usage: " << argv[0] << " <type> [deep [seed [k [n [N [Pe]]]]]]\n";
      cout << "Where: type = 1 for multiple-cycle with classic decoder\n";
      cout << "       type = 2 for multiple-cycle with alternative decoder\n";
      cout << "       type = 11 for single-cycle with classic decoder\n";
      cout << "       type = 12 for single-cycle with alternative decoder\n";
      cout << "Code settings deep,seed,n,k are used for all types;\n";
      cout << "   Defaults to deep 0, seed 0, k/n = 4/15\n";
      cout << "Block size N and error probability Pe are for single-cycle.\n";
      exit(1);
      }

   int i = 1;
   const int type  = atoi(argv[i]);
   const bool deep = ((argc > ++i) ? atoi(argv[i]) : 0) != 0;
   const int seed  = ((argc > ++i) ? atoi(argv[i]) : 0);
   const int k     = ((argc > ++i) ? atoi(argv[i]) : 4);
   const int n     = ((argc > ++i) ? atoi(argv[i]) : 15);
   const int N     = ((argc > ++i) ? atoi(argv[i]) : 5);
   const double Pe = ((argc > ++i) ? atof(argv[i]) : Plo);

   // show revision information
   cout << "URL: " << __WCURL__ << "\n";
   cout << "Version: " << __WCVER__ << "\n";

   // do what the user asked for
   switch(type)
      {
      case 1:
      case 2:
         // try short,medium,large codes for benchmarking at low error probability
         testcycle(type, seed, n, k, 10, Plo, deep, false);
         testcycle(type, seed, n, k, 100, Plo, deep, false);
         testcycle(type, seed, n, k, 1000, Plo, deep, false);
         // try short,medium codes for benchmarking at high error probability
         testcycle(type, seed, n, k, 10, Phi, deep, false);
         testcycle(type, seed, n, k, 100, Phi, deep, false);
         break;

      case 11:
      case 12:
         testcycle(type-10, seed, n, k, N, Pe, deep);
         break;

      default:
         cout << "Unknown type = " << type << "\n";
         break;
      }

   return 0;
   }
