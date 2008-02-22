/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "timer.h"
#include "logrealfast.h"
#include "watermarkcode.h"
#include "bsid.h"

#include <iostream>

using std::cout;
using std::cerr;

using libbase::timer;
using libbase::vector;
using libbase::matrix;
using libbase::logrealfast;

using libcomm::modulator;
using libcomm::channel;
using libcomm::watermarkcode;

channel<bool> *create_channel(int seed, int N, double Pe)
   {
   channel<bool> *chan = new libcomm::bsid(N);
   chan->seed(seed+1);
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

void print_signal(const char* desc, int n, vector<bool> tx)
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

void testcycle(int const seed, int const n, int const k, int const tau, double Pe=0, bool display=true)
   {
   const int N = tau*n;
   // create modem and channel
   watermarkcode<logrealfast> modem(n,k, N);
   modem.seed(seed);
   channel<bool> *chan = create_channel(seed, N, Pe);
   cout << modem.description() << "\n";

   // define an alternating encoded sequence
   vector<int> encoded = create_encoded(k, tau, display);
   // modulate it using the previously created watermarkcode
   vector<bool> tx = modulate_encoded(k, n, modem, encoded, display);
   // pass it through the channel
   vector<bool> rx = transmit_modulated(n, *chan, tx, display);
   // demodulate received signal
   matrix<double> ptable = demodulate_encoded(*chan, modem, rx, display);
   // count errors
   count_errors(encoded, ptable);

   delete chan;
   }

int main(int argc, char *argv[])
   {
   // error probabilities corresponding to SNR = 12dB and 1dB respectively
   const double Plo = 9.00601e-09;
   const double Phi = 0.056282;

   // user-defined parameters
   if(argc == 1)
      cout << "Usage: " << argv[0] << " [seed [n [k [tau [p]]]]]\n" << std::flush;
   const int seed = ((argc > 1) ? atoi(argv[1]) : 0);
   const int n    = ((argc > 2) ? atoi(argv[2]) : 3);
   const int k    = ((argc > 3) ? atoi(argv[3]) : 2);
   const int tau  = ((argc > 4) ? atoi(argv[4]) : 5);
   const double p = ((argc > 5) ? atof(argv[5]) : Plo);

   // do what the user asked for
   testcycle(seed, n, k, tau, p);

   // try short,medium,large codes for benchmarking at low error probability
   testcycle(seed, 15, 4, 10, Plo, false);
   testcycle(seed, 15, 4, 100, Plo, false);
   testcycle(seed, 15, 4, 1000, Plo, false);

   // try short,medium codes for benchmarking at high error probability
   testcycle(seed, 15, 4, 10, Phi, false);
   testcycle(seed, 15, 4, 100, Phi, false);

   return 0;
   }
