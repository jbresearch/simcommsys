/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "logrealfast.h"
#include "watermarkcode.h"
#include "bsid.h"
#include "timer.h"

#include <iostream>

using std::cout;
using std::cerr;

using libbase::timer;
using libbase::vector;
using libbase::matrix;
using libbase::logrealfast;

using libcomm::sigspace;
using libcomm::modulator;
using libcomm::watermarkcode;

vector<int> create_encoded(int k, int tau, bool display=true)
   {
   vector<int> encoded(tau);
   for(int i=0; i<tau; i++)
      encoded(i) = (i%(1<<k));
   if(display)
      cout << "Encoded: " << encoded << "\n";
   return encoded;
   }

vector<sigspace> modulate_encoded(int k, int n, modulator& modem, vector<int>& encoded, bool display=true)
   {
   vector<sigspace> tx;
   modem.modulate(1<<k, encoded, tx);
   if(display)
      {
      cout << "Tx:\n";
      for(int i=0; i<tx.size(); i++)
         cout << tx(i) << ((i%n == n-1) ? "\n" : "\t");
      }
   return tx;
   }

void demodulate_encoded(int N, modulator& modem, vector<sigspace>& tx, double snr, bool display=true)
   {
   // assume an error-free transmission
   libcomm::bsid chan(N);
   chan.seed(1);
   chan.set_snr(snr);
   // demodulate an error-free version
   matrix<double> ptable;
   modem.demodulate(chan, tx, ptable);
   if(display)
      cout << "Ptable: " << ptable << "\n";
   }

void test_errorfree(int const seed, int const n, int const k, int const tau, double snr=12, bool display=true)
   {
   const int N = tau*n;
   // create a watermark codec
   watermarkcode<logrealfast> modem(n,k,seed, N);
   cout << modem.description() << "\n";
   // define an alternating encoded sequence
   vector<int> encoded = create_encoded(k, tau, display);
   // modulate it using the previously created watermarkcode
   vector<sigspace> tx = modulate_encoded(k, n, modem, encoded, display);
   // demodulate an error-free version
   timer t;
   demodulate_encoded(N, modem, tx, snr, display);
   t.stop();
   cout << "Time taken: " << t << "\n";
   }

int main(int argc, char *argv[])
   {
   // user-defined parameters
   if(argc == 1)
      cout << "Usage: " << argv[0] << " [seed [n [k [tau]]]]\n";
   const int seed = ((argc > 1) ? atoi(argv[1]) : 0);
   const int n    = ((argc > 2) ? atoi(argv[2]) : 3);
   const int k    = ((argc > 3) ? atoi(argv[3]) : 2);
   const int tau  = ((argc > 4) ? atoi(argv[4]) : 5);
   // do what the user asked for
   test_errorfree(seed, n, k, tau);

   // try short,medium,large codes for benchmarking at high SNR
   test_errorfree(seed, 15, 4, 10, 12.0, false);
   test_errorfree(seed, 15, 4, 100, 12.0, false);
   test_errorfree(seed, 15, 4, 1000, 12.0, false);

   // try short,medium,large codes for benchmarking at low SNR
   test_errorfree(seed, 15, 4, 10, 1.0, false);
   test_errorfree(seed, 15, 4, 100, 1.0, false);
   test_errorfree(seed, 15, 4, 1000, 1.0, false);

   return 0;
   }
