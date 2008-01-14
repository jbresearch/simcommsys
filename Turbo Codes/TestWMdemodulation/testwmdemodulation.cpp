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

#include <iostream>

using std::cout;
using std::cerr;

using libbase::vector;
using libbase::matrix;
using libbase::logrealfast;
using libcomm::sigspace;
using libcomm::modulator;
using libcomm::watermarkcode;

vector<int> create_encoded(int k, int tau)
   {
   vector<int> encoded(tau);
   for(int i=0; i<tau; i++)
      encoded(i) = (i%(1<<k));
   cout << "Encoded: " << encoded << "\n";
   return encoded;
   }

vector<sigspace> modulate_encoded(int k, int n, modulator& modem, vector<int>& encoded)
   {
   vector<sigspace> tx;
   modem.modulate(1<<k, encoded, tx);
   cout << "Tx:\n";
   for(int i=0; i<tx.size(); i++)
      cout << tx(i) << ((i%n == n-1) ? "\n" : "\t");
   return tx;
   }

void demodulate_encoded(int N, modulator& modem, vector<sigspace>& tx)
   {
   // assume an error-free transmission
   libcomm::bsid chan(N);
   chan.seed(1);
   chan.set_snr(12);
   // demodulate an error-free version
   matrix<double> ptable;
   modem.demodulate(chan, tx, ptable);
   cout << "Ptable: " << ptable << "\n";
   }

void test_errorfree(int const seed, int const n, int const k, int const tau)
   {
   const int N = tau*n;
   // create a watermark codec
   watermarkcode<logrealfast> modem(n,k,seed, N);
   cout << modem.description() << "\n";
   // define an alternating encoded sequence
   vector<int> encoded = create_encoded(k, tau);
   // modulate it using the previously created watermarkcode
   vector<sigspace> tx = modulate_encoded(k, n, modem, encoded);
   // demodulate an error-free version
   demodulate_encoded(N, modem, tx);
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

   test_errorfree(seed, n, k, tau);

   return 0;
   }
