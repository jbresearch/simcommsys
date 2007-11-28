#include "logrealfast.h"
#include "watermarkcode.h"
#include "bsid.h"

#include <iostream>

int main(int argc, char *argv[])
   {
   using std::cout;
   using std::cerr;

   // fixed parameters
   const int I=3, xmax=5;
   // user-defined parameters
   const int seed = ((argc > 1) ? atoi(argv[1]) : 0);
   const int n    = ((argc > 2) ? atoi(argv[2]) : 3);
   const int k    = ((argc > 3) ? atoi(argv[3]) : 2);
   // create a watermark codec
   libcomm::watermarkcode<libbase::logrealfast> modem(n,k,seed, I,xmax);
   cout << modem.description() << "\n";
   
   // define an alternating encoded sequence
   using libbase::vector;
   const int tau = 5;
   const int N = 1<<k;
   vector<int> encoded(tau);
   for(int i=0; i<tau; i++)
      encoded(i) = (i%N);
   cout << "Encoded: " << encoded << "\n";

   // modulate it using the previously created watermarkcode
   using libcomm::sigspace;
   vector<sigspace> tx;
   modem.modulate(N, encoded, tx);
   cout << "Tx:\n";
   for(int i=0; i<tx.size(); i++)
      cout << tx(i) << ((i%n == n-1) ? "\n" : "\t");

   // assume an error-free transmission
   using libcomm::bsid;
   bsid chan(I,xmax);
   chan.seed(1);
   chan.set_snr(12);
   // demodulate an error-free version
   using libbase::matrix;
   matrix<double> ptable;
   modem.demodulate(chan, tx, ptable);
   cout << "Ptable: " << ptable << "\n";

   return 0;
   }
