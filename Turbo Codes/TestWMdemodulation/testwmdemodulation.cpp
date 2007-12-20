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

int main(int argc, char *argv[])
   {
   using std::cout;
   using std::cerr;

   // user-defined parameters
   if(argc == 1)
      cout << "Usage: " << argv[0] << " [seed [n [k [tau]]]]\n";
   const int seed = ((argc > 1) ? atoi(argv[1]) : 0);
   const int n    = ((argc > 2) ? atoi(argv[2]) : 3);
   const int k    = ((argc > 3) ? atoi(argv[3]) : 2);
   const int tau  = ((argc > 4) ? atoi(argv[4]) : 5);

   // create a watermark codec
   libcomm::watermarkcode<libbase::logrealfast> modem(n,k,seed, tau*n);
   cout << modem.description() << "\n";

   // define an alternating encoded sequence
   using libbase::vector;
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
   bsid chan(tau*n);
   chan.seed(1);
   chan.set_snr(12);
   // demodulate an error-free version
   using libbase::matrix;
   matrix<double> ptable;
   modem.demodulate(chan, tx, ptable);
   cout << "Ptable: " << ptable << "\n";

   return 0;
   }
