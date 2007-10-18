#include "logrealfast.h"
#include "watermarkcode.h"
#include "bsid.h"

#include <iostream>

int main(int argc, char *argv[])
   {
   using std::cout;
   using std::cerr;

   // create a watermark codec
   using libbase::logrealfast;
   using libcomm::watermarkcode;
   int N=100, n=5, k=3;
   if(argc < 3)
      cerr << "Usage: " << argv[0] << " [<n> <k>]\n";
   else
      {
      n = atoi(argv[1]);
      k = atoi(argv[2]);
      }
   watermarkcode<logrealfast> codec(N,n,k,0, 10,50, 0.01,0.01,0.01);
   cout << codec.description() << "\n";
   
   using libbase::vector;
   using libcomm::bsid;
   using libcomm::sigspace;
   // define an alternating input sequence
   const int tau = 5;
   vector<sigspace> tx(tau);
   for(int i=0; i<tau; i++)
      tx(i) = sigspace((i%2) ? -1 : 1, 0);
   // pass that through the channel
   vector<sigspace> rx1, rx2;
   bsid channel1, channel2(0.3,0.3);
   // channel1 is a substitution-only channel
   channel1.seed(1);
   channel1.set_snr(-12);
   channel1.transmit(tx, rx1);
   // channel1 is an insdel-only channel
   channel2.seed(1);
   channel2.set_snr(12);
   channel2.transmit(tx, rx2);
   // display all sequences
   cout << tx << "\n";
   cout << rx1 << "\n";
   cout << rx2 << "\n";
   
   return 0;
   }
