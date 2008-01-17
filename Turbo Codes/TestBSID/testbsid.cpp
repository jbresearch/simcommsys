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

   // create a test sequence and test BSID transmission
   using libbase::vector;
   using libcomm::bsid;
   using libcomm::sigspace;
   // define an alternating input sequence
   const int tau = 5;
   vector<sigspace> tx(tau);
   for(int i=0; i<tau; i++)
      tx(i) = sigspace((i%2) ? -1 : 1, 0);
   cout << "Tx: " << tx << "\n";
   // pass that through the channel
   vector<sigspace> rx1, rx2;
   // channel1 is a substitution-only channel
   bsid channel1(tau);
   channel1.seed(1);
   channel1.set_parameter(12);
   channel1.set_ps(0.3);
   channel1.transmit(tx, rx1);
   cout << "Rx1: " << rx1 << "\n";
   // channel1 is an insdel-only channel
   bsid channel2(tau);
   channel2.seed(1);
   channel2.set_parameter(12);
   channel2.set_pi(0.3);
   channel2.set_pd(0.3);
   channel2.transmit(tx, rx2);
   cout << "Rx2: " << rx2 << "\n";

   return 0;
   }
