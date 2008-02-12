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
using libcomm::bsid;
using libcomm::sigspace;

void visualtest()
   {
   // define an alternating input sequence
   const int tau = 5;
   vector<bool> tx(tau);
   for(int i=0; i<tau; i++)
      tx(i) = (i&1);
   cout << "Tx: " << tx << "\n";
   // pass that through the channel
   vector<bool> rx1, rx2;
   // probability of error corresponding to SNR=12
   const double p = 9.00601e-09;
   // channel1 is a substitution-only channel
   bsid channel1(tau);
   channel1.seed(1);
   channel1.set_parameter(p);
   channel1.set_ps(0.3);
   channel1.transmit(tx, rx1);
   cout << "Rx1: " << rx1 << "\n";
   // channel1 is an insdel-only channel
   bsid channel2(tau);
   channel2.seed(1);
   channel2.set_parameter(p);
   channel2.set_pi(0.3);
   channel2.set_pd(0.3);
   channel2.transmit(tx, rx2);
   cout << "Rx2: " << rx2 << "\n";
   }

int main(int argc, char *argv[])
   {
   // create a test sequence and test BSID transmission
   visualtest();
   return 0;
   }
