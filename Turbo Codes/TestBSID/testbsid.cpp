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
#include "rvstatistics.h"

#include <iostream>

using std::cout;
using std::cerr;
using std::flush;
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

void testinsertion(double p)
   {
   // define input sequence
   const int tau = 1000;
   vector<bool> tx(tau);
   // define insertion-only channel
   bsid channel(tau,false,false,true);
   channel.seed(0);
   channel.set_parameter(p);
   // run a number of transmissions with an all-zero source
   cout << "Testing insertions on an all-zero source:\n";
   cout << "      N:\t" << tau << "\n";
   cout << "      p:\t" << p << "\n";
   tx = bool(0);
   vector<bool> rx;
   libbase::rvstatistics drift, zeros, ones;
   for(int i=0; i<1000; i++)
      {
      channel.transmit(tx,rx);
      drift.insert(rx.size()-tau);
      int count = 0;
      for(int j=0; j<rx.size(); j++)
         count += rx(j);
      ones.insert(count);
      zeros.insert(rx.size()-count);
      }
   // show results
   cout << "  Drift:\t" << drift.mean() << "\t" << drift.sigma() << "\n";
   cout << "  Zeros:\t" << zeros.mean() << "\t" << zeros.sigma() << "\n";
   cout << "   Ones:\t" << ones.mean() << "\t" << ones.sigma() << "\n";
   }

int main(int argc, char *argv[])
   {
   // create a test sequence and test BSID transmission
   visualtest();
   // test error likelihoods and distribution
   testinsertion(0.01);

   return 0;
   }
