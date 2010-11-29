#include "logrealfast.h"
#include "modem/dminner.h"
#include "channel/bsid.h"
#include "randgen.h"
#include "rvstatistics.h"

#include <iostream>

namespace testbsid {

using std::cout;
using std::cerr;
using libbase::vector;
using libbase::randgen;
using libcomm::bsid;
using libcomm::sigspace;

void visualtest()
   {
   // define an alternating input sequence
   const int tau = 5;
   vector<bool> tx(tau);
   for (int i = 0; i < tau; i++)
      tx(i) = (i & 1);
   cout << "Tx: " << tx << std::endl;
   // pass that through the channel
   vector<bool> rx1, rx2;
   // probability of error corresponding to SNR=12
   const double p = 9.00601e-09;
   // seed generator
   randgen prng;
   prng.seed(0);
   // channel1 is a substitution-only channel
   bsid channel1(tau);
   channel1.seedfrom(prng);
   channel1.set_parameter(p);
   channel1.set_ps(0.3);
   channel1.transmit(tx, rx1);
   cout << "Rx1: " << rx1 << std::endl;
   // channel1 is an insdel-only channel
   bsid channel2(tau);
   channel2.seedfrom(prng);
   channel2.set_parameter(p);
   channel2.set_pi(0.3);
   channel2.set_pd(0.3);
   channel2.transmit(tx, rx2);
   cout << "Rx2: " << rx2 << std::endl;
   }

void testtransmission(int tau, double p, bool ins, bool del, bool sub, bool src)
   {
   // define channel according to specifications
   bsid channel(sub, del, ins);
   randgen prng;
   prng.seed(0);
   channel.seedfrom(prng);
   channel.set_parameter(p);
   // run a number of transmissions with an all-zero source
   cout << "Testing on an all-" << (src ? "one" : "zero") << " source:" << std::endl;
   cout << "   type:\t";
   if (ins)
      cout << "insertions ";
   if (del)
      cout << "deletions ";
   if (sub)
      cout << "substitutions";
   cout << std::endl;
   cout << "      N:\t" << tau << std::endl;
   cout << "      p:\t" << p << std::endl;
   // define input sequence
   vector<bool> tx(tau);
   tx = src;
   vector<bool> rx;
   libbase::rvstatistics drift, zeros, ones;
   for (int i = 0; i < 1000; i++)
      {
      channel.transmit(tx, rx);
      drift.insert(rx.size() - tau);
      int count = 0;
      for (int j = 0; j < rx.size(); j++)
         count += rx(j);
      ones.insert(count);
      zeros.insert(rx.size() - count);
      }
   // show results
   cout << "   Value\tMean\tSigma" << std::endl;
   cout << "  Drift:\t" << drift.mean() << "\t" << drift.sigma() << std::endl;
   cout << "  Zeros:\t" << zeros.mean() << "\t" << zeros.sigma() << std::endl;
   cout << "   Ones:\t" << ones.mean() << "\t" << ones.sigma() << std::endl;
   cout << std::endl;
   }

/*!
 * \brief   Test program for BSID channel
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

int main(int argc, char *argv[])
   {
   // create a test sequence and test BSID transmission
   visualtest();
   // test insertion-only channels
   testtransmission(1000, 0.01, true, false, false, 0);
   testtransmission(1000, 0.25, true, false, false, 0);
   // test deletion-only channels
   testtransmission(1000, 0.01, false, true, false, 0);
   testtransmission(1000, 0.25, false, true, false, 0);
   // test substitution-only channels
   testtransmission(1000, 0.1, false, false, true, 0);
   testtransmission(1000, 0.1, false, false, true, 1);
   // test insertion-deletion channels
   testtransmission(1000, 0.01, true, true, false, 0);
   testtransmission(1000, 0.25, true, true, false, 0);

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testbsid::main(argc, argv);
   }
