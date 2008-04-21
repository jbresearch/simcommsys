/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "qsc.h"
#include "gf.h"
#include "randgen.h"
#include <iostream>

using libcomm::channel;
using libcomm::qsc;
using libbase::gf;
using libbase::vector;
using libbase::randgen;

using std::cout;
using std::cerr;

template <class G> void ShowHistogram(vector<G> &x)
   {
   const int N = x.size();
   const int q = G::elements();
   vector<int> f(q);
   f = 0;
   for(int i=0; i<N; i++)
      f(x(i))++;
   assertalways(f.sum() == N);
   cout << f;
   }

template <class G> void TestChannel(channel<G> &chan, double p)
   {
   const int N = 100000;
   const int q = G::elements();
   cout << '\n' << chan.description() << '\n';
   randgen r;
   vector<G> tx(N);
   for(int i=0; i<N; i++)
      tx(i) = r.ival(q);
   cout << "Tx:\n";
   ShowHistogram(tx);
   vector<G> rx(N);
   chan.set_parameter(p);
   chan.transmit(tx,rx);
   cout << "Rx:\n";
   ShowHistogram(rx);
   }

template <int m, int poly> void TestQSC()
   {
   qsc< gf<m,poly> > chan;
   TestChannel(chan,0.1);
   }

int main(int argc, char *argv[])
   {
   //TestQSC<1,0x3>();
   //TestQSC<2,0x7>();
   TestQSC<3,0xB>();
   TestQSC<4,0x13>();
   //TestQSC<5,0x25>();
   //TestQSC<6,0x43>();
   //TestQSC<7,0x89>();
   //TestQSC<8,0x11D>();
   //TestQSC<9,0x211>();
   //TestQSC<10,0x409>();
   return 0;
   }
