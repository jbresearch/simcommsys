/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "channel/qsc.h"
#include "gf.h"
#include "randgen.h"
#include <iostream>

namespace testqsc {

using libcomm::channel;
using libcomm::qsc;
using libbase::gf;
using libbase::vector;
using libbase::randgen;

using std::cout;
using std::cerr;

template <class G>
void ShowHistogram(vector<G> &x)
   {
   const int N = x.size();
   const int q = G::elements();
   vector<int> f(q);
   f = 0;
   for (int i = 0; i < N; i++)
      f(x(i))++;
   assertalways(f.sum() == N);
   const double E = double(N) / double(q);
   for (int i = 0; i < q; i++)
      cout << i << "\t" << f(i) << "\t[" << 100.0 * (f(i) - E) / E << "%]" << std::endl;
   }

template <class G>
void TestChannel(channel<G> &chan, double p)
   {
   const int N = 100000;
   const int q = G::elements();
   cout << std::endl << chan.description() << std::endl;
   randgen r;
   r.seed(0);
   vector<G> tx(N);
   for (int i = 0; i < N; i++)
      tx(i) = r.ival(q);
   cout << "Tx:" << std::endl;
   ShowHistogram(tx);
   vector<G> rx(N);
   chan.seedfrom(r);
   chan.set_parameter(p);
   chan.transmit(tx, rx);
   cout << "Rx:" << std::endl;
   ShowHistogram(rx);
   }

template <int m, int poly>
void TestQSC()
   {
   qsc<gf<m, poly> > chan;
   TestChannel(chan, 0.1);
   }

/*!
 * \brief   Test program for q-ary symmetric channel
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   //TestQSC<1,0x3>();
   //TestQSC<2,0x7>();
   TestQSC<3, 0xB> ();
   TestQSC<4, 0x13> ();
   //TestQSC<5,0x25>();
   //TestQSC<6,0x43>();
   //TestQSC<7,0x89>();
   //TestQSC<8,0x11D>();
   //TestQSC<9,0x211>();
   //TestQSC<10,0x409>();
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testqsc::main(argc, argv);
   }
