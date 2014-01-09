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

#include "modem/mpsk.h"
#include "modem/qam.h"
#include "bitfield.h"
#include "itfunc.h"
#include <iostream>

namespace testmodem {

using libcomm::blockmodem;
using libcomm::mpsk;
using libcomm::qam;

using libbase::bitfield;
using libbase::gray;

using std::cout;
using std::cerr;
using std::hex;
using std::dec;

template <class S>
void TestModem(blockmodem<S> &mdm)
   {
   const int m = mdm.num_symbols();
   const int bits = int(log2(m));
   assert(m == 1<<bits);
   cout << std::endl << mdm.description() << std::endl;
   cout << "Average Energy/symbol: " << mdm.energy() << std::endl;
   for (int i = 0; i < m; i++)
      cout << bitfield(gray(i), bits) << '\t' << mdm.modulate(gray(i)) << std::endl;
   }

void TestMPSK(int m)
   {
   mpsk mdm(m);
   TestModem(mdm);
   }

void TestQAM(int m)
   {
   qam mdm(m);
   TestModem(mdm);
   }

/*!
 * \brief   Test program for modem class
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   TestMPSK(2);
   TestMPSK(4);
   TestMPSK(8);
   TestQAM(4);
   TestQAM(16);
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testmodem::main(argc, argv);
   }
