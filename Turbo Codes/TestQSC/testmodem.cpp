/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "mpsk.h"
#include "qam.h"
#include "bitfield.h"
#include "itfunc.h"
#include <iostream>

using libcomm::modulator;
using libcomm::mpsk;
using libcomm::qam;

using libbase::bitfield;
using libbase::gray;

using std::cout;
using std::cerr;
using std::hex;
using std::dec;

template <class S> void TestModem(modulator<S> &modem)
   {
   const int m = modem.num_symbols();
   const int bits = int(log2(m));
   assert(m == 1<<bits);
   cout << '\n' << modem.description() << '\n';
   cout << "Average Energy/symbol: " << modem.energy() << '\n';
   for(int i=0; i<m; i++)
      cout << bitfield(gray(i),bits) << '\t' << modem.modulate(gray(i)) << '\n';
   }

void TestMPSK(int m)
   {
   mpsk modem(m);
   TestModem(modem);
   }

void TestQAM(int m)
   {
   qam modem(m);
   TestModem(modem);
   }

int main(int argc, char *argv[])
   {
   TestMPSK(2);
   TestMPSK(4);
   TestMPSK(8);
   TestQAM(4);
   TestQAM(16);
   return 0;
   }
