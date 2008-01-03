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

using libcomm::mpsk;
using libcomm::qam;

using libbase::bitfield;
using libbase::gray;

using std::cout;
using std::cerr;
using std::hex;
using std::dec;

void TestMPSK(int m)
   {
   const int bits = int(log2(m));
   assert(m == 1<<bits);
   mpsk modem(m);
   cout << '\n' << modem.description() << '\n';
   for(int i=0; i<m; i++)
      cout << bitfield(gray(i),bits) << '\t' << modem.modulate(gray(i)) << '\n';
   }

int main(int argc, char *argv[])
   {
   TestMPSK(2);
   TestMPSK(4);
   TestMPSK(8);
   return 0;
   }
