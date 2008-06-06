/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "logrealfast.h"
#include "dminner.h"
#include "bsid.h"
#include "itfunc.h"
#include "bitfield.h"

#include <iostream>

int main(int argc, char *argv[])
   {
   using std::cin;
   using std::cout;
   using std::cerr;

   // create a watermark code to start with
   int n=5, k=3;
   libcomm::dminner<libbase::logrealfast> modem(n,k);
   cout << modem.description() << "\n";

   // get a new watermark from stdin
   cerr << "Enter watermark code details:\n";
   modem.serialize(cin);
   cout << modem.description() << "\n";

   // compute distance table
   n = modem.get_n();
   k = modem.get_k();
   libbase::matrix<int> c(1<<k,n);
   c = 0;
   for(int i=0; i<(1<<k); i++)
      for(int j=i+1; j<(1<<k); j++)
         {
         int t = libbase::weight(modem.get_lut(i) ^ modem.get_lut(j));
         c(i,t-1)++;
         c(j,t-1)++;
         }

   // display codebook and distance table
   cout << "d\ts\t";
   for(int t=1; t<=n; t++)
      cout << "c_" << t << (t==n ? '\n' : '\t');
   for(int i=0; i<(1<<k); i++)
      {
      cout << i << '\t' << libbase::bitfield(modem.get_lut(i),n) << '\t';
      for(int t=1; t<=n; t++)
         cout << c(i,t-1) << (t==n ? '\n' : '\t');
      }

   return 0;
   }
