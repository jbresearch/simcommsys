#include "logrealfast.h"
#include "dminner.h"
#include "bsid.h"
#include "itfunc.h"
#include "bitfield.h"

#include <iostream>

/*!
   \brief   Test program for DM inner code creation
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

int main(int argc, char *argv[])
   {
   using std::cin;
   using std::cout;
   using std::cerr;

   // create a watermark code to start with
   int n=5, k=3;
   libcomm::dminner<libbase::logrealfast,false> mdm(n,k);
   cout << mdm.description() << "\n";

   // get a new watermark from stdin
   cerr << "Enter watermark code details:\n";
   mdm.serialize(cin);
   cout << mdm.description() << "\n";

   // compute distance table
   n = mdm.get_n();
   k = mdm.get_k();
   libbase::matrix<int> c(1<<k,n);
   c = 0;
   for(int i=0; i<(1<<k); i++)
      for(int j=i+1; j<(1<<k); j++)
         {
         int t = libbase::weight(mdm.get_lut(i) ^ mdm.get_lut(j));
         c(i,t-1)++;
         c(j,t-1)++;
         }

   // display codebook and distance table
   cout << "d\ts\t";
   for(int t=1; t<=n; t++)
      cout << "c_" << t << (t==n ? '\n' : '\t');
   for(int i=0; i<(1<<k); i++)
      {
      cout << i << '\t' << libbase::bitfield(mdm.get_lut(i),n) << '\t';
      for(int t=1; t<=n; t++)
         cout << c(i,t-1) << (t==n ? '\n' : '\t');
      }

   return 0;
   }
