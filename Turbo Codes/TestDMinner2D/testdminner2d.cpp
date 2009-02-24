#include "logrealfast.h"
#include "dminner2d.h"
#include "itfunc.h"

#include <iostream>

/*!
   \brief   Test program for 2D DM inner code
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

int main(int argc, char *argv[])
   {
   using std::cin;
   using std::cout;
   using std::cerr;

   // create a code to start with
   libcomm::dminner2d<libbase::logrealfast,false> mdm;
   // get code definition from stdin
   cerr << "Enter code details:\n";
   mdm.serialize(cin);
   cout << mdm.description() << "\n";

   // compute distance table
   const int m = mdm.get_symbol_rows();
   const int n = mdm.get_symbol_cols();
   const int q = mdm.num_symbols();
   libbase::matrix<int> c(q,m*n);
   c = 0;
   for(int i=0; i<q; i++)
      for(int j=i+1; j<q; j++)
         {
         int t = libbase::weight(mdm.get_symbol(i) ^ mdm.get_symbol(j));
         c(i,t-1)++;
         c(j,t-1)++;
         }

   // display codebook and distance table
//    cout << "d\ts\t";
//    for(int t=1; t<=n; t++)
//       cout << "c_" << t << (t==n ? '\n' : '\t');
//    for(int i=0; i<(1<<k); i++)
//       {
//       cout << i << '\t' << libbase::bitfield(mdm.get_lut(i),n) << '\t';
//       for(int t=1; t<=n; t++)
//          cout << c(i,t-1) << (t==n ? '\n' : '\t');
//       }

   return 0;
   }
