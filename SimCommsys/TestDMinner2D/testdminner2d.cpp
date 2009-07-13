#include "logrealfast.h"
#include "modem/dminner2d.h"
#include "itfunc.h"

#include <iostream>

namespace testdminner2d {

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
   libcomm::dminner2d<libbase::logrealfast, false> mdm;
   // get code definition from stdin
   cerr << "Enter code details:\n";
   mdm.serialize(cin);
   cout << mdm.description() << "\n";

   // compute distance table
   const int m = mdm.get_symbol_rows();
   const int n = mdm.get_symbol_cols();
   const int q = mdm.num_symbols();
   libbase::matrix<int> c(q, m * n);
   c = 0;
   for (int i = 0; i < q; i++)
      for (int j = i + 1; j < q; j++)
         {
         int t = libbase::weight(mdm.get_symbol(i) ^ mdm.get_symbol(j));
         c(i, t - 1)++;
         c(j, t - 1)++;
         }

   // display codebook and distance table
   cout << "d\ts\t";
   for (int t = 1; t <= m * n; t++)
      cout << "c_" << t << (t == m * n ? '\n' : '\t');
   for (int d = 0; d < q; d++)
      {
      cout << d << '\t';
      for (int i = 0; i < m; i++)
         {
         for (int j = 0; j < n; j++)
            cout << mdm.get_symbol(d)(j, i);
         cout << (i == m - 1 ? '\t' : ',');
         }
      for (int t = 1; t <= m * n; t++)
         cout << c(d, t - 1) << (t == m * n ? '\n' : '\t');
      }

   return 0;
   }

}
; // end namespace

int main(int argc, char *argv[])
   {
   return testdminner2d::main(argc, argv);
   }
