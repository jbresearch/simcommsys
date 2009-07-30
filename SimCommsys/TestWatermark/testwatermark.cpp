#include "logrealfast.h"
#include "modem/dminner.h"
#include "channel/bsid.h"
#include "itfunc.h"
#include "bitfield.h"

#include <iostream>

namespace testwatermark {

/*!
 * \brief   Test program for DM inner code creation
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

int main(int argc, char *argv[])
   {
   using std::cin;
   using std::cout;
   using std::cerr;

   // create a watermark code to start with
   libcomm::dminner<libbase::logrealfast, false> mdm;
   // get a new watermark from stdin
   cerr << "Enter watermark code details:\n";
   mdm.serialize(cin);
   cout << mdm.description() << "\n";

   // compute distance table
   const int n = mdm.get_symbolsize();
   const int q = mdm.num_symbols();
   libbase::matrix<int> c(q, n);
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
   for (int t = 1; t <= n; t++)
      cout << "c_" << t << (t == n ? '\n' : '\t');
   for (int i = 0; i < q; i++)
      {
      cout << i << '\t';
      cout << libbase::bitfield(mdm.get_symbol(i), n) << '\t';
      for (int t = 1; t <= n; t++)
         cout << c(i, t - 1) << (t == n ? '\n' : '\t');
      }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testwatermark::main(argc, argv);
   }
