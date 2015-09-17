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

#include "logrealfast.h"
#include "modem/dminner.h"
#include "itfunc.h"
#include "bitfield.h"

#include <iostream>

namespace testwatermark {

/*!
 * \brief   Test program for DM inner code creation
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   using std::cin;
   using std::cout;
   using std::cerr;

   // create a watermark code to start with
   libcomm::dminner<libbase::logrealfast> mdm;
   // get a new watermark from stdin
   cerr << "Enter watermark code details:" << std::endl;
   mdm.serialize(cin);
   cout << mdm.description() << std::endl;

   // compute distance table
   const int n = mdm.get_symbolsize(0);
   const int q = mdm.num_symbols();
   libbase::matrix<int> c(q, n);
   c = 0;
   for (int i = 0; i < q; i++)
      for (int j = i + 1; j < q; j++)
         {
         int t = libbase::weight(mdm.get_symbol(0, i) ^ mdm.get_symbol(0, j));
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
      cout << libbase::bitfield(mdm.get_symbol(0, i), n) << '\t';
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
