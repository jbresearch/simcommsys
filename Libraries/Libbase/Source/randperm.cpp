/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "randperm.h"

namespace libbase {

void randperm::init(const int N, random& r)
   {
   assert(N >= 0);
   // initialize array to hold permuted positions
   lut.init(N);
   if(N == 0)
      return;
   lut = -1;
   // create the permutation vector
   for(int i=0; i<N; i++)
      {
      int j;
      do {
         j = r.ival(N);
         } while(lut(j)>=0);
      lut(j) = i;
      }
   }

}; // end namespace
