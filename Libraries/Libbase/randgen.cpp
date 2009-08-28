/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "randgen.h"

namespace libbase {

const int32s randgen::mbig = 1000000000L;
const int32s randgen::mseed = 161803398L;

void randgen::init(int32u s)
   {
   next = 0L;
   nextp = 31L;
   mj = (mseed - s) % mbig;
   ma[55] = mj;
   int32s mk = 1;
   for (int i = 1; i <= 54; i++)
      {
      int ii = (21 * i) % 55;
      ma[ii] = mk;
      mk = mj - mk;
      if (mk < 0)
         mk += mbig;
      mj = ma[ii];
      }
   for (int k = 1; k <= 4; k++)
      for (int i = 1; i <= 54; i++)
         {
         ma[i] -= ma[1 + (i + 30) % 55];
         if (ma[i] < 0)
            ma[i] += mbig;
         }
   }

void randgen::advance()
   {
   if (++next >= 56)
      next = 1;
   if (++nextp >= 56)
      nextp = 1;
   mj = ma[next] - ma[nextp];
   if (mj < 0)
      mj += mbig;
   ma[next] = mj;
   }

} // end namespace
