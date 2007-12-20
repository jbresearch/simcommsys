/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "randgen.h"

namespace libbase {

using std::clog;
using std::flush;

const int32s randgen::mbig = 1000000000L;
const int32s randgen::mseed = 161803398L;

randgen::randgen(int32u s)
   {
#ifdef DEBUG
   counter = 0;
   clog << "DEBUG: randgen (" << this << ") created.\n" << flush;
#endif
   seed(s);
   }

randgen::~randgen()
   {
#ifdef DEBUG
   clog << "DEBUG: randgen (" << this << ") destroyed after " << counter << " steps.\n" << flush;
#endif
   }

void randgen::seed(int32u s)
   {
   next = 0L;
   nextp = 31L;
   mj = (mseed - s) % mbig;
   ma[55] = mj;
   int32s mk = 1;
   for(int i=1; i<=54; i++)
      {
      int ii = (21*i) % 55;
      ma[ii] = mk;
      mk = mj - mk;
      if(mk < 0) mk += mbig;
      mj = ma[ii];
      }
   for(int k=1; k<=4; k++)
      for(int i=1; i<=54; i++)
         {
         ma[i] -= ma[1+(i+30)%55];
         if(ma[i] < 0) ma[i] += mbig;
         }
#ifdef DEBUG
   if(counter > 0)
       clog << "DEBUG: randgen (" << this << ") reseeded after " << counter << " steps.\n" << flush;
   counter = 0;
#endif
   ready = false;
   next_gval = 0.0;
   }

}; // end namespace
