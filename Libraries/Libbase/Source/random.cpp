/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "random.h"

namespace libbase {

random::random()
   {
#ifndef NDEBUG
   counter = 0;
   trace << "DEBUG: random (" << this << ") created.\n" << std::flush;
#endif
   next_gval_available = false;
   }

random::~random()
   {
#ifndef NDEBUG
   trace << "DEBUG: random (" << this << ") destroyed after " << counter << " steps.\n" << std::flush;
#endif
   }

int32u random::seed(int32u s)
   {
#ifndef NDEBUG
   if(counter > 0)
      trace << "DEBUG: random (" << this << ") reseeded after " << counter << " steps.\n" << std::flush;
   counter = 0;
#endif
   // this makes sure any stored gval is discarded
   next_gval_available = false;
   // initialize underlying generator
   init(s);
   // use the first generated value as suggested seed
   return ival();
   }

double random::gval()
   {
   if(next_gval_available)
      {
      next_gval_available = false;
      return next_gval;
      }

   double v1, v2, rsq;
   do {
      v1 = 2.0 * fval() - 1.0;
      v2 = 2.0 * fval() - 1.0;
      rsq = (v1*v1) + (v2*v2);
      } while(rsq >= 1.0 || rsq == 0.0);
   double fac = sqrt(-2.0*log(rsq)/rsq);
   next_gval = v2 * fac;
   next_gval_available = true;
   return (v1 * fac);
   }

}; // end namespace
