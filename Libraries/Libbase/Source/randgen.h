#ifndef __randgen_h
#define __randgen_h

#include "config.h"
#include <math.h>
#include <iostream>

namespace libbase {

/*!
   \brief   Random Generator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   A class which returns a double precision (64-bit) random value between
   0.0 and 1.0 (both inclusive) generated using the subtractive technique
   due to Knuth. This algorithm was found to give very good results in the
   communications lab during the third year.

   \note
   - The subtractive algorithm has a very long period (necessary for low
     bit error rates in the tested data stream)
   - It also does not suffer from low-order correlations (facilitating its
     use with a variable number of bits/code in the data stream)

   \version 1.01 (16 Nov 2001)
   moved 'ready' and 'next_gval' from static objects within gval() to member objects.

   \version 1.02 (23 Feb 2002)
   added flushes to all end-of-line clog outputs, to clean up text user interface.

   \version 1.03 (6 Mar 2002)
   changed vcs version variable from a global to a static class variable.
   also changed use of iostream from global to std namespace.

   \version 1.10 (26 Oct 2006)
   - defined class and associated data within "libbase" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.11 (22 Jan 2008)
   - Changed debug output to go to trace instead of clog.
*/

class randgen {
   static const int32s  mbig;
   static const int32s  mseed;
   int32s       next, nextp;
   int32s       ma[56], mj;
   inline void advance(void);
   bool ready;
   double next_gval;
#ifdef DEBUG
   int32u       counter;
#endif
public:
   randgen(int32u s =0);
   ~randgen();
   void seed(int32u s);
   inline int32u ival(int32u m);        // return unsigned integer modulo 'm'
   inline double fval();                // return floating point value in [0,1]
   inline double gval();                // return gaussian-distributed double (zero mean, unit variance)
   inline double gval(const double sigma);      // as gval(), but set std dev to given sigma
};

inline void randgen::advance()
   {
   if(++next >= 56) next = 1;
   if(++nextp >= 56) nextp = 1;
   mj = ma[next] - ma[nextp];
   if(mj < 0) mj += mbig;
   ma[next] = mj;
#ifdef DEBUG
   counter++;
   if(counter == 0)
      trace << "DEBUG: randgen (" << this << ") counter looped ***.\n" << std::flush;
#endif
   }

inline int32u randgen::ival(int32u m)
   {
   advance();
   return(mj % m);
   }

inline double randgen::fval()
   {
   advance();
   return(mj/(double)mbig);
   }

inline double randgen::gval()
   {
   if(!ready)
      {
      double v1, v2, rsq;
      do {
         v1 = 2.0 * fval() - 1.0;
         v2 = 2.0 * fval() - 1.0;
         rsq = (v1*v1) + (v2*v2);
         } while(rsq >= 1.0 || rsq == 0.0);
      double fac = sqrt(-2.0*log(rsq)/rsq);
      next_gval = v2 * fac;
      ready = true;
      return (v1 * fac);
      }

   ready = false;
   return next_gval;
   }

inline double randgen::gval(const double sigma)
   {
   return gval() * sigma;
   }

}; // end namespace

#endif
