/* randgen
 * ~~~~~~~
 * a class which returns a double precision (64-bit) random value between
 * 0.0 and 1.0 (both inclusive) generated using the subtractive technique
 * due to Knuth. This algorithm was found to give very good results in the
 * communications lab during the third year.
 *
 * notes:
 *   the subtractive algorithm has a very long period (necessary for low
 *   bit error rates in the tested data stream)
 *   it also does not suffer from low-order correlations (facilitating its
 *   use with a variable number of bits/code in the data stream)
 */

#ifndef __randgen_h
#define __randgen_h

#include "config.h"
#include "vcs.h"
#include <math.h>
#include <iostream.h>

extern const vcs randgen_version;

class randgen {
   static const int32s	mbig;
   static const int32s	mseed;
   double   next_gval;
   bool     ready;
   int32s	next, nextp;
   int32s	ma[56], mj;
   inline void advance(void);
#ifdef DEBUG   
   int32u	counter;
#endif
public:
   randgen(int32u s =0);
   ~randgen();
   void seed(int32u s);
   inline int32u ival(int32u m);	// return unsigned integer modulo 'm'
   inline double fval();		// return floating point value in [0,1]
   inline double gval();		// return gaussian-distributed double (zero mean, unit variance)
   inline double gval(const double sigma);	// as gval(), but set std dev to given sigma
};

inline void randgen::advance()
   {
   if(++next == 56) next = 1;
   if(++nextp == 56) nextp = 1;
   mj = ma[next] - ma[nextp];
   if(mj < 0) mj += mbig;
   ma[next] = mj;
#ifdef DEBUG   
   counter++;
   if(counter == 0)
      cerr << "DEBUG: randgen (" << this << ") counter looped ***.\n";
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

#endif
