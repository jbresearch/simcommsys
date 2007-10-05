#ifndef __pixel_h
#define __pixel_h

#include "config.h"
#include "vcs.h"
#include "itfunc.h"
#include <math.h>

/*
  Version 1.01 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.

  Version 1.02 (13 Oct 2006)
  added explicit conversion to integer in quant().

  Version 1.10 (10 Nov 2006)
  * defined class and associated data within "libimage" namespace.
*/

namespace libimage {

class pixel {
   static const libbase::vcs version;
   double r, g, b;
   double limit(const double x) const;
   int quant(const double x, const int bits) const;
public:
   pixel(libbase::int8u red=0, libbase::int8u green=0, libbase::int8u blue=0);
   pixel(double gray);

   void limit();
   void quantise(const int bpc);

   pixel& operator+=(const pixel& a);
   friend pixel operator+(const pixel& a, const pixel& b);

   libbase::int8u red() const   { return quant(r,8); };
   libbase::int8u green() const { return quant(g,8); };
   libbase::int8u blue() const  { return quant(b,8); };
   double luminance() const { return 0.299*r + 0.587*g + 0.114*b; };
};

// Internal Functions

inline double pixel::limit(const double x) const
   {
   if(x > 1.0)
      return 1.0;
   if(x < 0.0)
      return 0.0;
   return x;
   }

// converts a real value by clipping into [1,0] to an integer between (1<<bits)-1 and 0
inline int pixel::quant(const double x, const int bits) const
   {
   const int m = (1<<bits)-1;
   return int(libbase::round(m*limit(x)));
   }


// External Functions

inline pixel::pixel(libbase::int8u red, libbase::int8u green, libbase::int8u blue)
   {
   const double m = 0xff;
   r = red/m;
   g = green/m;
   b = blue/m;
   }

inline pixel::pixel(double gray)
   {
   r = g = b = gray;
   }

inline void pixel::limit()
   {
   r = limit(r);
   g = limit(g);
   b = limit(b);
   }

inline void pixel::quantise(const int bpc)
   {
   const double m = (1<<bpc)-1;
   using libbase::round;
   r = round(m*limit(r))/m;
   g = round(m*limit(g))/m;
   b = round(m*limit(b))/m;
   }

inline pixel& pixel::operator+=(const pixel& a)
   {
   r += a.r;
   g += a.g;
   b += a.b;
   return *this;
   }

inline pixel operator+(const pixel& a, const pixel& b)
   {
   pixel c = a;
   c += b;
   return c;
   }

}; // end namespace

#endif
