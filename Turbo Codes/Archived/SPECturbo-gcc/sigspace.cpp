#include "sigspace.h"

const vcs sigspace_version("Signal Space module (sigspace)", 1.00);

ostream& operator<<(ostream& s, const sigspace& x)
   {
   s.setf(ios::fixed, ios::floatfield);
   s.precision(6);
   s << '[' << x.inphase << ',';
   s.setf(ios::fixed, ios::floatfield);
   s.precision(6);
   s << x.quad << ']';
   return s;
   }
   
istream& operator>>(istream& s, sigspace& x)
   {
   double i = 0, q = 0;
   char c = 0;
   
   s >> c;
   if(c == '[')
      {
      s >> i >> c;
      if(c == ',')
         s >> q >> c;
      else
         s.clear(ios::badbit);
      if(c != ']')
         s.clear(ios::badbit);
      }
   else
      s.clear(ios::badbit);
   
   if(s)
      x = sigspace(i, q);
      
   return s;
   }


