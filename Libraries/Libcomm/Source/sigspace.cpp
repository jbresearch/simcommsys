/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "sigspace.h"

namespace libcomm {

const libbase::vcs sigspace::version("Signal Space module (sigspace)", 1.30);


// stream input / output

std::ostream& operator<<(std::ostream& s, const sigspace& x)
   {
   using std::ios;
   s.setf(ios::fixed, ios::floatfield);
   s.precision(6);
   s << '[' << x.inphase << ',';
   s.setf(ios::fixed, ios::floatfield);
   s.precision(6);
   s << x.quad << ']';
   return s;
   }
   
std::istream& operator>>(std::istream& s, sigspace& x)
   {
   using std::ios;

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

}; // end namespace
