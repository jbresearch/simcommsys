#include "sigspace.h"

namespace libcomm {

const libbase::vcs sigspace::version("Signal Space module (sigspace)", 1.30);


// creator / destructor

sigspace::sigspace(const double i, const double q)
   {
   inphase = i;
   quad = q;
   }   

// comparison operations

bool sigspace::operator==(const sigspace& a)
   {
   return(inphase == a.inphase && quad == a.quad);
   }

bool sigspace::operator!=(const sigspace& a)
   {
   return(inphase != a.inphase || quad != a.quad);
   }

// arithmetic operations

sigspace& sigspace::operator+=(const sigspace& a)
   {
   inphase += a.inphase;
   quad += a.quad;
   return *this;
   }

sigspace& sigspace::operator-=(const sigspace& a)
   {
   inphase -= a.inphase;
   quad -= a.quad;
   return *this;
   }

sigspace& sigspace::operator*=(const double a)
   {
   inphase *= a;
   quad *= a;
   return *this;
   }

sigspace& sigspace::operator/=(const double a)
   {
   inphase /= a;
   quad /= a;
   return *this;
   }

// arithmetic operations - friends

sigspace operator+(const sigspace& a, const sigspace& b)
   {
   sigspace c = a;
   c += b;
   return c;
   }
   
sigspace operator-(const sigspace& a, const sigspace& b)
   {
   sigspace c = a;
   c -= b;
   return c;
   }
   
sigspace operator*(const sigspace& a, const double b)
   {
   sigspace c = a;
   c *= b;
   return c;
   }
   
sigspace operator/(const sigspace& a, const double b)
   {
   sigspace c = a;
   c /= b;
   return c;
   }

sigspace operator*(const double a, const sigspace& b)
   {
   sigspace c = b;
   c *= a;
   return c;
   }
   
sigspace operator/(const double a, const sigspace& b)
   {
   sigspace c = b;
   c /= a;
   return c;
   }

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
