#include "logreal.h"

namespace libbase {

const vcs logreal::version("Logarithm Arithmetic module (logreal)", 1.11);

// Input/Output Operations

std::ostream& operator<<(std::ostream& s, const logreal& x)
   {        
   using std::ios;

   const double lg = -x.logval/log(10.0);

   const ios::fmtflags flags = s.flags();
   s.setf(ios::fixed, ios::floatfield);
   s << pow(10.0, lg-floor(lg));
   s.setf(ios::showpos);
   s << "e" << int(floor(lg));
   s.flags(flags);

   return s;
   }

}; // end namespace
