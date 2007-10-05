#include "logreal.h"

const vcs logreal_version("Logarithm Arithmetic module (logreal)", 1.00);

// Input/Output Operations

ostream& operator<<(ostream& s, const logreal& x)
   {        
   const double lg = -x.logval/log(10.0);

   const int flags = s.flags();
   s.setf(ios::fixed, ios::floatfield);
   s << pow(10.0, lg-floor(lg));
   s.setf(ios::showpos);
   s << "e" << int(floor(lg));
   s.flags(flags);

   return s;
   }

