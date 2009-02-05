/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "logreal.h"

namespace libbase {

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
