/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "plmod.h"

namespace libcomm {

double plmod(const double u)
   {
   if(u < 0.5)
      return u + 0.5;
   else if(u > 0.5)
      return u - 0.5;
   else
      return 0;
   }

}; // end namespace
