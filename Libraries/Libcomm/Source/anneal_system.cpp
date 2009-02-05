/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "anneal_system.h"

namespace libcomm {

std::ostream& operator<<(std::ostream& sout, const anneal_system& x)
   {
   return x.output(sout);
   }

}; // end namespace
