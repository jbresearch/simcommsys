/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "anneal_system.h"

namespace libcomm {

const libbase::vcs anneal_system::version("Simulated Annealing System Base module (anneal_system)", 1.30);

std::ostream& operator<<(std::ostream& sout, const anneal_system& x)
   {
   return x.output(sout);
   }

}; // end namespace
