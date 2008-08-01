/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "mapper.h"

namespace libcomm {

// Setup functions

void mapper::set_parameters(const int N, const int M, const int S)
   {
   mapper::N = N;
   mapper::M = M;
   mapper::S = S;
   setup();
   }

}; // end namespace
