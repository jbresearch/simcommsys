/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "mpgnu.h"
#include <stdlib.h>

namespace libbase {

const vcs mpgnu::version("GNU Multi-Precision Arithmetic module (mpgnu)", 1.10);

#ifdef GMP
mpf_t mpgnu::dblmin;
mpf_t mpgnu::dblmax;
#endif

void mpgnu::init()
   {
#ifndef GMP
   std::cerr << "FATAL ERROR (mpgnu): GNU Multi-Precision not implemented - cannot initialise.\n";
   exit(1);
#else
   static bool ready = false;
   if(!ready)
      {
      mpf_init_set_d(dblmin, DBL_MIN);
      mpf_init_set_d(dblmax, DBL_MAX);
      ready = true;
      }
#endif
   }

}; // end namespace
