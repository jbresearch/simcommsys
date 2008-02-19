/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_prof_sym.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>

namespace libcomm {

// commsys functions

void commsys_prof_sym::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   const int skip = count()/get_iter();
   // Update the count for every bit in error
   for(int t=0; t<get_symbolsperblock(); t++)
      if(source(t) != decoded(t))
         result(skip*i + source(t))++;
   }

}; // end namespace
