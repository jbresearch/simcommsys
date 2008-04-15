/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_prof_burst.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>

namespace libcomm {

// commsys functions

void commsys_prof_burst::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   assert(i >= 0 && i < get_iter());
   const int skip = count()/get_iter();
   // Update the relevant count for every symbol in error
   // Check the first symbol first
   if(source(0) != decoded(0))
      result(skip*i + 0) += 1.0;
   // For each remaining symbol
   for(int t=1; t<get_symbolsperblock(); t++)
      {
      if(source(t-1) != decoded(t-1))
         result(skip*i + 3) += 1.0 / double(get_symbolsperblock()-1);
      if(source(t) != decoded(t))
         {
         // Keep separate counts, depending on whether the previous symbol was in error
         if(source(t-1) != decoded(t-1))
            result(skip*i + 2) += 1.0 / double(get_symbolsperblock()-1);
         else
            result(skip*i + 1) += 1.0 / double(get_symbolsperblock()-1);
         }
      }
   }

}; // end namespace
