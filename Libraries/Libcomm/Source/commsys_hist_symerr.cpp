/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_hist_symerr.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>

namespace libcomm {

// commsys functions

void commsys_hist_symerr::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   const int skip = count()/get_iter();
   int symerrors = countsymerrors(source, decoded);
   // Update the count for that number of symbol errors (may be zero)
   result(skip*i + symerrors)++;
   }

}; // end namespace
