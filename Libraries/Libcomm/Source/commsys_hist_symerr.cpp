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

// constructor / destructor

commsys_hist_symerr::commsys_hist_symerr(libbase::randgen *src, codec *cdc, modulator<sigspace> *modem, puncture *punc, channel<sigspace> *chan) : \
   commsys<sigspace>(src, cdc, modem, punc, chan)
   {
   }

// commsys functions

void commsys_hist_symerr::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   const int skip = count()/iter;
   int biterrors = countbiterrors(source, decoded);
   // Update the count for that number of bit errors
   result(skip*i + biterrors)++;
   }

}; // end namespace
