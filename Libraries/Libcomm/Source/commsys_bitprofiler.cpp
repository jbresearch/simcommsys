/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_bitprofiler.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>

namespace libcomm {

// constructor / destructor

commsys_bitprofiler::commsys_bitprofiler(libbase::randgen *src, codec *cdc, modulator *modem, puncture *punc, channel<sigspace> *chan) : \
   commsys<sigspace>(src, cdc, modem, punc, chan)
   {
   }

// commsys functions

void commsys_bitprofiler::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   const int skip = count()/iter;
   // Update the count for every bit in error
   for(int t=0; t<tau-m; t++)
      if(source(t) != decoded(t))
         result(skip*i + t)++;
   }

}; // end namespace
