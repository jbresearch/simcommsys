/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_prof_pos.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>

namespace libcomm {

// constructor / destructor

commsys_prof_pos::commsys_prof_pos(libbase::randgen *src, codec *cdc, modulator<sigspace> *modem, puncture *punc, channel<sigspace> *chan) : \
   commsys<sigspace>(src, cdc, modem, punc, chan)
   {
   }

// commsys functions

void commsys_prof_pos::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   const int skip = count()/iter;
   // Update the count for every bit in error
   for(int t=0; t<tau-m; t++)
      if(source(t) != decoded(t))
         result(skip*i + t)++;
   }

}; // end namespace
