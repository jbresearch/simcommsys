/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_profiler.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>

namespace libcomm {

// constructor / destructor

commsys_profiler::commsys_profiler(libbase::randgen *src, codec *cdc, modulator *modem, puncture *punc, channel<sigspace> *chan) : \
   commsys(src, cdc, modem, punc, chan)
   {
   }

// commsys functions

void commsys_profiler::cycleonce(libbase::vector<double>& result)
   {
   // Create source stream
   libbase::vector<int> source = createsource();
   // Full cycle from Encode through Demodulate
   transmitandreceive(source);
   // For every iteration
   const int skip = count()/iter;
   for(int i=0; i<iter; i++)
      {
      // Decode & count errors
      libbase::vector<int> decoded;
      cdc->decode(decoded);
      int biterrors = countbiterrors(source, decoded);
      //int symerrors = countsymerrors(source, decoded);
      // Update the count for that number of bit errors
      result(skip*i + biterrors)++;
      }
   }

}; // end namespace
