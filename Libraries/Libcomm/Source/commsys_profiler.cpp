#include "commsys_profiler.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>

namespace libcomm {

const libbase::vcs commsys_profiler::version("Communication System Profiler module (commsys_profiler)", 1.50);

// constructor / destructor

commsys_profiler::commsys_profiler(libbase::randgen *src, codec *cdc, modulator *modem, puncture *punc, channel *chan) : \
   commsys(src, cdc, modem, punc, chan)
   {
   }

// commsys functions

void commsys_profiler::cycleonce(libbase::vector<double>& result)
   {
   // Create source stream
   createsource();
   // Full cycle from Encode through Demodulate
   transmitandreceive();

   // For every iteration possible
   const int skip = count()/iter;
   for(int i=0; i<iter; i++)
      {
      // Decode
      cdc->decode(decoded);

      // Count the number of errors
      int delta = 0;
      for(int t=0; t<tau-m; t++)
         delta += libbase::weight(source(t) ^ decoded(t));
      
      // Update the count for that number of bit errors
      result(skip*i + delta)++;
      }
   }

}; // end namespace
