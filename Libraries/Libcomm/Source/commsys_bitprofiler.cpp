#include "commsys_bitprofiler.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>

namespace libcomm {

const libbase::vcs commsys_bitprofiler::version("Communication System Bit Profiler module (commsys_bitprofiler)", 1.40);

// constructor / destructor

commsys_bitprofiler::commsys_bitprofiler(libbase::randgen *src, codec *cdc, modulator *modem, puncture *punc, channel *chan) : \
   commsys(src, cdc, modem, punc, chan)
   {
   }

// commsys functions

void commsys_bitprofiler::cycleonce(libbase::vector<double>& result)
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

      // Update the count for every bit in error
      for(int t=0; t<tau-m; t++)
         if(source(t) ^ decoded(t))
            result(skip*i + t)++;
      }
   }

}; // end namespace
