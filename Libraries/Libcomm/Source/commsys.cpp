#include "commsys.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>
#include <sstream>

namespace libcomm {

const libbase::vcs commsys::version("Communication System module (commsys)", 1.62);

const libbase::serializer commsys::shelper("experiment", "commsys", commsys::create);

// internal functions

void commsys::createsource()
   {
   int t;
   source.init(tau);
   for(t=0; t<tau-m; t++)
      source(t) = src->ival(K);
   for(t=tau-m; t<tau; t++)
      source(t) = fsm::tail;
   }

void commsys::transmitandreceive()
   {
   cdc->encode(source, encoded);
   modem->modulate(N, encoded, signal1);
   if(punc != NULL)
      {
      punc->transform(signal1, signal2);
      chan->transmit(signal2, signal2);
      modem->demodulate(*chan, signal2, ptable2);
      punc->inverse(ptable2, ptable1);
      }
   else
      {
      chan->transmit(signal1, signal1);
      modem->demodulate(*chan, signal1, ptable1);
      }
   cdc->translate(ptable1);
   }

void commsys::cycleonce(libbase::vector<double>& result)
   {
   // Create source stream
   createsource();
   // Full cycle from Encode through Demodulate
   transmitandreceive();

   // For every iteration
   int delta = 0;
   for(int i=0; i<iter; i++)
      {
      // Decode
      cdc->decode(decoded);
      // Count the number of errors
      delta = 0;
      for(int t=0; t<tau-m; t++)
         delta += libbase::weight(source(t) ^ decoded(t));
      // Estimate the BER
      result(2*i + 0) += delta / double((tau-m)*k);
      // Estimate the FER (Frame Error Rate)
      result(2*i + 1) += delta ? 1 : 0;
      }
   }

void commsys::init()
   {
   tau = cdc->block_size();
   m = cdc->tail_length();
   N = cdc->num_outputs();
   K = cdc->num_inputs();
   k = int(libbase::round(libbase::log2(double(K))));
   if(K != 1<<k)
      {
      std::cerr << "FATAL ERROR (commsys): can only estimate BER for a q-ary source (" << k << ", " << K << ").\n";
      exit(1);
      }
   iter = cdc->num_iter();
   // set up channel energy/bit (Eb)
   double rate = cdc->rate();
   if(punc != NULL)
      rate /= punc->rate();
   chan->set_eb(modem->bit_energy() / rate);
   }

void commsys::clear()
   {
   src = NULL;
   cdc = NULL;
   modem = NULL;
   punc = NULL;
   chan = NULL;
   internallyallocated = true;
   }

void commsys::free()
   {
   if(internallyallocated)
      {
      delete src;
      delete cdc;
      delete modem;
      delete punc;
      delete chan;
      }
   clear();
   }

commsys::commsys()
   {
   clear();
   }
   
// public constructor / destructor

commsys::commsys(libbase::randgen *src, codec *cdc, modulator *modem, puncture *punc, channel *chan)
   {
   commsys::src = src;
   commsys::cdc = cdc;
   commsys::modem = modem;
   commsys::punc = punc;
   commsys::chan = chan;
   internallyallocated = false;
   init();
   }

commsys::commsys(const commsys& c)
   {
   commsys::src = new libbase::randgen;
   commsys::cdc = c.cdc->clone();
   commsys::modem = c.modem->clone();
   commsys::punc = c.punc->clone();
   commsys::chan = c.chan->clone();
   internallyallocated = true;
   init();
   }

// experiment functions

void commsys::seed(int s)
   {
   src->seed(s);
   cdc->seed(s);
   chan->seed(s);
   }

void commsys::sample(libbase::vector<double>& result, int& samplecount)
   {
   // initialise result vector
   result.init(count());
   result = 0;

   // iterate for 500ms, which is a good compromise between efficiency and usability
   int passes=0;
   libbase::timer t;
   while(t.elapsed() < 0.5)
      {
      cycleonce(result);   // will update result
      passes++;
      samplecount++;
      }
   t.stop();   // to avoid expiry

   // update result
   result /= double(passes);
   }

// description output

std::string commsys::description() const
   {
   std::ostringstream sout;
   sout << "Communication System: ";
   sout << cdc->description() << ", ";
   sout << modem->description() << ", ";
   if(punc != NULL)
      sout << punc->description() << ", ";
   sout << chan->description();
   return sout.str();
   }

// object serialization - saving

std::ostream& commsys::serialize(std::ostream& sout) const
   {
   sout << cdc;
   sout << modem;
   const bool ispunctured = (punc != NULL);
   sout << int(ispunctured) << "\n";
   if(ispunctured)
      sout << punc;
   sout << chan;
   return sout;
   }

// object serialization - loading

std::istream& commsys::serialize(std::istream& sin)
   {
   free();
   src = new libbase::randgen;
   sin >> cdc;
   sin >> modem;
   int ispunctured;
   sin >> ispunctured;
   if(ispunctured != 0)
      sin >> punc;
   sin >> chan;
   init();
   return sin;
   }

}; // end namespace
