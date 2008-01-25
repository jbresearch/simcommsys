/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>
#include <sstream>

namespace libcomm {

const libbase::serializer commsys::shelper("experiment", "commsys", commsys::create);

// Setup functions

/*!
   \brief Initialize the communications system

   This function performs two things:
   - Setting up the values of the computed parameters (including any
     necessary validation on their values)
   - Sets the average energy per bit in the bound channel model
*/
void commsys::init()
   {
   tau = cdc->block_size();
   m = cdc->tail_length();
   N = cdc->num_outputs();
   K = cdc->num_inputs();
   k = int(round(log2(double(K))));
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

/*!
   \brief Sets up system with no bound objects.
*/
void commsys::clear()
   {
   src = NULL;
   cdc = NULL;
   modem = NULL;
   punc = NULL;
   chan = NULL;
   internallyallocated = true;
   }

/*!
   \brief Removes association with bound objects

   This function performs two things:
   - Deletes any internally-allocated bound objects
   - Sets up the system with no bound objects
*/
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

// Internal functions

/*!
   \brief Create source sequence to be encoded

   The source sequence consists of uniformly random symbols followed by a
   tail sequence if required by the given codec.
*/
void commsys::createsource()
   {
   int t;
   source.init(tau);
   for(t=0; t<tau-m; t++)
      source(t) = src->ival(K);
   for(t=tau-m; t<tau; t++)
      source(t) = fsm::tail;
   }

/*!
   \brief Perform a complete transmit/receive cycle, except for final decoding

   The cycle consists of the steps depicted in the following diagram:
   \dot
   digraph txrxcycle {
      // Make figure left-to-right
      rankdir = LR;
      // block definitions
      node [ shape=box ];
      encode [ label="Encode" ];
      modulate [ label="Modulate" ];
      puncture [ style=dotted,label="Puncture" ];
      transmit [ label="Transmit" ];
      demodulate [ label="Demodulate" ];
      unpuncture [ style=dotted,label="Inverse Puncture" ];
      translate [ label="Translate" ];
      // path definitions
      encode -> modulate;
      modulate -> transmit;
      transmit -> demodulate;
      demodulate -> translate;
      modulate -> puncture [ style=dotted ];
      puncture -> transmit [ style=dotted ];
      demodulate -> unpuncture [ style=dotted ];
      unpuncture -> translate [ style=dotted ];
   }
   \enddot

   The dotted lines and blocks indicate optional sections to support puncturing,
   which is currently done in signal-space.
*/
void commsys::transmitandreceive()
   {
   libbase::vector<int> encoded;
   cdc->encode(source, encoded);
   libbase::vector<sigspace> signal1;
   modem->modulate(N, encoded, signal1);
   libbase::matrix<double> ptable1;
   if(punc != NULL)
      {
      libbase::vector<sigspace> signal2;
      punc->transform(signal1, signal2);
      chan->transmit(signal2, signal2);
      libbase::matrix<double> ptable2;
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

/*!
   \brief Count the number of bit errors in the last encode/decode cycle
   \return Error count in bits
*/
int commsys::countbiterrors() const
   {
   int biterrors = 0;
   for(int t=0; t<tau-m; t++)
      biterrors += libbase::weight(source(t) ^ decoded(t));
   return biterrors;
   }

/*!
   \brief Count the number of symbol errors in the last encode/decode cycle
   \return Error count in symbols
*/
int commsys::countsymerrors()
   {
   int symerrors = 0;
   for(int t=0; t<tau-m; t++)
      if(source(t) != decoded(t))
         symerrors++;
   return symerrors;
   }

/*!
   \brief Perform a complete encode->transmit->receive cycle
   \param[out] result   Vector containing the set of results to be updated

   Results are organized as (BER,SER,FER), repeated for every iteration that
   needs to be performed.

   \note It is assumed that the result vector serves as an accumulator, so that
         every cycle effectively adds to this result. The caller is responsible
         to divide by the appropriate amount at the end to compute a meaningful
         average.
*/
void commsys::cycleonce(libbase::vector<double>& result)
   {
   // Create source stream
   createsource();
   // Full cycle from Encode through Demodulate
   transmitandreceive();
   // For every iteration
   for(int i=0; i<iter; i++)
      {
      // Decode & count errors
      cdc->decode(decoded);
      int biterrors = countbiterrors();
      int symerrors = countsymerrors();
      // Estimate the BER, SER, FER
      result(3*i + 0) += biterrors / double((tau-m)*k);
      result(3*i + 1) += symerrors / double((tau-m));
      result(3*i + 2) += symerrors ? 1 : 0;
      }
   }

// Constructors / Destructors

/*!
   \brief Main public constructor

   Initializes system with bound objects as supplied by user.
*/
commsys::commsys(libbase::randgen *src, codec *cdc, modulator *modem, puncture *punc, channel<sigspace> *chan)
   {
   commsys::src = src;
   commsys::cdc = cdc;
   commsys::modem = modem;
   commsys::punc = punc;
   commsys::chan = chan;
   internallyallocated = false;
   init();
   }

/*!
   \brief Copy constructor

   Initializes system with bound objects cloned from supplied system.

   \todo Fix cast when cloning channel: this should not be necessary.
*/
commsys::commsys(const commsys& c)
   {
   commsys::src = new libbase::randgen;
   commsys::cdc = c.cdc->clone();
   commsys::modem = c.modem->clone();
   commsys::punc = c.punc->clone();
   commsys::chan = (channel<sigspace> *)c.chan->clone();
   internallyallocated = true;
   init();
   }

// Experiment parameter handling

void commsys::seed(int s)
   {
   src->seed(s);
   cdc->seed(s+1);
   chan->seed(s+2);
   }

// Experiment handling

void commsys::sample(libbase::vector<double>& result)
   {
   // initialise result vector
   result.init(count());
   result = 0;
   // compute a single cycle
   cycleonce(result);
   }

// Description & Serialization

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

std::ostream& commsys::serialize(std::ostream& sout) const
   {
   sout << chan;
   sout << modem;
   sout << cdc;
   //const bool ispunctured = (punc != NULL);
   //sout << int(ispunctured) << "\n";
   //if(ispunctured)
   //   sout << punc;
   return sout;
   }

std::istream& commsys::serialize(std::istream& sin)
   {
   free();
   src = new libbase::randgen;
   sin >> chan;
   sin >> modem;
   sin >> cdc;
   //int ispunctured;
   //sin >> ispunctured;
   //if(ispunctured != 0)
   //   sin >> punc;
   internallyallocated = true;
   init();
   return sin;
   }

}; // end namespace
