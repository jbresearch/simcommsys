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

// Setup functions

/*!
   \brief Initialize internal computed parameters

   \note This function is only responsible for initializing parameters
         that are specific to this object/derivation. Anything else
         should get done automatically when the base serializer or
         constructor is called.
*/
template <class S> void basic_commsys<S>::init()
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
   }

/*!
   \brief Sets up system with no bound objects.

   \note This function is only responsible for clearing pointers to
         objects that are specific to this object/derivation.
         Anything else should get done automatically when the base
         serializer or constructor is called.
*/
template <class S> void basic_commsys<S>::clear()
   {
   src = NULL;
   cdc = NULL;
   chan = NULL;
   internallyallocated = true;
   }

/*!
   \brief Removes association with bound objects

   This function performs two things:
   - Deletes any internally-allocated bound objects
   - Sets up the system with no bound objects

   \note This function is only responsible for deleting bound
         objects that are specific to this object/derivation.
         Anything else should get done automatically when the base
         serializer or constructor is called.
*/
template <class S> void basic_commsys<S>::free()
   {
   if(internallyallocated)
      {
      delete src;
      delete cdc;
      delete chan;
      }
   clear();
   }

// Internal functions

/*!
   \brief Create source sequence to be encoded
   \return Source sequence of the required length

   The source sequence consists of uniformly random symbols followed by a
   tail sequence if required by the given codec.
*/
template <class S> libbase::vector<int> basic_commsys<S>::createsource()
   {
   libbase::vector<int> source(tau);
   for(int t=0; t<tau-m; t++)
      source(t) = src->ival(K);
   for(int t=tau-m; t<tau; t++)
      source(t) = fsm::tail;
   return source;
   }

/*!
   \brief Count the number of bit errors in the last encode/decode cycle
   \return Error count in bits
*/
template <class S> int basic_commsys<S>::countbiterrors(const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
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
template <class S> int basic_commsys<S>::countsymerrors(const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   int symerrors = 0;
   for(int t=0; t<tau-m; t++)
      if(source(t) != decoded(t))
         symerrors++;
   return symerrors;
   }

/*!
   \brief Update result set
   \param[out] result   Vector containing the set of results to be updated
   \param[in]  i        Iteration just performed
   \param[in]  source   Source data sequence
   \param[in]  decoded  Decoded data sequence

   Results are organized as (BER,SER,FER), repeated for every iteration that
   needs to be performed.
*/
template <class S> void basic_commsys<S>::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   assert(i >= 0 && i < iter);
   // Count errors
   int biterrors = countbiterrors(source, decoded);
   int symerrors = countsymerrors(source, decoded);
   // Estimate the BER, SER, FER
   result(3*i + 0) += biterrors / double((tau-m)*k);
   result(3*i + 1) += symerrors / double((tau-m));
   result(3*i + 2) += symerrors ? 1 : 0;
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
template <class S> void basic_commsys<S>::cycleonce(libbase::vector<double>& result)
   {
   assert(result.size() == count());
   // Create source stream
   libbase::vector<int> source = createsource();
   // Full cycle from Encode through Demodulate
   transmitandreceive(source);
   // For every iteration
   for(int i=0; i<iter; i++)
      {
      // Decode & update results
      libbase::vector<int> decoded;
      cdc->decode(decoded);
      updateresults(result, i, source, decoded);
      }
   }

// Constructors / Destructors

/*!
   \brief Main public constructor

   Initializes system with bound objects as supplied by user.
*/
template <class S> basic_commsys<S>::basic_commsys(libbase::randgen *src, codec *cdc, channel<S> *chan)
   {
   basic_commsys<S>::src = src;
   basic_commsys<S>::cdc = cdc;
   basic_commsys<S>::chan = chan;
   internallyallocated = false;
   init();
   }

/*!
   \brief Copy constructor

   Initializes system with bound objects cloned from supplied system.

   \todo Fix cast when cloning channel: this should not be necessary.
*/
template <class S> basic_commsys<S>::basic_commsys(const basic_commsys<S>& c)
   {
   basic_commsys<S>::src = new libbase::randgen;
   basic_commsys<S>::cdc = c.cdc->clone();
   basic_commsys<S>::chan = (channel<sigspace> *)c.chan->clone();
   internallyallocated = true;
   init();
   }

// Experiment parameter handling

template <class S> void basic_commsys<S>::seed(int s)
   {
   src->seed(s);
   cdc->seed(s+1);
   chan->seed(s+2);
   }

// Experiment handling

template <class S> void basic_commsys<S>::sample(libbase::vector<double>& result)
   {
   // initialise result vector
   result.init(count());
   result = 0;
   // compute a single cycle
   cycleonce(result);
   }

// Description & Serialization

template <class S> std::string basic_commsys<S>::description() const
   {
   std::ostringstream sout;
   sout << "Communication System: ";
   sout << cdc->description() << ", ";
   sout << chan->description();
   return sout.str();
   }

template <class S> std::ostream& basic_commsys<S>::serialize(std::ostream& sout) const
   {
   sout << chan;
   sout << cdc;
   return sout;
   }

template <class S> std::istream& basic_commsys<S>::serialize(std::istream& sin)
   {
   free();
   src = new libbase::randgen;
   sin >> chan;
   sin >> cdc;
   internallyallocated = true;
   init();
   return sin;
   }

// Explicit Realizations

template class basic_commsys<sigspace>;

// *** Specific to commsys<sigspace> ***

const libbase::serializer commsys<sigspace>::shelper("experiment", "commsys<sigspace>", commsys<sigspace>::create);

// Setup functions

/*!
   \copydoc basic_commsys<S>::init()

   This function sets the average energy per data bit in the bound channel model.
   The value depends on:
   - Rate of codec
   - Rate of puncturing
   - Average energy per uncoded bit in the modulation scheme
*/
void commsys<sigspace>::init()
   {
   // set up channel energy/bit (Eb)
   double rate = cdc->rate();
   if(punc != NULL)
      rate /= punc->rate();
   chan->set_eb(modem->bit_energy() / rate);
   }

void commsys<sigspace>::clear()
   {
   modem = NULL;
   punc = NULL;
   }

void commsys<sigspace>::free()
   {
   if(internallyallocated)
      {
      delete modem;
      delete punc;
      }
   clear();
   }

// Internal functions

/*!
   \copydoc basic_commsys<S>::transmitandreceive()

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
void commsys<sigspace>::transmitandreceive(libbase::vector<int>& source)
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

// Constructors / Destructors

commsys<sigspace>::commsys(libbase::randgen *src, codec *cdc, modulator *modem, puncture *punc, channel<sigspace> *chan) : basic_commsys<sigspace>(src, cdc, chan)
   {
   commsys::modem = modem;
   commsys::punc = punc;
   init();
   }

commsys<sigspace>::commsys(const commsys& c) : basic_commsys<sigspace>(c)
   {
   commsys::modem = c.modem->clone();
   commsys::punc = c.punc->clone();
   init();
   }

// Description & Serialization

std::string commsys<sigspace>::description() const
   {
   std::ostringstream sout;
   sout << basic_commsys<sigspace>::description() << ", ";
   sout << modem->description();
   if(punc != NULL)
      sout << ", " << punc->description();
   return sout.str();
   }

std::ostream& commsys<sigspace>::serialize(std::ostream& sout) const
   {
   sout << modem;
   //const bool ispunctured = (punc != NULL);
   //sout << int(ispunctured) << "\n";
   //if(ispunctured)
   //   sout << punc;
   basic_commsys<sigspace>::serialize(sout);
   return sout;
   }

std::istream& commsys<sigspace>::serialize(std::istream& sin)
   {
   free();
   sin >> modem;
   //int ispunctured;
   //sin >> ispunctured;
   //if(ispunctured != 0)
   //   sin >> punc;
   basic_commsys<sigspace>::serialize(sin);
   init();
   return sin;
   }

}; // end namespace
