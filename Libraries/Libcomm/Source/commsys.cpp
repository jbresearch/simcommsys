/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys.h"

#include "fsm.h"
#include "gf.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>
#include <sstream>

namespace libcomm {


// *** Result Set ***

/*!
   \brief Count the number of bit errors in the last encode/decode cycle
   \return Error count in bits
*/
int commsys_errorrates::countbiterrors(const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   int biterrors = 0;
   for(int t=0; t<get_symbolsperblock(); t++)
      biterrors += libbase::weight(source(t) ^ decoded(t));
   return biterrors;
   }

/*!
   \brief Count the number of symbol errors in the last encode/decode cycle
   \return Error count in symbols
*/
int commsys_errorrates::countsymerrors(const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   int symerrors = 0;
   for(int t=0; t<get_symbolsperblock(); t++)
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
void commsys_errorrates::updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const
   {
   assert(i >= 0 && i < get_iter());
   // Count errors
   int biterrors = countbiterrors(source, decoded);
   int symerrors = countsymerrors(source, decoded);
   // Estimate the BER, SER, FER
   result(3*i + 0) += biterrors / double(get_symbolsperblock()*get_bitspersymbol());
   result(3*i + 1) += symerrors / double(get_symbolsperblock());
   result(3*i + 2) += symerrors ? 1 : 0;
   }


// *** Templated Common Base ***

// Setup functions

/*!
   \brief Initialize internal computed parameters

   \note This function is only responsible for initializing parameters
         that are specific to this object/derivation. Anything else
         should get done automatically when the base serializer or
         constructor is called.
*/
template <class S, class R> void basic_commsys<S,R>::init()
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
template <class S, class R> void basic_commsys<S,R>::clear()
   {
   src = NULL;
   cdc = NULL;
   modem = NULL;
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
template <class S, class R> void basic_commsys<S,R>::free()
   {
   if(internallyallocated)
      {
      delete src;
      delete cdc;
      delete modem;
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
template <class S, class R> libbase::vector<int> basic_commsys<S,R>::createsource()
   {
   libbase::vector<int> source(tau);
   for(int t=0; t<tau-m; t++)
      source(t) = src->ival(K);
   for(int t=tau-m; t<tau; t++)
      source(t) = fsm::tail;
   return source;
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
template <class S, class R> void basic_commsys<S,R>::cycleonce(libbase::vector<double>& result)
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
      R::updateresults(result, i, source, decoded);
      }
   }

// Constructors / Destructors

/*!
   \brief Main public constructor

   Initializes system with bound objects as supplied by user.
*/
template <class S, class R> basic_commsys<S,R>::basic_commsys(libbase::randgen *src, codec *cdc, modulator<S> *modem, channel<S> *chan)
   {
   basic_commsys<S,R>::src = src;
   basic_commsys<S,R>::cdc = cdc;
   basic_commsys<S,R>::modem = modem;
   basic_commsys<S,R>::chan = chan;
   internallyallocated = false;
   init();
   }

/*!
   \brief Copy constructor

   Initializes system with bound objects cloned from supplied system.

   \todo Fix cast when cloning channel: this should not be necessary.
   \todo Fix cast when cloning modem: this should not be necessary.
*/
template <class S, class R> basic_commsys<S,R>::basic_commsys(const basic_commsys<S,R>& c)
   {
   basic_commsys<S,R>::src = new libbase::randgen;
   basic_commsys<S,R>::cdc = c.cdc->clone();
   basic_commsys<S,R>::modem = (modulator<S> *)c.modem->clone();
   basic_commsys<S,R>::chan = (channel<S> *)c.chan->clone();
   internallyallocated = true;
   init();
   }

// Experiment parameter handling

template <class S, class R> void basic_commsys<S,R>::seed(int s)
   {
   src->seed(s);
   cdc->seed(s+1);
   chan->seed(s+2);
   }

// Experiment handling

template <class S, class R> void basic_commsys<S,R>::sample(libbase::vector<double>& result)
   {
   // initialise result vector
   result.init(count());
   result = 0;
   // compute a single cycle
   cycleonce(result);
   }

// Description & Serialization

template <class S, class R> std::string basic_commsys<S,R>::description() const
   {
   std::ostringstream sout;
   sout << "Communication System: ";
   sout << cdc->description() << ", ";
   sout << modem->description() << ", ";
   sout << chan->description();
   return sout.str();
   }

template <class S, class R> std::ostream& basic_commsys<S,R>::serialize(std::ostream& sout) const
   {
   sout << chan;
   sout << modem;
   sout << cdc;
   return sout;
   }

template <class S, class R> std::istream& basic_commsys<S,R>::serialize(std::istream& sin)
   {
   free();
   src = new libbase::randgen;
   sin >> chan;
   sin >> modem;
   sin >> cdc;
   internallyallocated = true;
   init();
   return sin;
   }

// Explicit Realizations

template class basic_commsys<sigspace>;
template class basic_commsys<bool>;
template class basic_commsys< libbase::gf<1,0x3> >;
template class basic_commsys< libbase::gf<2,0x7> >;
template class basic_commsys< libbase::gf<3,0xB> >;
template class basic_commsys< libbase::gf<4,0x13> >;


// *** Templated Base ***

// Internal functions

/*!
   \copydoc basic_commsys::transmitandreceive()

   The cycle consists of the steps depicted in the following diagram:
   \dot
   digraph txrxcycle {
      // Make figure left-to-right
      rankdir = LR;
      // block definitions
      node [ shape=box ];
      encode [ label="Encode" ];
      modulate [ label="Modulate" ];
      transmit [ label="Transmit" ];
      demodulate [ label="Demodulate" ];
      translate [ label="Translate" ];
      // path definitions
      encode -> modulate;
      modulate -> transmit;
      transmit -> demodulate;
      demodulate -> translate;
   }
   \enddot
*/
template <class S, class R> void commsys<S,R>::transmitandreceive(libbase::vector<int>& source)
   {
   libbase::vector<int> encoded;
   this->cdc->encode(source, encoded);
   libbase::vector<S> signal;
   this->modem->modulate(this->N, encoded, signal);
   libbase::matrix<double> ptable;
   this->chan->transmit(signal, signal);
   this->modem->demodulate(*this->chan, signal, ptable);
   this->cdc->translate(ptable);
   }

// Explicit Realizations

template class commsys<bool>;
template <> const libbase::serializer commsys<bool>::shelper("experiment", "commsys<bool>", commsys<bool>::create);
template class commsys< libbase::gf<1,0x3> >;
template <> const libbase::serializer commsys< libbase::gf<1,0x3> >::shelper("experiment", "commsys<gf<1,0x3>>", commsys< libbase::gf<1,0x3> >::create);
template class commsys< libbase::gf<2,0x7> >;
template <> const libbase::serializer commsys< libbase::gf<2,0x7> >::shelper("experiment", "commsys<gf<2,0x7>>", commsys< libbase::gf<2,0x7> >::create);
template class commsys< libbase::gf<3,0xB> >;
template <> const libbase::serializer commsys< libbase::gf<3,0xB> >::shelper("experiment", "commsys<gf<3,0xB>>", commsys< libbase::gf<3,0xB> >::create);
template class commsys< libbase::gf<4,0x13> >;
template <> const libbase::serializer commsys< libbase::gf<4,0x13> >::shelper("experiment", "commsys<gf<4,0x13>>", commsys< libbase::gf<4,0x13> >::create);


// *** Specific to commsys<sigspace> ***

// Setup functions

/*!
   \copydoc basic_commsys::init()

   This function sets the average energy per data bit in the bound channel model.
   The value depends on:
   - Rate of codec
   - Rate of puncturing
   - Average energy per uncoded bit in the modulation scheme
*/
template <class R> void commsys<sigspace,R>::init()
   {
   // set up channel energy/bit (Eb)
   double rate = this->cdc->rate();
   if(punc != NULL)
      rate /= punc->rate();
   this->chan->set_eb(this->modem->bit_energy() / rate);
   }

template <class R> void commsys<sigspace,R>::clear()
   {
   punc = NULL;
   }

template <class R> void commsys<sigspace,R>::free()
   {
   if(this->internallyallocated)
      {
      delete punc;
      }
   clear();
   }

// Internal functions

/*!
   \copydoc basic_commsys::transmitandreceive()

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
template <class R> void commsys<sigspace,R>::transmitandreceive(libbase::vector<int>& source)
   {
   libbase::vector<int> encoded;
   this->cdc->encode(source, encoded);
   libbase::vector<sigspace> signal1;
   this->modem->modulate(this->N, encoded, signal1);
   libbase::matrix<double> ptable1;
   if(punc != NULL)
      {
      libbase::vector<sigspace> signal2;
      punc->transform(signal1, signal2);
      this->chan->transmit(signal2, signal2);
      libbase::matrix<double> ptable2;
      this->modem->demodulate(*this->chan, signal2, ptable2);
      punc->inverse(ptable2, ptable1);
      }
   else
      {
      this->chan->transmit(signal1, signal1);
      this->modem->demodulate(*this->chan, signal1, ptable1);
      }
   this->cdc->translate(ptable1);
   }

// Constructors / Destructors

template <class R> commsys<sigspace,R>::commsys(libbase::randgen *src, codec *cdc, modulator<sigspace> *modem, puncture *punc, channel<sigspace> *chan) : basic_commsys<sigspace,R>(src, cdc, modem, chan)
   {
   commsys::punc = punc;
   init();
   }

template <class R> commsys<sigspace,R>::commsys(const commsys<sigspace,R>& c) : basic_commsys<sigspace,R>(c)
   {
   commsys::punc = c.punc->clone();
   init();
   }

// Description & Serialization

template <class R> std::string commsys<sigspace,R>::description() const
   {
   std::ostringstream sout;
   sout << basic_commsys<sigspace,R>::description();
   if(punc != NULL)
      sout << ", " << punc->description();
   return sout.str();
   }

template <class R> std::ostream& commsys<sigspace,R>::serialize(std::ostream& sout) const
   {
   //const bool ispunctured = (punc != NULL);
   //sout << int(ispunctured) << "\n";
   //if(ispunctured)
   //   sout << punc;
   basic_commsys<sigspace,R>::serialize(sout);
   return sout;
   }

template <class R> std::istream& commsys<sigspace,R>::serialize(std::istream& sin)
   {
   free();
   //int ispunctured;
   //sin >> ispunctured;
   //if(ispunctured != 0)
   //   sin >> punc;
   basic_commsys<sigspace,R>::serialize(sin);
   init();
   return sin;
   }

// Explicit Realizations

template class commsys<sigspace>;
template <> const libbase::serializer commsys<sigspace>::shelper("experiment", "commsys<sigspace>", commsys<sigspace>::create);

}; // end namespace
