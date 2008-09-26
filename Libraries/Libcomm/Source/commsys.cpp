/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys.h"

#include "map_straight.h"
#include "fsm.h"
#include "gf.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>
#include <sstream>

namespace libcomm {


// *** Templated Common Base ***

// Setup functions

/*!
   \brief Initialize internal computed parameters

   \note This function is only responsible for initializing parameters
         that are specific to this object/derivation. Anything else
         should get done automatically when the base serializer or
         constructor is called.
*/
template <class S>
void basic_commsys<S>::init()
   {
   tau = cdc->block_size();
   m = cdc->tail_length();
   M = modem->num_symbols();
   N = cdc->num_outputs();
   K = cdc->num_inputs();
   k = int(round(log2(double(K))));
   // confirm that source is representable in binary
   assertalways(K == 1<<k);
   iter = cdc->num_iter();
   // set up mapper with required parameters
   map->set_parameters(N, M, cdc->num_symbols());
   }

/*!
   \brief Sets up system with no bound objects.

   \note This function is only responsible for clearing pointers to
         objects that are specific to this object/derivation.
         Anything else should get done automatically when the base
         serializer or constructor is called.
*/
template <class S>
void basic_commsys<S>::clear()
   {
   cdc = NULL;
   map = NULL;
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
template <class S>
void basic_commsys<S>::free()
   {
   if(internallyallocated)
      {
      delete cdc;
      delete map;
      delete modem;
      delete chan;
      }
   clear();
   }

// Internal functions

// Constructors / Destructors

/*!
   \brief Main public constructor

   Initializes system with bound objects as supplied by user.
*/
template <class S>
basic_commsys<S>::basic_commsys(codec *cdc, mapper *map, modulator<S> *modem, channel<S> *chan)
   {
   basic_commsys<S>::cdc = cdc;
   basic_commsys<S>::map = map;
   basic_commsys<S>::modem = modem;
   basic_commsys<S>::chan = chan;
   internallyallocated = false;
   init();
   }

/*!
   \brief Copy constructor

   Initializes system with bound objects cloned from supplied system.

   \todo Fix cast when cloning channel: this should not be necessary.
   \todo Fix cast when cloning modem: this should not be necessary.
*/
template <class S>
basic_commsys<S>::basic_commsys(const basic_commsys<S>& c)
   {
   basic_commsys<S>::cdc = c.cdc->clone();
   basic_commsys<S>::map = c.map->clone();
   basic_commsys<S>::modem = (modulator<S> *)c.modem->clone();
   basic_commsys<S>::chan = (channel<S> *)c.chan->clone();
   internallyallocated = true;
   init();
   }

// Communication System Setup

template <class S>
void basic_commsys<S>::seedfrom(libbase::random& r)
   {
   cdc->seedfrom(r);
   map->seedfrom(r);
   modem->seedfrom(r);
   chan->seedfrom(r);
   }

// Communication System Interface

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
      map [ label="Map" ];
      modulate [ label="Modulate" ];
      transmit [ label="Transmit" ];
      demodulate [ label="Demodulate" ];
      unmap [ label="Inverse Map" ];
      translate [ label="Translate" ];
      // path definitions
      encode -> map;
      map -> modulate;
      modulate -> transmit;
      transmit -> demodulate;
      demodulate -> unmap;
      unmap -> translate;
   }
   \enddot
*/
template <class S>
void basic_commsys<S>::transmitandreceive(libbase::vector<int>& source)
   {
   libbase::vector<int> encoded;
   this->cdc->encode(source, encoded);
   this->map->advance();
   libbase::vector<int> transmitted;
   this->map->transform(encoded, transmitted);
   libbase::vector<S> signal;
   this->modem->modulate(this->M, transmitted, signal);
   this->chan->transmit(signal, signal);
   libbase::matrix<double> pin;
   this->modem->demodulate(*this->chan, signal, pin);
   libbase::matrix<double> pout;
   this->map->inverse(pin, pout);
   this->cdc->translate(pout);
   }

// Description & Serialization

template <class S>
std::string basic_commsys<S>::description() const
   {
   std::ostringstream sout;
   sout << "Communication System: ";
   sout << cdc->description() << ", ";
   sout << map->description() << ", ";
   sout << modem->description() << ", ";
   sout << chan->description();
   return sout.str();
   }

template <class S>
std::ostream& basic_commsys<S>::serialize(std::ostream& sout) const
   {
   sout << chan;
   sout << modem;
   sout << map;
   sout << cdc;
   return sout;
   }

template <class S>
std::istream& basic_commsys<S>::serialize(std::istream& sin)
   {
   free();
   sin >> chan;
   sin >> modem;
   sin >> map;
   if(sin.fail())
      {
      assert(map == NULL);
      map = new map_straight;
      sin.clear();
      }
   sin >> cdc;
   internallyallocated = true;
   init();
   return sin;
   }

// Explicit Realizations

template class basic_commsys<bool>;
template class basic_commsys< libbase::gf<1,0x3> >;
template class basic_commsys< libbase::gf<2,0x7> >;
template class basic_commsys< libbase::gf<3,0xB> >;
template class basic_commsys< libbase::gf<4,0x13> >;
template class basic_commsys<sigspace>;


// *** General Communication System ***

// Serialization Support

template <class S>
std::ostream& commsys<S>::serialize(std::ostream& sout) const
   {
   return basic_commsys<S>::serialize(sout);
   }

template <class S>
std::istream& commsys<S>::serialize(std::istream& sin)
   {
   return basic_commsys<S>::serialize(sin);
   }

// Explicit Realizations

template class commsys<bool>;
template <>
const libbase::serializer commsys<bool>::shelper("commsys", "commsys<bool>", commsys<bool>::create);
template class commsys< libbase::gf<1,0x3> >;
template <>
const libbase::serializer commsys< libbase::gf<1,0x3> >::shelper("commsys", "commsys<gf<1,0x3>>", commsys< libbase::gf<1,0x3> >::create);
template class commsys< libbase::gf<2,0x7> >;
template <>
const libbase::serializer commsys< libbase::gf<2,0x7> >::shelper("commsys", "commsys<gf<2,0x7>>", commsys< libbase::gf<2,0x7> >::create);
template class commsys< libbase::gf<3,0xB> >;
template <>
const libbase::serializer commsys< libbase::gf<3,0xB> >::shelper("commsys", "commsys<gf<3,0xB>>", commsys< libbase::gf<3,0xB> >::create);
template class commsys< libbase::gf<4,0x13> >;
template <>
const libbase::serializer commsys< libbase::gf<4,0x13> >::shelper("commsys", "commsys<gf<4,0x13>>", commsys< libbase::gf<4,0x13> >::create);


// *** Specific to commsys<sigspace> ***

#if 0

// Setup functions

/*!
   \copydoc commsys::init()

   This function sets the average energy per data bit in the bound channel model.
   The value depends on:
   - Rate of codec
   - Rate of puncturing
   - Average energy per uncoded bit in the modulation scheme
*/

void commsys<sigspace>::init()
   {
   // set up channel energy/bit (Eb)
   double rate = this->cdc->rate() * this->map->rate();
   if(punc != NULL)
      rate /= punc->rate();
   this->chan->set_eb(this->modem->bit_energy() / rate);
   }


void commsys<sigspace>::clear()
   {
   punc = NULL;
   }


void commsys<sigspace>::free()
   {
   if(this->internallyallocated)
      {
      delete punc;
      }
   clear();
   }

// Internal functions

/*!
   \copydoc commsys::transmitandreceive()

   The cycle consists of the steps depicted in the following diagram:
   \dot
   digraph txrxcycle {
      // Make figure left-to-right
      rankdir = LR;
      // block definitions
      node [ shape=box ];
      encode [ label="Encode" ];
      map [ label="Map" ];
      modulate [ label="Modulate" ];
      puncture [ style=dotted,label="Puncture" ];
      transmit [ label="Transmit" ];
      demodulate [ label="Demodulate" ];
      unpuncture [ style=dotted,label="Inverse Puncture" ];
      unmap [ label="Inverse Map" ];
      translate [ label="Translate" ];
      // path definitions
      encode -> map;
      map -> modulate;
      modulate -> transmit;
      transmit -> demodulate;
      demodulate -> unmap;
      unmap -> translate;
      modulate -> puncture [ style=dotted ];
      puncture -> transmit [ style=dotted ];
      demodulate -> unpuncture [ style=dotted ];
      unpuncture -> unmap [ style=dotted ];
   }
   \enddot

   The dotted lines and blocks indicate optional sections to support puncturing,
   which is currently done in signal-space.
*/
void commsys<sigspace>::transmitandreceive(libbase::vector<int>& source)
   {
   libbase::vector<int> encoded;
   this->cdc->encode(source, encoded);
   this->map->advance();
   libbase::vector<int> transmitted;
   this->map->transform(encoded, transmitted);
   libbase::vector<sigspace> signal1;
   this->modem->modulate(this->M, transmitted, signal1);
   libbase::matrix<double> pin;
   if(punc != NULL)
      {
      libbase::vector<sigspace> signal2;
      punc->transform(signal1, signal2);
      this->chan->transmit(signal2, signal2);
      libbase::matrix<double> pchan;
      this->modem->demodulate(*this->chan, signal2, pchan);
      punc->inverse(pchan, pin);
      }
   else
      {
      this->chan->transmit(signal1, signal1);
      this->modem->demodulate(*this->chan, signal1, pin);
      }
   libbase::matrix<double> pout;
   this->map->inverse(pin, pout);
   this->cdc->translate(pout);
   }

// Constructors / Destructors

commsys<sigspace>::commsys(codec *cdc, mapper *map, modulator<sigspace> *modem, puncture *punc, channel<sigspace> *chan) : commsys<sigspace>(cdc, map, modem, chan)
   {
   commsys::punc = punc;
   init();
   }

commsys<sigspace>::commsys(const commsys<sigspace>& c) : commsys<sigspace>(c)
   {
   commsys::punc = c.punc->clone();
   init();
   }

// Description & Serialization

std::string commsys<sigspace>::description() const
   {
   std::ostringstream sout;
   sout << commsys<sigspace>::description();
   if(punc != NULL)
      sout << ", " << punc->description();
   return sout.str();
   }

std::ostream& commsys<sigspace>::serialize(std::ostream& sout) const
   {
   commsys<sigspace>::serialize(sout);
   const bool ispunctured = (punc != NULL);
   sout << ispunctured << "\n";
   if(ispunctured)
      sout << punc;
   return sout;
   }

std::istream& commsys<sigspace>::serialize(std::istream& sin)
   {
   free();
   commsys<sigspace>::serialize(sin);
   bool ispunctured;
   sin >> ispunctured;
   // handle old-format files
   if(sin.fail())
      {
      ispunctured = false;
      sin.clear();
      }
   if(ispunctured)
      sin >> punc;
   init();
   return sin;
   }

#endif

// Explicit Realizations

template class commsys<sigspace>;
template <>
const libbase::serializer commsys<sigspace>::shelper("commsys", "commsys<sigspace>", commsys<sigspace>::create);

}; // end namespace
