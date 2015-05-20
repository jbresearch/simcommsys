/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "commsys.h"

#include "mapper/map_straight.h"
#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Stop when an error is introduced to a correctly-decoded frame
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// *** Templated Common Base ***

// Setup functions

/*!
 * \brief Initialize internal computed parameters
 *
 * \note This function is only responsible for initializing parameters
 * that are specific to this object/derivation. Anything else
 * should get done automatically when the base serializer or
 * constructor is called.
 */
template <class S, template <class > class C>
void basic_commsys<S, C>::init()
   {
   const int M = mdm->num_symbols();
   const int N = cdc->num_outputs();
   // set up mapper with required parameters
   map->set_parameters(N, M);
   map->set_blocksize(cdc->output_block_size());
   // set up modem with appropriate block size
   mdm->set_blocksize(map->output_block_size());
   }

/*!
 * \brief Sets up system with no bound objects.
 *
 * \note This function is only responsible for clearing pointers to
 * objects that are specific to this object/derivation.
 * Anything else should get done automatically when the base
 * serializer or constructor is called.
 */
template <class S, template <class > class C>
void basic_commsys<S, C>::clear()
   {
   cdc = NULL;
   map = NULL;
   mdm = NULL;
   txchan = NULL;
   rxchan = NULL;
   }

/*!
 * \brief Removes association with bound objects
 *
 * This function performs two things:
 * - Deletes any internally-allocated bound objects
 * - Sets up the system with no bound objects
 *
 * \note This function is only responsible for deleting bound
 * objects that are specific to this object/derivation.
 * Anything else should get done automatically when the base
 * serializer or constructor is called.
 */
template <class S, template <class > class C>
void basic_commsys<S, C>::free()
   {
   // note: delete can be safely called with null pointers
   delete cdc;
   delete map;
   delete mdm;
   delete txchan;
   delete rxchan;
   clear();
   }

// Internal functions

// Constructors / Destructors

/*!
 * \brief Copy constructor
 *
 * Initializes system with bound objects cloned from supplied system.
 */
template <class S, template <class > class C>
basic_commsys<S, C>::basic_commsys(const basic_commsys<S, C>& c) :
   cdc(dynamic_cast<codec<C>*> (c.cdc->clone())), map(
         dynamic_cast<mapper<C>*> (c.map->clone())), mdm(
         dynamic_cast<blockmodem<S, C>*> (c.mdm->clone())), txchan(
         dynamic_cast<channel<S, C>*> (c.txchan->clone())), rxchan(
         dynamic_cast<channel<S, C>*> (c.rxchan->clone())), singlechannel(
         c.singlechannel)
   {
   init();
   }

// Communication System Setup

template <class S, template <class > class C>
void basic_commsys<S, C>::seedfrom(libbase::random& r)
   {
   cdc->seedfrom(r);
   map->seedfrom(r);
   mdm->seedfrom(r);
   txchan->seedfrom(r);
   rxchan->seedfrom(r);
   }

// Communication System Interface

/*!
 * The encode process consists of the steps depicted in the following diagram:
 * \dot
 * digraph encode {
 * // Make figure left-to-right
 * rankdir = LR;
 * // block definitions
 * node [ shape=box ];
 * encode [ label="Encode" ];
 * map [ label="Map" ];
 * modulate [ label="Modulate" ];
 * // path definitions
 * encode -> map -> modulate;
 * }
 * \enddot
 */
template <class S, template <class > class C>
C<S> basic_commsys<S, C>::encode_path(const C<int>& source)
   {
   // Keep track of what we're transmitting
#if DEBUG>=2
   lastsource = source;
#endif
   // Encode
   C<int> encoded;
   this->cdc->reset_timers();
   this->cdc->encode(source, encoded);
   this->add_timers(*this->cdc);
   // Map
   C<int> mapped;
   this->map->reset_timers();
   this->map->transform(encoded, mapped);
   this->add_timers(*this->map);
   // Modulate
   const int M = this->mdm->num_symbols();
   C<S> transmitted;
   this->mdm->reset_timers();
   this->mdm->modulate(M, mapped, transmitted);
   this->add_timers(*this->mdm);
   return transmitted;
   }

/*!
 * The cycle consists of the steps depicted in the following diagram:
 * \dot
 * digraph transmit {
 * // Make figure left-to-right
 * rankdir = LR;
 * // block definitions
 * node [ shape=box ];
 * transmit [ label="Transmit" ];
 * // path definitions
 * -> transmit ->;
 * }
 * \enddot
 */
template <class S, template <class > class C>
C<S> basic_commsys<S, C>::transmit(const C<S>& transmitted)
   {
   C<S> received;
   this->txchan->reset_timers();
   this->txchan->transmit(transmitted, received);
   this->add_timers(*this->txchan);
   return received;
   }

/*!
 * The receive path consists of the steps depicted in the following diagram:
 * \dot
 * digraph decode {
 * // Make figure left-to-right
 * rankdir = LR;
 * // block definitions
 * node [ shape=box ];
 * demodulate [ label="Demodulate" ];
 * unmap [ label="Inverse Map" ];
 * init_decoder [ label="Initialize Decoder" ];
 * // path definitions
 * demodulate -> unmap -> init_decoder;
 * }
 * \enddot
 */
template <class S, template <class > class C>
void basic_commsys<S, C>::receive_path(const C<S>& received)
   {
   // Demodulate
   C<array1d_t> ptable_mapped;
   this->mdm->reset_timers();
   this->mdm->demodulate(*this->rxchan, received, ptable_mapped);
   this->add_timers(*this->mdm);
   // After-demodulation receive path
   softreceive_path(ptable_mapped);
   }

/*!
 * The after-demodulation receive path consists of the steps depicted in the
 * following diagram:
 * \dot
 * digraph decode {
 * // Make figure left-to-right
 * rankdir = LR;
 * // block definitions
 * node [ shape=box ];
 * unmap [ label="Inverse Map" ];
 * init_decoder [ label="Initialize Decoder" ];
 * // path definitions
 * unmap -> init_decoder;
 * }
 * \enddot
 */
template <class S, template <class > class C>
void basic_commsys<S, C>::softreceive_path(const C<array1d_t>& ptable_mapped)
   {
   // Inverse Map
   C<array1d_t> ptable_encoded;
   this->map->reset_timers();
   this->map->inverse(ptable_mapped, ptable_encoded);
   this->add_timers(*this->map);
   // Translate
   this->cdc->reset_timers();
   this->cdc->init_decoder(ptable_encoded);
   this->add_timers(*this->cdc);
   // This frame has not been decoded yet
#if DEBUG>=2
   lastframecorrect = false;
#endif
   }

template <class S, template <class > class C>
void basic_commsys<S, C>::decode(C<int>& decoded)
   {
   // Decode
   this->cdc->reset_timers();
   this->cdc->decode(decoded);
   this->add_timers(*this->cdc);
   // Keep track of correct decodings
#if DEBUG>=2
   if(lastsource.size() > 0)
      {
      bool thisframecorrect = decoded.isequalto(lastsource);
      assert(!(lastframecorrect && !thisframecorrect));
      lastframecorrect = thisframecorrect;
      }
#endif
   }

// Description & Serialization

template <class S, template <class > class C>
std::string basic_commsys<S, C>::description() const
   {
   std::ostringstream sout;
   sout << "Communication System: ";
   sout << cdc->description() << ", ";
   sout << map->description() << ", ";
   sout << mdm->description() << ", ";
   if (singlechannel)
      sout << txchan->description();
   else
      {
      sout << "TX: " << txchan->description() << ", ";
      sout << "RX: " << rxchan->description();
      }
   return sout.str();
   }

// object serialization - saving

template <class S, template <class > class C>
std::ostream& basic_commsys<S, C>::serialize(std::ostream& sout) const
   {
   // format version
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "# Single channel?" << std::endl;
   sout << singlechannel << std::endl;
   if (singlechannel)
      {
      sout << "## Channel" << std::endl;
      sout << txchan;
      }
   else
      {
      sout << "## TX Channel" << std::endl;
      sout << txchan;
      sout << "## RX Channel" << std::endl;
      sout << rxchan;
      }
   sout << "## Modem" << std::endl;
   sout << mdm;
   sout << "## Mapper" << std::endl;
   sout << map;
   sout << "## Codec" << std::endl;
   sout << cdc;
   return sout;
   }

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added version numbering; added split channel model
 */
template <class S, template <class > class C>
std::istream& basic_commsys<S, C>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   free();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
   // handle old-format files
   if (sin.fail())
      {
      version = 0;
      sin.clear();
      }
   // get channel split flag
   if (version >= 1)
      sin >> libbase::eatcomments >> singlechannel >> libbase::verify;
   else
      singlechannel = true;
   // get channel model(s)
   sin >> libbase::eatcomments >> txchan >> libbase::verify;
   assertalways(txchan);
   if (singlechannel)
      rxchan = dynamic_cast<channel<S, C>*> (txchan->clone());
   else
      sin >> libbase::eatcomments >> rxchan >> libbase::verify;
   assertalways(rxchan);
   // get modem
   sin >> libbase::eatcomments >> mdm >> libbase::verify;
   assertalways(mdm);
   // get mapper (if present)
   sin >> libbase::eatcomments >> map;
   if (version == 0 && sin.fail())
      {
      assert(map == NULL);
      map = new map_straight<C> ;
      sin.clear();
      }
   sin >> libbase::verify;
   assertalways(map);
   // get codec
   sin >> libbase::eatcomments >> cdc >> libbase::verify;
   assertalways(cdc);
   // initialize and return
   init();
   assertalways(sin.good());
   return sin;
   }

// *** General Communication System ***

// Serialization Support

template <class S, template <class > class C>
std::ostream& commsys<S, C>::serialize(std::ostream& sout) const
   {
   return basic_commsys<S, C>::serialize(sout);
   }

template <class S, template <class > class C>
std::istream& commsys<S, C>::serialize(std::istream& sin)
   {
   return basic_commsys<S, C>::serialize(sin);
   }

// *** Specific to commsys<sigspace> ***

// Setup functions

/*!
 * \copydoc basic_commsys::init()
 *
 * This function sets the average energy per data bit in the bound channel model.
 * The value depends on:
 * - Rate of codec
 * - Rate of puncturing
 * - Average energy per uncoded bit in the modulation scheme
 */

template <template <class > class C>
void commsys<sigspace, C>::init()
   {
   // set up channel energy/bit (Eb)
   libbase::trace << "DEBUG: overall code rate = " << this->rate() << std::endl;
   this->txchan->set_eb(this->mdm->bit_energy() / this->rate());
   this->rxchan->set_eb(this->mdm->bit_energy() / this->rate());
   }

// Serialization Support

template <template <class > class C>
std::ostream& commsys<sigspace, C>::serialize(std::ostream& sout) const
   {
   return basic_commsys<sigspace, C>::serialize(sout);
   }

template <template <class > class C>
std::istream& commsys<sigspace, C>::serialize(std::istream& sin)
   {
   basic_commsys<sigspace, C>::serialize(sin);
   init();
   return sin;
   }

} // end namespace

#include "gf.h"
#include "erasable.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::erasable;
using libbase::matrix;
using libbase::vector;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define FINITE_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ

#define ADD_ERASABLE(r, x, type) \
   (type)(erasable<type>)

#define ALL_FINITE_TYPE_SEQ \
   BOOST_PP_SEQ_FOR_EACH(ADD_ERASABLE, x, FINITE_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace) \
   ALL_FINITE_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)(matrix)

/* Serialization string: commsys<type,container>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      container = vector | matrix
 */
#define INSTANTIATE(r, args) \
      template class basic_commsys<BOOST_PP_SEQ_ENUM(args)>; \
      template class commsys<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer commsys<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "commsys", \
            "commsys<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            commsys<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ))

} // end namespace
