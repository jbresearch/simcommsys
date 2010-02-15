/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "commsys.h"

#include "mapper/map_straight.h"
#include "fsm.h"
#include "gf.h"
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
   const int K = cdc->num_inputs();
   const int k = int(round(log2(double(K))));
   // confirm that source is representable in binary
   assertalways(K == 1<<k);
   // set up mapper with required parameters
   map->set_parameters(N, M, cdc->num_symbols());
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
   chan = NULL;
   internallyallocated = true;
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
   if (internallyallocated)
      {
      delete cdc;
      delete map;
      delete mdm;
      delete chan;
      }
   clear();
   }

// Internal functions

// Constructors / Destructors

/*!
 * \brief Main public constructor
 * 
 * Initializes system with bound objects as supplied by user.
 */
template <class S, template <class > class C>
basic_commsys<S, C>::basic_commsys(codec<C> *cdc, mapper<C> *map, blockmodem<S,
      C> *mdm, channel<S, C> *chan)
   {
   this->cdc = cdc;
   this->map = map;
   this->mdm = mdm;
   this->chan = chan;
   internallyallocated = false;
   init();
   }

/*!
 * \brief Copy constructor
 * 
 * Initializes system with bound objects cloned from supplied system.
 */
template <class S, template <class > class C>
basic_commsys<S, C>::basic_commsys(const basic_commsys<S, C>& c)
   {
   this->cdc = dynamic_cast<codec<C>*> (c.cdc->clone());
   this->map = dynamic_cast<mapper<C>*> (c.map->clone());
   this->mdm = dynamic_cast<blockmodem<S, C>*> (c.mdm->clone());
   this->chan = dynamic_cast<channel<S, C>*> (c.chan->clone());
   internallyallocated = true;
   init();
   }

// Communication System Setup

template <class S, template <class > class C>
void basic_commsys<S, C>::seedfrom(libbase::random& r)
   {
   cdc->seedfrom(r);
   map->seedfrom(r);
   mdm->seedfrom(r);
   chan->seedfrom(r);
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
   this->cdc->encode(source, encoded);
   // Map
   C<int> mapped;
   this->map->transform(encoded, mapped);
   // Modulate
   const int M = this->mdm->num_symbols();
   C<S> transmitted;
   this->mdm->modulate(M, mapped, transmitted);
   return transmitted;
   }

template <class S, template <class > class C>
C<S> basic_commsys<S, C>::transmit(const C<S>& transmitted)
   {
   C<S> received;
   this->chan->transmit(transmitted, received);
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
   this->mdm->demodulate(*this->chan, received, ptable_mapped);
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
   this->map->inverse(ptable_mapped, ptable_encoded);
   // Translate
   this->cdc->init_decoder(ptable_encoded);
   // This frame has not been decoded yet
#if DEBUG>=2
   lastframecorrect = false;
#endif
   }

template <class S, template <class > class C>
void basic_commsys<S, C>::decode(C<int>& decoded)
   {
   // Decode
   this->cdc->decode(decoded);
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

/*!
 * The cycle consists of the steps depicted in the following diagram:
 * \dot
 * digraph txrxcycle {
 * // Make figure left-to-right
 * rankdir = LR;
 * // block definitions
 * node [ shape=box ];
 * encode [ label="Encode" ];
 * map [ label="Map" ];
 * modulate [ label="Modulate" ];
 * transmit [ label="Transmit" ];
 * demodulate [ label="Demodulate" ];
 * unmap [ label="Inverse Map" ];
 * init_decoder [ label="Initialize Decoder" ];
 * // path definitions
 * encode -> map -> modulate;
 * modulate -> transmit -> demodulate;
 * demodulate -> unmap -> init_decoder;
 * }
 * \enddot
 */
template <class S, template <class > class C>
void basic_commsys<S, C>::transmitandreceive(const C<int>& source)
   {
   // Encode -> Map -> Modulate
   C<S> transmitted = encode_path(source);
   // Transmit
   C<S> received = transmit(transmitted);
   // Demodulate -> Inverse Map -> Translate
   receive_path(received);
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
   sout << chan->description();
   return sout.str();
   }

template <class S, template <class > class C>
std::ostream& basic_commsys<S, C>::serialize(std::ostream& sout) const
   {
   sout << chan;
   sout << mdm;
   sout << map;
   sout << cdc;
   return sout;
   }

template <class S, template <class > class C>
std::istream& basic_commsys<S, C>::serialize(std::istream& sin)
   {
   free();
   sin >> libbase::eatcomments >> chan;
   if (chan == NULL)
      failwith("Failed to load channel.");
   sin >> libbase::eatcomments >> mdm;
   if (mdm == NULL)
      failwith("Failed to load modem.");
   sin >> libbase::eatcomments >> map;
   if (sin.fail())
      {
      assert(map == NULL);
      map = new map_straight<C> ;
      sin.clear();
      }
   sin >> libbase::eatcomments >> cdc;
   if (cdc == NULL)
      failwith("Failed to load codec.");
   internallyallocated = true;
   init();
   return sin;
   }

// Explicit Realizations

using libbase::gf;
using libbase::matrix;

template class basic_commsys<sigspace> ;
template class basic_commsys<bool> ;
template class basic_commsys<gf<1, 0x3> > ;
template class basic_commsys<gf<2, 0x7> > ;
template class basic_commsys<gf<3, 0xB> > ;
template class basic_commsys<gf<4, 0x13> > ;
template class basic_commsys<gf<5, 0x25> > ;
template class basic_commsys<gf<6, 0x43> > ;
template class basic_commsys<gf<7, 0x89> > ;
template class basic_commsys<gf<8, 0x11D> > ;
template class basic_commsys<gf<9, 0x211> > ;
template class basic_commsys<gf<10, 0x409> > ;

template class basic_commsys<sigspace, matrix> ;
template class basic_commsys<bool, matrix> ;
template class basic_commsys<gf<1, 0x3> , matrix> ;
template class basic_commsys<gf<2, 0x7> , matrix> ;
template class basic_commsys<gf<3, 0xB> , matrix> ;
template class basic_commsys<gf<4, 0x13> , matrix> ;
template class basic_commsys<gf<5, 0x25> , matrix> ;
template class basic_commsys<gf<6, 0x43> , matrix> ;
template class basic_commsys<gf<7, 0x89> , matrix> ;
template class basic_commsys<gf<8, 0x11D> , matrix> ;
template class basic_commsys<gf<9, 0x211> , matrix> ;
template class basic_commsys<gf<10, 0x409> , matrix> ;

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
   libbase::trace << "DEBUG: overall code rate = " << this->rate() << "\n";
   this->chan->set_eb(this->mdm->bit_energy() / this->rate());
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

// Explicit Realizations

using libbase::serializer;
using libbase::gf;
using libbase::matrix;

template class commsys<sigspace> ;
template <>
const serializer commsys<sigspace>::shelper("commsys", "commsys<sigspace>",
      commsys<sigspace>::create);

template class commsys<bool> ;
template <>
const serializer commsys<bool>::shelper("commsys", "commsys<bool>", commsys<
      bool>::create);

template class commsys<gf<1, 0x3> > ;
template <>
const serializer commsys<gf<1, 0x3> >::shelper("commsys", "commsys<gf<1,0x3>>",
      commsys<gf<1, 0x3> >::create);

template class commsys<gf<2, 0x7> > ;
template <>
const serializer commsys<gf<2, 0x7> >::shelper("commsys", "commsys<gf<2,0x7>>",
      commsys<gf<2, 0x7> >::create);

template class commsys<gf<3, 0xB> > ;
template <>
const serializer commsys<gf<3, 0xB> >::shelper("commsys", "commsys<gf<3,0xB>>",
      commsys<gf<3, 0xB> >::create);

template class commsys<gf<4, 0x13> > ;
template <>
const serializer commsys<gf<4, 0x13> >::shelper("commsys",
      "commsys<gf<4,0x13>>", commsys<gf<4, 0x13> >::create);

template class commsys<gf<5, 0x25> > ;
template <>
const serializer commsys<gf<5, 0x25> >::shelper("commsys",
      "commsys<gf<5,0x25>>", commsys<gf<5, 0x25> >::create);

template class commsys<gf<6, 0x43> > ;
template <>
const serializer commsys<gf<6, 0x43> >::shelper("commsys",
      "commsys<gf<6,0x43>>", commsys<gf<6, 0x43> >::create);

template class commsys<gf<7, 0x89> > ;
template <>
const serializer commsys<gf<7, 0x89> >::shelper("commsys",
      "commsys<gf<7,0x89>>", commsys<gf<7, 0x89> >::create);

template class commsys<gf<8, 0x11D> > ;
template <>
const serializer commsys<gf<8, 0x11D> >::shelper("commsys",
      "commsys<gf<8,0x11D>>", commsys<gf<8, 0x11D> >::create);

template class commsys<gf<9, 0x211> > ;
template <>
const serializer commsys<gf<9, 0x211> >::shelper("commsys",
      "commsys<gf<9,0x211>>", commsys<gf<9, 0x211> >::create);

template class commsys<gf<10, 0x409> > ;
template <>
const serializer commsys<gf<10, 0x409> >::shelper("commsys",
      "commsys<gf<10,0x409>>", commsys<gf<10, 0x409> >::create);

template class commsys<bool, matrix> ;
template <>
const serializer commsys<bool, matrix>::shelper("commsys",
      "commsys<bool,matrix>", commsys<bool, matrix>::create);

template class commsys<gf<1, 0x3> , matrix> ;
template <>
const serializer commsys<gf<1, 0x3> , matrix>::shelper("commsys",
      "commsys<gf<1,0x3>,matrix>", commsys<gf<1, 0x3> , matrix>::create);

template class commsys<gf<2, 0x7> , matrix> ;
template <>
const serializer commsys<gf<2, 0x7> , matrix>::shelper("commsys",
      "commsys<gf<2,0x7>,matrix>", commsys<gf<2, 0x7> , matrix>::create);

template class commsys<gf<3, 0xB> , matrix> ;
template <>
const serializer commsys<gf<3, 0xB> , matrix>::shelper("commsys",
      "commsys<gf<3,0xB>,matrix>", commsys<gf<3, 0xB> , matrix>::create);

template class commsys<gf<4, 0x13> , matrix> ;
template <>
const serializer commsys<gf<4, 0x13> , matrix>::shelper("commsys",
      "commsys<gf<4,0x13>,matrix>", commsys<gf<4, 0x13> , matrix>::create);

template class commsys<gf<5, 0x25> , matrix> ;
template <>
const serializer commsys<gf<5, 0x25> , matrix>::shelper("commsys",
      "commsys<gf<5,0x25>,matrix>", commsys<gf<5, 0x25> , matrix>::create);

template class commsys<gf<6, 0x43> , matrix> ;
template <>
const serializer commsys<gf<6, 0x43> , matrix>::shelper("commsys",
      "commsys<gf<6,0x43>,matrix>", commsys<gf<6, 0x43> , matrix>::create);

template class commsys<gf<7, 0x89> , matrix> ;
template <>
const serializer commsys<gf<7, 0x89> , matrix>::shelper("commsys",
      "commsys<gf<7,0x89>,matrix>", commsys<gf<7, 0x89> , matrix>::create);

template class commsys<gf<8, 0x11D> , matrix> ;
template <>
const serializer commsys<gf<8, 0x11D> , matrix>::shelper("commsys",
      "commsys<gf<8,0x11D>,matrix>", commsys<gf<8, 0x11D> , matrix>::create);

template class commsys<gf<9, 0x211> , matrix> ;
template <>
const serializer commsys<gf<9, 0x211> , matrix>::shelper("commsys",
      "commsys<gf<9,0x211>,matrix>", commsys<gf<9, 0x211> , matrix>::create);

template class commsys<gf<10, 0x409> , matrix> ;
template <>
const serializer commsys<gf<10, 0x409> , matrix>::shelper("commsys",
      "commsys<gf<10,0x409>,matrix>", commsys<gf<10, 0x409> , matrix>::create);

} // end namespace
