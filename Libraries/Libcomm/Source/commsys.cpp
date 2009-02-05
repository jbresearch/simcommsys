/*!
   \file

   \section svn Version Control
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
template <class S, template<class> class C>
void basic_commsys<S,C>::init()
   {
   M = mdm->num_symbols();
   N = cdc->num_outputs();
   K = cdc->num_inputs();
   k = int(round(log2(double(K))));
   // confirm that source is representable in binary
   assertalways(K == 1<<k);
   iter = cdc->num_iter();
   // set up mapper with required parameters
   map->set_parameters(N, M, cdc->num_symbols());
   map->set_blocksize(cdc->output_block_size());
   // set up modem with appropriate block size
   mdm->set_blocksize(map->output_block_size());
   }

/*!
   \brief Sets up system with no bound objects.

   \note This function is only responsible for clearing pointers to
         objects that are specific to this object/derivation.
         Anything else should get done automatically when the base
         serializer or constructor is called.
*/
template <class S, template<class> class C>
void basic_commsys<S,C>::clear()
   {
   cdc = NULL;
   map = NULL;
   mdm = NULL;
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
template <class S, template<class> class C>
void basic_commsys<S,C>::free()
   {
   if(internallyallocated)
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
   \brief Main public constructor

   Initializes system with bound objects as supplied by user.
*/
template <class S, template<class> class C>
basic_commsys<S,C>::basic_commsys(codec<C> *cdc, mapper *map, blockmodem<S> *mdm, channel<S> *chan)
   {
   basic_commsys<S,C>::cdc = cdc;
   basic_commsys<S,C>::map = map;
   basic_commsys<S,C>::mdm = mdm;
   basic_commsys<S,C>::chan = chan;
   internallyallocated = false;
   init();
   }

/*!
   \brief Copy constructor

   Initializes system with bound objects cloned from supplied system.

   \todo Fix cast when cloning channel: this should not be necessary.
   \todo Fix cast when cloning modem: this should not be necessary.
*/
template <class S, template<class> class C>
basic_commsys<S,C>::basic_commsys(const basic_commsys<S,C>& c)
   {
   basic_commsys<S,C>::cdc = c.cdc->clone();
   basic_commsys<S,C>::map = c.map->clone();
   basic_commsys<S,C>::mdm = (blockmodem<S> *)c.mdm->clone();
   basic_commsys<S,C>::chan = (channel<S> *)c.chan->clone();
   internallyallocated = true;
   init();
   }

// Communication System Setup

template <class S, template<class> class C>
void basic_commsys<S,C>::seedfrom(libbase::random& r)
   {
   cdc->seedfrom(r);
   map->seedfrom(r);
   mdm->seedfrom(r);
   chan->seedfrom(r);
   }

// Communication System Interface

/*!
   The encode process consists of the steps depicted in the following diagram:
   \dot
   digraph encode {
      // Make figure left-to-right
      rankdir = LR;
      // block definitions
      node [ shape=box ];
      encode [ label="Encode" ];
      map [ label="Map" ];
      modulate [ label="Modulate" ];
      // path definitions
      encode -> map -> modulate;
   }
   \enddot
*/
template <class S, template<class> class C>
libbase::vector<S> basic_commsys<S,C>::encode(const libbase::vector<int>& source)
   {
   // Encode
   libbase::vector<int> encoded;
   this->cdc->encode(source, encoded);
   // Map
   libbase::vector<int> mapped;
   this->map->transform(encoded, mapped);
   // Modulate
   libbase::vector<S> transmitted;
   this->mdm->modulate(this->M, mapped, transmitted);
   return transmitted;
   }

/*!
   The translate process consists of the steps depicted in the following diagram:
   \dot
   digraph decode {
      // Make figure left-to-right
      rankdir = LR;
      // block definitions
      node [ shape=box ];
      demodulate [ label="Demodulate" ];
      unmap [ label="Inverse Map" ];
      translate [ label="Translate" ];
      // path definitions
      demodulate -> unmap -> translate;
   }
   \enddot
*/
template <class S, template<class> class C>
void basic_commsys<S,C>::translate(const libbase::vector<S>& received)
   {
   // Demodulate
   libbase::vector< libbase::vector<double> > ptable_mapped;
   this->mdm->demodulate(*this->chan, received, ptable_mapped);
   // Inverse Map
   libbase::vector< libbase::vector<double> > ptable_encoded;
   this->map->inverse(ptable_mapped, ptable_encoded);
   // Translate
   this->cdc->translate(ptable_encoded);
   }

/*!
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
      encode -> map -> modulate;
      modulate -> transmit -> demodulate;
      demodulate -> unmap -> translate;
   }
   \enddot
*/
template <class S, template<class> class C>
void basic_commsys<S,C>::transmitandreceive(const libbase::vector<int>& source)
   {
   // Encode -> Map -> Modulate
   libbase::vector<S> transmitted = encode(source);
   // Transmit
   libbase::vector<S> received;
   this->chan->transmit(transmitted, received);
   // Demodulate -> Inverse Map -> Translate
   translate(received);
   }

// Description & Serialization

template <class S, template<class> class C>
std::string basic_commsys<S,C>::description() const
   {
   std::ostringstream sout;
   sout << "Communication System: ";
   sout << cdc->description() << ", ";
   sout << map->description() << ", ";
   sout << mdm->description() << ", ";
   sout << chan->description();
   return sout.str();
   }

template <class S, template<class> class C>
std::ostream& basic_commsys<S,C>::serialize(std::ostream& sout) const
   {
   sout << chan;
   sout << mdm;
   sout << map;
   sout << cdc;
   return sout;
   }

template <class S, template<class> class C>
std::istream& basic_commsys<S,C>::serialize(std::istream& sin)
   {
   free();
   sin >> chan;
   sin >> mdm;
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

template <class S, template<class> class C>
std::ostream& commsys<S,C>::serialize(std::ostream& sout) const
   {
   return basic_commsys<S,C>::serialize(sout);
   }

template <class S, template<class> class C>
std::istream& commsys<S,C>::serialize(std::istream& sin)
   {
   return basic_commsys<S,C>::serialize(sin);
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

// Setup functions

/*!
   \copydoc basic_commsys::init()

   This function sets the average energy per data bit in the bound channel model.
   The value depends on:
   - Rate of codec
   - Rate of puncturing
   - Average energy per uncoded bit in the modulation scheme
*/

template <template<class> class C>
void commsys<sigspace,C>::init()
   {
   // set up channel energy/bit (Eb)
   libbase::trace << "DEBUG: overall code rate = " << this->rate() << "\n";
   this->chan->set_eb(this->mdm->bit_energy() / this->rate());
   }

// Serialization Support

template <template<class> class C>
std::ostream& commsys<sigspace,C>::serialize(std::ostream& sout) const
   {
   return basic_commsys<sigspace,C>::serialize(sout);
   }

template <template<class> class C>
std::istream& commsys<sigspace,C>::serialize(std::istream& sin)
   {
   basic_commsys<sigspace,C>::serialize(sin);
   init();
   return sin;
   }

// Explicit Realizations

template class commsys<sigspace>;
template <>
const libbase::serializer commsys<sigspace>::shelper("commsys", "commsys<sigspace>", commsys<sigspace>::create);

}; // end namespace
