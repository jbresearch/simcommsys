/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_simulator.h"
#include "commsys_prof_burst.h"
#include "commsys_prof_pos.h"
#include "commsys_prof_sym.h"
#include "commsys_hist_symerr.h"

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
   \brief Sets up system with no bound objects.

   \note This function is only responsible for clearing pointers to
         objects that are specific to this object/derivation.
         Anything else should get done automatically when the base
         serializer or constructor is called.
*/
template <class S, class R>
void basic_commsys_simulator<S,R>::clear()
   {
   src = NULL;
   sys = NULL;
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
template <class S, class R>
void basic_commsys_simulator<S,R>::free()
   {
   if(internallyallocated)
      {
      delete src;
      delete sys;
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
template <class S, class R>
libbase::vector<int> basic_commsys_simulator<S,R>::createsource()
   {
   const int tau = sys->getcodec()->input_block_size();
   libbase::vector<int> source(tau);
   for(int t=0; t<tau; t++)
      source(t) = src->ival(get_alphabetsize());
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
template <class S, class R>
void basic_commsys_simulator<S,R>::cycleonce(libbase::vector<double>& result)
   {
   assert(result.size() == count());
   // Create source stream
   libbase::vector<int> source = createsource();
   // Full cycle from Encode through Demodulate
   sys->transmitandreceive(source);
   // For every iteration
   libbase::vector<int> decoded;
   for(int i=0; i<get_iter(); i++)
      {
      // Decode & update results
      sys->getcodec()->decode(decoded);
      R::updateresults(result, i, source, decoded);
      }
   // Keep record of what we last simulated
   const int tau = sys->getcodec()->input_block_size();
   assert(source.size() == tau);
   assert(decoded.size() == tau);
   last_event.init(2*tau);
   for(int i=0; i<tau; i++)
      {
      last_event(i) = source(i);
      last_event(i+tau) = decoded(i);
      }
   }

// Constructors / Destructors

/*!
   \brief Main public constructor

   Initializes system with bound objects as supplied by user.
*/
template <class S, class R>
basic_commsys_simulator<S,R>::basic_commsys_simulator(libbase::randgen *src, commsys<S> *sys)
   {
   basic_commsys_simulator<S,R>::src = src;
   basic_commsys_simulator<S,R>::sys = sys;
   internallyallocated = false;
   }

/*!
   \brief Copy constructor

   Initializes system with bound objects cloned from supplied system.

   \todo Fix cast when cloning channel: this should not be necessary.
   \todo Fix cast when cloning modem: this should not be necessary.
*/
template <class S, class R>
basic_commsys_simulator<S,R>::basic_commsys_simulator(const basic_commsys_simulator<S,R>& c)
   {
   basic_commsys_simulator<S,R>::src = new libbase::randgen;
   basic_commsys_simulator<S,R>::sys = (commsys<S> *)c.sys->clone();
   internallyallocated = true;
   }

// Experiment parameter handling

template <class S, class R>
void basic_commsys_simulator<S,R>::seedfrom(libbase::random& r)
   {
   src->seed(r.ival());
   sys->seedfrom(r);
   }

// Experiment handling

template <class S, class R>
void basic_commsys_simulator<S,R>::sample(libbase::vector<double>& result)
   {
   // initialise result vector
   result.init(count());
   result = 0;
   // compute a single cycle
   cycleonce(result);
   }

// Description & Serialization

template <class S, class R>
std::string basic_commsys_simulator<S,R>::description() const
   {
   std::ostringstream sout;
   sout << "Simulator for ";
   sout << sys->description();
   return sout.str();
   }

template <class S, class R>
std::ostream& basic_commsys_simulator<S,R>::serialize(std::ostream& sout) const
   {
   sout << sys;
   return sout;
   }

template <class S, class R>
std::istream& basic_commsys_simulator<S,R>::serialize(std::istream& sin)
   {
   free();
   src = new libbase::randgen;
   sin >> sys;
   internallyallocated = true;
   return sin;
   }

// Explicit Realizations

template class basic_commsys_simulator<sigspace>;
template class basic_commsys_simulator<bool>;
template class basic_commsys_simulator< libbase::gf<1,0x3> >;
template class basic_commsys_simulator< libbase::gf<2,0x7> >;
template class basic_commsys_simulator< libbase::gf<3,0xB> >;
template class basic_commsys_simulator< libbase::gf<4,0x13> >;

template class basic_commsys_simulator<bool,commsys_prof_burst>;
template class basic_commsys_simulator<bool,commsys_prof_pos>;
template class basic_commsys_simulator<bool,commsys_prof_sym>;
template class basic_commsys_simulator<bool,commsys_hist_symerr>;


// *** Templated Base ***

// Serialization Support

template <class S, class R>
std::ostream& commsys_simulator<S,R>::serialize(std::ostream& sout) const
   {
   return basic_commsys_simulator<S,R>::serialize(sout);
   }

template <class S, class R>
std::istream& commsys_simulator<S,R>::serialize(std::istream& sin)
   {
   return basic_commsys_simulator<S,R>::serialize(sin);
   }

// Explicit Realizations

template class commsys_simulator<bool>;
template <>
const libbase::serializer commsys_simulator<bool>::shelper("experiment", "commsys_simulator<bool>", commsys_simulator<bool>::create);
template class commsys_simulator< libbase::gf<1,0x3> >;
template <>
const libbase::serializer commsys_simulator< libbase::gf<1,0x3> >::shelper("experiment", "commsys_simulator<gf<1,0x3>>", commsys_simulator< libbase::gf<1,0x3> >::create);
template class commsys_simulator< libbase::gf<2,0x7> >;
template <>
const libbase::serializer commsys_simulator< libbase::gf<2,0x7> >::shelper("experiment", "commsys_simulator<gf<2,0x7>>", commsys_simulator< libbase::gf<2,0x7> >::create);
template class commsys_simulator< libbase::gf<3,0xB> >;
template <>
const libbase::serializer commsys_simulator< libbase::gf<3,0xB> >::shelper("experiment", "commsys_simulator<gf<3,0xB>>", commsys_simulator< libbase::gf<3,0xB> >::create);
template class commsys_simulator< libbase::gf<4,0x13> >;
template <>
const libbase::serializer commsys_simulator< libbase::gf<4,0x13> >::shelper("experiment", "commsys_simulator<gf<4,0x13>>", commsys_simulator< libbase::gf<4,0x13> >::create);

template class commsys_simulator<bool,commsys_prof_burst>;
template <>
const libbase::serializer commsys_simulator<bool,commsys_prof_burst>::shelper("experiment", "commsys_simulator<bool,prof_burst>", commsys_simulator<bool,commsys_prof_burst>::create);
template class commsys_simulator<bool,commsys_prof_pos>;
template <>
const libbase::serializer commsys_simulator<bool,commsys_prof_pos>::shelper("experiment", "commsys_simulator<bool,prof_pos>", commsys_simulator<bool,commsys_prof_pos>::create);
template class commsys_simulator<bool,commsys_prof_sym>;
template <>
const libbase::serializer commsys_simulator<bool,commsys_prof_sym>::shelper("experiment", "commsys_simulator<bool,prof_sym>", commsys_simulator<bool,commsys_prof_sym>::create);
template class commsys_simulator<bool,commsys_hist_symerr>;
template <>
const libbase::serializer commsys_simulator<bool,commsys_hist_symerr>::shelper("experiment", "commsys_simulator<bool,hist_symerr>", commsys_simulator<bool,commsys_hist_symerr>::create);

template class commsys_simulator<sigspace>;
template <>
const libbase::serializer commsys_simulator<sigspace>::shelper("experiment", "commsys_simulator<sigspace>", commsys_simulator<sigspace>::create);

}; // end namespace
