/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "exit_computer.h"

#include "mapper/map_straight.h"
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
 * \brief Sets up system with no bound objects.
 * 
 * \note This function is only responsible for clearing pointers to
 * objects that are specific to this object/derivation.
 * Anything else should get done automatically when the base
 * serializer or constructor is called.
 */
template <class S>
void exit_computer<S>::clear()
   {
   src = NULL;
   sys = NULL;
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
template <class S>
void exit_computer<S>::free()
   {
   if (internallyallocated)
      {
      delete src;
      delete sys;
      }
   clear();
   }

// Internal functions

/*!
 * \brief Create source sequence to be encoded
 * \return Source sequence of the required length
 * 
 * The source sequence consists of uniformly random symbols followed by a
 * tail sequence if required by the given codec.
 */
template <class S>
libbase::vector<int> exit_computer<S>::createsource()
   {
   const int tau = sys->input_block_size();
   libbase::vector<int> source(tau);
   for (int t = 0; t < tau; t++)
      source(t) = src->ival(sys->num_inputs());
   return source;
   }

/*!
 * \brief Perform a complete encode->transmit->receive cycle
 * \param[out] result   Vector containing the set of results to be updated
 * 
 * Results are organized as (BER,SER,FER), repeated for every iteration that
 * needs to be performed.
 * 
 * \note It is assumed that the result vector serves as an accumulator, so that
 * every cycle effectively adds to this result. The caller is responsible
 * to divide by the appropriate amount at the end to compute a meaningful
 * average.
 */
template <class S>
void exit_computer<S>::cycleonce(libbase::vector<double>& result)
   {
   assert(result.size() == count());
   // Create source stream
   libbase::vector<int> source = createsource();
   // Full cycle from Encode through Demodulate
   sys->transmitandreceive(source);
   // For every iteration
   libbase::vector<int> decoded;
   for (int i = 0; i < sys->num_iter(); i++)
      {
      // Decode & update results
      sys->decode(decoded);
      //R::updateresults(result, i, source, decoded);
      }
   // Keep record of what we last simulated
   const int tau = sys->input_block_size();
   assert(source.size() == tau);
   assert(decoded.size() == tau);
   last_event.init(2 * tau);
   for (int i = 0; i < tau; i++)
      {
      last_event(i) = source(i);
      last_event(i + tau) = decoded(i);
      }
   }

// Constructors / Destructors

/*!
 * \brief Main public constructor
 * 
 * Initializes system with bound objects as supplied by user.
 */
template <class S>
exit_computer<S>::exit_computer(libbase::randgen *src, commsys<S> *sys)
   {
   this->src = src;
   this->sys = sys;
   internallyallocated = false;
   }

/*!
 * \brief Copy constructor
 * 
 * Initializes system with bound objects cloned from supplied system.
 */
template <class S>
exit_computer<S>::exit_computer(const exit_computer<S>& c)
   {
   this->src = new libbase::randgen;
   this->sys = c.sys->clone();
   internallyallocated = true;
   }

// Experiment parameter handling

template <class S>
void exit_computer<S>::seedfrom(libbase::random& r)
   {
   src->seed(r.ival());
   sys->seedfrom(r);
   }

// Experiment handling

template <class S>
void exit_computer<S>::sample(libbase::vector<double>& result)
   {
   // initialise result vector
   result.init(count());
   result = 0;
   // compute a single cycle
   cycleonce(result);
   }

// Description & Serialization

template <class S>
std::string exit_computer<S>::description() const
   {
   std::ostringstream sout;
   sout << "Simulator for ";
   sout << sys->description();
   return sout.str();
   }

template <class S>
std::ostream& exit_computer<S>::serialize(std::ostream& sout) const
   {
   sout << sys;
   return sout;
   }

template <class S>
std::istream& exit_computer<S>::serialize(std::istream& sin)
   {
   free();
   src = new libbase::randgen;
   sin >> libbase::eatcomments >> sys;
   internallyallocated = true;
   return sin;
   }

// Explicit Realizations

using libbase::serializer;
using libbase::gf;

template class exit_computer<sigspace> ;
template <>
const serializer exit_computer<sigspace>::shelper("experiment",
      "exit_computer<sigspace>", exit_computer<sigspace>::create);

template class exit_computer<bool> ;
template <>
const serializer exit_computer<bool>::shelper("experiment",
      "exit_computer<bool>", exit_computer<bool>::create);

template class exit_computer<gf<1, 0x3> > ;
template <>
const serializer exit_computer<gf<1, 0x3> >::shelper("experiment",
      "exit_computer<gf<1,0x3>>", exit_computer<gf<1, 0x3> >::create);

template class exit_computer<gf<2, 0x7> > ;
template <>
const serializer exit_computer<gf<2, 0x7> >::shelper("experiment",
      "exit_computer<gf<2,0x7>>", exit_computer<gf<2, 0x7> >::create);

template class exit_computer<gf<3, 0xB> > ;
template <>
const serializer exit_computer<gf<3, 0xB> >::shelper("experiment",
      "exit_computer<gf<3,0xB>>", exit_computer<gf<3, 0xB> >::create);

template class exit_computer<gf<4, 0x13> > ;
template <>
const serializer exit_computer<gf<4, 0x13> >::shelper("experiment",
      "exit_computer<gf<4,0x13>>", exit_computer<gf<4, 0x13> >::create);

template class exit_computer<gf<5, 0x25> > ;
template <>
const serializer exit_computer<gf<5, 0x25> >::shelper("experiment",
      "exit_computer<gf<5,0x25>>", exit_computer<gf<5, 0x25> >::create);

template class exit_computer<gf<6, 0x43> > ;
template <>
const serializer exit_computer<gf<6, 0x43> >::shelper("experiment",
      "exit_computer<gf<6,0x43>>", exit_computer<gf<6, 0x43> >::create);

template class exit_computer<gf<7, 0x89> > ;
template <>
const serializer exit_computer<gf<7, 0x89> >::shelper("experiment",
      "exit_computer<gf<7,0x89>>", exit_computer<gf<7, 0x89> >::create);

template class exit_computer<gf<8, 0x11D> > ;
template <>
const serializer exit_computer<gf<8, 0x11D> >::shelper("experiment",
      "exit_computer<gf<8,0x11D>>", exit_computer<gf<8, 0x11D> >::create);

template class exit_computer<gf<9, 0x211> > ;
template <>
const serializer exit_computer<gf<9, 0x211> >::shelper("experiment",
      "exit_computer<gf<9,0x211>>", exit_computer<gf<9, 0x211> >::create);

template class exit_computer<gf<10, 0x409> > ;
template <>
const serializer exit_computer<gf<10, 0x409> >::shelper("experiment",
      "exit_computer<gf<10,0x409>>", exit_computer<gf<10, 0x409> >::create);

// realizations for non-default containers

// template class exit_computer<bool,matrix>;
// template <>
// const serializer exit_computer<bool,matrix>::shelper("experiment", "exit_computer<bool,matrix>", exit_computer<bool,matrix>::create);

} // end namespace
