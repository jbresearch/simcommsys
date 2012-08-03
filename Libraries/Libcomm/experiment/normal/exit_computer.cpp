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
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "exit_computer.h"

#include "mapper/map_straight.h"
#include "fsm.h"
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
   // Encode -> Map -> Modulate
   libbase::vector<S> transmitted = sys->encode_path(source);
   // Transmit
   libbase::vector<S> received = sys->transmit(transmitted);
   // Demodulate -> Inverse Map -> Translate
   sys->receive_path(received);
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
exit_computer<S>::exit_computer(const exit_computer<S>& c) :
   internallyallocated(true), src(new libbase::randgen), sys(
         dynamic_cast<commsys<S> *> (c.sys->clone()))
   {
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
   sin >> libbase::eatcomments >> sys >> libbase::verify;
   internallyallocated = true;
   return sin;
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ

/* Serialization string: exit_computer<type>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 */
#define INSTANTIATE(r, x, type) \
      template class exit_computer<type>; \
      template <> \
      const serializer exit_computer<type>::shelper( \
            "experiment", \
            "exit_computer<" BOOST_PP_STRINGIZE(type) ">", \
            exit_computer<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // end namespace
