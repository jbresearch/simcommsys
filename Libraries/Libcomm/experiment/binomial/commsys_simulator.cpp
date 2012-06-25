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

#include "commsys_simulator.h"

#include "mapper/map_straight.h"
#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>
#include <sstream>

namespace libcomm {

// *** Templated Common Base ***

// Internal functions

/*!
 * \brief Create source sequence to be encoded
 * \return Source sequence of the required length
 * 
 * The source sequence consists of uniformly random symbols followed by a
 * tail sequence if required by the given codec.
 */
template <class S, class R>
libbase::vector<int> commsys_simulator<S, R>::createsource()
   {
   const int tau = sys->input_block_size();
   libbase::vector<int> source(tau);
   for (int t = 0; t < tau; t++)
      source(t) = src.ival(sys->num_inputs());
   return source;
   }

// Experiment handling

/*!
 * \brief Perform a complete encode->transmit->receive cycle
 * \param[out] result   Vector containing the set of results to be updated
 *
 * Results are organized according to the collector used, as a function of
 * the iteration count.
 *
 * \note The results collector assumes that the result vector is an accumulator,
 * so that every call adds to the existing result. This explains the need to
 * initialize the result vector to zero.
 */
template <class S, class R>
void commsys_simulator<S, R>::sample(libbase::vector<double>& result)
   {
   // Reset timers
   this->reset_timers();
   // Initialise result vector
   result.init(count());
   result = 0;
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
      R::updateresults(result, i, source, decoded);
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

// Description & Serialization

template <class S, class R>
std::string commsys_simulator<S, R>::description() const
   {
   std::ostringstream sout;
   sout << "Simulator for ";
   sout << sys->description();
   return sout.str();
   }

template <class S, class R>
std::ostream& commsys_simulator<S, R>::serialize(std::ostream& sout) const
   {
   sout << sys;
   return sout;
   }

template <class S, class R>
std::istream& commsys_simulator<S, R>::serialize(std::istream& sin)
   {
   free();
   sin >> libbase::eatcomments >> sys >> libbase::verify;
   return sin;
   }

} // end namespace

#include "gf.h"
#include "result_collector/commsys/errors_hamming.h"
#include "result_collector/commsys/errors_levenshtein.h"
#include "result_collector/commsys/prof_burst.h"
#include "result_collector/commsys/prof_pos.h"
#include "result_collector/commsys/prof_sym.h"
#include "result_collector/commsys/hist_symerr.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ
#define COLLECTOR_TYPE_SEQ \
   (errors_hamming) \
   (errors_levenshtein) \
   (prof_burst) \
   (prof_pos) \
   (prof_sym) \
   (hist_symerr)

/* Serialization string: commsys_simulator<type,collector>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      collector = errors_hamming | errors_levenshtein | ...
 */
#define INSTANTIATE(r, args) \
      template class commsys_simulator<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer commsys_simulator<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "experiment", \
            "commsys_simulator<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            commsys_simulator<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(COLLECTOR_TYPE_SEQ))

} // end namespace
