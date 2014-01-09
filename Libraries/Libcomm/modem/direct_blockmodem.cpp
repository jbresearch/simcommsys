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

#include "direct_blockmodem.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

using libbase::serializer;
using libbase::vector;
using libbase::matrix;

// *** Vector GF(q) blockmodem ***

// Block modem operations

template <class G, class dbl>
void direct_blockmodem_implementation<G, vector, dbl>::domodulate(const int N,
      const vector<int>& encoded, vector<G>& tx)
   {
   // Inherit sizes
   const int tau = encoded.size();
   // Initialize results vector
   tx.init(tau);
   // Modulate encoded stream
   for (int t = 0; t < tau; t++)
      tx(t) = Implementation::modulate(encoded(t));
   }

template <class G, class dbl>
void direct_blockmodem_implementation<G, vector, dbl>::dodemodulate(
      const channel<G, vector>& chan, const vector<G>& rx,
      vector<array1d_t>& ptable)
   {
   // Inherit sizes
   const int M = this->num_symbols();
   // Allocate space for temporary results
   vector<vector<double> > ptable_double;
      {
      // Create a matrix of all possible transmitted symbols
      vector<G> tx(M);
      for (int x = 0; x < M; x++)
         tx(x) = Implementation::modulate(x);
      // Work out the probabilities of each possible signal
      chan.receive(tx, rx, ptable_double);
      }
   // Convert result
   ptable = ptable_double;
   }

// *** Templated GF(q) blockmodem ***

// Description

template <class G, template <class > class C, class dbl>
std::string direct_blockmodem<G, C, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Blockwise " << Implementation::description();
   return sout.str();
   }

// Serialization Support

template <class G, template <class > class C, class dbl>
std::ostream& direct_blockmodem<G, C, dbl>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class G, template <class > class C, class dbl>
std::istream& direct_blockmodem<G, C, dbl>::serialize(std::istream& sin)
   {
   return sin;
   }

} // end namespace

#include "gf.h"
#include "erasable.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::erasable;
using libbase::logrealfast;
using libbase::matrix;
using libbase::vector;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)
   //(vector)(matrix)
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

#define ADD_ERASABLE(r, x, type) \
   (type)(erasable<type>)

#define ALL_SYMBOL_TYPE_SEQ \
   BOOST_PP_SEQ_FOR_EACH(ADD_ERASABLE, x, SYMBOL_TYPE_SEQ)

/* Serialization string: direct_blockmodem<type,container,real>
 * where:
 *      type = bool | gf2 | gf4 ... (or erasable<type> for all these)
 *      container = vector | matrix
 *      real = float | double | logrealfast
 */
#define INSTANTIATE(r, args) \
      template class direct_blockmodem_implementation<BOOST_PP_SEQ_ENUM(args)>; \
      template class direct_blockmodem<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer direct_blockmodem<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "blockmodem", \
            "direct_blockmodem<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            direct_blockmodem<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (ALL_SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
