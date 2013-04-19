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

#include "map_dividing.h"
#include "vectorutils.h"
#include "vector_itfunc.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Output mapping indices during transform and inverse
//     (note this produces a lot of output)
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*** Vector Specialization ***/

// Interface with mapper
template <class dbl, class dbl2>
void map_dividing<libbase::vector, dbl, dbl2>::dotransform(const array1i_t& in,
      array1i_t& out) const
   {
   assertalways(in.size() == This::input_block_size());
   // Initialize results vector
   out.init(This::output_block_size());
#if DEBUG>=2
   std::cerr << "DEBUG (map_dividing::transform): mapping " << in.size()
         << " -> " << out.size() << std::endl;
#endif
   // Repeat for each q-ary encoded symbol
   for (int t = 0; t < in.size(); t++)
      // Divide this q-ary encoded symbol into 'k' M-ary modulated symbols
      for (int i = 0, div = 1; i < k; i++, div *= Base::M)
         {
         // t: current q-ary encoded symbol index
         // i: index into k-set of M-ary modulated symbols, least-significant first
         // div: corresponding divider, = M^i
         // x: starts as current q-ary encoded symbol value, shifted down so
         //    that component from index i is in lsb position, = in(t) / M^i
         // j: corresponding index into full sequence of M-ary modulated symbols
         const int j = t * k + i;
         out(j) = (in(t) / div) % Base::M;
#if DEBUG>=2
         std::cerr << "DEBUG: out(" << j << ") = in(" << t << ") / " << div
               << " mod " << Base::M << " = " << out(j) << std::endl;
#endif
         }
   }

template <class dbl, class dbl2>
void map_dividing<libbase::vector, dbl, dbl2>::dotransform(
      const array1vd_t& pin, array1vd_t& pout) const
   {
   // Confirm input symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0).size() == Base::q);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::input_block_size());
   // Initialize results vector
   libbase::allocate(pout, This::output_block_size(), Base::M);
#if DEBUG>=2
   std::cerr << "DEBUG (map_dividing::transform): mapping " << pin.size()
         << " -> " << pout.size() << std::endl;
#endif
   // Space for internal results
   libbase::vector<dbl2> ptemp;
   ptemp.init(Base::M);
   // Repeat for each M-ary modulation symbol (index)
   for (int t = 0, mul = 1; t < This::output_block_size(); t++, mul *= Base::M)
      {
      const int j = t / k;
      const int i = t % k;
      if (i == 0)
         mul = 1;
      // Repeat for each M-ary modulation symbol (value)
      for (int x = 0; x < Base::M; x++)
         {
         // Aggregate the probabilities for the q-ary encoded symbols with
         // index and values corresponding to this M-ary modulation symbol
         // t: current M-ary modulation symbol index
         // x: current M-ary modulation symbol value
         // j: corresponding index into full sequence of q-ary encoded symbols
         //    (remains the same for each k-set of M-ary modulation symbols)
         // i: index into k-set of M-ary modulation symbols, least-significant first
         // mul: corresponding multiplier, = M^i
         // d: q-ary encoded symbol being considered
         dbl2 p = 0;
         for (int d = 0; d < Base::q; d++)
            if ((d / mul) % Base::M == x)
               {
               p += dbl2(pin(j)(d));
#if DEBUG>=2
               std::cerr << "DEBUG: pout(" << t << ")(" << x << ") += pin(" << j
                     << ")(" << d << ")" << std::endl;
#endif
               }
         ptemp(x) = p;
         }
      // Normalize and copy results
      libbase::normalize(ptemp, pout(t));
      }
   }

template <class dbl, class dbl2>
void map_dividing<libbase::vector, dbl, dbl2>::doinverse(const array1vd_t& pin,
      array1vd_t& pout) const
   {
   // Confirm modulation symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0).size() == Base::M);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::output_block_size());
   // Initialize results vector
   libbase::allocate(pout, This::input_block_size(), Base::q);
#if DEBUG>=2
   std::cerr << "DEBUG (map_dividing::inverse): mapping " << pin.size()
         << " -> " << pout.size() << std::endl;
#endif
   // Space for internal results
   libbase::vector<dbl2> ptemp;
   ptemp.init(Base::q);
   // Repeat for each q-ary encoded symbol (index and value)
   for (int t = 0; t < This::input_block_size(); t++)
      {
      for (int x = 0; x < Base::q; x++)
         {
         // Aggregate the probabilities for the M-ary modulation symbols with
         // index and values corresponding to this q-ary encoded symbol
         // t: current q-ary encoded symbol index
         // x: current q-ary encoded symbol value
         // i: index into k-set of M-ary modulation symbols, least-significant first
         // thisx: x, shifted down so that component from index i is in lsb
         //    position, = x / M^i
         // j: corresponding index into full sequence of M-ary modulation symbols
         dbl2 p = 1;
         for (int i = 0, thisx = x; i < k; i++, thisx /= Base::M)
            {
            const int j = t * k + i;
            const int y = thisx % Base::M;
            p *= dbl2(pin(j)(y));
#if DEBUG>=2
            std::cerr << "DEBUG: pout(" << t << ")(" << x << ") *= pin(" << j
                  << ")(" << y << ")" << std::endl;
#endif
            }
         ptemp(x) = p;
         }
      // Normalize and copy results
      libbase::normalize(ptemp, pout(t));
      }
   }

// Description

template <class dbl, class dbl2>
std::string map_dividing<libbase::vector, dbl, dbl2>::description() const
   {
   std::ostringstream sout;
   sout << "Dividing Mapper (Vector)";
   sout << " [" << this->input_block_size() << "<->"
         << this->output_block_size() << "]";
   return sout.str();
   }

// Serialization Support

template <class dbl, class dbl2>
std::ostream& map_dividing<libbase::vector, dbl, dbl2>::serialize(
      std::ostream& sout) const
   {
   return sout;
   }

template <class dbl, class dbl2>
std::istream& map_dividing<libbase::vector, dbl, dbl2>::serialize(
      std::istream& sin)
   {
   return sin;
   }

} // end namespace

#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;
using libbase::vector;

#define CONTAINER_TYPE_SEQ \
   (vector)
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: map_dividing<container,dbl,dbl2>
 * where:
 *      container = vector
 *      dbl = float | double | logrealfast
 *              [dbl is the interface arithmetic type]
 *      dbl2 = float | double | logrealfast
 *              [dbl2 is the internal arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class map_dividing<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer map_dividing<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "mapper", \
            "map_dividing<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            map_dividing<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
      (CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
