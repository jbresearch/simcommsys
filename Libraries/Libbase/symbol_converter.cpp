/*!
 * \file
 *
 * Copyright (c) 2015 Johann A. Briffa
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

#include "symbol_converter.h"
#include "vector_itfunc.h"
#include "vectorutils.h"

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Output mapping indices during transforms (produces a lot of output)
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class dbl, class dbl2>
void symbol_converter<dbl, dbl2>::aggregate_symbols(const array1i_t& in,
      array1i_t& out) const
   {
   // Initialize results vector
   out.init(in.size() / k);
   assertalways(in.size() == out.size() * k);
#if DEBUG>=2
   std::cerr << "DEBUG (symbol_converter): mapping " << in.size()
         << " -> " << out.size() << std::endl;
#endif
   // Repeat for each large output symbol
   for (int t = 0; t < out.size(); t++)
      {
      // Aggregate the next 'k' small input symbols
      // t: current large output symbol index
      // i: index into k-set of small input symbols, least-significant first
      // mul: corresponding multiplier, = S^i
      // j: corresponding index into full sequence of small input symbols
      int d = 0;
      for (int i = 0, mul = 1; i < k; i++, mul *= S)
         {
         const int j = t * k + i;
         assert(in(j) >= 0 && in(j) < S);
         d += in(j) * mul;
#if DEBUG>=2
         std::cerr << "DEBUG: out(" << t << ") += in(" << j << ") * " << mul
               << std::endl;
#endif
         }
      assert(d >= 0 && d < L);
      out(t) = d;
      }
   }

template <class dbl, class dbl2>
void symbol_converter<dbl, dbl2>::aggregate_probabilities(const array1vd_t& pin,
      array1vd_t& pout) const
   {
   // Confirm input symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0).size() == S);
   // Initialize results vector
   libbase::allocate(pout, pin.size() / k, L);
   assertalways(pin.size() == pout.size() * k);
#if DEBUG>=2
   std::cerr << "DEBUG (symbol_converter): mapping " << pin.size()
         << " -> " << pout.size() << std::endl;
#endif
   // Space for internal results
   vector<dbl2> ptemp;
   ptemp.init(L);
   // Repeat for each large output symbol (index)
   for (int t = 0; t < pout.size(); t++)
      {
      // Repeat for each large output symbol (value)
      for (int x = 0; x < L; x++)
         {
         // Aggregate the probabilities for the next 'k' small input symbols
         // with values corresponding to this large output symbol
         // t: current large output symbol index
         // x: current large output symbol value
         // i: index into k-set of small input symbols, least-significant first
         // thisx: x, shifted down so that component from index i is in lsb
         //    position, = x / S^i
         // j: corresponding index into full sequence of small input symbols
         dbl2 p = 1;
         for (int i = 0, thisx = x; i < k; i++, thisx /= S)
            {
            const int j = t * k + i;
            const int y = thisx % S;
            p *= dbl2(pin(j)(y));
#if DEBUG>=2
            std::cerr << "DEBUG: pout(" << t << ")(" << x << ") *= pin(" << j
                  << ")(" << y << ")" << std::endl;
#endif
            }
         ptemp(x) = p;
         }
      // Normalize and copy results
      normalize(ptemp, pout(t));
      }
   }

template <class dbl, class dbl2>
void symbol_converter<dbl, dbl2>::divide_symbols(const array1i_t& in,
      array1i_t& out) const
   {
   // Initialize results vector
   out.init(in.size() * k);
#if DEBUG>=2
   std::cerr << "DEBUG (symbol_converter): mapping " << in.size()
         << " -> " << out.size() << std::endl;
#endif
   // Repeat for each large input symbol
   for (int t = 0; t < in.size(); t++)
      {
      assert(in(t) >= 0 && in(t) < L);
      // Divide this large input symbol into 'k' small output symbols
      for (int i = 0, div = 1; i < k; i++, div *= S)
         {
         // t: current large input symbol index
         // i: index into k-set of small output symbols, least-significant first
         // div: corresponding divider, = S^i
         // x: starts as current large input symbol value, shifted down so
         //    that component from index i is in lsb position, = in(t) / S^i
         // j: corresponding index into full sequence of small output symbols
         const int j = t * k + i;
         out(j) = (in(t) / div) % S;
#if DEBUG>=2
         std::cerr << "DEBUG: out(" << j << ") = in(" << t << ") / " << div
               << " mod " << S << " = " << out(j) << std::endl;
#endif
         }
      }
   }

template <class dbl, class dbl2>
void symbol_converter<dbl, dbl2>::divide_probabilities(const array1vd_t& pin,
      array1vd_t& pout) const
   {
   // Confirm input symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0).size() == L);
   // Initialize results vector
   libbase::allocate(pout, pin.size() * k, S);
#if DEBUG>=2
   std::cerr << "DEBUG (symbol_converter): mapping " << pin.size()
         << " -> " << pout.size() << std::endl;
#endif
   // Space for internal results
   vector<dbl2> ptemp;
   ptemp.init(S);
   // Repeat for each small output symbol (index)
   for (int t = 0, mul = 1; t < pout.size(); t++, mul *= S)
      {
      const int j = t / k;
      const int i = t % k;
      if (i == 0)
         mul = 1;
      // Repeat for each small output symbol (value)
      for (int x = 0; x < S; x++)
         {
         // Aggregate the probabilities for the large input symbols with
         // index and values corresponding to this small output symbol
         // t: current small output symbol index
         // x: current small output symbol value
         // j: corresponding index into full sequence of large input symbols
         //    (remains the same for each k-set of small output symbols)
         // i: index into k-set of small output symbols, least-significant first
         // mul: corresponding multiplier, = S^i
         // d: large input symbol being considered
         dbl2 p = 0;
         for (int d = 0; d < L; d++)
            if ((d / mul) % S == x)
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
      normalize(ptemp, pout(t));
      }
   }

} /* namespace libbase */

#include "logrealfast.h"

namespace libbase {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each_product.hpp>

#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

#define INSTANTIATE(r, args) \
      template class symbol_converter<BOOST_PP_SEQ_ENUM(args)>;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (REAL_TYPE_SEQ)(REAL_TYPE_SEQ))

} /* namespace libbase */
