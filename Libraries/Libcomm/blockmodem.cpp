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

#include "blockmodem.h"
#include "gf.h"
#include "logrealfast.h"
#include "cputimer.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// *** Blockwise Modulator Common Interface ***

// Block modem operations

template <class S, template <class > class C, class dbl>
void basic_blockmodem<S, C, dbl>::modulate(const int N, const C<int>& encoded,
      C<S>& tx)
   {
   test_invariant();
   //libbase::cputimer t("t_modulate");
   advance_always();
   domodulate(N, encoded, tx);
   //add_timer(t);
   }

template <class S, template <class > class C, class dbl>
void basic_blockmodem<S, C, dbl>::demodulate(const channel<S, C>& chan,
      const C<S>& rx, C<array1d_t>& ptable)
   {
   test_invariant();
   libbase::cputimer t("t_demodulate");
   advance_if_dirty();
   dodemodulate(chan, rx, ptable);
   mark_as_dirty();
   add_timer(t);
   }

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::vector;
using libbase::matrix;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)(matrix)
#define REAL_TYPE_SEQ \
   (double)(logrealfast)

#define INSTANTIATE(r, args) \
      template class basic_blockmodem<BOOST_PP_SEQ_ENUM(args)>; \
      template class blockmodem<BOOST_PP_SEQ_ENUM(args)>;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
