/*!
 * \file
 * $Id$
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

#include "informed_modulator.h"

namespace libcomm {

// Block modem operations

template <class S, template <class > class C>
void informed_modulator<S, C>::demodulate(const channel<S, C>& chan,
      const C<S>& rx, const C<array1d_t>& app, C<array1d_t>& ptable)
   {
   this->advance_if_dirty();
   dodemodulate(chan, rx, app, ptable);
   this->mark_as_dirty();
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>

using libbase::matrix;
using libbase::vector;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)(matrix)

#define INSTANTIATE(r, args) \
      template class informed_modulator<BOOST_PP_SEQ_ENUM(args)>;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ))

} // end namespace
