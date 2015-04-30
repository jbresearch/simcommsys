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

/* \note This file contains some of the explicit realizations for fba2-implementation.h.
 * For this module it was necessary to split the realizations over separate
 * units, or ptxas would complain with excessive cmem usage.
 */
#include "fba2-implementation.h"

// Explicit Realizations
#include "modem/tvb-receiver.h"
#include "gf.h"
#include "mpgnu.h"
#include "logrealfast.h"

namespace libcomm {

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/first_n.hpp>
#include <boost/preprocessor/seq/elem.hpp>

using libbase::mpgnu;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#define REAL_PAIRS_SEQ \
   ((mpgnu)(mpgnu)) \
   ((logrealfast)(logrealfast)) \
   ((double)(double)) \
   ((double)(float)) \
   ((float)(float))
#define FLAG_SEQ \
   (true)(false)

// *** Instantiations for tvb ***

#define INSTANTIATE3(args) \
      template class fba2<tvb_receiver< \
         BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_FIRST_N(3,args))> , \
         BOOST_PP_SEQ_ENUM(args)> ;

#define INSTANTIATE2(r, flags, reals) \
      INSTANTIATE3( (BOOST_PP_SEQ_ELEM(5,SYMBOL_TYPE_SEQ)) reals flags )

#define INSTANTIATE1(r, flags) \
      BOOST_PP_SEQ_FOR_EACH(INSTANTIATE2, flags, REAL_PAIRS_SEQ)

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE1,
      (FLAG_SEQ)(FLAG_SEQ)(FLAG_SEQ))

} // end namespace
