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

/* \note This file contains some of the explicit realizations for
 * fba2-implementation.h. For this module it was necessary to split the
 * realizations over separate units, or ptxas would complain with excessive cmem
 * usage.
 */
#include "fba2-fss-implementation.h"

// Explicit Realizations
#include "gf.h"
#include "logrealfast.h"
#include "modem/tvb-fss-receiver.h"
#include "mpgnu.h"

namespace libcomm
{

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/first_n.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/subseq.hpp>

using libbase::logrealfast;
using libbase::mpgnu;

// clang-format off
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

#define INSTANTIATE2(args) \
      template class fba2_fss<tvb_fss_receiver< \
         BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_FIRST_N(3,args))> , \
         BOOST_PP_SEQ_ENUM(args)> ;

#define INSTANTIATE1(r, args) \
      INSTANTIATE2( (BOOST_PP_SEQ_ELEM(0,args)) \
            BOOST_PP_SEQ_ELEM(1,args) \
            (BOOST_PP_SEQ_ELEM(2,args)) )
// clang-format on

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE1,
                              (SYMBOL_TYPE_SEQ)(REAL_PAIRS_SEQ)(FLAG_SEQ))

} // namespace libcomm
