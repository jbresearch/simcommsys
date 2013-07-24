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

/* \note This file contains some of the explicit realizations for fba2-cuda.cu.
 * For this module it was necessary to split the realizations over separate
 * units, or ptxas would complain with excessive cmem usage.
 */
#include "fba2-cuda.cu"

// Explicit Realizations

#include "modem/tvb-receiver-cuda.h"
#include "gf.h"

namespace cuda {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#define REAL_TYPE_SEQ \
   (float)(double)
#define REAL2_TYPE_SEQ \
   (double)

// *** Instantiations for tvb: gf types only ***

#define INSTANTIATE_TVB(r, args) \
      template class fba2<tvb_receiver<BOOST_PP_SEQ_ENUM(args)> , \
         BOOST_PP_SEQ_ENUM(args)> ; \
      template class value<fba2<tvb_receiver<BOOST_PP_SEQ_ENUM(args)> , \
         BOOST_PP_SEQ_ENUM(args)>::metric_computer> ; \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_TVB, (SYMBOL_TYPE_SEQ)(REAL_TYPE_SEQ)(REAL2_TYPE_SEQ))

} // end namespace
