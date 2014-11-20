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

#include "fba2-factory.h"

#ifdef USE_CUDA
#  include "fba2-cuda.h"
#  define FBA_TYPE cuda::fba2
#else
#  include "fba2.h"
#  define FBA_TYPE fba2
#endif
#include "fba2-fss.h"

#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/enum.hpp>

namespace libcomm {

template <class receiver_t, class sig, class real, class real2>
boost::shared_ptr<fba2_interface<receiver_t, sig, real> > fba2_factory<
      receiver_t, sig, real, real2>::get_instance(bool fss, bool thresholding,
      bool lazy, bool globalstore)
   {
   boost::shared_ptr<fba2_interface<receiver_t, sig, real> > fba_ptr;

#define FLAG_SEQ \
(true)(false)

#define CONDITIONAL(r, args) \
   if (BOOST_PP_SEQ_ELEM(0,args) == thresholding && \
         BOOST_PP_SEQ_ELEM(1,args) == lazy && \
         BOOST_PP_SEQ_ELEM(2,args) == globalstore) \
         fba_ptr.reset(new FBA_TYPE<receiver_t, sig, real, real2, BOOST_PP_SEQ_ENUM(args)>);

   if (!fss)
      {
      BOOST_PP_SEQ_FOR_EACH_PRODUCT(CONDITIONAL,
            (FLAG_SEQ)(FLAG_SEQ)(FLAG_SEQ))
      }
   else
      {
      if (globalstore)
         fba_ptr.reset(new fba2_fss<receiver_t, sig, real, real2, true>);
      else
         fba_ptr.reset(new fba2_fss<receiver_t, sig, real, real2, false>);
      }

#undef CONDITIONAL

   assertalways(fba_ptr);
   return fba_ptr;
   }

} // end namespace


// Explicit Realizations
#ifdef USE_CUDA
#  include "modem/tvb-receiver-cuda.h"
#  define RECV_TYPE cuda::tvb_receiver
#else
#  include "modem/tvb-receiver.h"
#  define RECV_TYPE tvb_receiver
#endif
#include "gf.h"
#include "mpgnu.h"
#include "logrealfast.h"

namespace libcomm {

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/first_n.hpp>

using libbase::mpgnu;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#ifdef USE_CUDA
#define REAL_PAIRS_SEQ \
   ((double)(double)) \
   ((double)(float)) \
   ((float)(float))
#else
#define REAL_PAIRS_SEQ \
   ((mpgnu)(mpgnu)) \
   ((logrealfast)(logrealfast)) \
   ((double)(double)) \
   ((double)(float)) \
   ((float)(float))
#endif

// *** Instantiations for tvb ***

#define INSTANTIATE3(args) \
      template class fba2_factory< \
         RECV_TYPE<BOOST_PP_SEQ_ENUM(args)> , \
         BOOST_PP_SEQ_ENUM(args)> ;

#define INSTANTIATE2(r, symbol, reals) \
      INSTANTIATE3( symbol reals )

#define INSTANTIATE1(r, symbol) \
      BOOST_PP_SEQ_FOR_EACH(INSTANTIATE2, symbol, REAL_PAIRS_SEQ)

// NOTE: we *have* to use for-each product here as we cannot nest for-each
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE1, (SYMBOL_TYPE_SEQ))

} // end namespace
