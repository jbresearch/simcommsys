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

#include "codec_reshaped.h"

namespace libcomm {

// object serialization - saving

template <class base_codec>
std::ostream& codec_reshaped<base_codec>::serialize(std::ostream& sout) const
   {
   return base.serialize(sout);
   }

// object serialization - loading

template <class base_codec>
std::istream& codec_reshaped<base_codec>::serialize(std::istream& sin)
   {
   return base.serialize(sin);
   }

} // end namespace

#include "turbo.h"
#include "uncoded.h"
#include "ldpc.h"

#include "gf.h"
#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define REAL_TYPE_SEQ \
   (float)(double) \
   (mpreal)(mpgnu) \
   (logreal)(logrealfast)

/*** Turbo codes ***/

/* Serialization string: codec_reshaped<turbo<real1,real2>>
 * where:
 *      real1 = float | double | mpreal | mpgnu | logreal | logrealfast
 *      real1 is the internal arithmetic type
 *      real2 is the inter-iteration statistics type (must be double)
 */
#define INSTANTIATE_TURBO(r, x, type) \
      template class codec_reshaped<turbo<type, double> >; \
      template <> \
      const serializer codec_reshaped<turbo<type, double> >::shelper( \
            "codec", \
            "codec_reshaped<turbo<" BOOST_PP_STRINGIZE(type) ",double>>", \
            codec_reshaped<turbo<type, double> >::create); \

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TURBO, x, REAL_TYPE_SEQ)

/*** Uncoded/repetition codes ***/

/* Serialization string: codec_reshaped<uncoded<real>>
 * where:
 *      real = float | double | mpreal | mpgnu | logreal | logrealfast
 */
#define INSTANTIATE_UNCODED(r, x, type) \
      template class codec_reshaped<uncoded<type> >; \
      template <> \
      const serializer codec_reshaped<uncoded<type> >::shelper( \
            "codec", \
            "codec_reshaped<uncoded<" BOOST_PP_STRINGIZE(type) ">>", \
            codec_reshaped<uncoded<type> >::create); \

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_UNCODED, x, (double))

/*** LDPC codes ***/

#undef  REAL_TYPE_SEQ
#define REAL_TYPE_SEQ \
   (double)(mpreal)

/* Serialization string: ldpc<type,real>
 * where:
 *      type = gf2 | gf4 ...
 *      real = double | mpreal
 */
#define INSTANTIATE_LDPC(r, args) \
      template class codec_reshaped<ldpc<BOOST_PP_SEQ_ENUM(args)> >; \
      template <> \
      const serializer codec_reshaped<ldpc<BOOST_PP_SEQ_ENUM(args)> >::shelper( \
            "codec", \
            "codec_reshaped<ldpc<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">>", \
            codec_reshaped<ldpc<BOOST_PP_SEQ_ENUM(args)> >::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_LDPC, (GF_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
