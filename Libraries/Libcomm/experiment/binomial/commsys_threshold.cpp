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

#include "commsys_threshold.h"

#include <sstream>

namespace libcomm {

// Experiment parameter handling

template <class S, class R>
void commsys_threshold<S, R>::set_parameter(const double x)
   {
   parametric& m = dynamic_cast<parametric&> (*this->sys->getmodem());
   m.set_parameter(x);
   }

template <class S, class R>
double commsys_threshold<S, R>::get_parameter() const
   {
   const parametric& m =
         dynamic_cast<const parametric&> (*this->sys->getmodem());
   return m.get_parameter();
   }

// Description & Serialization

template <class S, class R>
std::string commsys_threshold<S, R>::description() const
   {
   std::ostringstream sout;
   sout << "Modem-threshold-varying ";
   sout << Base::description();
   return sout.str();
   }

template <class S, class R>
std::ostream& commsys_threshold<S, R>::serialize(std::ostream& sout) const
   {
   sout << Base::get_parameter() << std::endl;
   Base::serialize(sout);
   return sout;
   }

template <class S, class R>
std::istream& commsys_threshold<S, R>::serialize(std::istream& sin)
   {
   double x;
   sin >> libbase::eatcomments >> x >> libbase::verify;
   Base::serialize(sin);
   Base::set_parameter(x);
   return sin;
   }

} // end namespace

#include "gf.h"
#include "result_collector/commsys/errors_hamming.h"
#include "result_collector/commsys/errors_levenshtein.h"
#include "result_collector/commsys/prof_burst.h"
#include "result_collector/commsys/prof_pos.h"
#include "result_collector/commsys/prof_sym.h"
#include "result_collector/commsys/hist_symerr.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ
#define COLLECTOR_TYPE_SEQ \
   (errors_hamming) \
   (errors_levenshtein) \
   (prof_burst) \
   (prof_pos) \
   (prof_sym) \
   (hist_symerr)

/* Serialization string: commsys_threshold<type,collector>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      collector = errors_hamming | errors_levenshtein | ...
 */
#define INSTANTIATE(r, args) \
      template class commsys_threshold<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer commsys_threshold<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "experiment", \
            "commsys_threshold<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            commsys_threshold<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(COLLECTOR_TYPE_SEQ))

} // end namespace
