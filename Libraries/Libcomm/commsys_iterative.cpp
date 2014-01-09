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

#include "commsys_iterative.h"

#include "modem/informed_modulator.h"
#include "gf.h"
#include <sstream>

namespace libcomm {

// Communication System Interface

template <class S, template <class > class C>
void commsys_iterative<S, C>::receive_path(const C<S>& received)
   {
   // Demodulate
   C<array1d_t> ptable_mapped;
   informed_modulator<S>& m = dynamic_cast<informed_modulator<S>&> (*this->mdm);
   for (int i = 0; i < iter; i++)
      {
      libbase::trace
            << "DEBUG (commsys_iterative): Starting demodulation iteration "
            << i << std::endl;
      m.demodulate(*this->rxchan, received, ptable_mapped, ptable_mapped);
      m.mark_as_clean();
      }
   m.mark_as_dirty();
   // After-demodulation receive path
   this->softreceive_path(ptable_mapped);
   }

// Description & Serialization

template <class S, template <class > class C>
std::string commsys_iterative<S, C>::description() const
   {
   std::ostringstream sout;
   sout << "Iterative ";
   sout << commsys<S, C>::description() << ", ";
   sout << iter << " iterations";
   return sout.str();
   }

template <class S, template <class > class C>
std::ostream& commsys_iterative<S, C>::serialize(std::ostream& sout) const
   {
   sout << iter << std::endl;
   commsys<S, C>::serialize(sout);
   return sout;
   }

template <class S, template <class > class C>
std::istream& commsys_iterative<S, C>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> iter >> libbase::verify;
   commsys<S, C>::serialize(sin);
   return sin;
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::matrix;
using libbase::vector;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)
   //(vector)(matrix)

/* Serialization string: commsys_iterative<type,container>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      container = vector | matrix
 */
#define INSTANTIATE(r, args) \
      template class commsys_iterative<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer commsys_iterative<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "commsys", \
            "commsys_iterative<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            commsys_iterative<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ))

} // end namespace
