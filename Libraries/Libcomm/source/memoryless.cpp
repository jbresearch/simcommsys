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

#include "memoryless.h"
#include <sstream>

namespace libcomm {

// initialization

template <class S, template <class > class C>
void memoryless<S, C>::init(libbase::vector<float> symbol_probabilities)
   {
   const int n = symbol_probabilities.size();
   float sum = 0;
   for (int i = 0; i < n; i++)
      {
      sum += symbol_probabilities(i);
      cpt(i) = sum;
      }
   assertalways(sum == 1.0);
   }

// object serialization - saving

template <class S, template <class > class C>
std::ostream& memoryless<S, C>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "#: symbol count" << std::endl;
   sout << cpt.size() << std::endl;
   sout << "#: symbol probabilities" << std::endl;
   float sum = 0;
   for (int i = 0; i < cpt.size(); i++)
      {
      sout << cpt(i) - sum << std::endl;
      sum = cpt(i);
      }
   return sout;
   }

// object serialization - loading

template <class S, template <class > class C>
std::istream& memoryless<S, C>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
   // read count of input symbols
   int temp;
   sin >> libbase::eatcomments >> temp >> libbase::verify;
   // read input symbols from stream
   libbase::vector<float> symbol_probabilities;
   symbol_probabilities.init(temp);
   sin >> libbase::eatcomments;
   symbol_probabilities.serialize(sin);
   libbase::verify(sin);
   // initialize
   init(symbol_probabilities);
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

#define SYMBOL_TYPE_SEQ \
   (int) \
   GF_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)
   //(vector)(matrix)

/* Serialization string: memoryless<type,container>
 * where:
 *      type = int | gf2 | gf4 ...
 *      container = vector | matrix
 */
#define INSTANTIATE(r, args) \
      template class memoryless<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer memoryless<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "source", \
            "memoryless<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            memoryless<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ))

} // end namespace
