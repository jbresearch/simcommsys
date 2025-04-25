/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
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


 #include "gaussian.h"
 #include <sstream>

 namespace libcomm
 {

 // Save mean and variance to stream
 template <class S, template <class> class C>
 std::ostream& gaussian<S, C>::serialize(std::ostream& sout) const
 {
     sout << "# Mean" << std::endl;
     sout << mean << std::endl;
     sout << "# Variance" << std::endl;
     sout << stddev << std::endl;
     return sout;
 }

 // Load mean and variance from stream
 template <class S, template <class> class C>
 std::istream& gaussian<S, C>::serialize(std::istream& sin)
 {
     assertalways(sin.good());
     sin >> libbase::eatcomments >> mean;
     sin >> libbase::eatcomments >> stddev;

     return sin;
 }

 } // namespace libcomm

 // --- Explicit template instantiations ---

 #include <boost/preprocessor/seq/enum.hpp>
 #include <boost/preprocessor/seq/for_each_product.hpp>
 #include <boost/preprocessor/stringize.hpp>

 using libbase::serializer;
 using libbase::vector;

 namespace libcomm
 {

 #define SYMBOL_TYPE_SEQ (gaussian_state)
 #define CONTAINER_TYPE_SEQ (vector)

 #define INSTANTIATE(r, args) \
     template class gaussian<BOOST_PP_SEQ_ENUM(args)>; \
     template <> \
     const serializer gaussian<BOOST_PP_SEQ_ENUM(args)>::shelper( \
         "source", \
         "gaussian<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, args)) "," \
         BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1, args)) ">", \
         gaussian<BOOST_PP_SEQ_ENUM(args)>::create);

 BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ))

 } // namespace libcomm
