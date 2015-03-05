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

#include "berrou.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// initialization

template <class real>
void berrou<real>::init(const int M)
   {
   berrou<real>::M = M;

   if (libbase::weight(M) != 1)
      {
      std::cerr << "FATAL ERROR (berrou): M must be an integral power of 2." << std::endl;
      exit(1);
      }
   int tau = M * M;
   this->lut.init(tau);
   const int P[] = {17, 37, 19, 29, 41, 23, 13, 7};
   for (int i = 0; i < M; i++)
      for (int j = 0; j < M; j++)
         {
         int ir = (M / 2 + 1) * (i + j) % M;
         int xi = (i + j) % 8;
         int jr = (P[xi] * (j + 1) - 1) % M;
         this->lut(i * M + j) = ir * M + jr;
         }
   }

// description output

template <class real>
std::string berrou<real>::description() const
   {
   std::ostringstream sout;
   sout << "Berrou Interleaver (" << M << "×" << M << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& berrou<real>::serialize(std::ostream& sout) const
   {
   sout << M << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& berrou<real>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> M >> libbase::verify;
   init(M);
   return sin;
   }

} // end namespace

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;

#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: berrou<real>
 * where:
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, x, type) \
   template class berrou<type>; \
   template <> \
   const serializer berrou<type>::shelper( \
         "interleaver", \
         "berrou<" BOOST_PP_STRINGIZE(type) ">", \
         berrou<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
