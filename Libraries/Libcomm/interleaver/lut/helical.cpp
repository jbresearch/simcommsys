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

#include "helical.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// initialisation functions

template <class real>
void helical<real>::init(const int tau, const int rows, const int cols)
   {
   helical<real>::rows = rows;
   helical<real>::cols = cols;

   int blklen = rows * cols;
   if (blklen > tau)
      {
      std::cerr
            << "FATAL ERROR (helical): Interleaver block size cannot be greater than BCJR block." << std::endl;
      exit(1);
      }
   this->lut.init(tau);
   int row = rows - 1, col = 0;
   int i;
   for (i = 0; i < blklen; i++)
      {
      this->lut(i) = row * cols + col;
      row = (row - 1 + rows) % rows;
      col = (col + 1) % cols;
      }
   for (i = blklen; i < tau; i++)
      this->lut(i) = i;
   }

// description output

template <class real>
std::string helical<real>::description() const
   {
   std::ostringstream sout;
   sout << "Helical " << rows << "×" << cols << " Interleaver";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& helical<real>::serialize(std::ostream& sout) const
   {
   sout << "# Interleaver size" << std::endl;
   sout << this->lut.size() << std::endl;
   sout << "# Number of rows" << std::endl;
   sout << rows << std::endl;
   sout << "# Number of cols" << std::endl;
   sout << cols << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& helical<real>::serialize(std::istream& sin)
   {
   int tau;
   sin >> libbase::eatcomments >> tau >> libbase::verify;
   sin >> libbase::eatcomments >> rows >> libbase::verify;
   sin >> libbase::eatcomments >> cols >> libbase::verify;
   init(tau, rows, cols);
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

/* Serialization string: helical<real>
 * where:
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, x, type) \
   template class helical<type>; \
   template <> \
   const serializer helical<type>::shelper( \
         "interleaver", \
         "helical<" BOOST_PP_STRINGIZE(type) ">", \
         helical<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
