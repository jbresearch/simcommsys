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

#include "shift_lut.h"
#include <sstream>

namespace libcomm {

// initialisation functions

template <class real>
void shift_lut<real>::init(const int amount, const int tau)
   {
   shift_lut<real>::amount = amount;

   this->lut.init(tau);
   for (int i = 0; i < tau; i++)
      this->lut(i) = (i + amount) % tau;
   }

// description output

template <class real>
std::string shift_lut<real>::description() const
   {
   std::ostringstream sout;
   sout << "Shift by " << amount << " Interleaver";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& shift_lut<real>::serialize(std::ostream& sout) const
   {
   sout << this->lut.size() << std::endl;
   sout << amount << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& shift_lut<real>::serialize(std::istream& sin)
   {
   int tau, amount;
   sin >> libbase::eatcomments >> tau >> libbase::verify;
   sin >> libbase::eatcomments >> amount >> libbase::verify;
   init(amount, tau);
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

/* Serialization string: shift_lut<real>
 * where:
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, x, type) \
   template class shift_lut<type>; \
   template <> \
   const serializer shift_lut<type>::shelper( \
         "interleaver", \
         "shift_lut<" BOOST_PP_STRINGIZE(type) ">", \
         shift_lut<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
