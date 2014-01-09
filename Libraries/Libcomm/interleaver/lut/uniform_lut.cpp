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

#include "uniform_lut.h"
#include <sstream>

namespace libcomm {

// initialisation

template <class real>
void uniform_lut<real>::init(const int tau, const int m)
   {
   uniform_lut<real>::tau = tau;
   uniform_lut<real>::m = m;
   this->lut.init(tau);
   }

// intra-frame functions

template <class real>
void uniform_lut<real>::advance()
   {
   // create array to hold 'used' status of possible lut values
   libbase::vector<bool> used(tau - m);
   used = false;
   // fill in lut
   int t;
   for (t = 0; t < tau - m; t++)
      {
      int tdash;
      do
         {
         tdash = r.ival(tau - m);
         } while (used(tdash));
      used(tdash) = true;
      this->lut(t) = tdash;
      }
   for (t = tau - m; t < tau; t++)
      this->lut(t) = fsm::tail;
   }

// description output

template <class real>
std::string uniform_lut<real>::description() const
   {
   std::ostringstream sout;
   sout << "Uniform Interleaver";
   if (m > 0)
      sout << " (Forced tail length " << m << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& uniform_lut<real>::serialize(std::ostream& sout) const
   {
   sout << "# Interleaver size" << std::endl;
   sout << this->lut.size() << std::endl;
   sout << "# Forced tail length" << std::endl;
   sout << m << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& uniform_lut<real>::serialize(std::istream& sin)
   {
   int tau, m;
   sin >> libbase::eatcomments >> tau >> libbase::verify;
   sin >> libbase::eatcomments >> m >> libbase::verify;
   init(tau, m);
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

/* Serialization string: uniform_lut<real>
 * where:
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, x, type) \
   template class uniform_lut<type>; \
   template <> \
   const serializer uniform_lut<type>::shelper( \
         "interleaver", \
         "uniform_lut<" BOOST_PP_STRINGIZE(type) ">", \
         uniform_lut<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
