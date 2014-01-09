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

#include "rand_lut.h"
#include "vector.h"
#include <sstream>

namespace libcomm {

// initialisation

template <class real>
void rand_lut<real>::init(const int tau, const int m)
   {
   p = (1 << m) - 1;
   if (tau % p != 0)
      {
      std::cerr
            << "FATAL ERROR (rand_lut): interleaver length must be a multiple of the encoder impulse respone length." << std::endl;
      exit(1);
      }
   this->lut.init(tau);
   }

// intra-frame functions

template <class real>
void rand_lut<real>::seedfrom(libbase::random& r)
   {
   this->r.seed(r.ival());
   advance();
   }

template <class real>
void rand_lut<real>::advance()
   {
   const int tau = this->lut.size();
   // create array to hold 'used' status of possible lut values
   libbase::vector<bool> used(tau);
   used = false;
   // fill in lut
   for (int t = 0; t < tau; t++)
      {
      int tdash;
      do
         {
         tdash = int(r.ival(tau) / p) * p + t % p;
         } while (used(tdash));
      used(tdash) = true;
      this->lut(t) = tdash;
      }
   }

// description output

template <class real>
std::string rand_lut<real>::description() const
   {
   std::ostringstream sout;
   sout << "Random Interleaver (self-terminating for m=" << int(log2(p + 1))
         << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& rand_lut<real>::serialize(std::ostream& sout) const
   {
   sout << this->lut.size() << std::endl;
   sout << int(log2(p + 1)) << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& rand_lut<real>::serialize(std::istream& sin)
   {
   int tau, m;
   sin >> libbase::eatcomments >> tau >> m >> libbase::verify;
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

/* Serialization string: rand_lut<real>
 * where:
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, x, type) \
   template class rand_lut<type>; \
   template <> \
   const serializer rand_lut<type>::shelper( \
         "interleaver", \
         "rand_lut<" BOOST_PP_STRINGIZE(type) ">", \
         rand_lut<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
