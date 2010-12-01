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
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "padded.h"
#include <sstream>

namespace libcomm {

using libbase::vector;
using libbase::matrix;

// construction and destruction

template <class real>
padded<real>::padded() :
   otp(NULL), inter(NULL)
   {
   }

template <class real>
padded<real>::padded(const interleaver<real>& inter, const fsm& encoder,
      const bool terminated, const bool renewable)
   {
   otp = new onetimepad<real> (encoder, inter.size(), terminated, renewable);
   padded<real>::inter = dynamic_cast<interleaver<real> *> (inter.clone());
   }

template <class real>
padded<real>::padded(const padded& x)
   {
   inter = dynamic_cast<interleaver<real> *> (x.inter->clone());
   otp = dynamic_cast<interleaver<real> *> (x.otp->clone());
   }

template <class real>
padded<real>::~padded()
   {
   if (otp != NULL)
      delete otp;
   if (inter != NULL)
      delete inter;
   }

// inter-frame operations

template <class real>
void padded<real>::seedfrom(libbase::random& r)
   {
   assertalways(otp);
   otp->seedfrom(r);
   }

template <class real>
void padded<real>::advance()
   {
   assertalways(otp);
   otp->advance();
   }

// transform functions

template <class real>
void padded<real>::transform(const vector<int>& in, vector<int>& out) const
   {
   vector<int> temp;
   inter->transform(in, temp);
   otp->transform(temp, out);
   }

template <class real>
void padded<real>::transform(const matrix<real>& in, matrix<real>& out) const
   {
   matrix<real> temp;
   inter->transform(in, temp);
   otp->transform(temp, out);
   }

template <class real>
void padded<real>::inverse(const matrix<real>& in, matrix<real>& out) const
   {
   matrix<real> temp;
   otp->inverse(in, temp);
   inter->inverse(temp, out);
   }

// description output

template <class real>
std::string padded<real>::description() const
   {
   std::ostringstream sout;
   sout << "Padded Interleaver [" << inter->description() << " + "
         << otp->description() << "]";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& padded<real>::serialize(std::ostream& sout) const
   {
   sout << otp;
   sout << inter;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& padded<real>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> otp;
   sin >> libbase::eatcomments >> inter;
   return sin;
   }

// Explicit instantiations

template class padded<float> ;
template <>
const libbase::serializer padded<float>::shelper("interleaver",
      "padded<float>", padded<float>::create);

template class padded<double> ;
template <>
const libbase::serializer padded<double>::shelper("interleaver",
      "padded<double>", padded<double>::create);

template class padded<libbase::logrealfast> ;
template <>
const libbase::serializer padded<libbase::logrealfast>::shelper("interleaver",
      "padded<logrealfast>", padded<libbase::logrealfast>::create);

} // end namespace
