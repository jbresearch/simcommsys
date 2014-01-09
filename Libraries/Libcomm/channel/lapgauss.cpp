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

#include "lapgauss.h"

namespace libcomm {

const libbase::serializer lapgauss::shelper("channel", "lapgauss",
      lapgauss::create);

// constructors / destructors

lapgauss::lapgauss()
   {
   }

// handle functions

void lapgauss::compute_parameters(const double Eb, const double No)
   {
   sigma = sqrt(Eb * No);
   }

// channel handle functions

sigspace lapgauss::corrupt(const sigspace& s)
   {
   const double x = r.gval(sigma);
   const double y = r.gval(sigma);
   return s + sigspace(x, y);
   }

double lapgauss::pdf(const sigspace& tx, const sigspace& rx) const
   {
   using libbase::gauss;
   sigspace n = rx - tx;
   return gauss(n.i() / sigma) * gauss(n.q() / sigma);
   }

// description output

std::string lapgauss::description() const
   {
   return "Laplacian-Gaussian channel";
   }

// object serialization - saving

std::ostream& lapgauss::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& lapgauss::serialize(std::istream& sin)
   {
   return sin;
   }

} // end namespace
