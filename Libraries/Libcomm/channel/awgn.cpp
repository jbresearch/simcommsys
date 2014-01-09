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

#include "awgn.h"

namespace libcomm {

const libbase::serializer awgn::shelper("channel", "awgn", awgn::create);

// handle functions

void awgn::compute_parameters(const double Eb, const double No)
   {
   sigma = sqrt(Eb * No);
   }

// channel handle functions

sigspace awgn::corrupt(const sigspace& s)
   {
   const double x = r.gval(sigma);
   const double y = r.gval(sigma);
   return s + sigspace(x, y);
   }

double awgn::pdf(const sigspace& tx, const sigspace& rx) const
   {
   sigspace n = rx - tx;
   using libbase::gauss;
   return gauss(n.i() / sigma) * gauss(n.q() / sigma);
   }

// Description

std::string awgn::description() const
   {
   return "AWGN channel";
   }

// Serialization Support

std::ostream& awgn::serialize(std::ostream& sout) const
   {
   return sout;
   }

std::istream& awgn::serialize(std::istream& sin)
   {
   return sin;
   }

} // end namespace
