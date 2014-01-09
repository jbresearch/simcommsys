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

#include "sigspace.h"

namespace libcomm {

// stream input / output

std::ostream& operator<<(std::ostream& s, const sigspace& x)
   {
   using std::ios;
   s.setf(ios::fixed, ios::floatfield);
   s.precision(6);
   s << '[' << x.inphase << ',';
   s.setf(ios::fixed, ios::floatfield);
   s.precision(6);
   s << x.quad << ']';
   return s;
   }

std::istream& operator>>(std::istream& s, sigspace& x)
   {
   double i = 0, q = 0;
   char c = 0;

   s >> c;
   if (c == '[')
      {
      s >> i >> c;
      if (c == ',')
         s >> q >> c;
      else
         s.clear(std::ios::failbit);
      if (c != ']')
         s.clear(std::ios::failbit);
      }
   else
      s.clear(std::ios::failbit);

   if (s)
      x = sigspace(i, q);

   return s;
   }

} // end namespace
