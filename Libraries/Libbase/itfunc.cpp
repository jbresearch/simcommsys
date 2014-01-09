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

#include "itfunc.h"
#include <iostream>
#include <sstream>

namespace libbase {

using std::cerr;
using std::string;

/*! \brief Binary Hamming weight
 */
int weight(int cw)
   {
   int c = cw;
   int w = 0;
   while (c)
      {
      w += (c & 1);
      c >>= 1;
      }
   return w;
   }

/*! \brief Inverse Gray code
 */
int32u igray(int32u n)
   {
   int32u r = n;
   for (int i = 1; i < 32; i <<= 1)
      r ^= r >> i;
   return r;
   }

/*! \brief Greatest common divisor
 * GCD function based on Euclid's algorithm.
 */
int gcd(int a, int b)
   {
   while (b != 0)
      {
      int t = b;
      b = a % b;
      a = t;
      }
   return a;
   }

/*! \brief Converts a string to its hex representation
 */
string hexify(const string input)
   {
   std::ostringstream sout;
   sout << std::hex;
   for (size_t i = 0; i < input.length(); i++)
      {
      sout.width(2);
      sout.fill('0');
      sout << int(int8u(input.at(i)));
      }
   string output = sout.str();
   //trace << "(itfunc) hexify: (" << input << ") = " << output << ", length = " << output.length() << std::endl;
   return output;
   }

/*! \brief Reconstructs a string from its hex representation
 */
string dehexify(const string input)
   {
   string output;
   for (size_t i = 0; i < input.length(); i += 2)
      {
      string s = input.substr(i, 2);
      if (s.length() == 1)
         s += '0';
      output += char(strtoul(s.c_str(), NULL, 16));
      }
   //trace << "(itfunc) dehexify: " << input << " = (" << output << "), length = " << output.length() << std::endl;
   return output;
   }

} // end namespace
