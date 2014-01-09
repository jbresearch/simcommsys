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

#include "logrealfast.h"

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of out-of-range values (infinity and zero)
// 3 - Log difference values and the errors for all LUT access to a file
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

const int logrealfast::lutsize = 1 << 17;
const double logrealfast::lutrange = 12.0;
double *logrealfast::lut_add;
double *logrealfast::lut_sub;
bool logrealfast::lutready = false;
#if DEBUG>=3
std::ofstream logrealfast::file;
#endif

// LUT constructor

void logrealfast::buildlut()
   {
   // set up LUT for addition operation
   lut_add = new double[lutsize];
   for (int i = 0; i < lutsize; i++)
      lut_add[i] = log(1 + exp(-lutrange * i / (lutsize - 1)));
   // set up LUT for subtraction operation
   lut_sub = new double[lutsize];
   for (int i = 0; i < lutsize; i++)
      lut_sub[i] = log(1 - exp(-lutrange * i / (lutsize - 1)));
   // flag that we're done
   lutready = true;
#if DEBUG>=3
   // set up file to log difference and error values for LUT access
   file.open("logrealfast-table.txt");
   file.precision(6);
#endif
   }

// conversion

double logrealfast::convertfromdouble(const double m)
   {
   // trap infinity
   const int inf = isinf(m);
   if (inf < 0)
      {
      failwith("Negative infinity cannot be represented");
      }
   else if (inf > 0)
      {
#if DEBUG>=2
      std::cerr << "DEBUG (logrealfast): +Inf cannot be represented." << std::endl;
#endif
      return -std::numeric_limits<double>::infinity();
      }
   // trap NaN
   else if (isnan(m))
      {
      failwith("NaN cannot be represented");
      }
   // trap negative numbers
   else if (m < 0)
      {
      failwith("Negative numbers cannot be represented");
      }
   // trap zero
   else if (m == 0)
      {
#if DEBUG>=2
      std::cerr << "DEBUG (logrealfast): Zero cannot be represented." << std::endl;
#endif
      return std::numeric_limits<double>::infinity();
      }
   // finally convert (value must be ok)
   return -log(m);
   }

// Input/Output Operations

std::ostream& operator<<(std::ostream& sout, const logrealfast& x)
   {
   // trap infinity
   const int inf = isinf(x.logval);
   if (inf < 0)
      {
      sout << "+Inf";
      }
   else if (inf > 0)
      {
      sout << "0";
      }
   // finite values
   else
      {
      const double lv10 = -x.logval / log(10.0);
      const double exponent = floor(lv10);
      const double mantissa = lv10 - exponent;

      const std::ios::fmtflags flags = sout.flags();
      sout.setf(std::ios::fixed, std::ios::floatfield);
      sout << ::pow(10.0, mantissa);
      sout.setf(std::ios::showpos);
      sout << "e" << int(exponent);
      sout.flags(flags);
      }
   return sout;
   }

std::istream& operator>>(std::istream& sin, logrealfast& x)
   {
   assertalways(sin.good());
   // get the number representation as a string
   using std::string;
   string sval;
   sin >> sval;
   // split into mantissa and exponent
   size_t pos = sval.find('e');
   double mantissa;
   int exponent;
   if (pos != string::npos)
      {
      mantissa = atof(sval.substr(0, pos).c_str());
      exponent = atoi(sval.substr(pos + 1).c_str());
      }
   else
      {
      mantissa = atof(sval.c_str());
      exponent = 0;
      }
   // convert to logvalue
   x.logval = logrealfast::convertfromdouble(mantissa);
   x.logval -= exponent * log(10.0);

   assertalways(sin.good());
   return sin;
   }

} // end namespace
