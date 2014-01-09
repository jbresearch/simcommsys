/*!
 * \file
 *
 * Copyright (c) 2010 Stephan Wesemeyer
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

#ifndef GF_FAST_H_
#define GF_FAST_H_

#include "config.h"
#include <iostream>
#include <string>

namespace libbase {

template <int m, int poly>
class gf_fast {
   static const int log_lut[];
   static const int pow_lut[];
public:
   /*! \name Class parameters */
   //! Number of elements in the field
   static int elements()
      {
      return 1 << m;
      }

   //! dimension of the field over GF(2)
   static int dimension()
      {
      return m;
      }
   // @}

private:
   /*! \name Object representation */
   //! Representation of this element by its polynomial coefficients
   int value;
   //! Representation of this element by its power of the primitive element
   int pow_of_alpha;
   // @}

   /*! \name Internal functions */
   void init(int value);
   void init(const char *s);
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   gf_fast(int val = 0)
      {
      this->init(val);
      }
   gf_fast(const char *s)
      {
      init(s);
      }
   // @}

   /*! \name Type conversion */
   operator int() const
      {
      return value;
      }
   operator std::string() const;
   // @}

   /*! \name Arithmetic operations */
   gf_fast& operator+=(const gf_fast& x);
   gf_fast& operator-=(const gf_fast& x);
   gf_fast& operator*=(const gf_fast& x);
   gf_fast& operator/=(const gf_fast& x);
   gf_fast inverse() const;
   gf_fast power(int y) const; //!returns the y-th power of this element
   int log_gf() const//!returns the log of this element wrt \alpha=2, the primitive element
      {
      return this->pow_of_alpha;
      }

   // @}

};

/*! \name Arithmetic operations */

template <int m, int poly>
gf_fast<m, poly> operator+(const gf_fast<m, poly>& a, const gf_fast<m, poly>& b)
   {
   gf_fast<m, poly> c = a;
   return c += b;
   }

template <int m, int poly>
gf_fast<m, poly> operator-(const gf_fast<m, poly>& a, const gf_fast<m, poly>& b)
   {
   gf_fast<m, poly> c = a;
   return c -= b;
   }

template <int m, int poly>
gf_fast<m, poly> operator*(const gf_fast<m, poly>& a, const gf_fast<m, poly>& b)
   {
   gf_fast<m, poly> c = a;
   return c *= b;
   }

template <int m, int poly>
gf_fast<m, poly> operator/(const gf_fast<m, poly>& a, const gf_fast<m, poly>& b)
   {
   gf_fast<m, poly> c = a;
   return c /= b;
   }

// @}

/*! \name Stream Input/Output */

template <int m, int poly>
std::ostream& operator<<(std::ostream& s, const gf_fast<m, poly>& b)
   {
   s << std::string(b);
   return s;
   }

template <int m, int poly>
std::istream& operator>>(std::istream& s, gf_fast<m, poly>& b)
   {
   std::string str;
   s >> str;
   b = str.c_str();
   return s;
   }

// @}

} // end namespace

#endif
