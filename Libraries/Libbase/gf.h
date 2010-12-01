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

#ifndef __gf_h
#define __gf_h

#include "config.h"
#include <iostream>
#include <string>

namespace libbase {

/*!
 * \brief   Galois Field Element.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements extensions of the binary field: \f$ GF(2^n) \f$.
 *
 * Realizations:
 * - gf<8,283> Rijndael,
 * - gf<2>..gf<10> Lin & Costello
 *
 * \param   m     Order of the binary field extension; that is, the field will
 * be \f$ GF(2^m) \f$.
 * \param   poly  Primitive polynomial used to define the field elements
 *
 * In integer representations of polynomials (e.g \c poly), higher-order bits in
 * the integer represent higher-order powers of the polynomial representation.
 * For example:
 * \f[ x^6 + x^4 + x^2 + x^1 + 1 = \{ 01010111 \}_2 = \{ 57 \}_16 = \{ 87 \}_10 \f]
 *
 * \warning Due to the internal representation, this class is limited to
 * \f$ GF(2^31) \f$.
 */

template <int m, int poly>
class gf {
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
   // @}

   /*! \name Internal functions */
   /*!
    * \brief Initialization
    * \param   value Representation of element by its polynomial coefficients
    *
    * \todo Validate \c poly - this should be a primitive polynomial [cf. Lin & Costello, 2004, p.41]
    */
   void init(int value)
      {
      assert(m < 32);
      assert(value >=0 && value < (1<<m));
      gf::value = value;
      }
   void init(const char *s);
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   gf(int value = 0)
      {
      init(value);
      }
   gf(const char *s)
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
   /*!
    * \brief Addition
    * \param   x  Field element we want to add to this one.
    *
    * Addition within extensions of a field is the addition of the corresponding coefficients
    * in the polynomial representation. When the field characteristic is 2 (ie. for extensions
    * of a binary field), addition of the coefficients is equivalent to an XOR operation.
    */
   gf& operator+=(const gf& x)
      {
      value ^= x.value;
      return *this;
      }
   /*!
    * \brief Subtraction
    * \param   x  Field element we want to subtract from this one.
    *
    * Subtraction within extensions of a field is the subtraction of the corresponding coefficients
    * in the polynomial representation. When the field characteristic is 2 (ie. for extensions
    * of a binary field), subtraction of the coefficients is equivalent to an XOR operation, and
    * therefore equivalent to addition.
    */
   gf& operator-=(const gf& x)
      {
      value ^= x.value;
      return *this;
      }
   gf& operator*=(const gf& x);
   /*!
    * \brief Division
    * \param   x  Field element we want to divide this one by (i.e. divisor).
    *
    * Division in a finite field can be performed by:
    * - finding the multiplicative inverse, and multiplying by it
    * - obtaining the logarithms of the two values, performing a subtraction, and
    * then computing the inverse logarithm
    *
    * In this implementation, we use the multiplicatve inverse method.
    */
   gf& operator/=(const gf& x)
      {
      return *this *= x.inverse();
      }
   gf inverse() const;
   // @}

};

/*! \name Arithmetic operations */

template <int m, int poly>
gf<m, poly> operator+(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c += b;
   }

template <int m, int poly>
gf<m, poly> operator-(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c -= b;
   }

template <int m, int poly>
gf<m, poly> operator*(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c *= b;
   }

template <int m, int poly>
gf<m, poly> operator/(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c /= b;
   }

// @}

/*! \name Stream Input/Output */

template <int m, int poly>
std::ostream& operator<<(std::ostream& s, const gf<m, poly>& b)
   {
   s << std::string(b);
   return s;
   }

template <int m, int poly>
std::istream& operator>>(std::istream& s, gf<m, poly>& b)
   {
   std::string str;
   s >> str;
   b = str.c_str();
   return s;
   }

// @}

} // end namespace

#endif
