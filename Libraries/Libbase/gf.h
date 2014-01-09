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
 *
 * \note This class has CUDA device support.
 */

template <int m, int poly>
class gf {
public:
   /*! \name Class parameters */
   //! Number of elements in the field
#ifdef __CUDACC__
   __device__ __host__
#endif
   static int elements()
      {
      return 1 << m;
      }

   //! dimension of the field over GF(2)
#ifdef __CUDACC__
   __device__ __host__
#endif
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
#ifdef __CUDACC__
   __device__ __host__
#endif
   void init(int value)
      {
      assert(m < 32);
      assert(value >= 0 && value < (1 << m));
      gf::value = value;
      }
   void init(const char *s);
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
#ifdef __CUDACC__
   __device__ __host__
#endif
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
#ifdef __CUDACC__
   __device__ __host__
#endif
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
#ifdef __CUDACC__
   __device__ __host__
#endif
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
#ifdef __CUDACC__
   __device__ __host__
#endif
   gf& operator-=(const gf& x)
      {
      value ^= x.value;
      return *this;
      }
   /*!
    * \brief Multiplication
    * \param   x  Field element we want to multiply to this one (ie. multiplicand).
    *
    * Multiplication within extensions of a field is the multiplication of the polynomials
    * representing the two values. This can be done by the usual long-multiplication
    * algorithm. Every time the result overflows, we need to subtract the modular polynomial;
    * for extensions of a binary field, this is achieved by an XOR operation.
    *
    * [cf. Gladman, "A Specification for Rijndael, the AES Algorithm", 2003, pp.3-4]
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   gf& operator*=(const gf& x)
      {
      // Copy the multiplier (A) and multiplicand (B)
      int32u A = value;
      int32u B = x.value;
      // Initialize result
      value = 0;
      // Loop over all bits in multiplicand
      for (int i = 0; i < m && B != 0; i++)
         {
         // If the corresponding bit in the multiplicand is set,
         // add (XOR) the shifted multiplier
         if (B & 1)
            value ^= A;
         // Shift the multiplicand
         B >>= 1;
         // Shift the multiplier, subtracting the polynomial on overflow
         A <<= 1;
         if (A & (1 << m))
            A ^= poly;
         }
      return *this;
      }
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
#ifdef __CUDACC__
   __device__ __host__
#endif
   gf& operator/=(const gf& x)
      {
      return *this *= x.inverse();
      }
   /*!
    * \brief Multiplicative inverse
    *
    * The multiplicative inverse \f$ b^{-1} \f$ of \f$ b \f$ is such that:
    * \f[ b^{-1} a = 1 \f]
    *
    * In this implementation, we use the brute force search method.
    */
#ifdef __CUDACC__
   __device__ __host__
#endif
   gf inverse() const
      {
      const gf<m, poly> one = 1;
      gf<m, poly> result = 1;
      for (int i = 1; i < elements(); i++)
         {
         if (result * *this == one)
            break;
         result *= 2;
         }
      assert(result * *this == one);
      return result;
      }
   // @}

};

/*! \name Arithmetic operations */

template <int m, int poly>
#ifdef __CUDACC__
__device__ __host__
#endif
gf<m, poly> operator+(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c += b;
   }

template <int m, int poly>
#ifdef __CUDACC__
__device__ __host__
#endif
gf<m, poly> operator-(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c -= b;
   }

template <int m, int poly>
#ifdef __CUDACC__
__device__ __host__
#endif
gf<m, poly> operator*(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c *= b;
   }

template <int m, int poly>
#ifdef __CUDACC__
__device__ __host__
#endif
gf<m, poly> operator/(const gf<m, poly>& a, const gf<m, poly>& b)
   {
   gf<m, poly> c = a;
   return c /= b;
   }

// @}

/*! \name Stream Input/Output */

template <int m, int poly>
std::ostream& operator<<(std::ostream& os, const gf<m, poly>& b)
   {
   os << std::string(b);
   return os;
   }

template <int m, int poly>
std::istream& operator>>(std::istream& is, gf<m, poly>& b)
   {
   std::string str;
   // skip any initial whitespace
   is >> libbase::eatwhite;
   // read up to 'm' digits from stream
   char c;
   for (int i = 0; i < m && is.get(c); i++)
      {
      if (isspace(c))
         {
         is.putback(c);
         break;
         }
      str += c;
      }
   // convert
   b = str.c_str();
   return is;
   }

// @}

// Typedefs for explicit instantiations

// Degenerate case GF(2):
typedef gf<1, 0x3> gf2; // 1 { 1 }

// Lin & Costello, 2004, App. A:
typedef gf<2, 0x7> gf4; // 1 { 11 }
typedef gf<3, 0xB> gf8; // 1 { 011 }
typedef gf<4, 0x13> gf16; // 1 { 0011 }
typedef gf<5, 0x25> gf32; // 1 { 0 0101 }
typedef gf<6, 0x43> gf64; // 1 { 00 0011 }
typedef gf<7, 0x89> gf128; // 1 { 000 1001 }
typedef gf<8, 0x11D> gf256; // 1 { 0001 1101 }
typedef gf<9, 0x211> gf512; // 1 { 0 0001 0001 }
typedef gf<10, 0x409> gf1024; // 1 { 00 0000 1001 }

// Rijndael field cf. Gladman, 2003, p.5:
typedef gf<8, 0x11B> gf256aes; // 1 { 0001 1011 }

} // end namespace


// Pre-processor sequence for explicit instantiations

#define GF_TYPE_SEQ \
      (gf2) \
      (gf4) \
      (gf8) \
      (gf16) \
      (gf32) \
      (gf64) \
      (gf128) \
      (gf256) \
      (gf512) \
      (gf1024) \
      (gf256aes)

#endif
