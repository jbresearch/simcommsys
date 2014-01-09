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

#include "gf_fast.h"

namespace libbase {
//Lookup tables for GF(2)
template <> const int gf_fast<1, 0x3>::log_lut[2] = {0, 0};
template <> const int gf_fast<1, 0x3>::pow_lut[2] = {1, 1};

//lookup tables for GF(4)
template <> const int gf_fast<2, 0x7>::log_lut[4] = {0, 0, 1, 2};
template <> const int gf_fast<2, 0x7>::pow_lut[4] = {1, 2, 3, 1};

//lookup tables for GF(8)
template <> const int gf_fast<3, 0xB>::log_lut[8] = {0, 0, 1, 3, 2, 6, 4, 5};
template <> const int gf_fast<3, 0xB>::pow_lut[8] = {1, 2, 4, 3, 6, 7, 5, 1};

//lookup table for GF(16)
template <> const int gf_fast<4, 0x13>::log_lut[16] = {0, 0, 1, 4, 2, 8, 5, 10,
      3, 14, 9, 7, 6, 13, 11, 12};
template <> const int gf_fast<4, 0x13>::pow_lut[16] = {1, 2, 4, 8, 3, 6, 12,
      11, 5, 10, 7, 14, 15, 13, 9, 1};

//lookup tables for GF(32)
template <> const int gf_fast<5, 0x25>::log_lut[32] = {0, 0, 1, 18, 2, 5, 19,
      11, 3, 29, 6, 27, 20, 8, 12, 23, 4, 10, 30, 17, 7, 22, 28, 26, 21, 25, 9,
      16, 13, 14, 24, 15};

template <> const int gf_fast<5, 0x25>::pow_lut[32] = {1, 2, 4, 8, 16, 5, 10,
      20, 13, 26, 17, 7, 14, 28, 29, 31, 27, 19, 3, 6, 12, 24, 21, 15, 30, 25,
      23, 11, 22, 9, 18, 1};

//lookup tables for GF(64)
template <> const int gf_fast<6, 0x43>::log_lut[64] = {0, 0, 1, 6, 2, 12, 7,
      26, 3, 32, 13, 35, 8, 48, 27, 18, 4, 24, 33, 16, 14, 52, 36, 54, 9, 45,
      49, 38, 28, 41, 19, 56, 5, 62, 25, 11, 34, 31, 17, 47, 15, 23, 53, 51,
      37, 44, 55, 40, 10, 61, 46, 30, 50, 22, 39, 43, 29, 60, 42, 21, 20, 59,
      57, 58};

template <> const int
      gf_fast<6, 0x43>::pow_lut[64] = {1, 2, 4, 8, 16, 32, 3, 6, 12, 24, 48,
            35, 5, 10, 20, 40, 19, 38, 15, 30, 60, 59, 53, 41, 17, 34, 7, 14,
            28, 56, 51, 37, 9, 18, 36, 11, 22, 44, 27, 54, 47, 29, 58, 55, 45,
            25, 50, 39, 13, 26, 52, 43, 21, 42, 23, 46, 31, 62, 63, 61, 57, 49,
            33, 1};

//lookup tables for GF(128)
template <> const int gf_fast<7, 0x89>::log_lut[128] = {0, 0, 1, 31, 2, 62, 32,
      103, 3, 7, 63, 15, 33, 84, 104, 93, 4, 124, 8, 121, 64, 79, 16, 115, 34,
      11, 85, 38, 105, 46, 94, 51, 5, 82, 125, 60, 9, 44, 122, 77, 65, 67, 80,
      42, 17, 69, 116, 23, 35, 118, 12, 28, 86, 25, 39, 57, 106, 19, 47, 89,
      95, 71, 52, 110, 6, 14, 83, 92, 126, 30, 61, 102, 10, 37, 45, 50, 123,
      120, 78, 114, 66, 41, 68, 22, 81, 59, 43, 76, 18, 88, 70, 109, 117, 27,
      24, 56, 36, 49, 119, 113, 13, 91, 29, 101, 87, 108, 26, 55, 40, 21, 58,
      75, 107, 54, 20, 74, 48, 112, 90, 100, 96, 97, 72, 98, 53, 73, 111, 99};

template <> const int gf_fast<7, 0x89>::pow_lut[128] = {1, 2, 4, 8, 16, 32, 64,
      9, 18, 36, 72, 25, 50, 100, 65, 11, 22, 44, 88, 57, 114, 109, 83, 47, 94,
      53, 106, 93, 51, 102, 69, 3, 6, 12, 24, 48, 96, 73, 27, 54, 108, 81, 43,
      86, 37, 74, 29, 58, 116, 97, 75, 31, 62, 124, 113, 107, 95, 55, 110, 85,
      35, 70, 5, 10, 20, 40, 80, 41, 82, 45, 90, 61, 122, 125, 115, 111, 87,
      39, 78, 21, 42, 84, 33, 66, 13, 26, 52, 104, 89, 59, 118, 101, 67, 15,
      30, 60, 120, 121, 123, 127, 119, 103, 71, 7, 14, 28, 56, 112, 105, 91,
      63, 126, 117, 99, 79, 23, 46, 92, 49, 98, 77, 19, 38, 76, 17, 34, 68, 1};

//lookup tables for GF(256)
template <> const int gf_fast<8, 0x11D>::log_lut[256] = {0, 0, 1, 25, 2, 50,
      26, 198, 3, 223, 51, 238, 27, 104, 199, 75, 4, 100, 224, 14, 52, 141,
      239, 129, 28, 193, 105, 248, 200, 8, 76, 113, 5, 138, 101, 47, 225, 36,
      15, 33, 53, 147, 142, 218, 240, 18, 130, 69, 29, 181, 194, 125, 106, 39,
      249, 185, 201, 154, 9, 120, 77, 228, 114, 166, 6, 191, 139, 98, 102, 221,
      48, 253, 226, 152, 37, 179, 16, 145, 34, 136, 54, 208, 148, 206, 143,
      150, 219, 189, 241, 210, 19, 92, 131, 56, 70, 64, 30, 66, 182, 163, 195,
      72, 126, 110, 107, 58, 40, 84, 250, 133, 186, 61, 202, 94, 155, 159, 10,
      21, 121, 43, 78, 212, 229, 172, 115, 243, 167, 87, 7, 112, 192, 247, 140,
      128, 99, 13, 103, 74, 222, 237, 49, 197, 254, 24, 227, 165, 153, 119, 38,
      184, 180, 124, 17, 68, 146, 217, 35, 32, 137, 46, 55, 63, 209, 91, 149,
      188, 207, 205, 144, 135, 151, 178, 220, 252, 190, 97, 242, 86, 211, 171,
      20, 42, 93, 158, 132, 60, 57, 83, 71, 109, 65, 162, 31, 45, 67, 216, 183,
      123, 164, 118, 196, 23, 73, 236, 127, 12, 111, 246, 108, 161, 59, 82, 41,
      157, 85, 170, 251, 96, 134, 177, 187, 204, 62, 90, 203, 89, 95, 176, 156,
      169, 160, 81, 11, 245, 22, 235, 122, 117, 44, 215, 79, 174, 213, 233,
      230, 231, 173, 232, 116, 214, 244, 234, 168, 80, 88, 175};

template <> const int gf_fast<8, 0x11D>::pow_lut[256] = {1, 2, 4, 8, 16, 32,
      64, 128, 29, 58, 116, 232, 205, 135, 19, 38, 76, 152, 45, 90, 180, 117,
      234, 201, 143, 3, 6, 12, 24, 48, 96, 192, 157, 39, 78, 156, 37, 74, 148,
      53, 106, 212, 181, 119, 238, 193, 159, 35, 70, 140, 5, 10, 20, 40, 80,
      160, 93, 186, 105, 210, 185, 111, 222, 161, 95, 190, 97, 194, 153, 47,
      94, 188, 101, 202, 137, 15, 30, 60, 120, 240, 253, 231, 211, 187, 107,
      214, 177, 127, 254, 225, 223, 163, 91, 182, 113, 226, 217, 175, 67, 134,
      17, 34, 68, 136, 13, 26, 52, 104, 208, 189, 103, 206, 129, 31, 62, 124,
      248, 237, 199, 147, 59, 118, 236, 197, 151, 51, 102, 204, 133, 23, 46,
      92, 184, 109, 218, 169, 79, 158, 33, 66, 132, 21, 42, 84, 168, 77, 154,
      41, 82, 164, 85, 170, 73, 146, 57, 114, 228, 213, 183, 115, 230, 209,
      191, 99, 198, 145, 63, 126, 252, 229, 215, 179, 123, 246, 241, 255, 227,
      219, 171, 75, 150, 49, 98, 196, 149, 55, 110, 220, 165, 87, 174, 65, 130,
      25, 50, 100, 200, 141, 7, 14, 28, 56, 112, 224, 221, 167, 83, 166, 81,
      162, 89, 178, 121, 242, 249, 239, 195, 155, 43, 86, 172, 69, 138, 9, 18,
      36, 72, 144, 61, 122, 244, 245, 247, 243, 251, 235, 203, 139, 11, 22, 44,
      88, 176, 125, 250, 233, 207, 131, 27, 54, 108, 216, 173, 71, 142, 1};

// Internal functions

/*!
 \brief Initialization
 \param   value Representation of element by its polynomial coefficients

 \todo Validate \c poly - this should be a primitive polynomial [cf. Lin & Costello, 2004, p.41]
 */
template <int m, int poly>
void gf_fast<m, poly>::init(int value)
   {
   assert(m < 32);
   assert(value >= 0 && value < (1 << m));
   this->value = value;
   this->pow_of_alpha = gf_fast<m, poly>::log_lut[this->value];
   }

/*!
 \brief Conversion from string
 \param   s     String representation of element by its polynomial coefficients (binary)

 This function converts the string to an integer and calls init().
 The string must only contain 1's and 0's.
 */
template <int m, int poly>
void gf_fast<m, poly>::init(const char *s)
   {
   int32u value = 0;
   const char *p;
   for (p = s; *p == '1' || *p == '0'; p++)
      {
      value <<= 1;
      if (*p == '1')
         value |= 1;
      }
   assert(*p == '\0');
   this->init(value);
   }

// Conversion operations

template <int m, int poly>
gf_fast<m, poly>::operator std::string() const
   {
   std::string sTemp;
   for (int i = m - 1; i >= 0; i--)
      sTemp += '0' + ((value >> i) & 1);
   return sTemp;
   }

// Arithmetic operations

/*!
 \brief Addition
 \param   x  Field element we want to add to this one.

 Addition within extensions of a field is the addition of the corresponding coefficients
 in the polynomial representation. When the field characteristic is 2 (ie. for extensions
 of a binary field), addition of the coefficients is equivalent to an XOR operation.
 */
template <int m, int poly>
gf_fast<m, poly>& gf_fast<m, poly>::operator+=(const gf_fast<m, poly>& x)
   {
   this->value ^= x.value;
   this ->pow_of_alpha = gf_fast<m, poly>::log_lut[this->value];
   return *this;
   }

/*!
 \brief Subtraction
 \param   x  Field element we want to subtract from this one.

 Subtraction within extensions of a field is the subtraction of the corresponding coefficients
 in the polynomial representation. When the field characteristic is 2 (ie. for extensions
 of a binary field), subtraction of the coefficients is equivalent to an XOR operation, and
 therefore equivalent to addition.
 */
template <int m, int poly>
gf_fast<m, poly>& gf_fast<m, poly>::operator-=(const gf_fast<m, poly>& x)
   {
   value ^= x.value;
   this ->pow_of_alpha = gf_fast<m, poly>::log_lut[this->value];
   return *this;
   }

/*!
 \brief Multiplication
 \param   x  Field element we want to multiply to this one (ie. multiplicand).

 Multiplication is the equivalent of summing the powers of the elements

 [cf. Gladman, "A Specification for Rijndael, the AES Algorithm", 2003, pp.3-4]
 */
template <int m, int poly>
gf_fast<m, poly>& gf_fast<m, poly>::operator*=(const gf_fast<m, poly>& x)
   {
   //only  do something if we are not 0
   if (0 != this->value)
      {
      //are we multiplying by 0
      if (0 == x)
         {
         this->value = 0;
         this->pow_of_alpha = 0; //by convention
         }
      else
         {
         this->pow_of_alpha = (this->pow_of_alpha + x.log_gf())
               % (this->elements() - 1);
         this->value = gf_fast<m, poly>::pow_lut[this->pow_of_alpha];
         }
      }
   return *this;
   }

/*!
 \brief Division
 \param   x  Field element we want to divide this one by (i.e. divisor).
 Division is done by finding the difference of the powers

 */
template <int m, int poly>
gf_fast<m, poly>& gf_fast<m, poly>::operator/=(const gf_fast<m, poly>& x)
   {
   //ensure we do not divide by 0
   assertalways(0!=x);
   //only need to do any work if we are not 0
   if (0 != this->value)
      {
      this->pow_of_alpha += (this->elements() - 1);
      this->pow_of_alpha = (this->pow_of_alpha - x.log_gf())
            % (this->elements() - 1);
      this->value = gf_fast<m, poly>::pow_lut[this->pow_of_alpha];

      }
   return *this;
   }

/*!
 \brief Multiplicative inverse

 The multiplicative inverse \f$ b^{-1} \f$ of \f$ b \f$ is such that:
 \f[ b^{-1} a = 1 \f]
 In this method we simply use the power of alpha to determine its inverse

 */
template <int m, int poly>
gf_fast<m, poly> gf_fast<m, poly>::inverse() const
   {
   //0 does not have an inverse
   assertalways(0!=this->value);
   int tmp = (this->elements() - 1 - this->pow_of_alpha);
   return gf_fast<m, poly> (gf_fast<m, poly>::pow_lut[tmp]);
   }

/*!
 *  \brief power function
 *  This function simply works out the product of the powers mod (2^m-1)
 */
template <int m, int poly>
gf_fast<m, poly> gf_fast<m, poly>::power(int y) const
   {
   int pow = (this->pow_of_alpha * y) % (gf_fast<m, poly>::elements() - 1);
   if (pow < 0)
      {
      //Ensure the power is positive
      pow += (gf_fast<m, poly>::elements() - 1);
      }
   gf_fast<m, poly> tmp(this->pow_lut[pow]);
   return tmp;
   }

// Explicit Realizations

// Degenerate case GF(2)

template class gf_fast<1, 0x3> ; // 1 { 1 }

// cf. Lin & Costello, 2004, App. A

template class gf_fast<2, 0x7> ; // 1 { 11 }
template class gf_fast<3, 0xB> ; // 1 { 011 }
template class gf_fast<4, 0x13> ; // 1 { 0011 }
template class gf_fast<5, 0x25> ; // 1 { 0 0101 }
template class gf_fast<6, 0x43> ; // 1 { 00 0011 }
template class gf_fast<7, 0x89> ; // 1 { 000 1001 }
template class gf_fast<8, 0x11D> ; // 1 { 0001 1101 }
/*
 template class gf_fast<9, 0x211> ; // 1 { 0 0001 0001 }
 template class gf_fast<10, 0x409> ; // 1 { 00 0000 1001 }

 // Rijndael field cf. Gladman, 2003, p.5

 template class gf_fast<8, 0x11B> ; // 1 { 0001 1011 }
 */
} // end namespace


